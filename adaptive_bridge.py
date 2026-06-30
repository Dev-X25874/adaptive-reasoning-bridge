import torch
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import logging
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdaptiveReasoningBridge:
    """
    State-of-the-Art Single-File Controller implementing:
    1. Wait-Style Budget Forcing (s1 paper) 
    2. Sectional Entropy-Based Adaptive Scaling [10, 27]
    3. Verifiable Reasoning Format Control 
    """
    
    def __init__(
        self, 
        model_name: str, 
        tp_size: int = 1,
        entropy_threshold: float = 1.35,
        max_context_length: int = 32768
    ):
        """
        Initialize the Adaptive Reasoning Bridge.
        
        Args:
            model_name: HuggingFace model identifier
            tp_size: Tensor parallel size for vLLM
            entropy_threshold: Threshold for adaptive scaling
            max_context_length: Maximum context window size
        """
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = LLM(
            model=model_name, 
            tensor_parallel_size=tp_size,
            trust_remote_code=True
        )
        self.entropy_threshold = entropy_threshold
        self.max_context_length = max_context_length
        
        # Control tokens as defined in Open-R1 and s1 [16, 20]
        self.THINK_START = "<|im_start|>think"
        self.THINK_END = "<|im_end|>"
        self.WAIT_SIGNAL = "\nWait, let me rethink the previous step for potential errors..."

    def _calculate_entropy(self, logprobs_list: List[Dict]) -> float:
        """
        Calculates Shannon Entropy as an uncertainty proxy. 
        H = -sum(p * log(p))
        
        Args:
            logprobs_list: List of token logprobs from vLLM
            
        Returns:
            Mean entropy across all tokens
        """
        if not logprobs_list:
            return 0.0
            
        entropies = []
        for token_logprobs in logprobs_list:
            if not token_logprobs:
                continue
                
            # vLLM logprobs are {token_id: log_prob}
            probs = np.exp(list(token_logprobs.values()))
            probs = probs / np.sum(probs)  # Re-normalize
            entropy = -np.sum(probs * np.log(probs + 1e-9))
            entropies.append(entropy)
            
        return float(np.mean(entropies)) if entropies else 0.0

    def _sectional_analysis(self, logprobs: List[Dict]) -> bool:
        """
        Implements Sectional Entropy Analysis (0.2, 0.6, 0.2 weights). 
        Heuristic: High uncertainty in the 'middle' reasoning requires scaling.
        
        Args:
            logprobs: List of token logprobs
            
        Returns:
            True if weighted entropy exceeds threshold
        """
        if not logprobs or len(logprobs) < 10:
            return False
        
        split = len(logprobs) // 3
        sec1 = self._calculate_entropy(logprobs[:split])
        sec2 = self._calculate_entropy(logprobs[split:2*split])
        sec3 = self._calculate_entropy(logprobs[2*split:])
        
        weighted_score = (0.2 * sec1) + (0.6 * sec2) + (0.2 * sec3)
        logger.debug(f"Sectional entropy: {weighted_score:.3f} (threshold: {self.entropy_threshold})")
        
        return weighted_score > self.entropy_threshold

    def _check_context_length(self, text: str) -> bool:
        """Check if text exceeds maximum context length."""
        tokens = self.tokenizer.encode(text)
        return len(tokens) < self.max_context_length

    def generate_intellect(
        self, 
        query: str, 
        budget_limit: int = 2,
        max_new_tokens: int = 4096,
        temperature: float = 0.8,
        top_p: float = 0.95
    ) -> Dict:
        """
        Core reasoning loop with iterative budget forcing. 
        
        Args:
            query: User query/problem to solve
            budget_limit: Maximum reasoning iterations
            max_new_tokens: Max tokens per generation
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Dictionary containing thought process, answer, and metadata
        """
        # Construct the initial R1-style prompt 
        prompt = f"<|im_start|>user\n{query}\n<|im_end|>\n{self.THINK_START}\n"
        full_thought_trace = ""
        current_step = 0
        
        # Sampling configuration for reasoning [34, 35]
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            logprobs=5,
            stop=[self.THINK_END]
        )

        logger.info(f"Starting reasoning with budget limit: {budget_limit}")

        while current_step < budget_limit:
            try:
                # Check context length before generation
                if not self._check_context_length(prompt):
                    logger.warning("Context length exceeded, stopping early")
                    break
                
                outputs = self.llm.generate([prompt], sampling_params)
                gen_text = outputs[0].outputs[0].text
                logprobs = outputs[0].outputs[0].logprobs
                
                if not gen_text:
                    logger.warning("Empty generation, stopping")
                    break
                
                full_thought_trace += gen_text
                current_step += 1
                
                logger.info(f"Step {current_step}/{budget_limit} completed")
                
                # Decision: Should we force more thinking? [23, 26]
                # Continue if: budget remaining AND high uncertainty detected
                should_continue = (
                    current_step < budget_limit and 
                    logprobs is not None and 
                    self._sectional_analysis(logprobs)
                )
                
                if should_continue:
                    prompt += gen_text + self.WAIT_SIGNAL
                    logger.info("High entropy detected, forcing additional reasoning")
                else:
                    prompt += gen_text
                    break
                    
            except Exception as e:
                logger.error(f"Generation error at step {current_step}: {e}")
                break

        # Final extraction phase [16, 34]
        logger.info("Extracting final answer")
        final_prompt = f"{prompt}\n{self.THINK_END}\n<|im_start|>assistant\n"
        final_params = SamplingParams(temperature=0.1, max_tokens=1024)
        
        try:
            final_output = self.llm.generate([final_prompt], final_params)
            final_answer = final_output[0].outputs[0].text
        except Exception as e:
            logger.error(f"Final extraction error: {e}")
            final_answer = "Error generating final answer"
        
        return {
            "query": query,
            "thought": full_thought_trace,
            "answer": final_answer,
            "steps_scaled": current_step,
            "budget_limit": budget_limit
        }

    def generate_simple(
        self,
        query: str,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> str:
        """
        Simple generation without adaptive reasoning (baseline).
        
        Args:
            query: User query
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response
        """
        prompt = f"<|im_start|>user\n{query}\n<|im_end|>\n<|im_start|>assistant\n"
        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text

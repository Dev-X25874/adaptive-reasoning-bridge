"""
Basic usage examples for Adaptive Reasoning Bridge
"""

import sys
sys.path.append('..')

from adaptive_bridge import AdaptiveReasoningBridge
import json

def example_math_problem():
    """Example: Complex math reasoning"""
    print("=" * 80)
    print("Example 1: Math Problem with Adaptive Reasoning")
    print("=" * 80)
    
    bridge = AdaptiveReasoningBridge(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        tp_size=1,
        entropy_threshold=1.35
    )
    
    query = """
    A train leaves Station A at 2:00 PM traveling at 60 mph toward Station B.
    Another train leaves Station B at 2:30 PM traveling at 80 mph toward Station A.
    The stations are 350 miles apart. At what time will the trains meet?
    """
    
    result = bridge.generate_intellect(
        query=query,
        budget_limit=3,
        max_new_tokens=2048
    )
    
    print(f"\nQuery: {result['query']}")
    print(f"\nThought Process ({result['steps_scaled']} steps):")
    print("-" * 80)
    print(result['thought'])
    print("\n" + "=" * 80)
    print("Final Answer:")
    print("-" * 80)
    print(result['answer'])
    print("\n")

def example_logic_puzzle():
    """Example: Logic puzzle requiring multi-step reasoning"""
    print("=" * 80)
    print("Example 2: Logic Puzzle")
    print("=" * 80)
    
    bridge = AdaptiveReasoningBridge(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    )
    
    query = """
    In a room, there are 3 light switches. Each switch controls one of three light bulbs 
    in another room. You cannot see the bulbs from the switch room. You can flip the 
    switches as many times as you want, but you can only enter the bulb room once. 
    How can you determine which switch controls which bulb?
    """
    
    result = bridge.generate_intellect(query=query, budget_limit=2)
    
    print(f"\nSteps taken: {result['steps_scaled']}")
    print(f"\nFinal Answer:\n{result['answer']}\n")

def example_code_debugging():
    """Example: Code debugging with reasoning"""
    print("=" * 80)
    print("Example 3: Code Debugging")
    print("=" * 80)
    
    bridge = AdaptiveReasoningBridge(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        entropy_threshold=1.2  # Lower threshold for code tasks
    )
    
    query = """
    Debug this Python function:
    
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    
    The function works but is extremely slow for large n. 
    Explain why and provide an optimized version.
    """
    
    result = bridge.generate_intellect(query=query, budget_limit=2)
    
    print(f"\nAnswer:\n{result['answer']}\n")

def example_comparison():
    """Example: Compare adaptive vs simple generation"""
    print("=" * 80)
    print("Example 4: Adaptive vs Simple Generation Comparison")
    print("=" * 80)
    
    bridge = AdaptiveReasoningBridge(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    )
    
    query = "What is 15% of 240?"
    
    # Simple generation
    print("\n[Simple Generation]")
    simple_answer = bridge.generate_simple(query, max_tokens=512)
    print(simple_answer)
    
    # Adaptive reasoning
    print("\n[Adaptive Reasoning]")
    adaptive_result = bridge.generate_intellect(query, budget_limit=1)
    print(f"Answer: {adaptive_result['answer']}")
    print(f"Steps: {adaptive_result['steps_scaled']}")

def save_results_to_file():
    """Example: Save results to JSON"""
    print("=" * 80)
    print("Example 5: Saving Results")
    print("=" * 80)
    
    bridge = AdaptiveReasoningBridge(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    )
    
    queries = [
        "What is 25 * 36?",
        "Explain quantum entanglement in simple terms",
        "Write a haiku about artificial intelligence"
    ]
    
    results = []
    for query in queries:
        result = bridge.generate_intellect(query, budget_limit=2)
        results.append(result)
    
    # Save to file
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to results.json")

if __name__ == "__main__":
    # Run examples
    print("\n🚀 Adaptive Reasoning Bridge - Examples\n")
    
    # Uncomment the examples you want to run:
    example_math_problem()
    # example_logic_puzzle()
    # example_code_debugging()
    # example_comparison()
    # save_results_to_file()

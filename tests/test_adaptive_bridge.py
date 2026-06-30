"""
Unit tests for Adaptive Reasoning Bridge
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
sys.path.append('..')

from adaptive_bridge import AdaptiveReasoningBridge

class TestAdaptiveReasoningBridge:
    
    @pytest.fixture
    def mock_bridge(self):
        """Create a mocked bridge for testing"""
        with patch('adaptive_bridge.LLM'), \
             patch('adaptive_bridge.AutoTokenizer'):
            bridge = AdaptiveReasoningBridge(
                model_name="test-model",
                entropy_threshold=1.35
            )
            return bridge
    
    def test_initialization(self, mock_bridge):
        """Test bridge initialization"""
        assert mock_bridge.entropy_threshold == 1.35
        assert mock_bridge.THINK_START == "<|im_start|>think"
        assert mock_bridge.THINK_END == "<|im_end|>"
    
    def test_calculate_entropy_empty(self, mock_bridge):
        """Test entropy calculation with empty input"""
        result = mock_bridge._calculate_entropy([])
        assert result == 0.0
    
    def test_calculate_entropy_single_token(self, mock_bridge):
        """Test entropy calculation with single token"""
        logprobs = [
            {1: -0.5, 2: -1.0, 3: -2.0}
        ]
        result = mock_bridge._calculate_entropy(logprobs)
        assert result > 0.0
        assert isinstance(result, float)
    
    def test_calculate_entropy_multiple_tokens(self, mock_bridge):
        """Test entropy calculation with multiple tokens"""
        logprobs = [
            {1: -0.1, 2: -0.2, 3: -0.3},
            {4: -0.5, 5: -0.5, 6: -2.0},
            {7: -1.0, 8: -1.0, 9: -1.0}
        ]
        result = mock_bridge._calculate_entropy(logprobs)
        assert result > 0.0
        assert isinstance(result, float)
    
    def test_sectional_analysis_empty(self, mock_bridge):
        """Test sectional analysis with empty input"""
        result = mock_bridge._sectional_analysis([])
        assert result is False
    
    def test_sectional_analysis_short_sequence(self, mock_bridge):
        """Test sectional analysis with short sequence"""
        logprobs = [{1: -0.5}] * 5
        result = mock_bridge._sectional_analysis(logprobs)
        assert result is False
    
    def test_sectional_analysis_low_entropy(self, mock_bridge):
        """Test sectional analysis with low entropy (confident predictions)"""
        # Low entropy = confident predictions
        logprobs = [{1: -0.01, 2: -10.0, 3: -10.0}] * 30
        result = mock_bridge._sectional_analysis(logprobs)
        assert result is False
    
    def test_sectional_analysis_high_entropy(self, mock_bridge):
        """Test sectional analysis with high entropy (uncertain predictions)"""
        # High entropy = uncertain predictions
        logprobs = [{1: -1.0, 2: -1.0, 3: -1.0, 4: -1.0}] * 30
        result = mock_bridge._sectional_analysis(logprobs)
        assert result is True
    
    def test_context_length_check(self, mock_bridge):
        """Test context length checking"""
        mock_bridge.tokenizer.encode = Mock(return_value=list(range(100)))
        mock_bridge.max_context_length = 1000
        
        result = mock_bridge._check_context_length("test text")
        assert result is True
        
        mock_bridge.tokenizer.encode = Mock(return_value=list(range(2000)))
        result = mock_bridge._check_context_length("test text")
        assert result is False
    
    def test_generate_intellect_basic(self, mock_bridge):
        """Test basic generate_intellect call"""
        # Mock the LLM generate method
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock()]
        mock_output.outputs[0].text = "This is a test response"
        mock_output.outputs[0].logprobs = [{1: -0.5, 2: -1.0}] * 20
        
        mock_bridge.llm.generate = Mock(return_value=[mock_output])
        mock_bridge._check_context_length = Mock(return_value=True)
        
        result = mock_bridge.generate_intellect(
            query="Test query",
            budget_limit=1
        )
        
        assert "query" in result
        assert "thought" in result
        assert "answer" in result
        assert "steps_scaled" in result
        assert result["query"] == "Test query"
    
    def test_generate_intellect_budget_exhausted(self, mock_bridge):
        """Test that budget limit is respected"""
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock()]
        mock_output.outputs[0].text = "Response"
        mock_output.outputs[0].logprobs = [{1: -1.0, 2: -1.0}] * 30  # High entropy
        
        mock_bridge.llm.generate = Mock(return_value=[mock_output])
        mock_bridge._check_context_length = Mock(return_value=True)
        
        result = mock_bridge.generate_intellect(
            query="Test",
            budget_limit=2
        )
        
        # Should stop at budget limit even with high entropy
        assert result["steps_scaled"] <= 2
    
    def test_generate_simple(self, mock_bridge):
        """Test simple generation without reasoning"""
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock()]
        mock_output.outputs[0].text = "Simple answer"
        
        mock_bridge.llm.generate = Mock(return_value=[mock_output])
        
        result = mock_bridge.generate_simple(
            query="What is 2+2?",
            max_tokens=512
        )
        
        assert result == "Simple answer"
    
    def test_error_handling_empty_generation(self, mock_bridge):
        """Test handling of empty generation"""
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock()]
        mock_output.outputs[0].text = ""
        mock_output.outputs[0].logprobs = []
        
        mock_bridge.llm.generate = Mock(return_value=[mock_output])
        mock_bridge._check_context_length = Mock(return_value=True)
        
        result = mock_bridge.generate_intellect(query="Test")
        
        # Should handle empty generation gracefully
        assert "answer" in result
    
    def test_error_handling_none_logprobs(self, mock_bridge):
        """Test handling of None logprobs"""
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock()]
        mock_output.outputs[0].text = "Response"
        mock_output.outputs[0].logprobs = None
        
        mock_bridge.llm.generate = Mock(return_value=[mock_output])
        mock_bridge._check_context_length = Mock(return_value=True)
        
        result = mock_bridge.generate_intellect(query="Test", budget_limit=2)
        
        # Should not crash with None logprobs
        assert result is not None
    
    def test_custom_parameters(self, mock_bridge):
        """Test custom generation parameters"""
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock()]
        mock_output.outputs[0].text = "Response"
        mock_output.outputs[0].logprobs = [{1: -0.5}] * 20
        
        mock_bridge.llm.generate = Mock(return_value=[mock_output])
        mock_bridge._check_context_length = Mock(return_value=True)
        
        result = mock_bridge.generate_intellect(
            query="Test",
            budget_limit=5,
            max_new_tokens=8192,
            temperature=0.9,
            top_p=0.99
        )
        
        assert result["budget_limit"] == 5

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

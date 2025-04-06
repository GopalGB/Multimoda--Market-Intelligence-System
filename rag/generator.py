# rag/generator.py
import torch
import os
import logging
import json
import time
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer
)
import threading
from queue import Queue

logger = logging.getLogger(__name__)

class StopOnTokens(StoppingCriteria):
    """Stopping criteria based on specific tokens being generated."""
    def __init__(self, stop_token_ids: List[int]):
        self.stop_token_ids = stop_token_ids
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

class ContentGenerator:
    """
    Generator component for the RAG system.
    
    This class provides methods to:
    - Generate content based on context from the retriever
    - Format prompts with context
    - Control generation parameters
    - Provide chat-like interactions
    - Support streaming generation
    
    It uses language models optimized for content generation with
    context-aware prompting to produce high-quality results.
    """
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3-8B-Instruct",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        max_new_tokens: int = 512,
        prompt_template: Optional[str] = None
    ):
        """
        Initialize the content generator.
        
        Args:
            model_name: Name of the language model to use
            cache_dir: Directory for caching models
            device: Device to run the model on (cpu, cuda:0, etc.)
            load_in_8bit: Whether to load model in 8-bit precision
            max_new_tokens: Default maximum tokens to generate
            prompt_template: Custom prompt template (None for model-specific default)
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_new_tokens = max_new_tokens
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Set model loading parameters
        self.load_in_8bit = load_in_8bit
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self._initialize_model()
        
        # Set prompt template
        self.prompt_template = prompt_template or self._get_default_prompt_template()
        
        # Check for streaming compatibility
        self.streaming_compatible = hasattr(self.model.generate, "__kwdefaults__") and "streamer" in self.model.generate.__kwdefaults__
        
        # Load stop tokens
        self.stop_token_ids = self._get_stop_token_ids()
    
    def _initialize_model(self) -> None:
        """Initialize the language model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                padding_side="left"
            )
            
            # Ensure tokenizer has pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate quantization
            if self.load_in_8bit:
                # Load model in 8-bit precision
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    device_map=self.device,
                    load_in_8bit=True,
                    torch_dtype=torch.float16
                )
            else:
                # Load model in default precision
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    device_map=self.device
                )
            
            logger.info(f"Model {self.model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _get_default_prompt_template(self) -> str:
        """Get the default prompt template based on model type."""
        # Detect model type for appropriate template
        model_type = self.model_name.lower()
        
        if "llama-3" in model_type:
            # Llama 3 Chat template
            return """<|begin_of_text|><|user|>
{query}

{context}

Please provide your response based on the information above.
<|assistant|>
{response}"""
        
        elif "llama-2" in model_type:
            # Llama 2 Chat template
            return """<s>[INST] 
{query}

{context}

Please provide a response based on the information above. 
[/INST] 
{response}"""
        
        elif "mistral" in model_type:
            # Mistral Instruct template
            return """<s>[INST] 
{query}

{context}

Using the context above, answer the query.
[/INST]
{response}"""
        
        elif "vicuna" in model_type or "wizardlm" in model_type:
            # Vicuna/WizardLM template
            return """USER: {query}

Context information:
{context}

Please provide a response based on the context information.

A: {response}"""
        
        else:
            # Default template
            return """User: {query}

Context:
{context}

Please respond to the query using the context above.
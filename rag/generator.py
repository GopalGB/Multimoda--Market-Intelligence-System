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

Please respond to the query using the context above."""

    def _get_stop_token_ids(self) -> List[int]:
        """
        Get stop token IDs for the model.
        
        Returns:
            List of token IDs to use as stopping criteria
        """
        # Get common stop tokens based on the model type
        stop_tokens = []
        
        # Add model-specific stop tokens
        model_type = self.model_name.lower()
        
        if "llama" in model_type:
            # Llama models
            stop_words = ["</s>", "<|end_of_text|>", "<|im_end|>", "<|endoftext|>"]
            stop_tokens.extend([self.tokenizer.eos_token_id])
        elif "mistral" in model_type:
            # Mistral models
            stop_words = ["</s>", "<|im_end|>"]
            stop_tokens.extend([self.tokenizer.eos_token_id])
        else:
            # Default stop tokens
            stop_tokens.extend([self.tokenizer.eos_token_id])
            stop_words = ["</s>", "<|endoftext|>"]
        
        # Convert stop words to token IDs
        for word in stop_words:
            try:
                ids = self.tokenizer.encode(word, add_special_tokens=False)
                if len(ids) == 1:  # Only add single token markers
                    stop_tokens.append(ids[0])
            except:
                pass
        
        # Remove duplicates and return
        return list(set(stop_tokens))
    
    def format_prompt(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
        context_format: str = "{text}",
        context_separator: str = "\n\n",
        max_context_length: int = 3800
    ) -> str:
        """
        Format a prompt with query and context information.
        
        Args:
            query: User query
            contexts: List of context dictionaries from the retriever
            context_format: Format string for each context item
            context_separator: Separator between context items
            max_context_length: Maximum character length for combined contexts
            
        Returns:
            Formatted prompt string
        """
        # Format each context item
        formatted_contexts = []
        total_length = 0
        
        for ctx in contexts:
            # Extract text from context
            if "text" in ctx:
                text = ctx["text"]
            elif "content" in ctx.get("document", {}):
                text = ctx["document"]["content"]
            elif "text" in ctx.get("document", {}):
                text = ctx["document"]["text"]
            elif "body" in ctx.get("document", {}):
                text = ctx["document"]["body"]
            else:
                # Use the entire context as fallback
                text = str(ctx)
            
            # Format the context
            formatted_ctx = context_format.format(text=text, score=ctx.get("score", 1.0))
            
            # Check if adding this context would exceed the maximum length
            if total_length + len(formatted_ctx) > max_context_length:
                # If we already have some contexts, break to use what we have
                if formatted_contexts:
                    break
                # Otherwise, truncate this context
                formatted_ctx = formatted_ctx[:max_context_length]
            
            formatted_contexts.append(formatted_ctx)
            total_length += len(formatted_ctx)
        
        # Combine contexts
        combined_context = context_separator.join(formatted_contexts)
        
        # Format the full prompt
        prompt = self.prompt_template.format(
            query=query,
            context=combined_context,
            response=""
        )
        
        return prompt
    
    def generate(
        self,
        query: str,
        contexts: List[Dict[str, Any]] = None,
        prompt: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        return_prompt: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate content based on query and contexts.
        
        Args:
            query: User query
            contexts: List of context dictionaries from the retriever
            prompt: Pre-formatted prompt (will be used instead of query+contexts if provided)
            generation_kwargs: Additional keyword arguments for generation
            return_prompt: Whether to return the prompt in the response
            
        Returns:
            Generated text or dictionary with generation details
        """
        # Default contexts if None
        contexts = contexts or []
        
        # Format prompt if not provided
        if prompt is None:
            prompt = self.format_prompt(query, contexts)
        
        # Set up generation parameters
        params = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "num_return_sequences": 1
        }
        
        # Update with user-provided parameters
        if generation_kwargs:
            params.update(generation_kwargs)
        
        # Set up stopping criteria
        stopping_criteria = StoppingCriteriaList([StopOnTokens(self.stop_token_ids)])
        params["stopping_criteria"] = stopping_criteria
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            start_time = time.time()
            
            output = self.model.generate(**inputs, **params)
            
            # Calculate generation time
            gen_time = time.time() - start_time
        
        # Decode output
        output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract response by removing the prompt
        response = output_text[len(self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):].strip()
        
        # Return result
        if return_prompt:
            return {
                "response": response,
                "prompt": prompt,
                "generation_time": gen_time,
                "num_tokens": len(output[0]) - len(inputs.input_ids[0])
            }
        else:
            return response
    
    def stream_generate(
        self,
        query: str,
        contexts: List[Dict[str, Any]] = None,
        prompt: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Generate content with streaming output.
        
        Args:
            query: User query
            contexts: List of context dictionaries from the retriever
            prompt: Pre-formatted prompt (will be used instead of query+contexts if provided)
            generation_kwargs: Additional keyword arguments for generation
            callback: Function to call with each token as it's generated
            
        Returns:
            Complete generated text
        """
        if not self.streaming_compatible:
            logger.warning("Streaming not fully supported for this model, falling back to non-streaming")
            return self.generate(query, contexts, prompt, generation_kwargs)
        
        # Default contexts if None
        contexts = contexts or []
        
        # Format prompt if not provided
        if prompt is None:
            prompt = self.format_prompt(query, contexts)
        
        # Set up generation parameters
        params = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "do_sample": True
        }
        
        # Update with user-provided parameters
        if generation_kwargs:
            params.update(generation_kwargs)
        
        # Set up stopping criteria
        stopping_criteria = StoppingCriteriaList([StopOnTokens(self.stop_token_ids)])
        params["stopping_criteria"] = stopping_criteria
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Set up streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=10.0
        )
        params["streamer"] = streamer
        
        # Generate in a separate thread
        generation_thread = threading.Thread(
            target=self._generate_with_streamer,
            args=(inputs, params)
        )
        generation_thread.start()
        
        # Collect streamed output
        generated_text = ""
        
        # Process stream
        for text in streamer:
            generated_text += text
            
            # Call callback if provided
            if callback:
                callback(text)
        
        # Wait for generation to complete
        generation_thread.join()
        
        return generated_text
    
    def _generate_with_streamer(
        self,
        inputs: Dict[str, torch.Tensor],
        params: Dict[str, Any]
    ) -> None:
        """
        Helper method to run generation in a separate thread for streaming.
        
        Args:
            inputs: Tokenized inputs
            params: Generation parameters
        """
        with torch.no_grad():
            self.model.generate(**inputs, **params)
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        contexts: Optional[List[Dict[str, Any]]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        return_prompt: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate a response in a chat conversation with context.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            contexts: List of context dictionaries from the retriever
            generation_kwargs: Additional keyword arguments for generation
            return_prompt: Whether to return the prompt in the response
            
        Returns:
            Generated text or dictionary with generation details
        """
        # Extract the latest user query from messages
        user_queries = [msg['content'] for msg in messages if msg['role'] == 'user']
        
        if not user_queries:
            raise ValueError("No user messages found in chat history")
        
        # Use the most recent user query
        query = user_queries[-1]
        
        # Format chat history for the prompt
        chat_history = self._format_chat_history(messages)
        
        # Format context information
        context_info = ""
        if contexts:
            context_texts = []
            for ctx in contexts:
                if "text" in ctx:
                    context_texts.append(ctx["text"])
                elif "document" in ctx and "text" in ctx["document"]:
                    context_texts.append(ctx["document"]["text"])
                elif "document" in ctx and "content" in ctx["document"]:
                    context_texts.append(ctx["document"]["content"])
            
            if context_texts:
                context_info = "Context information:\n" + "\n\n".join(context_texts)
        
        # Format full prompt
        if context_info:
            prompt = f"{chat_history}\n\n{context_info}\n\nAssistant: "
        else:
            prompt = f"{chat_history}\n\nAssistant: "
        
        # Generate response
        return self.generate(
            query=query,
            prompt=prompt,
            generation_kwargs=generation_kwargs,
            return_prompt=return_prompt
        )
    
    def _format_chat_history(self, messages: List[Dict[str, str]]) -> str:
        """
        Format a chat history from message dictionaries.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Formatted chat history string
        """
        chat_str = ""
        
        for msg in messages:
            role = msg.get('role', '').lower()
            content = msg.get('content', '')
            
            if role == 'user':
                chat_str += f"User: {content}\n\n"
            elif role == 'assistant':
                chat_str += f"Assistant: {content}\n\n"
            elif role == 'system':
                # For system messages, add a special prefix
                chat_str += f"System: {content}\n\n"
        
        return chat_str.strip()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the generator.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "model_name": self.model_name,
            "device": self.device,
            "max_new_tokens": self.max_new_tokens,
            "streaming_compatible": self.streaming_compatible,
            "loaded_in_8bit": self.load_in_8bit,
            "model_type": self.model.__class__.__name__,
            "tokenizer_type": self.tokenizer.__class__.__name__,
            "tokenizer_vocab_size": len(self.tokenizer)
        }
        
        return stats
# models/optimization/quantization.py
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import copy
import os
import time
import json

class ModelQuantizer:
    """
    Quantize PyTorch models for improved inference performance.
    
    This class provides methods for quantizing deep learning models using
    various techniques:
    - Static quantization (requires calibration data)
    - Dynamic quantization (no calibration data needed)
    - Quantization-aware training (for trained models)
    - Half-precision conversion (FP16/BF16)
    
    Additionally, it provides benchmarking utilities to measure performance
    improvements and model size reductions.
    """
    def __init__(self, model: nn.Module):
        """
        Initialize the model quantizer.
        
        Args:
            model: PyTorch model to quantize
        """
        self.model = model
        self.quantized_model = None
        self.dtype_mapping = {
            "int8": torch.qint8,
            "uint8": torch.quint8,
            "int32": torch.qint32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16  # Available on newer GPUs
        }
    
    def static_quantization(
        self,
        dtype: str = "int8",
        calibration_data: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> nn.Module:
        """
        Apply static quantization to the model.
        
        Static quantization requires calibration data to determine optimal
        quantization parameters. It's best for inference-only scenarios where
        the input distribution is known.
        
        Args:
            dtype: Quantization data type ('int8', 'uint8', or 'int32')
            calibration_data: Data for calibration (tuple of input tensors)
            
        Returns:
            Quantized model
            
        Raises:
            ValueError: If an unsupported dtype is provided
        """
        if dtype not in ["int8", "uint8", "int32"]:
            raise ValueError(f"Unsupported quantization dtype for static quantization: {dtype}")
            
        # Make a copy of the model for quantization
        model_copy = copy.deepcopy(self.model)
        
        # Prepare for quantization
        model_copy.eval()
        
        # Fuse modules if possible (conv + bn + relu)
        model_copy = torch.quantization.fuse_modules(model_copy, self._find_fusable_modules(model_copy))
        
        # Set qconfig
        model_copy.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare model for quantization
        torch.quantization.prepare(model_copy, inplace=True)
        
        # Calibrate with data if provided
        if calibration_data is not None:
            with torch.no_grad():
                model_copy(*calibration_data)
        
        # Convert to quantized model
        self.quantized_model = torch.quantization.convert(model_copy, inplace=False)
        
        return self.quantized_model
    
    def dynamic_quantization(
        self,
        dtype: str = "int8"
    ) -> nn.Module:
        """
        Apply dynamic quantization to the model.
        
        Dynamic quantization doesn't require calibration data and quantizes
        weights ahead of time but quantizes activations dynamically at runtime.
        It's best for RNNs and LSTMs and models where input distributions vary.
        
        Args:
            dtype: Quantization data type ('int8', 'uint8', or 'int32')
            
        Returns:
            Quantized model
            
        Raises:
            ValueError: If an unsupported dtype is provided
        """
        if dtype not in ["int8", "uint8", "int32"]:
            raise ValueError(f"Unsupported quantization dtype for dynamic quantization: {dtype}")
            
        # Choose quantization function based on dtype
        qtype = self.dtype_mapping[dtype]
        
        # Apply dynamic quantization
        self.quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear, nn.LSTM, nn.LSTMCell, nn.RNNCell, nn.GRUCell},
            dtype=qtype
        )
        
        return self.quantized_model
    
    def quantization_aware_training(
        self,
        train_fn: callable,
        dtype: str = "int8"
    ) -> nn.Module:
        """
        Apply quantization-aware training.
        
        QAT simulates the effects of quantization during training to allow
        the model to adapt to quantization-induced errors. The model remains
        in floating point but learns to be more robust to quantization.
        
        Args:
            train_fn: Function to train the model (takes model as parameter)
            dtype: Quantization data type ('int8', 'uint8', or 'int32')
            
        Returns:
            Quantized model
            
        Raises:
            ValueError: If an unsupported dtype is provided
        """
        if dtype not in ["int8", "uint8", "int32"]:
            raise ValueError(f"Unsupported quantization dtype for QAT: {dtype}")
            
        # Make a copy of the model for QAT
        model_copy = copy.deepcopy(self.model)
        
        # Fuse modules if possible
        model_copy = torch.quantization.fuse_modules(model_copy, self._find_fusable_modules(model_copy))
        
        # Set qconfig
        model_copy.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        # Prepare model for QAT
        torch.quantization.prepare_qat(model_copy, inplace=True)
        
        # Train with QAT
        train_fn(model_copy)
        
        # Convert to quantized model
        model_copy.eval()
        self.quantized_model = torch.quantization.convert(model_copy, inplace=False)
        
        return self.quantized_model
    
    def half_precision(self, dtype: str = "fp16") -> nn.Module:
        """
        Convert model to lower precision floating point.
        
        Args:
            dtype: Precision type ('fp16' or 'bf16')
            
        Returns:
            Lower precision model
            
        Raises:
            ValueError: If an unsupported dtype is provided
            RuntimeError: If BF16 is requested but not supported by hardware
        """
        if dtype not in ["fp16", "bf16"]:
            raise ValueError(f"Unsupported dtype for half precision: {dtype}")
        
        model_copy = copy.deepcopy(self.model)
        
        if dtype == "fp16":
            self.quantized_model = model_copy.half()
        else:  # bf16
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError("BF16 is not supported on this hardware")
            self.quantized_model = model_copy.to(torch.bfloat16)
        
        return self.quantized_model
    
    def _find_fusable_modules(self, model: nn.Module) -> List[List[str]]:
        """
        Find modules that can be fused for quantization.
        
        Looks for common patterns in the model that can be fused together
        for better quantization results:
        - Conv2d -> BatchNorm2d -> ReLU
        - Linear -> ReLU
        - Conv2d -> BatchNorm2d
        
        Args:
            model: PyTorch model
            
        Returns:
            List of lists with fusable module names
        """
        fusable_modules = []
        
        # Find all named modules
        named_modules = dict(model.named_modules())
        
        # Find Conv2d -> BatchNorm2d -> ReLU sequences
        for name, module in named_modules.items():
            if isinstance(module, nn.Conv2d):
                # Check if followed by BatchNorm2d and ReLU
                conv_name = name
                parent_path = conv_name.rsplit('.', 1)[0] if '.' in conv_name else ''
                
                # Handle different naming conventions
                bn_candidates = [
                    f"{parent_path}.bn" if parent_path else "bn",
                    f"{parent_path}.batch_norm" if parent_path else "batch_norm",
                    f"{parent_path}.norm" if parent_path else "norm",
                    f"{conv_name}_bn"
                ]
                
                relu_candidates = [
                    f"{parent_path}.relu" if parent_path else "relu",
                    f"{parent_path}.activation" if parent_path else "activation",
                    f"{conv_name}_relu"
                ]
                
                # Find BatchNorm
                bn_name = None
                for candidate in bn_candidates:
                    if candidate in named_modules and isinstance(named_modules[candidate], nn.BatchNorm2d):
                        bn_name = candidate
                        break
                
                # Find ReLU
                relu_name = None
                for candidate in relu_candidates:
                    if candidate in named_modules and isinstance(named_modules[candidate], (nn.ReLU, nn.ReLU6)):
                        relu_name = candidate
                        break
                
                # Create fusion groups
                if bn_name and relu_name:
                    fusable_modules.append([conv_name, bn_name, relu_name])
                elif bn_name:
                    fusable_modules.append([conv_name, bn_name])
            
            # Find Linear -> ReLU sequences
            elif isinstance(module, nn.Linear):
                linear_name = name
                parent_path = linear_name.rsplit('.', 1)[0] if '.' in linear_name else ''
                
                # ReLU candidates
                relu_candidates = [
                    f"{parent_path}.relu" if parent_path else "relu",
                    f"{parent_path}.activation" if parent_path else "activation",
                    f"{linear_name}_relu"
                ]
                
                # Find ReLU
                for candidate in relu_candidates:
                    if candidate in named_modules and isinstance(named_modules[candidate], (nn.ReLU, nn.ReLU6)):
                        fusable_modules.append([linear_name, candidate])
                        break
        
        return fusable_modules
    
    def benchmark(
        self,
        input_data: Tuple[torch.Tensor, ...],
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark the original and quantized models.
        
        Compares inference speed, memory usage, and model size between
        the original model and the quantized version.
        
        Args:
            input_data: Input data for benchmarking (tuple of tensors)
            num_runs: Number of runs for benchmarking
            warmup_runs: Number of warmup runs before timing
            
        Returns:
            Dictionary with benchmark results
            
        Raises:
            ValueError: If no quantized model is available
        """
        if self.quantized_model is None:
            raise ValueError("No quantized model available. Apply quantization first.")
        
        # Ensure models are in eval mode
        self.model.eval()
        self.quantized_model.eval()
        
        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Clone inputs to avoid modifications
        input_data = tuple(inp.clone() for inp in input_data)
        
        # Move inputs to device if needed
        if device.type == "cuda":
            input_data = tuple(inp.to(device) for inp in input_data)
        
        # Warmup runs for original model
        with torch.no_grad():
            for _ in range(warmup_runs):
                self.model(*input_data)
        
        # Benchmark original model
        torch.cuda.synchronize() if device.type == "cuda" else None
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                self.model(*input_data)
                torch.cuda.synchronize() if device.type == "cuda" else None
        original_time = time.time() - start_time
        
        # Warmup runs for quantized model
        with torch.no_grad():
            for _ in range(warmup_runs):
                self.quantized_model(*input_data)
        
        # Benchmark quantized model
        torch.cuda.synchronize() if device.type == "cuda" else None
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                self.quantized_model(*input_data)
                torch.cuda.synchronize() if device.type == "cuda" else None
        quantized_time = time.time() - start_time
        
        # Calculate speedup
        speedup = original_time / quantized_time if quantized_time > 0 else float('inf')
        
        # Calculate model sizes
        original_size = self._get_model_size(self.model)
        quantized_size = self._get_model_size(self.quantized_model)
        
        # Size reduction
        size_reduction = 1.0 - (quantized_size / original_size) if original_size > 0 else 0.0
        
        # Gather memory usage statistics if CUDA is available
        memory_stats = {}
        if device.type == "cuda":
            # Reset CUDA memory
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Measure memory usage for original model
            with torch.no_grad():
                _ = self.model(*input_data)
            original_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            
            # Reset CUDA memory
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Measure memory usage for quantized model
            with torch.no_grad():
                _ = self.quantized_model(*input_data)
            quantized_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            
            memory_stats = {
                "original_memory_mb": original_memory,
                "quantized_memory_mb": quantized_memory,
                "memory_reduction": 1.0 - (quantized_memory / original_memory) if original_memory > 0 else 0.0
            }
        
        # Prepare results
        results = {
            "original_time_s": original_time,
            "quantized_time_s": quantized_time,
            "speedup_factor": speedup,
            "original_size_mb": original_size,
            "quantized_size_mb": quantized_size,
            "size_reduction": size_reduction,
            "num_runs": num_runs,
            "device": device.type
        }
        
        # Add memory stats if available
        if memory_stats:
            results.update(memory_stats)
        
        return results
    
    def _get_model_size(self, model: nn.Module) -> float:
        """
        Calculate the model size in MB.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model size in MB
        """
        # Save model to a temporary file to get accurate size
        import tempfile
        with tempfile.NamedTemporaryFile() as tmp:
            torch.save(model.state_dict(), tmp.name)
            # Get file size in MB
            size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
        
        return size_mb
    
    def export_model(
        self,
        path: str,
        input_data: Optional[Tuple[torch.Tensor, ...]] = None,
        include_benchmark: bool = True
    ) -> Dict[str, Any]:
        """
        Export the quantized model to disk.
        
        Args:
            path: Directory path to save the model
            input_data: Optional input data for benchmark info
            include_benchmark: Whether to include benchmark info
            
        Returns:
            Dictionary with export information
            
        Raises:
            ValueError: If no quantized model is available
        """
        if self.quantized_model is None:
            raise ValueError("No quantized model available. Apply quantization first.")
        
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save the quantized model
        model_path = os.path.join(path, "quantized_model.pt")
        torch.save(self.quantized_model, model_path)
        
        # Save model state dict separately for more flexibility
        state_dict_path = os.path.join(path, "quantized_model_state_dict.pt")
        torch.save(self.quantized_model.state_dict(), state_dict_path)
        
        # Get model info
        model_info = {
            "model_path": model_path,
            "state_dict_path": state_dict_path,
            "model_size_mb": self._get_model_size(self.quantized_model),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Run benchmark if requested and input data is provided
        if include_benchmark and input_data is not None:
            benchmark_results = self.benchmark(input_data)
            model_info["benchmark"] = benchmark_results
        
        # Save model info as JSON
        info_path = os.path.join(path, "model_info.json")
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        return model_info

    def trace_and_optimize(
        self,
        example_input: torch.Tensor,
        optimize_for_mobile: bool = False
    ) -> torch.jit.ScriptModule:
        """
        Create an optimized TorchScript model through tracing.

        Args:
            example_input: Example input for tracing
            optimize_for_mobile: Whether to optimize for mobile deployment

        Returns:
            Traced and optimized ScriptModule

        Raises:
            ValueError: If no quantized model is available
        """
        if self.quantized_model is None:
            raise ValueError("No quantized model available. Apply quantization first.")

        # Set model to evaluation mode
        self.quantized_model.eval()
        
        # Trace the model
        traced_model = torch.jit.trace(self.quantized_model, example_input)
        
        # Optimize the model
        with torch.no_grad():
            traced_model = torch.jit.optimize_for_inference(traced_model)
            
            # Further optimize for mobile if requested
            if optimize_for_mobile:
                try:
                    from torch.utils.mobile_optimizer import optimize_for_mobile
                    traced_model = optimize_for_mobile(traced_model)
                except ImportError:
                    print("Warning: torch.utils.mobile_optimizer not available. Skipping mobile optimization.")
        
        return traced_model
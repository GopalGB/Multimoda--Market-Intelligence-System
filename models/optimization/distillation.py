# models/optimization/onnx_export.py
import torch
import torch.nn as nn
import onnx
import os
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import onnxruntime as ort
import copy

class ONNXExporter:
    """
    Export PyTorch models to ONNX format for deployment.
    
    This class provides utilities for:
    - Converting PyTorch models to ONNX format
    - Optimizing ONNX models for inference
    - Validating conversions
    - Benchmarking ONNX models vs PyTorch models
    - Supporting quantized ONNX models
    
    ONNX models can be deployed across various platforms and hardware
    (CPU, GPU, specialized accelerators) with consistent behavior.
    """
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    ):
        """
        Initialize the ONNX exporter.
        
        Args:
            model: PyTorch model to export
            device: Device to run the model on ('cpu', 'cuda', etc.)
            dynamic_axes: Dynamic axes configuration for variable-sized inputs
        """
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Copy the model and move to device
        self.model = copy.deepcopy(model).to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Set dynamic axes
        self.dynamic_axes = dynamic_axes
        
        # Store the exported model path
        self.onnx_model_path = None
        self.onnx_session = None
    
    def export(
        self,
        dummy_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        output_path: str,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        opset_version: int = 12,
        verbose: bool = False,
        optimize: bool = True
    ) -> str:
        """
        Export the PyTorch model to ONNX format.
        
        Args:
            dummy_input: Example input for tracing
            output_path: Path to save the ONNX model
            input_names: Names for input tensors
            output_names: Names for output tensors
            opset_version: ONNX opset version
            verbose: Whether to print detailed export information
            optimize: Whether to optimize the ONNX model after export
            
        Returns:
            Path to the exported ONNX model
            
        Raises:
            RuntimeError: If export fails
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Move dummy inputs to device
        if isinstance(dummy_input, torch.Tensor):
            dummy_input = dummy_input.to(self.device)
        elif isinstance(dummy_input, tuple):
            dummy_input = tuple(x.to(self.device) for x in dummy_input)
        
        # Set default input and output names if not provided
        if input_names is None:
            if isinstance(dummy_input, torch.Tensor):
                input_names = ['input']
            else:
                input_names = [f'input_{i}' for i in range(len(dummy_input))]
                
        if output_names is None:
            output_names = ['output']
        
        # Export the model
        with torch.no_grad():
            try:
                torch.onnx.export(
                    self.model,
                    dummy_input,
                    output_path,
                    verbose=verbose,
                    input_names=input_names,
                    output_names=output_names,
                    opset_version=opset_version,
                    dynamic_axes=self.dynamic_axes,
                    do_constant_folding=True,  # Fold constants for optimization
                    export_params=True  # Store model parameters in the model
                )
                
                if verbose:
                    print(f"Model exported to {output_path}")
                
                # Optimize if requested
                if optimize:
                    self._optimize_onnx_model(output_path, output_path)
                
                # Store the path
                self.onnx_model_path = output_path
                
                # Validate the exported model
                self._validate_onnx_model(output_path, verbose)
                
                # Initialize ONNX session
                self.onnx_session = ort.InferenceSession(output_path)
                
                return output_path
                
            except Exception as e:
                raise RuntimeError(f"Failed to export model to ONNX: {str(e)}")
    
    def _optimize_onnx_model(
        self,
        input_path: str,
        output_path: str
    ) -> None:
        """
        Optimize the ONNX model for inference.
        
        Args:
            input_path: Path to the input ONNX model
            output_path: Path to save the optimized ONNX model
        """
        try:
            import onnxoptimizer
            
            # Load the model
            model = onnx.load(input_path)
            
            # Apply optimizations
            passes = [
                'eliminate_deadend',
                'eliminate_identity',
                'eliminate_nop_dropout',
                'extract_constant_to_initializer',
                'fuse_add_bias_into_conv',
                'fuse_bn_into_conv',
                'fuse_consecutive_concats',
                'fuse_consecutive_reduces',
                'fuse_matmul_add_bias_into_gemm',
                'fuse_pad_into_conv',
                'fuse_pad_into_pool',
                'fuse_transpose_into_gemm'
            ]
            
            optimized_model = onnxoptimizer.optimize(model, passes)
            
            # Save the optimized model
            onnx.save(optimized_model, output_path)
            
        except ImportError:
            print("Warning: onnxoptimizer not available. Skipping optimization.")
    
    def _validate_onnx_model(
        self,
        model_path: str,
        verbose: bool = False
    ) -> None:
        """
        Validate the exported ONNX model.
        
        Args:
            model_path: Path to the ONNX model
            verbose: Whether to print detailed validation information
        """
        try:
            # Load and check the model
            model = onnx.load(model_path)
            onnx.checker.check_model(model)
            
            if verbose:
                print("ONNX model validation successful")
                
        except Exception as e:
            print(f"Warning: ONNX model validation failed: {str(e)}")
    
    def quantize_onnx_model(
        self,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        quantization_mode: str = 'dynamic',
        per_channel: bool = False,
        reduce_range: bool = False,
        optimize_model: bool = True
    ) -> str:
        """
        Quantize the ONNX model for reduced size and faster inference.
        
        Args:
            input_path: Path to the input ONNX model (uses stored path if None)
            output_path: Path to save the quantized ONNX model
            quantization_mode: Quantization mode ('dynamic' or 'static')
            per_channel: Whether to quantize per channel
            reduce_range: Whether to reduce range for activation quantization
            optimize_model: Whether to optimize the model before quantization
            
        Returns:
            Path to the quantized ONNX model
            
        Raises:
            ImportError: If onnxruntime-tools is not available
            ValueError: If no ONNX model is available
        """
        try:
            from onnxruntime.quantization import quantize_dynamic, quantize_static
        except ImportError:
            raise ImportError("onnxruntime-tools is required for quantization. Install with 'pip install onnxruntime-tools'")
        
        # Use stored path if not provided
        if input_path is None:
            if self.onnx_model_path is None:
                raise ValueError("No ONNX model available. Export a model first.")
            input_path = self.onnx_model_path
        
        # Create default output path if not provided
        if output_path is None:
            base_path = os.path.splitext(input_path)[0]
            output_path = f"{base_path}_quantized.onnx"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Optimize the model if requested
        model_to_quantize = input_path
        if optimize_model:
            optimized_path = f"{os.path.splitext(input_path)[0]}_optimized.onnx"
            self._optimize_onnx_model(input_path, optimized_path)
            model_to_quantize = optimized_path
        
        # Quantize the model
        if quantization_mode == 'dynamic':
            quantize_dynamic(
                model_input=model_to_quantize,
                model_output=output_path,
                per_channel=per_channel,
                reduce_range=reduce_range
            )
        elif quantization_mode == 'static':
            # Static quantization requires calibration data
            # This is a simplified implementation
            quantize_static(
                model_input=model_to_quantize,
                model_output=output_path,
                per_channel=per_channel,
                reduce_range=reduce_range
            )
        else:
            raise ValueError(f"Unsupported quantization mode: {quantization_mode}")
        
        # Update the ONNX session
        self.onnx_model_path = output_path
        self.onnx_session = ort.InferenceSession(output_path)
        
        return output_path
    
    def benchmark(
        self,
        input_data: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark the PyTorch model vs the ONNX model.
        
        Args:
            input_data: Input data for benchmarking
            num_runs: Number of runs for benchmarking
            warmup_runs: Number of warmup runs before timing
            
        Returns:
            Dictionary with benchmark results
            
        Raises:
            ValueError: If no ONNX model is available
        """
        if self.onnx_session is None:
            raise ValueError("No ONNX session available. Export a model first.")
        
        # Ensure PyTorch model is in eval mode
        self.model.eval()
        
        # Prepare input data for ONNX
        onnx_inputs = {}
        
        if isinstance(input_data, torch.Tensor):
            onnx_inputs[self.onnx_session.get_inputs()[0].name] = input_data.cpu().numpy()
        else:
            for i, inp in enumerate(input_data):
                onnx_inputs[self.onnx_session.get_inputs()[i].name] = inp.cpu().numpy()
        
        # Warmup for PyTorch
        with torch.no_grad():
            for _ in range(warmup_runs):
                if isinstance(input_data, torch.Tensor):
                    _ = self.model(input_data)
                else:
                    _ = self.model(*input_data)
        
        # Benchmark PyTorch
        torch.cuda.synchronize() if self.device == "cuda" else None
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                if isinstance(input_data, torch.Tensor):
                    _ = self.model(input_data)
                else:
                    _ = self.model(*input_data)
                torch.cuda.synchronize() if self.device == "cuda" else None
        pytorch_time = time.time() - start_time
        
        # Warmup for ONNX
        for _ in range(warmup_runs):
            _ = self.onnx_session.run(None, onnx_inputs)
        
        # Benchmark ONNX
        start_time = time.time()
        for _ in range(num_runs):
            _ = self.onnx_session.run(None, onnx_inputs)
        onnx_time = time.time() - start_time
        
        # Calculate speedup
        speedup = pytorch_time / onnx_time if onnx_time > 0 else float('inf')
        
        # Get model sizes
        pytorch_size = self._get_model_size_mb(self.model)
        onnx_size = os.path.getsize(self.onnx_model_path) / (1024 * 1024)  # MB
        
        # Prepare results
        results = {
            "pytorch_time_s": pytorch_time,
            "onnx_time_s": onnx_time,
            "speedup_factor": speedup,
            "pytorch_size_mb": pytorch_size,
            "onnx_size_mb": onnx_size,
            "size_reduction": 1.0 - (onnx_size / pytorch_size) if pytorch_size > 0 else 0.0,
            "num_runs": num_runs,
            "device": self.device
        }
        
        return results
    
    def _get_model_size_mb(self, model: nn.Module) -> float:
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
    
    def create_inference_session(
        self,
        providers: Optional[List[str]] = None,
        optimization_level: int = 99
    ) -> ort.InferenceSession:
        """
        Create an optimized ONNX inference session.
        
        Args:
            providers: List of execution providers
            optimization_level: Optimization level (0-99)
            
        Returns:
            ONNX inference session
            
        Raises:
            ValueError: If no ONNX model is available
        """
        if self.onnx_model_path is None:
            raise ValueError("No ONNX model available. Export a model first.")
        
        # Set default providers based on device
        if providers is None:
            if self.device == "cuda" and 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
        
        # Create session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = optimization_level
        
        # Create session
        session = ort.InferenceSession(
            self.onnx_model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        # Update the stored session
        self.onnx_session = session
        
        return session
    
    def inference(
        self,
        input_data: Union[np.ndarray, torch.Tensor, Dict[str, np.ndarray]],
        output_names: Optional[List[str]] = None
    ) -> List[np.ndarray]:
        """
        Run inference using the ONNX model.
        
        Args:
            input_data: Input data for inference
            output_names: Names of outputs to return
            
        Returns:
            List of output arrays
            
        Raises:
            ValueError: If no ONNX session is available
        """
        if self.onnx_session is None:
            raise ValueError("No ONNX session available. Export a model first.")
        
        # Prepare input data
        if isinstance(input_data, dict):
            # Input is already a dictionary
            onnx_inputs = input_data
        elif isinstance(input_data, torch.Tensor):
            # Convert torch tensor to numpy
            onnx_inputs = {self.onnx_session.get_inputs()[0].name: input_data.cpu().numpy()}
        elif isinstance(input_data, np.ndarray):
            # Use numpy array directly
            onnx_inputs = {self.onnx_session.get_inputs()[0].name: input_data}
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
        
        # Run inference
        outputs = self.onnx_session.run(output_names, onnx_inputs)
        
        return outputs
    
    def compare_outputs(
        self,
        input_data: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        rtol: float = 1e-3,
        atol: float = 1e-5
    ) -> Dict[str, Any]:
        """
        Compare outputs from PyTorch and ONNX models.
        
        Args:
            input_data: Input data for comparison
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison
            
        Returns:
            Dictionary with comparison results
            
        Raises:
            ValueError: If no ONNX session is available
        """
        if self.onnx_session is None:
            raise ValueError("No ONNX session available. Export a model first.")
        
        # Get PyTorch outputs
        self.model.eval()
        with torch.no_grad():
            if isinstance(input_data, torch.Tensor):
                pytorch_outputs = self.model(input_data)
            else:
                pytorch_outputs = self.model(*input_data)
        
        # Convert to list if single output
        if not isinstance(pytorch_outputs, tuple):
            pytorch_outputs = (pytorch_outputs,)
        
        # Convert PyTorch outputs to numpy
        pytorch_outputs_np = [out.cpu().numpy() for out in pytorch_outputs]
        
        # Prepare input data for ONNX
        onnx_inputs = {}
        if isinstance(input_data, torch.Tensor):
            onnx_inputs[self.onnx_session.get_inputs()[0].name] = input_data.cpu().numpy()
        else:
            for i, inp in enumerate(input_data):
                onnx_inputs[self.onnx_session.get_inputs()[i].name] = inp.cpu().numpy()
        
        # Get ONNX outputs
        onnx_outputs = self.onnx_session.run(None, onnx_inputs)
        
        # Compare outputs
        results = []
        for i, (pt_out, onnx_out) in enumerate(zip(pytorch_outputs_np, onnx_outputs)):
            # Check if shapes match
            shape_match = pt_out.shape == onnx_out.shape
            
            # Calculate differences
            if shape_match:
                max_diff = np.max(np.abs(pt_out - onnx_out))
                mean_diff = np.mean(np.abs(pt_out - onnx_out))
                is_close = np.allclose(pt_out, onnx_out, rtol=rtol, atol=atol)
            else:
                max_diff = float('inf')
                mean_diff = float('inf')
                is_close = False
            
            results.append({
                "output_idx": i,
                "shape_match": shape_match,
                "pytorch_shape": pt_out.shape,
                "onnx_shape": onnx_out.shape,
                "max_abs_diff": float(max_diff),
                "mean_abs_diff": float(mean_diff),
                "is_close": is_close
            })
        
        # Overall result
        return {
            "outputs": results,
            "all_close": all(r["is_close"] for r in results),
            "rtol": rtol,
            "atol": atol
        }
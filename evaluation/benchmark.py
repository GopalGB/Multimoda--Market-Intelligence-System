# evaluation/benchmark.py
import os
import json
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime

# Import project modules
from models.fusion.fusion_model import MultimodalFusionModel
from models.visual.clip_model import CLIPWrapper
from models.text.roberta_model import RoBERTaWrapper
from causal.structural_model import StructuralCausalModel
from evaluation.metrics import EngagementMetrics, CausalMetrics, MultimodalMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("benchmarking")

class BenchmarkDataset(Dataset):
    """
    Dataset for benchmarking multimodal models.
    
    Loads multimodal data (text + image) for inference benchmarking.
    """
    
    def __init__(
        self,
        data_path: str,
        text_field: str = "text_content",
        image_field: str = "image_path",
        label_field: Optional[str] = "engagement",
        max_samples: Optional[int] = None,
        transform = None
    ):
        """
        Initialize benchmark dataset.
        
        Args:
            data_path: Path to dataset (CSV or JSON)
            text_field: Field name for text content
            image_field: Field name for image path
            label_field: Field name for labels (None if not available)
            max_samples: Maximum number of samples to load
            transform: Optional transform for images
        """
        self.text_field = text_field
        self.image_field = image_field
        self.label_field = label_field
        self.transform = transform
        
        # Load data
        if data_path.endswith('.csv'):
            self.data = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                self.data = pd.DataFrame(json.load(f))
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Limit samples if needed
        if max_samples is not None:
            self.data = self.data.head(max_samples)
            
        # Check required fields
        if text_field not in self.data.columns:
            raise ValueError(f"Text field '{text_field}' not found in data")
        if image_field not in self.data.columns:
            logger.warning(f"Image field '{image_field}' not found in data. Running with text only.")
            self.image_field = None
        
        logger.info(f"Loaded {len(self.data)} samples for benchmarking")
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with sample data
        """
        row = self.data.iloc[idx]
        
        sample = {
            'text': row[self.text_field],
            'idx': idx
        }
        
        # Add image if available
        if self.image_field is not None:
            image_path = row[self.image_field]
            if os.path.exists(image_path):
                from PIL import Image
                try:
                    image = Image.open(image_path).convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                    sample['image'] = image
                except Exception as e:
                    logger.warning(f"Error loading image {image_path}: {e}")
        
        # Add label if available
        if self.label_field is not None and self.label_field in row:
            sample['label'] = row[self.label_field]
        
        return sample


class FusionModelBenchmark:
    """
    Benchmark for the multimodal fusion model.
    
    Evaluates inference speed, memory usage, and prediction accuracy
    across different modalities and model configurations.
    """
    
    def __init__(
        self,
        fusion_model: MultimodalFusionModel,
        clip_model: Optional[CLIPWrapper] = None,
        roberta_model: Optional[RoBERTaWrapper] = None,
        device: Optional[str] = None,
        output_dir: str = "benchmark_results"
    ):
        """
        Initialize the benchmark.
        
        Args:
            fusion_model: Multimodal fusion model to benchmark
            clip_model: CLIP model for visual features
            roberta_model: RoBERTa model for text features
            device: Device to run benchmark on
            output_dir: Directory to save benchmark results
        """
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Save models
        self.fusion_model = fusion_model
        self.clip_model = clip_model
        self.roberta_model = roberta_model
        
        # Ensure models are on the right device
        self.fusion_model.to(self.device)
        if self.clip_model:
            self.clip_model.model.to(self.device)
        if self.roberta_model:
            self.roberta_model.model.to(self.device)
        
        # Create output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Metrics
        self.metrics = MultimodalMetrics()
        
        # Benchmark results
        self.results = {
            "inference_speed": {},
            "memory_usage": {},
            "accuracy": {},
            "timestamp": datetime.datetime.now().isoformat(),
            "device": self.device,
            "model_info": {
                "fusion_model": fusion_model.__class__.__name__,
                "clip_model": clip_model.model_name if clip_model else None,
                "roberta_model": roberta_model.model_name if roberta_model else None
            }
        }
    
    def benchmark_inference_speed(
        self,
        dataset: Union[Dataset, DataLoader],
        batch_size: int = 16,
        num_workers: int = 2,
        modalities: List[str] = ["text", "visual", "fusion"],
        num_runs: int = 3
    ) -> Dict[str, float]:
        """
        Benchmark inference speed.
        
        Args:
            dataset: Dataset or DataLoader for benchmarking
            batch_size: Batch size for inference
            num_workers: Number of workers for data loading
            modalities: Modalities to benchmark
            num_runs: Number of benchmark runs
            
        Returns:
            Dictionary with inference speed metrics
        """
        # Prepare DataLoader
        if isinstance(dataset, Dataset):
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True
            )
        else:
            dataloader = dataset
        
        logger.info(f"Benchmarking inference speed on {len(dataloader.dataset)} samples with batch size {batch_size}")
        
        # Run benchmark for each modality
        speed_results = {}
        
        for modality in modalities:
            if modality == "text" and self.roberta_model is None:
                logger.warning("Skipping text modality benchmark: RoBERTa model not available")
                continue
            if modality == "visual" and self.clip_model is None:
                logger.warning("Skipping visual modality benchmark: CLIP model not available")
                continue
            if modality == "fusion" and (self.roberta_model is None or self.clip_model is None):
                logger.warning("Skipping fusion modality benchmark: Text or visual model not available")
                continue
            
            # Run multiple times and take average
            run_times = []
            samples_per_second = []
            
            for run in range(num_runs):
                logger.info(f"Running {modality} inference benchmark (run {run+1}/{num_runs})")
                
                # Warm-up run
                self._run_inference(dataloader, modality, num_batches=min(5, len(dataloader)))
                
                # Benchmark run
                start_time = time.time()
                total_samples = 0
                
                with torch.no_grad():
                    for batch in tqdm(dataloader, desc=f"{modality} inference"):
                        batch_size = self._get_batch_size(batch)
                        self._run_inference(batch, modality)
                        total_samples += batch_size
                
                # Calculate metrics
                elapsed_time = time.time() - start_time
                run_times.append(elapsed_time)
                samples_per_second.append(total_samples / elapsed_time)
                
                logger.info(f"Run {run+1}: {elapsed_time:.2f}s, {samples_per_second[-1]:.2f} samples/sec")
            
            # Calculate average metrics
            avg_time = np.mean(run_times)
            avg_sps = np.mean(samples_per_second)
            
            speed_results[modality] = {
                "avg_time_seconds": float(avg_time),
                "avg_samples_per_second": float(avg_sps),
                "batch_size": batch_size,
                "total_samples": len(dataloader.dataset)
            }
            
            logger.info(f"{modality} inference: {avg_time:.2f}s, {avg_sps:.2f} samples/sec")
        
        # Update and return results
        self.results["inference_speed"] = speed_results
        return speed_results
    
    def benchmark_memory_usage(
        self,
        batch_sizes: List[int] = [1, 4, 16, 32, 64],
        sample_text_length: int = 256,
        modalities: List[str] = ["text", "visual", "fusion"]
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Benchmark GPU memory usage at different batch sizes.
        
        Args:
            batch_sizes: List of batch sizes to test
            sample_text_length: Length of sample text
            modalities: Modalities to benchmark
            
        Returns:
            Dictionary with memory usage metrics
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available. Skipping memory usage benchmark.")
            return {}
        
        logger.info("Benchmarking memory usage at different batch sizes")
        
        # Create dummy data for benchmark
        sample_text = "This is a sample text for benchmarking." * (sample_text_length // 10)
        
        # Initialize results
        memory_results = {modality: {"batch_sizes": batch_sizes, "memory_usage_mb": []} for modality in modalities}
        
        # Benchmark each modality
        for modality in modalities:
            if modality == "text" and self.roberta_model is None:
                logger.warning("Skipping text modality benchmark: RoBERTa model not available")
                continue
            if modality == "visual" and self.clip_model is None:
                logger.warning("Skipping visual modality benchmark: CLIP model not available")
                continue
            if modality == "fusion" and (self.roberta_model is None or self.clip_model is None):
                logger.warning("Skipping fusion modality benchmark: Text or visual model not available")
                continue
            
            logger.info(f"Benchmarking {modality} memory usage")
            
            for batch_size in batch_sizes:
                # Clear cache
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # Create batch data
                if modality == "text":
                    batch = {
                        "text": [sample_text] * batch_size
                    }
                    
                    # Run inference
                    with torch.no_grad():
                        features = self.roberta_model.encode_text(batch["text"])
                        _ = self.fusion_model.predict_engagement(
                            torch.zeros((batch_size, 768), device=self.device),
                            features
                        )
                
                elif modality == "visual":
                    # Create dummy image tensor
                    img_tensor = torch.randn(batch_size, 3, 224, 224, device=self.device)
                    
                    # Run inference
                    with torch.no_grad():
                        _ = self.clip_model.model(img_tensor)
                
                elif modality == "fusion":
                    # Create text features
                    text_features = self.roberta_model.encode_text([sample_text] * batch_size)
                    
                    # Create visual features
                    visual_features = torch.randn(batch_size, 768, device=self.device)
                    
                    # Run fusion
                    with torch.no_grad():
                        _ = self.fusion_model.predict_engagement(visual_features, text_features)
                
                # Measure memory
                memory_used = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
                memory_results[modality]["memory_usage_mb"].append(float(memory_used))
                
                logger.info(f"{modality} batch size {batch_size}: {memory_used:.2f} MB")
        
        # Update results
        self.results["memory_usage"] = memory_results
        return memory_results
    
    def benchmark_accuracy(
        self,
        dataset: Union[Dataset, DataLoader],
        batch_size: int = 16,
        num_workers: int = 2,
        modalities: List[str] = ["text", "visual", "fusion"],
        task_type: str = "regression"
    ) -> Dict[str, Dict[str, float]]:
        """
        Benchmark prediction accuracy.
        
        Args:
            dataset: Dataset or DataLoader with labeled data
            batch_size: Batch size for inference
            num_workers: Number of workers for data loading
            modalities: Modalities to benchmark
            task_type: Type of prediction task ('regression' or 'classification')
            
        Returns:
            Dictionary with accuracy metrics
        """
        # Prepare DataLoader
        if isinstance(dataset, Dataset):
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True
            )
        else:
            dataloader = dataset
        
        logger.info(f"Benchmarking accuracy on {len(dataloader.dataset)} samples")
        
        # Check if dataset has labels
        has_labels = False
        for batch in dataloader:
            if 'label' in batch:
                has_labels = True
                break
        
        if not has_labels:
            logger.warning("Dataset does not contain labels. Skipping accuracy benchmark.")
            return {}
        
        # Collect predictions for each modality
        predictions = {}
        labels = []
        
        # Get predictions
        for modality in modalities:
            if modality == "text" and self.roberta_model is None:
                logger.warning("Skipping text modality benchmark: RoBERTa model not available")
                continue
            if modality == "visual" and self.clip_model is None:
                logger.warning("Skipping visual modality benchmark: CLIP model not available")
                continue
            if modality == "fusion" and (self.roberta_model is None or self.clip_model is None):
                logger.warning("Skipping fusion modality benchmark: Text or visual model not available")
                continue
            
            logger.info(f"Getting {modality} predictions")
            
            modality_preds = []
            
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"{modality} predictions"):
                    # Get predictions
                    batch_preds = self._run_inference(batch, modality)
                    modality_preds.extend(batch_preds)
                    
                    # Collect labels (only need to do this once)
                    if modality == modalities[0]:
                        if isinstance(batch['label'], torch.Tensor):
                            batch_labels = batch['label'].cpu().numpy()
                        else:
                            batch_labels = batch['label']
                        labels.extend(batch_labels)
            
            predictions[modality] = np.array(modality_preds)
        
        # Convert labels to numpy array
        labels = np.array(labels)
        
        # Evaluate accuracy for each modality
        accuracy_results = {}
        
        for modality, preds in predictions.items():
            # Evaluate
            metrics = self.metrics.evaluate_modality(modality, labels, preds, task_type)
            accuracy_results[modality] = metrics
            
            # Log results
            if task_type == "regression":
                logger.info(f"{modality} RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
            else:
                logger.info(f"{modality} Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        
        # Evaluate fusion gain if multiple modalities present
        if len(predictions) > 1 and "fusion" in predictions:
            single_modality_preds = {k: v for k, v in predictions.items() if k != "fusion"}
            fusion_metrics = self.metrics.evaluate_fusion_gain(
                labels, single_modality_preds, predictions["fusion"], task_type
            )
            
            # Log fusion gains
            for k, v in fusion_metrics.items():
                if k.startswith("gain_over_"):
                    logger.info(f"Fusion gain over {k.replace('gain_over_', '')}: {v:.2%}")
        
        # Update results
        self.results["accuracy"] = {
            modality: {k: float(v) for k, v in metrics.items()}
            for modality, metrics in accuracy_results.items()
        }
        
        return accuracy_results
    
    def _run_inference(
        self,
        data: Union[Dict[str, Any], DataLoader],
        modality: str,
        num_batches: Optional[int] = None
    ) -> Optional[List[float]]:
        """
        Run inference on a batch or DataLoader.
        
        Args:
            data: Input data (batch dictionary or DataLoader)
            modality: Modality to use ('text', 'visual', or 'fusion')
            num_batches: Number of batches to process (None for all)
            
        Returns:
            List of predictions (if batch) or None (if DataLoader)
        """
        # Process DataLoader
        if isinstance(data, DataLoader):
            batch_iter = iter(data)
            for i in range(num_batches or len(data)):
                try:
                    batch = next(batch_iter)
                    self._run_inference(batch, modality)
                except StopIteration:
                    break
            return None
        
        # Process batch
        batch = data
        
        if modality == "text":
            # Process text
            text_features = self.roberta_model.encode_text(batch["text"])
            
            # Create dummy visual features
            batch_size = self._get_batch_size(batch)
            visual_features = torch.zeros((batch_size, 768), device=self.device)
            
            # Run fusion model with text only
            predictions = self.fusion_model.predict_engagement(visual_features, text_features)
            
            # Extract predictions
            if isinstance(predictions, dict) and "class_label" in predictions:
                return predictions["class_label"]
            elif isinstance(predictions, dict) and "engagement_score" in predictions:
                return predictions["engagement_score"]
            else:
                return predictions
            
        elif modality == "visual":
            # Process images if available
            batch_size = self._get_batch_size(batch)
            
            if "image" in batch:
                # Convert to tensor if needed
                if not isinstance(batch["image"], torch.Tensor):
                    # Handle list of PIL images
                    from torchvision import transforms
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    images = [transform(img) for img in batch["image"]]
                    image_tensor = torch.stack(images).to(self.device)
                else:
                    image_tensor = batch["image"].to(self.device)
                
                # Extract visual features
                visual_features = self.clip_model.encode_images(image_tensor)
            else:
                # Use random features if images not available
                visual_features = torch.randn(batch_size, 768, device=self.device)
            
            # Create dummy text features
            text_features = torch.zeros((batch_size, 768), device=self.device)
            
            # Run fusion model with visual only
            predictions = self.fusion_model.predict_engagement(visual_features, text_features)
            
            # Extract predictions
            if isinstance(predictions, dict) and "class_label" in predictions:
                return predictions["class_label"]
            elif isinstance(predictions, dict) and "engagement_score" in predictions:
                return predictions["engagement_score"]
            else:
                return predictions
            
        elif modality == "fusion":
            # Process text
            text_features = self.roberta_model.encode_text(batch["text"])
            
            # Process images if available
            batch_size = self._get_batch_size(batch)
            
            if "image" in batch:
                # Convert to tensor if needed
                if not isinstance(batch["image"], torch.Tensor):
                    # Handle list of PIL images
                    from torchvision import transforms
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    images = [transform(img) for img in batch["image"]]
                    image_tensor = torch.stack(images).to(self.device)
                else:
                    image_tensor = batch["image"].to(self.device)
                
                # Extract visual features
                visual_features = self.clip_model.encode_images(image_tensor)
            else:
                # Use random features if images not available
                visual_features = torch.randn(batch_size, 768, device=self.device)
            
            # Run fusion model
            predictions = self.fusion_model.predict_engagement(visual_features, text_features)
            
            # Extract predictions
            if isinstance(predictions, dict) and "class_label" in predictions:
                return predictions["class_label"]
            elif isinstance(predictions, dict) and "engagement_score" in predictions:
                return predictions["engagement_score"]
            else:
                return predictions
        
        else:
            raise ValueError(f"Unsupported modality: {modality}")
    
    def _get_batch_size(self, batch: Dict[str, Any]) -> int:
        """Get batch size from batch dictionary."""
        if "text" in batch:
            if isinstance(batch["text"], list):
                return len(batch["text"])
            else:
                return batch["text"].shape[0]
        elif "image" in batch:
            if isinstance(batch["image"], list):
                return len(batch["image"])
            else:
                return batch["image"].shape[0]
        elif "label" in batch:
            if isinstance(batch["label"], list):
                return len(batch["label"])
            else:
                return batch["label"].shape[0]
        else:
            return 1  # Default if can't determine
    
    def plot_inference_speed(
        self,
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        Plot inference speed comparison.
        
        Args:
            figsize: Figure size
        """
        if not self.results["inference_speed"]:
            logger.warning("No inference speed results to plot")
            return
        
        # Extract data
        modalities = list(self.results["inference_speed"].keys())
        speeds = [self.results["inference_speed"][m]["avg_samples_per_second"] for m in modalities]
        
        # Create plot
        plt.figure(figsize=figsize)
        plt.bar(modalities, speeds)
        plt.ylabel("Samples per Second")
        plt.title("Inference Speed by Modality")
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(speeds):
            plt.text(i, v + 0.1, f"{v:.1f}", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "inference_speed.png"))
        plt.show()
    
    def plot_memory_usage(
        self,
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        Plot memory usage across batch sizes.
        
        Args:
            figsize: Figure size
        """
        if not self.results["memory_usage"]:
            logger.warning("No memory usage results to plot")
            return
        
        # Create plot
        plt.figure(figsize=figsize)
        
        for modality, data in self.results["memory_usage"].items():
            batch_sizes = data["batch_sizes"]
            memory_usage = data["memory_usage_mb"]
            plt.plot(batch_sizes, memory_usage, marker='o', label=modality)
        
        plt.xlabel("Batch Size")
        plt.ylabel("GPU Memory Usage (MB)")
        plt.title("GPU Memory Usage by Batch Size")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "memory_usage.png"))
        plt.show()
    
    def plot_accuracy(
        self,
        metric: str = "rmse",
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        Plot accuracy comparison.
        
        Args:
            metric: Metric to plot
            figsize: Figure size
        """
        if not self.results["accuracy"]:
            logger.warning("No accuracy results to plot")
            return
        
        # Check if metric exists in results
        for modality, metrics in self.results["accuracy"].items():
            if metric not in metrics:
                logger.warning(f"Metric {metric} not found in results. Available metrics: {list(metrics.keys())}")
                return
        
        # Extract data
        modalities = list(self.results["accuracy"].keys())
        values = [self.results["accuracy"][m][metric] for m in modalities]
        
        # Determine if higher or lower is better
        higher_better = metric not in ["mse", "rmse", "mae", "mape"]
        
        # Create plot
        plt.figure(figsize=figsize)
        bars = plt.bar(modalities, values)
        plt.ylabel(metric.upper())
        plt.title(f"{metric.upper()} by Modality")
        plt.grid(True, alpha=0.3)
        
        # Add color based on higher/lower better
        if higher_better:
            # Green = good (high values)
            colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(values)))
            for i, bar in enumerate(bars):
                bar.set_color(colors[i])
        else:
            # Red = bad (high values)
            colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(values)))
            for i, bar in enumerate(bars):
                bar.set_color(colors[i])
        
        # Add value labels
        for i, v in enumerate(values):
            plt.text(i, v + (0.05 * v if v > 0 else -0.05), f"{v:.4f}", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"accuracy_{metric}.png"))
        plt.show()
    
    def save_results(
        self,
        filename: str = "benchmark_results.json"
    ) -> str:
        """
        Save benchmark results to file.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved results
        """
        output_path = os.path.join(self.output_dir, filename)
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {output_path}")
        return output_path


class CausalModelBenchmark:
    """
    Benchmark for causal inference models.
    
    Evaluates causal discovery, effect estimation, and counterfactual
    prediction performance.
    """
    
    def __init__(
        self,
        causal_model: StructuralCausalModel,
        output_dir: str = "benchmark_results"
    ):
        """
        Initialize the benchmark.
        
        Args:
            causal_model: Structural causal model to benchmark
            output_dir: Directory to save benchmark results
        """
        self.causal_model = causal_model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Metrics
        self.metrics = CausalMetrics()
        
        # Benchmark results
        self.results = {
            "causal_discovery": {},
            "effect_estimation": {},
            "counterfactual_prediction": {},
            "inference_time": {},
            "timestamp": datetime.datetime.now().isoformat(),
            "model_info": {
                "discovery_method": causal_model.discovery_method,
                "num_features": len(causal_model.feature_names)
            }
        }
    
    def benchmark_causal_discovery(
        self,
        data: pd.DataFrame,
        true_graph: Dict[str, List[str]],
        outcome_var: str
    ) -> Dict[str, float]:
        """
        Benchmark causal discovery performance.
        
        Args:
            data: Dataset for causal discovery
            true_graph: Ground truth causal graph
            outcome_var: Outcome variable name
            
        Returns:
            Dictionary with discovery metrics
        """
        logger.info("Benchmarking causal discovery")
        
        # Run causal discovery
        start_time = time.time()
        self.causal_model.discover_graph(data, outcome_var)
        discovery_time = time.time() - start_time
        
        # Convert model graph to dictionary for evaluation
        estimated_graph = {}
        for node in self.causal_model.causal_graph.graph.nodes():
            parents = list(self.causal_model.causal_graph.get_parents(node))
            estimated_graph[node] = parents
        
        # Evaluate discovery
        metrics = self.metrics.evaluate_causal_graph(true_graph, estimated_graph)
        metrics["discovery_time_seconds"] = discovery_time
        
        # Update results
        self.results["causal_discovery"] = metrics
        
        logger.info(f"Causal discovery: F1={metrics['f1']:.4f}, SHD={metrics['shd']:.1f}")
        return metrics
    
    def benchmark_effect_estimation(
        self,
        data: pd.DataFrame,
        true_effects: Dict[str, float],
        outcome_var: str
    ) -> Dict[str, float]:
        """
        Benchmark causal effect estimation.
        
        Args:
            data: Dataset for effect estimation
            true_effects: Ground truth causal effects
            outcome_var: Outcome variable name
            
        Returns:
            Dictionary with estimation metrics
        """
        logger.info("Benchmarking causal effect estimation")
        
        # Get feature names from true effects
        feature_names = list(true_effects.keys())
        
        # Estimate causal effects
        start_time = time.time()
        estimated_effects = self.causal_model.estimate_all_effects(data, outcome_var)
        estimation_time = time.time() - start_time
        
        # Extract effect sizes
        effect_sizes = {
            feature: effect_info["causal_effect"]
            for feature, effect_info in estimated_effects.items()
        }
        
        # Evaluate estimation
        metrics = self.metrics.evaluate_causal_effect(true_effects, effect_sizes)
        metrics["estimation_time_seconds"] = estimation_time
        
        # Update results
        self.results["effect_estimation"] = metrics
        
        logger.info(f"Effect estimation: MAE={metrics['mae']:.4f}, Coverage={metrics['coverage']:.2f}")
        return metrics
    
    def benchmark_counterfactual_prediction(
        self,
        data: pd.DataFrame,
        interventions: List[Dict[str, float]],
        true_outcomes: List[float],
        outcome_var: str
    ) -> Dict[str, float]:
        """
        Benchmark counterfactual prediction.
        
        Args:
            data: Dataset for counterfactual prediction
            interventions: List of intervention dictionaries
            true_outcomes: Ground truth counterfactual outcomes
            outcome_var: Outcome variable name
            
        Returns:
            Dictionary with prediction metrics
        """
        logger.info("Benchmarking counterfactual prediction")
        
        # Make counterfactual predictions
        start_time = time.time()
        predicted_outcomes = []
        
        for i, row_data in enumerate(data.iterrows()):
            idx, row = row_data
            intervention = interventions[i]
            
            # Create instance
            instance = pd.Series(row)
            
            # Make prediction
            result = self.causal_model.counterfactual_analysis(
                instance, intervention, outcome_var
            )
            
            predicted_outcomes.append(result["counterfactual_outcome"])
        
        prediction_time = time.time() - start_time
        
        # Evaluate predictions
        metrics = self.metrics.evaluate_counterfactual(
            np.array(true_outcomes),
            np.array(predicted_outcomes)
        )
        metrics["prediction_time_seconds"] = prediction_time
        
        # Update results
        self.results["counterfactual_prediction"] = metrics
        
        logger.info(f"Counterfactual prediction: MAE={metrics['mae']:.4f}, Correlation={metrics['correlation']:.4f}")
        return metrics
    
    def benchmark_inference_time(
        self,
        data: pd.DataFrame,
        interventions: Dict[str, float],
        outcome_var: str,
        num_runs: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark inference time.
        
        Args:
            data: Dataset for inference
            interventions: Intervention dictionary
            outcome_var: Outcome variable name
            num_runs: Number of benchmark runs
            
        Returns:
            Dictionary with timing metrics
        """
        logger.info("Benchmarking inference time")
        
        # Sample a few rows for benchmarking
        sample_size = min(10, len(data))
        sample_data = data.sample(sample_size)
        
        # Run multiple times and take average
        do_times = []
        counterfactual_times = []
        
        for run in range(num_runs):
            # Benchmark do-operator
            start_time = time.time()
            for _ in range(sample_size):
                self.causal_model.do_intervention(sample_data, interventions, outcome_var)
            do_time = (time.time() - start_time) / sample_size
            do_times.append(do_time)
            
            # Benchmark counterfactual analysis
            start_time = time.time()
            for i, row_data in enumerate(sample_data.iterrows()):
                idx, row = row_data
                instance = pd.Series(row)
                self.causal_model.counterfactual_analysis(instance, interventions, outcome_var)
            counterfactual_time = (time.time() - start_time) / sample_size
            counterfactual_times.append(counterfactual_time)
        
        # Calculate average times
        avg_do_time = np.mean(do_times)
        avg_counterfactual_time = np.mean(counterfactual_times)
        
        # Update results
        time_metrics = {
            "do_operator_time_seconds": float(avg_do_time),
            "counterfactual_time_seconds": float(avg_counterfactual_time),
            "sample_size": sample_size,
            "num_runs": num_runs
        }
        self.results["inference_time"] = time_metrics
        
        logger.info(f"Do-operator: {avg_do_time*1000:.2f}ms, Counterfactual: {avg_counterfactual_time*1000:.2f}ms")
        return time_metrics
    
    def save_results(
        self,
        filename: str = "causal_benchmark_results.json"
    ) -> str:
        """
        Save benchmark results to file.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved results
        """
        output_path = os.path.join(self.output_dir, filename)
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {output_path}")
        return output_path
    
    def plot_effect_estimation(
        self,
        true_effects: Dict[str, float],
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        Plot effect estimation results.
        
        Args:
            true_effects: Ground truth causal effects
            figsize: Figure size
        """
        # Extract estimated effects from model
        estimated_effects = {
            feature: effect_info["causal_effect"]
            for feature, effect_info in self.causal_model.causal_effects.items()
        }
        
        # Plot comparison
        self.metrics.plot_effect_comparison(true_effects, estimated_effects, figsize=figsize)
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, "effect_estimation.png"))
        
        
        
        
     lksefpsoiejfpwosgpov

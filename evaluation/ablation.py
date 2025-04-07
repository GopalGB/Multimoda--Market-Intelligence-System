# evaluation/ablation.py
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from tqdm import tqdm
import logging
import json
from pathlib import Path
import os
import datetime

from models.fusion.fusion_model import MultimodalFusionModel
from models.visual.clip_model import CLIPWrapper
from models.text.roberta_model import RoBERTaWrapper
from causal.structural_model import StructuralCausalModel
from evaluation.metrics import EngagementMetrics, MultimodalMetrics

logger = logging.getLogger(__name__)

class AblationStudy:
    """
    Conducts ablation studies to analyze the contribution of different
    model components, feature groups, and architectural choices on performance.
    
    This helps identify which parts of the CAIP system contribute most to its
    effectiveness and can guide future optimization efforts.
    """
    
    def __init__(
        self,
        base_model: MultimodalFusionModel,
        evaluation_data: pd.DataFrame,
        metrics: Optional[EngagementMetrics] = None,
        output_dir: str = "ablation_results",
        device: Optional[str] = None
    ):
        """
        Initialize the ablation study.
        
        Args:
            base_model: Reference fusion model to ablate
            evaluation_data: Data for evaluating model variations
            metrics: Metrics for evaluation (defaults to EngagementMetrics)
            output_dir: Directory to save results
            device: Device to run models on
        """
        self.base_model = base_model
        self.evaluation_data = evaluation_data
        self.metrics = metrics or EngagementMetrics()
        self.output_dir = output_dir
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize ablation results
        self.ablation_results = {}
        self.base_performance = None
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def evaluate_base_model(
        self,
        text_column: str = "text_content",
        image_column: Optional[str] = "image_path",
        target_column: str = "engagement",
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Evaluate the base model as a reference point.
        
        Args:
            text_column: Column name for text content
            image_column: Column name for image paths (None for text-only)
            target_column: Column name for target values
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating base model performance")
        
        # Initialize CLIP and RoBERTa models if needed
        if not hasattr(self, 'clip_model') and image_column:
            self.clip_model = CLIPWrapper(model_name=self.base_model.visual_model_name, device=self.device)
        
        if not hasattr(self, 'roberta_model'):
            self.roberta_model = RoBERTaWrapper(model_name=self.base_model.text_model_name, device=self.device)
        
        # Prepare data
        texts = self.evaluation_data[text_column].tolist()
        
        if image_column:
            image_paths = self.evaluation_data[image_column].tolist()
            has_images = True
        else:
            has_images = False
        
        targets = self.evaluation_data[target_column].values
        
        # Process in batches
        predictions = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating base model"):
            batch_texts = texts[i:i+batch_size]
            
            # Process text
            text_features = self.roberta_model.encode_text(batch_texts)
            
            # Process images if available
            if has_images:
                batch_images = image_paths[i:i+batch_size]
                batch_visual_features = []
                
                for img_path in batch_images:
                    if pd.isna(img_path) or not os.path.exists(img_path):
                        # Use zero vector for missing images
                        batch_visual_features.append(torch.zeros(self.clip_model.model.config.projection_dim))
                    else:
                        # Load and encode image
                        visual_features = self.clip_model.encode_images(img_path)
                        batch_visual_features.append(visual_features[0])
                
                visual_features = torch.stack(batch_visual_features)
            else:
                # Use zero vectors for all images
                visual_features = torch.zeros((len(batch_texts), 768), device=self.device)
            
            # Make predictions
            with torch.no_grad():
                batch_preds = self.base_model.predict_engagement(visual_features, text_features)
                
                # Extract numeric predictions from dict if needed
                if isinstance(batch_preds, dict) and "engagement_score" in batch_preds:
                    batch_preds = batch_preds["engagement_score"]
                
                predictions.extend(batch_preds)
        
        # Evaluate performance
        predictions = np.array(predictions)
        self.base_performance = self.metrics.evaluate(targets, predictions)
        
        logger.info(f"Base model performance: {self.base_performance}")
        return self.base_performance
    
    def ablate_modality(
        self,
        text_column: str = "text_content",
        image_column: Optional[str] = "image_path",
        target_column: str = "engagement",
        batch_size: int = 32
    ) -> Dict[str, Dict[str, float]]:
        """
        Ablate each modality to measure its contribution.
        
        Args:
            text_column: Column name for text content
            image_column: Column name for image paths
            target_column: Column name for target values
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of modality ablation results
        """
        logger.info("Performing modality ablation study")
        
        # Ensure base model has been evaluated
        if self.base_performance is None:
            self.evaluate_base_model(text_column, image_column, target_column, batch_size)
        
        # Ablation variations to test
        ablations = {
            "text_only": {"modality": "visual", "ablate": True},
            "visual_only": {"modality": "text", "ablate": True},
        }
        
        modality_results = {}
        
        # Process each ablation
        for name, config in ablations.items():
            logger.info(f"Testing ablation: {name}")
            
            # Prepare data
            texts = self.evaluation_data[text_column].tolist()
            
            if image_column:
                image_paths = self.evaluation_data[image_column].tolist()
                has_images = True
            else:
                has_images = False
            
            targets = self.evaluation_data[target_column].values
            
            # Process in batches
            predictions = []
            
            for i in tqdm(range(0, len(texts), batch_size), desc=f"Evaluating {name}"):
                batch_texts = texts[i:i+batch_size]
                
                # Process text
                if config["modality"] == "text" and config["ablate"]:
                    # Ablate text by using zero vectors
                    text_features = torch.zeros((len(batch_texts), 768), device=self.device)
                else:
                    text_features = self.roberta_model.encode_text(batch_texts)
                
                # Process images if available
                if has_images and not (config["modality"] == "visual" and config["ablate"]):
                    batch_images = image_paths[i:i+batch_size]
                    batch_visual_features = []
                    
                    for img_path in batch_images:
                        if pd.isna(img_path) or not os.path.exists(img_path):
                            # Use zero vector for missing images
                            batch_visual_features.append(torch.zeros(self.clip_model.model.config.projection_dim))
                        else:
                            # Load and encode image
                            visual_features = self.clip_model.encode_images(img_path)
                            batch_visual_features.append(visual_features[0])
                    
                    visual_features = torch.stack(batch_visual_features)
                else:
                    # Ablate visual or when not available
                    visual_features = torch.zeros((len(batch_texts), 768), device=self.device)
                
                # Make predictions
                with torch.no_grad():
                    batch_preds = self.base_model.predict_engagement(visual_features, text_features)
                    
                    # Extract numeric predictions from dict if needed
                    if isinstance(batch_preds, dict) and "engagement_score" in batch_preds:
                        batch_preds = batch_preds["engagement_score"]
                    
                    predictions.extend(batch_preds)
            
            # Evaluate performance
            predictions = np.array(predictions)
            perf = self.metrics.evaluate(targets, predictions)
            modality_results[name] = perf
            
            logger.info(f"{name} performance: {perf}")
        
        # Save results
        self.ablation_results["modality"] = modality_results
        self._save_results("modality_ablation.json", modality_results)
        
        return modality_results
    
    def ablate_fusion_methods(
        self,
        fusion_methods: List[str] = ["attention", "concat", "sum", "product"],
        text_column: str = "text_content",
        image_column: Optional[str] = "image_path",
        target_column: str = "engagement",
        batch_size: int = 32
    ) -> Dict[str, Dict[str, float]]:
        """
        Ablate different fusion methods to compare approaches.
        
        Args:
            fusion_methods: List of fusion methods to test
            text_column: Column name for text content
            image_column: Column name for image paths
            target_column: Column name for target values
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of fusion method ablation results
        """
        logger.info("Performing fusion method ablation study")
        
        # Ensure base model has been evaluated
        if self.base_performance is None:
            self.evaluate_base_model(text_column, image_column, target_column, batch_size)
        
        fusion_results = {}
        
        # Process each fusion method
        for method in fusion_methods:
            logger.info(f"Testing fusion method: {method}")
            
            # Create a modified model with the selected fusion method
            modified_model = self._create_model_with_fusion(method)
            
            # Skip if model creation failed
            if modified_model is None:
                logger.warning(f"Skipping fusion method '{method}': Failed to create model")
                continue
            
            # Prepare data
            texts = self.evaluation_data[text_column].tolist()
            
            if image_column:
                image_paths = self.evaluation_data[image_column].tolist()
                has_images = True
            else:
                has_images = False
            
            targets = self.evaluation_data[target_column].values
            
            # Process in batches
            predictions = []
            
            for i in tqdm(range(0, len(texts), batch_size), desc=f"Evaluating fusion: {method}"):
                batch_texts = texts[i:i+batch_size]
                
                # Process text
                text_features = self.roberta_model.encode_text(batch_texts)
                
                # Process images if available
                if has_images:
                    batch_images = image_paths[i:i+batch_size]
                    batch_visual_features = []
                    
                    for img_path in batch_images:
                        if pd.isna(img_path) or not os.path.exists(img_path):
                            # Use zero vector for missing images
                            batch_visual_features.append(torch.zeros(self.clip_model.model.config.projection_dim))
                        else:
                            # Load and encode image
                            visual_features = self.clip_model.encode_images(img_path)
                            batch_visual_features.append(visual_features[0])
                    
                    visual_features = torch.stack(batch_visual_features)
                else:
                    # Use zero vectors when images not available
                    visual_features = torch.zeros((len(batch_texts), 768), device=self.device)
                
                # Make predictions with modified model
                with torch.no_grad():
                    batch_preds = modified_model.predict_engagement(visual_features, text_features)
                    
                    # Extract numeric predictions from dict if needed
                    if isinstance(batch_preds, dict) and "engagement_score" in batch_preds:
                        batch_preds = batch_preds["engagement_score"]
                    
                    predictions.extend(batch_preds)
            
            # Evaluate performance
            predictions = np.array(predictions)
            perf = self.metrics.evaluate(targets, predictions)
            fusion_results[method] = perf
            
            logger.info(f"{method} fusion performance: {perf}")
        
        # Save results
        self.ablation_results["fusion"] = fusion_results
        self._save_results("fusion_ablation.json", fusion_results)
        
        return fusion_results
    
    def ablate_features(
        self,
        feature_groups: Dict[str, List[str]],
        text_column: str = "text_content",
        image_column: Optional[str] = "image_path",
        target_column: str = "engagement",
        batch_size: int = 32
    ) -> Dict[str, Dict[str, float]]:
        """
        Ablate feature groups to measure their contribution.
        
        Args:
            feature_groups: Dictionary mapping group names to feature lists
            text_column: Column name for text content
            image_column: Column name for image paths
            target_column: Column name for target values
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of feature ablation results
        """
        logger.info("Performing feature ablation study")
        
        # Ensure base model has been evaluated
        if self.base_performance is None:
            self.evaluate_base_model(text_column, image_column, target_column, batch_size)
        
        feature_results = {}
        all_features = []
        
        # Collect all features
        for group, features in feature_groups.items():
            all_features.extend(features)
        
        # Ablate each feature group
        for group, features_to_ablate in feature_groups.items():
            logger.info(f"Ablating feature group: {group}")
            
            # Create a model with these features ablated
            # This is a simplified approach - in practice would need
            # to modify the model architecture to truly ablate features
            model_with_ablation = self._create_model_with_ablated_features(features_to_ablate)
            
            if model_with_ablation is None:
                logger.warning(f"Skipping feature group '{group}': Failed to create model")
                continue
            
            # Prepare data
            texts = self.evaluation_data[text_column].tolist()
            
            if image_column:
                image_paths = self.evaluation_data[image_column].tolist()
                has_images = True
            else:
                has_images = False
            
            targets = self.evaluation_data[target_column].values
            
            # Process in batches
            predictions = []
            
            for i in tqdm(range(0, len(texts), batch_size), desc=f"Evaluating without {group}"):
                batch_texts = texts[i:i+batch_size]
                
                # Process text
                text_features = self.roberta_model.encode_text(batch_texts)
                
                # Process images if available
                if has_images:
                    batch_images = image_paths[i:i+batch_size]
                    batch_visual_features = []
                    
                    for img_path in batch_images:
                        if pd.isna(img_path) or not os.path.exists(img_path):
                            # Use zero vector for missing images
                            batch_visual_features.append(torch.zeros(self.clip_model.model.config.projection_dim))
                        else:
                            # Load and encode image
                            visual_features = self.clip_model.encode_images(img_path)
                            batch_visual_features.append(visual_features[0])
                    
                    visual_features = torch.stack(batch_visual_features)
                else:
                    # Use zero vectors when images not available
                    visual_features = torch.zeros((len(batch_texts), 768), device=self.device)
                
                # Make predictions with modified model
                with torch.no_grad():
                    batch_preds = model_with_ablation.predict_engagement(visual_features, text_features)
                    
                    # Extract numeric predictions from dict if needed
                    if isinstance(batch_preds, dict) and "engagement_score" in batch_preds:
                        batch_preds = batch_preds["engagement_score"]
                    
                    predictions.extend(batch_preds)
            
            # Evaluate performance
            predictions = np.array(predictions)
            perf = self.metrics.evaluate(targets, predictions)
            feature_results[group] = perf
            
            logger.info(f"Performance without {group}: {perf}")
        
        # Save results
        self.ablation_results["features"] = feature_results
        self._save_results("feature_ablation.json", feature_results)
        
        return feature_results
    
    def ablate_causal_relationships(
        self,
        causal_model: StructuralCausalModel,
        intervention_features: List[str],
        text_column: str = "text_content",
        target_column: str = "engagement",
        batch_size: int = 32
    ) -> Dict[str, Dict[str, float]]:
        """
        Ablate causal relationships through interventions.
        
        Args:
            causal_model: Structural causal model
            intervention_features: Features to intervene on
            text_column: Column name for text content
            target_column: Column name for target values
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of causal ablation results
        """
        logger.info("Performing causal relationship ablation study")
        
        causal_results = {}
        data = self.evaluation_data.copy()
        targets = data[target_column].values
        
        # Baseline without interventions
        predictions = causal_model.predict(data, target_column)
        baseline_perf = self.metrics.evaluate(targets, predictions)
        causal_results["baseline"] = baseline_perf
        
        # Ablate each feature through intervention
        for feature in intervention_features:
            logger.info(f"Testing intervention on: {feature}")
            
            # Calculate mean value for the feature
            mean_value = data[feature].mean()
            
            # Perform do-intervention: set feature to its mean
            interventions = {feature: mean_value}
            
            # Predict with intervention
            intervention_predictions = []
            
            for i in range(0, len(data), batch_size):
                batch_data = data.iloc[i:i+batch_size]
                batch_preds = causal_model.do_intervention(batch_data, interventions, target_column)
                
                if isinstance(batch_preds, list) or isinstance(batch_preds, np.ndarray):
                    intervention_predictions.extend(batch_preds)
                else:
                    # Handle scalar output from do_intervention
                    intervention_predictions.extend([batch_preds] * len(batch_data))
            
            # Evaluate performance
            intervention_predictions = np.array(intervention_predictions)
            perf = self.metrics.evaluate(targets, intervention_predictions)
            causal_results[f"intervene_{feature}"] = perf
            
            logger.info(f"Performance with intervention on {feature}: {perf}")
        
        # Save results
        self.ablation_results["causal"] = causal_results
        self._save_results("causal_ablation.json", causal_results)
        
        return causal_results
    
    def visualize_results(self, output_format: str = "png") -> None:
        """
        Visualize ablation study results.
        
        Args:
            output_format: Format for saving visualizations
        """
        if not self.ablation_results:
            logger.warning("No ablation results to visualize")
            return
        
        # Create visualization directory
        vis_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Set plotting style
        plt.style.use("seaborn-v0_8-whitegrid")
        
        # Plot modality ablation if available
        if "modality" in self.ablation_results:
            self._plot_modality_results(vis_dir, output_format)
        
        # Plot fusion method ablation if available
        if "fusion" in self.ablation_results:
            self._plot_fusion_results(vis_dir, output_format)
        
        # Plot feature ablation if available
        if "features" in self.ablation_results:
            self._plot_feature_results(vis_dir, output_format)
        
        # Plot causal ablation if available
        if "causal" in self.ablation_results:
            self._plot_causal_results(vis_dir, output_format)
        
        logger.info(f"Visualizations saved to {vis_dir}")
    
    def _plot_modality_results(self, vis_dir: str, output_format: str) -> None:
        """Plot modality ablation results."""
        modality_results = self.ablation_results["modality"]
        
        plt.figure(figsize=(10, 6))
        
        # Select the first metric as primary
        first_metric = next(iter(self.base_performance.keys()))
        metric_name = first_metric.upper()
        
        # Extract values
        values = {}
        values["base_model"] = self.base_performance[first_metric]
        for name, result in modality_results.items():
            values[name] = result[first_metric]
        
        # Create bar chart
        labels = list(values.keys())
        heights = [values[label] for label in labels]
        
        # Set color scheme based on whether higher or lower is better
        higher_better = first_metric in ["accuracy", "f1", "r2", "correlation"]
        if higher_better:
            colors = plt.cm.YlGnBu(np.linspace(0.3, 0.9, len(labels)))
        else:
            colors = plt.cm.YlOrRd(np.linspace(0.9, 0.3, len(labels)))
        
        ax = plt.bar(labels, heights, color=colors)
        
        # Add value labels
        for i, rect in enumerate(ax):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2., height + 0.01,
                    f"{heights[i]:.3f}", ha='center', va='bottom', fontweight='bold')
        
        plt.ylabel(metric_name)
        plt.title(f"Impact of Modality Ablation on {metric_name}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(vis_dir, f"modality_ablation.{output_format}"))
        plt.close()
    
    def _plot_fusion_results(self, vis_dir: str, output_format: str) -> None:
        """Plot fusion method ablation results."""
        fusion_results = self.ablation_results["fusion"]
        
        plt.figure(figsize=(12, 6))
        
        # Select the first metric as primary
        first_metric = next(iter(fusion_results[next(iter(fusion_results))].keys()))
        metric_name = first_metric.upper()
        
        # Extract values
        methods = list(fusion_results.keys())
        values = [fusion_results[method][first_metric] for method in methods]
        
        # Create bar chart for fusion methods
        higher_better = first_metric in ["accuracy", "f1", "r2", "correlation"]
        if higher_better:
            colors = plt.cm.YlGnBu(np.linspace(0.3, 0.9, len(methods)))
        else:
            colors = plt.cm.YlOrRd(np.linspace(0.9, 0.3, len(methods)))
            
        bars = plt.bar(methods, values, color=colors)
        
        # Add baseline reference
        if self.base_performance is not None:
            plt.axhline(y=self.base_performance[first_metric], color='r', linestyle='--', 
                       label=f"Base Model: {self.base_performance[first_metric]:.3f}")
            plt.legend()
        
        # Add value labels
        for i, rect in enumerate(bars):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2., height + 0.01,
                    f"{values[i]:.3f}", ha='center', va='bottom')
        
        plt.ylabel(metric_name)
        plt.title(f"Performance of Different Fusion Methods ({metric_name})")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(vis_dir, f"fusion_ablation.{output_format}"))
        plt.close()
    
    def _plot_feature_results(self, vis_dir: str, output_format: str) -> None:
        """Plot feature ablation results."""
        feature_results = self.ablation_results["features"]
        
        plt.figure(figsize=(12, 6))
        
        # Select the first metric as primary
        first_metric = next(iter(feature_results[next(iter(feature_results))].keys()))
        metric_name = first_metric.upper()
        
        # Extract values
        feature_groups = list(feature_results.keys())
        values = [feature_results[group][first_metric] for group in feature_groups]
        
        # Calculate feature importance (difference from baseline)
        if self.base_performance is not None:
            baseline = self.base_performance[first_metric]
            # Importance as absolute difference from baseline
            importance = [abs(baseline - val) for val in values]
            
            # Sort feature groups by importance
            sorted_indices = np.argsort(importance)[::-1]
            feature_groups = [feature_groups[i] for i in sorted_indices]
            values = [values[i] for i in sorted_indices]
            importance = [importance[i] for i in sorted_indices]
        
        # Create bar chart
        higher_better = first_metric in ["accuracy", "f1", "r2", "correlation"]
        if not higher_better:
            # For metrics where lower is better, reverse the order
            feature_groups = feature_groups[::-1]
            values = values[::-1]
        
        # Use colors based on difference from baseline
        if self.base_performance is not None:
            # Normalize importance for color mapping
            norm_importance = np.array(importance) / max(importance)
            colors = plt.cm.viridis(norm_importance)
        else:
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(feature_groups)))
            
        bars = plt.bar(feature_groups, values, color=colors)
        
        # Add baseline reference
        if self.base_performance is not None:
            plt.axhline(y=baseline, color='r', linestyle='--', 
                       label=f"Base Model: {baseline:.3f}")
            plt.legend()
        
        # Add value labels
        for i, rect in enumerate(bars):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2., height + 0.01,
                    f"{values[i]:.3f}", ha='center', va='bottom')
        
        plt.ylabel(metric_name)
        plt.title(f"Impact of Feature Ablation on {metric_name}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(vis_dir, f"feature_ablation.{output_format}"))
        plt.close()
        
        # Create feature importance plot
        if self.base_performance is not None:
            plt.figure(figsize=(10, 6))
            plt.barh(feature_groups, importance, color=plt.cm.viridis(norm_importance))
            plt.xlabel("Absolute Impact")
            plt.title("Feature Group Importance")
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"feature_importance.{output_format}"))
            plt.close()
    
    def _plot_causal_results(self, vis_dir: str, output_format: str) -> None:
        """Plot causal ablation results."""
        causal_results = self.ablation_results["causal"]
        
        plt.figure(figsize=(12, 6))
        
        # Select the first metric as primary
        first_metric = next(iter(causal_results[next(iter(causal_results))].keys()))
        metric_name = first_metric.upper()
        
        # Extract values
        interventions = list(causal_results.keys())
        values = [causal_results[intervention][first_metric] for intervention in interventions]
        
        # Create bar chart
        higher_better = first_metric in ["accuracy", "f1", "r2", "correlation"]
        colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(interventions)))
            
        bars = plt.bar(interventions, values, color=colors)
        
        # Add value labels
        for i, rect in enumerate(bars):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2., height + 0.01,
                    f"{values[i]:.3f}", ha='center', va='bottom')
        
        plt.ylabel(metric_name)
        plt.title(f"Impact of Causal Interventions on {metric_name}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(vis_dir, f"causal_ablation.{output_format}"))
        plt.close()
        
        # Calculate intervention effects
        if "baseline" in causal_results:
            baseline = causal_results["baseline"][first_metric]
            effects = {}
            
            for intervention in interventions:
                if intervention != "baseline":
                    effects[intervention] = causal_results[intervention][first_metric] - baseline
            
            # Plot intervention effects
            plt.figure(figsize=(10, 6))
            
            effect_labels = list(effects.keys())
            effect_values = list(effects.values())
            
            # Set colors based on effect direction
            effect_colors = ['green' if x > 0 else 'red' for x in effect_values]
            if not higher_better:
                effect_colors = ['red' if x > 0 else 'green' for x in effect_values]
                
            plt.barh(effect_labels, effect_values, color=effect_colors)
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.7)
            plt.xlabel(f"Effect on {metric_name}")
            plt.title("Causal Intervention Effects")
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(vis_dir, f"causal_effects.{output_format}"))
            plt.close()
    
    def _save_results(self, filename: str, results: Dict) -> None:
        """Save ablation results to a JSON file."""
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert any non-serializable values (e.g., numpy types)
        def convert_for_json(obj):
            if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (datetime.date, datetime.datetime)):
                return obj.isoformat()
            return obj
        
        # Process nested dictionaries
        def process_dict(d):
            result = {}
            for key, value in d.items():
                if isinstance(value, dict):
                    result[key] = process_dict(value)
                else:
                    result[key] = convert_for_json(value)
            return result
        
        # Save processed results
        with open(filepath, 'w') as f:
            json.dump(process_dict(results), f, indent=2)
    
    def _create_model_with_fusion(self, fusion_method: str) -> Optional[MultimodalFusionModel]:
        """
        Create a modified model with a specific fusion method.
        
        This would need to be implemented based on the actual model architecture.
        Here we just show the interface.
        
        Args:
            fusion_method: Fusion method name
            
        Returns:
            Modified model or None if creation failed
        """
        # This is a stub - in real implementation would modify the model architecture
        # For a proper implementation:
        # 1. Clone the base model
        # 2. Modify its fusion method
        # 3. Return the modified model
        
        # For example, assuming the model has a fusion_method attribute:
        try:
            modified_model = deepcopy(self.base_model)  # Not defined here, just conceptual
            modified_model.fusion_method = fusion_method
            return modified_model
        except Exception as e:
            logger.error(f"Failed to create model with fusion method '{fusion_method}': {str(e)}")
            return None
    
    def _create_model_with_ablated_features(self, features_to_ablate: List[str]) -> Optional[MultimodalFusionModel]:
        """
        Create a model with specific features ablated.
        
        This would need to be implemented based on the actual model architecture.
        Here we just show the interface.
        
        Args:
            features_to_ablate: List of feature names to ablate
            
        Returns:
            Modified model or None if creation failed
        """
        # This is a stub - in real implementation would modify the model architecture
        # For a proper implementation:
        # 1. Clone the base model
        # 2. Modify its feature processing to ablate specified features
        # 3. Return the modified model
        
        try:
            modified_model = deepcopy(self.base_model)  # Not defined here, just conceptual
            modified_model.ablated_features = features_to_ablate
            return modified_model
        except Exception as e:
            logger.error(f"Failed to create model with ablated features: {str(e)}")
            return None


def compare_ablation_studies(
    study_results: Dict[str, Dict],
    output_dir: str = "ablation_results",
    primary_metric: str = "rmse",
    output_format: str = "png"
) -> None:
    """
    Compare results from multiple ablation studies.
    
    Args:
        study_results: Dictionary mapping study names to results
        output_dir: Directory to save comparison visualizations
        primary_metric: Primary metric for comparison
        output_format: Format for saving visualizations
    """
    if not study_results:
        logger.warning("No ablation studies to compare")
        return
    
    # Create visualization directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plotting style
    plt.style.use("seaborn-v0_8-whitegrid")
    
    # Compare modality ablation across studies
    if all("modality" in results for results in study_results.values()):
        _compare_modality_results(study_results, output_dir, primary_metric, output_format)
    
    # Compare fusion ablation across studies
    if all("fusion" in results for results in study_results.values()):
        _compare_fusion_results(study_results, output_dir, primary_metric, output_format)
    
    # Compare feature ablation across studies
    if all("features" in results for results in study_results.values()):
        _compare_feature_results(study_results, output_dir, primary_metric, output_format)
    
    # Compare causal ablation across studies
    if all("causal" in results for results in study_results.values()):
        _compare_causal_results(study_results, output_dir, primary_metric, output_format)
    
    logger.info(f"Ablation comparison visualizations saved to {output_dir}")


def _compare_modality_results(
    study_results: Dict[str, Dict],
    output_dir: str,
    primary_metric: str,
    output_format: str
) -> None:
    """Compare modality ablation results across studies."""
    plt.figure(figsize=(12, 8))
    
    # Extract data for comparison
    studies = list(study_results.keys())
    modalities = set()
    
    for study, results in study_results.items():
        modalities.update(results["modality"].keys())
    
    modalities = sorted(list(modalities))
    
    # Create grouped bar chart
    bar_width = 0.8 / len(studies)
    positions = np.arange(len(modalities))
    
    for i, study in enumerate(studies):
        modality_results = study_results[study]["modality"]
        values = [modality_results.get(modality, {}).get(primary_metric, float('nan')) 
                 for modality in modalities]
        
        x_pos = positions + (i - len(studies)/2 + 0.5) * bar_width
        plt.bar(x_pos, values, width=bar_width, label=study)
    
    plt.xlabel("Modality")
    plt.ylabel(primary_metric.upper())
    plt.title(f"Comparison of Modality Ablation Studies ({primary_metric.upper()})")
    plt.xticks(positions, modalities, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f"modality_comparison.{output_format}"))
    plt.close()


def _compare_fusion_results(
    study_results: Dict[str, Dict],
    output_dir: str,
    primary_metric: str,
    output_format: str
) -> None:
    """Compare fusion ablation results across studies."""
    plt.figure(figsize=(12, 8))
    
    # Extract data for comparison
    studies = list(study_results.keys())
    fusion_methods = set()
    
    for study, results in study_results.items():
        fusion_methods.update(results["fusion"].keys())
    
    fusion_methods = sorted(list(fusion_methods))
    
    # Create grouped bar chart
    bar_width = 0.8 / len(studies)
    positions = np.arange(len(fusion_methods))
    
    for i, study in enumerate(studies):
        fusion_results = study_results[study]["fusion"]
        values = [fusion_results.get(method, {}).get(primary_metric, float('nan')) 
                 for method in fusion_methods]
        
        x_pos = positions + (i - len(studies)/2 + 0.5) * bar_width
        plt.bar(x_pos, values, width=bar_width, label=study)
    
    plt.xlabel("Fusion Method")
    plt.ylabel(primary_metric.upper())
    plt.title(f"Comparison of Fusion Method Studies ({primary_metric.upper()})")
    plt.xticks(positions, fusion_methods, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f"fusion_comparison.{output_format}"))
    plt.close()


def _compare_feature_results(
    study_results: Dict[str, Dict],
    output_dir: str,
    primary_metric: str,
    output_format: str
) -> None:
    """Compare feature ablation results across studies."""
    plt.figure(figsize=(14, 8))
    
    # Extract data for comparison
    studies = list(study_results.keys())
    feature_groups = set()
    
    for study, results in study_results.items():
        feature_groups.update(results["features"].keys())
    
    feature_groups = sorted(list(feature_groups))
    
    # Create grouped bar chart
    bar_width = 0.8 / len(studies)
    positions = np.arange(len(feature_groups))
    
    for i, study in enumerate(studies):
        feature_results = study_results[study]["features"]
        values = [feature_results.get(group, {}).get(primary_metric, float('nan')) 
                 for group in feature_groups]
        
        x_pos = positions + (i - len(studies)/2 + 0.5) * bar_width
        plt.bar(x_pos, values, width=bar_width, label=study)
    
    plt.xlabel("Feature Group")
    plt.ylabel(primary_metric.upper())
    plt.title(f"Comparison of Feature Ablation Studies ({primary_metric.upper()})")
    plt.xticks(positions, feature_groups, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f"feature_comparison.{output_format}"))
    plt.close()


def _compare_causal_results(
    study_results: Dict[str, Dict],
    output_dir: str,
    primary_metric: str,
    output_format: str
) -> None:
    """Compare causal ablation results across studies."""
    plt.figure(figsize=(14, 8))
    
    # Extract data for comparison
    studies = list(study_results.keys())
    interventions = set()
    
    for study, results in study_results.items():
        interventions.update(results["causal"].keys())
    
    interventions = sorted(list(interventions))
    
    # Create grouped bar chart
    bar_width = 0.8 / len(studies)
    positions = np.arange(len(interventions))
    
    for i, study in enumerate(studies):
        causal_results = study_results[study]["causal"]
        values = [causal_results.get(intervention, {}).get(primary_metric, float('nan')) 
                 for intervention in interventions]
        
        x_pos = positions + (i - len(studies)/2 + 0.5) * bar_width
        plt.bar(x_pos, values, width=bar_width, label=study)
    
    plt.xlabel("Intervention")
    plt.ylabel(primary_metric.upper())
    plt.title(f"Comparison of Causal Ablation Studies ({primary_metric.upper()})")
    plt.xticks(positions, interventions, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f"causal_comparison.{output_format}"))
    plt.close()
# evaluation/metrics.py
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Any, Optional, Tuple
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import torch

class EngagementMetrics:
    """
    Evaluation metrics for engagement prediction.
    
    Provides functions to evaluate both regression and classification
    engagement predictions across different modalities.
    """
    
    def __init__(self, task_type: str = "regression"):
        """
        Initialize engagement metrics.
        
        Args:
            task_type: Type of prediction task ('regression' or 'classification')
        """
        self.task_type = task_type
        self.metrics_history = []
        
    def evaluate(
        self,
        y_true: Union[np.ndarray, List],
        y_pred: Union[np.ndarray, List],
        sample_weights: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate predictions against ground truth.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            sample_weights: Optional weights for samples
            
        Returns:
            Dictionary of metric scores
        """
        # Ensure numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate metrics based on task type
        if self.task_type == "regression":
            metrics = self._evaluate_regression(y_true, y_pred, sample_weights)
        elif self.task_type == "classification":
            metrics = self._evaluate_classification(y_true, y_pred, sample_weights)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        
        # Store metrics history
        self.metrics_history.append(metrics)
        
        return metrics
    
    def _evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Evaluate regression predictions."""
        mse = mean_squared_error(y_true, y_pred, sample_weight=sample_weights)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred, sample_weight=sample_weights)
        r2 = r2_score(y_true, y_pred, sample_weight=sample_weights)
        
        # Calculate additional metrics
        # Mean absolute percentage error (handle zeros)
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-7))) * 100
        
        # Correlation coefficient
        corr, p_value = (0.0, 1.0) if len(y_true) <= 1 else \
                        pd.Series(y_true).corr(pd.Series(y_pred)), 0.0
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape),
            'correlation': float(corr),
            'p_value': float(p_value)
        }
    
    def _evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Evaluate classification predictions."""
        # For probability predictions, convert to class labels
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred_proba = y_pred.copy()
            y_pred = np.argmax(y_pred, axis=1)
        else:
            # For binary classification with probability outputs
            if np.max(y_pred) <= 1.0 and np.min(y_pred) >= 0.0 and len(np.unique(y_pred)) > 2:
                y_pred_proba = np.column_stack((1 - y_pred, y_pred))
                y_pred = (y_pred >= 0.5).astype(int)
            else:
                y_pred_proba = None
        
        # Calculate metrics
        acc = accuracy_score(y_true, y_pred, sample_weight=sample_weights)
        
        # Handle binary vs multiclass
        if len(np.unique(y_true)) == 2:
            precision = precision_score(y_true, y_pred, sample_weight=sample_weights)
            recall = recall_score(y_true, y_pred, sample_weight=sample_weights)
            f1 = f1_score(y_true, y_pred, sample_weight=sample_weights)
            
            # Calculate ROC AUC if probability predictions available
            if y_pred_proba is not None:
                try:
                    auc = roc_auc_score(y_true, y_pred_proba[:, 1], sample_weight=sample_weights)
                except (ValueError, IndexError):
                    auc = 0.5
            else:
                auc = 0.5
        else:
            # Multiclass metrics
            precision = precision_score(y_true, y_pred, average='weighted', sample_weight=sample_weights)
            recall = recall_score(y_true, y_pred, average='weighted', sample_weight=sample_weights)
            f1 = f1_score(y_true, y_pred, average='weighted', sample_weight=sample_weights)
            
            # Calculate ROC AUC for multiclass if probability predictions available
            if y_pred_proba is not None:
                try:
                    auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', sample_weight=sample_weights)
                except (ValueError, IndexError):
                    auc = 0.5
            else:
                auc = 0.5
        
        return {
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc)
        }
    
    def plot_predictions(
        self,
        y_true: Union[np.ndarray, List],
        y_pred: Union[np.ndarray, List],
        title: str = "Prediction vs Ground Truth",
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        Plot predictions against ground truth.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            title: Plot title
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        if self.task_type == "regression":
            # Scatter plot for regression
            plt.scatter(y_true, y_pred, alpha=0.5)
            
            # Add perfect prediction line
            min_val = min(np.min(y_true), np.min(y_pred))
            max_val = max(np.max(y_true), np.max(y_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            plt.xlabel("Ground Truth")
            plt.ylabel("Prediction")
            plt.title(title)
            
            # Add regression metrics as text
            metrics = self._evaluate_regression(y_true, y_pred)
            metrics_text = (f"RMSE: {metrics['rmse']:.4f}\n"
                           f"MAE: {metrics['mae']:.4f}\n"
                           f"R²: {metrics['r2']:.4f}")
            plt.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction',
                        verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
            
        else:
            # Confusion matrix for classification
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(title)
            
            # Add classification metrics as text
            metrics = self._evaluate_classification(y_true, y_pred)
            metrics_text = (f"Accuracy: {metrics['accuracy']:.4f}\n"
                           f"F1: {metrics['f1']:.4f}\n"
                           f"AUC: {metrics['auc']:.4f}")
            plt.annotate(metrics_text, xy=(0.05, 0.05), xycoords='axes fraction',
                        verticalalignment='bottom', bbox=dict(boxstyle='round', alpha=0.1))
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def compare_models(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        Compare metrics across multiple models.
        
        Args:
            results: Dictionary mapping model names to metric dictionaries
            metrics: Specific metrics to compare (None for all)
            figsize: Figure size
        """
        # Determine metrics to compare
        if metrics is None:
            # Get metrics common to all models
            metric_sets = [set(model_metrics.keys()) for model_metrics in results.values()]
            metrics = list(set.intersection(*metric_sets))
        
        # Create DataFrame for plotting
        df = pd.DataFrame({model: {metric: values[metric] for metric in metrics if metric in values}
                          for model, values in results.items()}).T
        
        # Plot comparison
        plt.figure(figsize=figsize)
        ax = df.plot(kind='bar', figsize=figsize)
        plt.title("Model Comparison")
        plt.ylabel("Score")
        plt.grid(True, alpha=0.3)
        plt.legend(title="Metrics")
        plt.tight_layout()
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)
        
        plt.show()


class CausalMetrics:
    """
    Evaluation metrics for causal inference.
    
    Provides functions to evaluate causal effect estimation,
    counterfactual predictions, and causal graph discovery.
    """
    
    def __init__(self):
        """Initialize causal metrics."""
        self.metrics_history = []
    
    def evaluate_causal_effect(
        self,
        true_effects: Dict[str, float],
        estimated_effects: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Evaluate causal effect estimation accuracy.
        
        Args:
            true_effects: Dictionary of ground truth causal effects
            estimated_effects: Dictionary of estimated causal effects
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get common variables
        common_vars = set(true_effects.keys()) & set(estimated_effects.keys())
        
        if not common_vars:
            return {
                'mae': float('nan'),
                'rmse': float('nan'),
                'bias': float('nan'),
                'r2': float('nan'),
                'coverage': 0.0
            }
        
        # Extract values for common variables
        true_vals = np.array([true_effects[var] for var in common_vars])
        est_vals = np.array([estimated_effects[var] for var in common_vars])
        
        # Calculate metrics
        mae = mean_absolute_error(true_vals, est_vals)
        rmse = np.sqrt(mean_squared_error(true_vals, est_vals))
        bias = np.mean(est_vals - true_vals)
        
        # R² score
        r2 = r2_score(true_vals, est_vals)
        
        # Coverage (proportion of variables with estimated effects)
        coverage = len(common_vars) / len(true_effects) if true_effects else 0.0
        
        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'bias': float(bias),
            'r2': float(r2),
            'coverage': float(coverage)
        }
        
        self.metrics_history.append(metrics)
        return metrics
    
    def evaluate_counterfactual(
        self,
        true_outcomes: np.ndarray,
        counterfactual_outcomes: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate counterfactual prediction accuracy.
        
        Args:
            true_outcomes: Ground truth counterfactual outcomes
            counterfactual_outcomes: Predicted counterfactual outcomes
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Calculate metrics
        mae = mean_absolute_error(true_outcomes, counterfactual_outcomes)
        rmse = np.sqrt(mean_squared_error(true_outcomes, counterfactual_outcomes))
        
        # Correlation
        corr, p_value = (0.0, 1.0) if len(true_outcomes) <= 1 else \
                        pd.Series(true_outcomes).corr(pd.Series(counterfactual_outcomes)), 0.0
        
        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'correlation': float(corr),
            'p_value': float(p_value)
        }
        
        return metrics
    
    def evaluate_causal_graph(
        self,
        true_graph: Dict[str, List[str]],
        estimated_graph: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """
        Evaluate causal graph discovery accuracy.
        
        Args:
            true_graph: Dictionary mapping nodes to their parent nodes in the true graph
            estimated_graph: Dictionary mapping nodes to their parent nodes in the estimated graph
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Calculate edge-level metrics
        true_edges = set()
        for node, parents in true_graph.items():
            for parent in parents:
                true_edges.add((parent, node))
        
        est_edges = set()
        for node, parents in estimated_graph.items():
            for parent in parents:
                est_edges.add((parent, node))
        
        # True positives, false positives, false negatives
        tp = len(true_edges & est_edges)
        fp = len(est_edges - true_edges)
        fn = len(true_edges - est_edges)
        
        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Structural Hamming Distance (SHD)
        shd = fp + fn
        
        metrics = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'shd': float(shd)
        }
        
        return metrics
    
    def plot_effect_comparison(
        self,
        true_effects: Dict[str, float],
        estimated_effects: Dict[str, float],
        title: str = "Causal Effect Estimation",
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        Plot comparison of true vs. estimated causal effects.
        
        Args:
            true_effects: Dictionary of ground truth causal effects
            estimated_effects: Dictionary of estimated causal effects
            title: Plot title
            figsize: Figure size
        """
        # Get common variables
        common_vars = sorted(set(true_effects.keys()) & set(estimated_effects.keys()))
        
        if not common_vars:
            print("No common variables to plot")
            return
        
        # Extract values
        true_vals = [true_effects[var] for var in common_vars]
        est_vals = [estimated_effects[var] for var in common_vars]
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Plot bar chart
        x = np.arange(len(common_vars))
        width = 0.35
        
        plt.bar(x - width/2, true_vals, width, label='True Effect')
        plt.bar(x + width/2, est_vals, width, label='Estimated Effect')
        
        plt.xlabel("Variables")
        plt.ylabel("Causal Effect")
        plt.title(title)
        plt.xticks(x, common_vars, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add metrics as text
        metrics = self.evaluate_causal_effect(true_effects, estimated_effects)
        metrics_text = (f"MAE: {metrics['mae']:.4f}\n"
                       f"RMSE: {metrics['rmse']:.4f}\n"
                       f"Coverage: {metrics['coverage']:.4f}")
        plt.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction',
                    verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
        
        plt.tight_layout()
        plt.show()


class MultimodalMetrics:
    """
    Evaluation metrics for multimodal models.
    
    Provides functions to evaluate performance of different modalities
    and fusion methods.
    """
    
    def __init__(self):
        """Initialize multimodal metrics."""
        self.engagement_metrics = EngagementMetrics()
        self.modality_results = {}
    
    def evaluate_modality(
        self,
        modality: str,
        y_true: Union[np.ndarray, List],
        y_pred: Union[np.ndarray, List],
        task_type: str = "regression"
    ) -> Dict[str, float]:
        """
        Evaluate performance of a specific modality.
        
        Args:
            modality: Name of the modality ('text', 'visual', 'fusion', etc.)
            y_true: Ground truth values
            y_pred: Predicted values
            task_type: Type of prediction task ('regression' or 'classification')
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Set task type
        self.engagement_metrics.task_type = task_type
        
        # Evaluate
        metrics = self.engagement_metrics.evaluate(y_true, y_pred)
        
        # Store results for this modality
        self.modality_results[modality] = metrics
        
        return metrics
    
    def compare_modalities(
        self,
        metrics: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        Compare performance across modalities.
        
        Args:
            metrics: Specific metrics to compare (None for all)
            figsize: Figure size
        """
        if not self.modality_results:
            print("No modality results to compare")
            return
        
        self.engagement_metrics.compare_models(self.modality_results, metrics, figsize)
    
    def evaluate_fusion_gain(
        self,
        y_true: Union[np.ndarray, List],
        single_modality_preds: Dict[str, np.ndarray],
        fusion_pred: np.ndarray,
        task_type: str = "regression"
    ) -> Dict[str, float]:
        """
        Evaluate the performance gain from multimodal fusion.
        
        Args:
            y_true: Ground truth values
            single_modality_preds: Dictionary mapping modality names to predictions
            fusion_pred: Predictions from the fusion model
            task_type: Type of prediction task ('regression' or 'classification')
            
        Returns:
            Dictionary of fusion gain metrics
        """
        # Set task type
        self.engagement_metrics.task_type = task_type
        
        # Evaluate each modality
        for modality, pred in single_modality_preds.items():
            self.evaluate_modality(modality, y_true, pred, task_type)
        
        # Evaluate fusion
        fusion_metrics = self.evaluate_modality("fusion", y_true, fusion_pred, task_type)
        
        # Calculate fusion gain over each modality
        fusion_gains = {}
        
        if task_type == "regression":
            # For regression, lower error is better
            base_metric = 'rmse'
            for modality, metrics in self.modality_results.items():
                if modality != "fusion":
                    # Calculate relative improvement in RMSE
                    modality_error = metrics[base_metric]
                    fusion_error = fusion_metrics[base_metric]
                    
                    # Relative improvement (positive is better)
                    if modality_error > 0:
                        rel_gain = (modality_error - fusion_error) / modality_error
                    else:
                        rel_gain = 0.0
                    
                    fusion_gains[f"gain_over_{modality}"] = float(rel_gain)
        else:
            # For classification, higher accuracy/F1/AUC is better
            base_metric = 'f1'
            for modality, metrics in self.modality_results.items():
                if modality != "fusion":
                    # Calculate relative improvement in F1
                    modality_score = metrics[base_metric]
                    fusion_score = fusion_metrics[base_metric]
                    
                    # Relative improvement (positive is better)
                    if modality_score > 0:
                        rel_gain = (fusion_score - modality_score) / modality_score
                    else:
                        rel_gain = fusion_score > 0
                    
                    fusion_gains[f"gain_over_{modality}"] = float(rel_gain)
        
        # Add fusion gains to fusion metrics
        fusion_metrics.update(fusion_gains)
        self.modality_results["fusion"] = fusion_metrics
        
        return fusion_metrics
    
    def plot_fusion_gain(
        self,
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        Plot fusion gain over single modalities.
        
        Args:
            figsize: Figure size
        """
        if "fusion" not in self.modality_results:
            print("No fusion results to plot")
            return
        
        # Extract fusion gains
        gains = {k: v for k, v in self.modality_results["fusion"].items() if k.startswith("gain_over_")}
        
        if not gains:
            print("No fusion gains to plot")
            return
        
        # Plot gains
        plt.figure(figsize=figsize)
        
        modalities = [k.replace("gain_over_", "") for k in gains.keys()]
        gain_values = list(gains.values())
        
        # Create bar chart
        plt.bar(modalities, gain_values)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        plt.xlabel("Modality")
        plt.ylabel("Relative Gain from Fusion")
        plt.title("Multimodal Fusion Performance Gain")
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(gain_values):
            plt.text(i, v + 0.01 * (1 if v >= 0 else -1),
                    f"{v:.1%}", ha='center')
        
        plt.tight_layout()
        plt.show()
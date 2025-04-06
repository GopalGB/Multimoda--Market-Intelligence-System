# causal/counterfactual.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from .structural_model import CausalGraph, StructuralCausalModel

class CounterfactualAnalyzer:
    """
    Generate and analyze counterfactual scenarios for audience engagement.
    
    This class helps answer "what-if" questions about audience engagement:
    - What if we changed the content features?
    - How would engagement change if we modify specific attributes?
    - Which feature changes would have the largest impact?
    
    It uses the structural causal model to simulate interventions and
    estimate their effects on engagement metrics.
    """
    def __init__(
        self,
        causal_model: StructuralCausalModel,
        feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        """
        Initialize the counterfactual analyzer.
        
        Args:
            causal_model: Structural causal model
            feature_ranges: Dictionary mapping features to (min, max) ranges
        """
        self.causal_model = causal_model
        self.feature_ranges = feature_ranges or {}
        self.counterfactual_results = {}
    
    def generate_counterfactual(
        self,
        data: pd.DataFrame,
        interventions: Dict[str, float],
        outcome_var: str,
        reference_row: Optional[int] = None,
        reference_values: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Generate a counterfactual scenario based on interventions.
        
        Args:
            data: DataFrame containing original data
            interventions: Dictionary mapping features to intervention values
            outcome_var: Name of the outcome variable
            reference_row: Index of reference row in data
            reference_values: Dictionary with reference values
            
        Returns:
            Dictionary with counterfactual results
        """
        # Validate inputs
        if reference_row is None and reference_values is None:
            raise ValueError("Either reference_row or reference_values must be provided")
            
        # Get reference values
        if reference_values is None:
            if reference_row not in data.index:
                raise ValueError(f"Reference row {reference_row} not found in data")
                
            reference_values = data.loc[reference_row].to_dict()
        
        # Ensure outcome variable is in data
        if outcome_var not in data.columns:
            raise ValueError(f"Outcome variable {outcome_var} not found in data")
            
        # Generate counterfactual
        factual_outcome = reference_values.get(outcome_var, None)
        if factual_outcome is None and reference_row is not None:
            factual_outcome = data.loc[reference_row, outcome_var]
        
        # Apply interventions
        counterfactual_values = reference_values.copy()
        for feature, value in interventions.items():
            # Apply feature range constraints if available
            if feature in self.feature_ranges:
                min_val, max_val = self.feature_ranges[feature]
                value = max(min_val, min(max_val, value))
                
            counterfactual_values[feature] = value
        
        # Predict counterfactual outcome
        counterfactual_outcome = self.causal_model.do_intervention(
            data, interventions, outcome_var
        )
        
        # Store counterfactual scenario
        result = {
            "factual_values": {k: v for k, v in reference_values.items() if k != outcome_var},
            "factual_outcome": factual_outcome,
            "interventions": interventions,
            "counterfactual_values": {k: v for k, v in counterfactual_values.items() if k != outcome_var},
            "counterfactual_outcome": counterfactual_outcome,
            "outcome_change": counterfactual_outcome - factual_outcome if factual_outcome is not None else None,
            "outcome_change_percent": ((counterfactual_outcome - factual_outcome) / factual_outcome * 100) if factual_outcome is not None and factual_outcome != 0 else None
        }
        
        # Add to results history
        scenario_id = len(self.counterfactual_results) + 1
        self.counterfactual_results[scenario_id] = result
        
        return result
    
    def generate_multiple_counterfactuals(
        self,
        data: pd.DataFrame,
        feature_interventions: Dict[str, List[float]],
        outcome_var: str,
        reference_row: Optional[int] = None,
        reference_values: Optional[Dict[str, float]] = None,
        max_combinations: int = 100
    ) -> Dict[int, Dict[str, Any]]:
        """
        Generate multiple counterfactual scenarios with different interventions.
        
        Args:
            data: DataFrame containing original data
            feature_interventions: Dictionary mapping features to lists of intervention values
            outcome_var: Name of the outcome variable
            reference_row: Index of reference row in data
            reference_values: Dictionary with reference values
            max_combinations: Maximum number of combinations to generate
            
        Returns:
            Dictionary mapping scenario IDs to counterfactual results
        """
        import itertools
        
        # Get all feature combinations
        features = list(feature_interventions.keys())
        feature_values = [feature_interventions[feature] for feature in features]
        
        # Generate combinations
        all_combinations = list(itertools.product(*feature_values))
        
        # Limit number of combinations if needed
        if len(all_combinations) > max_combinations:
            import random
            random.seed(42)  # For reproducibility
            all_combinations = random.sample(all_combinations, max_combinations)
        
        # Generate counterfactuals for each combination
        counterfactuals = {}
        
        for i, values in enumerate(all_combinations, 1):
            # Create intervention dictionary
            interventions = {feature: value for feature, value in zip(features, values)}
            
            # Generate counterfactual
            result = self.generate_counterfactual(
                data, interventions, outcome_var, reference_row, reference_values
            )
            
            # Store with sequential ID
            counterfactuals[i] = result
        
        return counterfactuals
    
    def compare_counterfactuals(
        self,
        scenario_ids: Optional[List[int]] = None,
        sort_by: str = "outcome_change",
        ascending: bool = False
    ) -> pd.DataFrame:
        """
        Compare multiple counterfactual scenarios.
        
        Args:
            scenario_ids: List of scenario IDs to compare (None for all)
            sort_by: Column to sort by
            ascending: Whether to sort in ascending order
            
        Returns:
            DataFrame with counterfactual comparison
        """
        if not self.counterfactual_results:
            raise ValueError("No counterfactual results available")
        
        # Select scenarios
        if scenario_ids is None:
            scenarios = self.counterfactual_results
        else:
            scenarios = {
                id: self.counterfactual_results[id]
                for id in scenario_ids
                if id in self.counterfactual_results
            }
        
        # Extract comparison data
        comparison_data = []
        
        for scenario_id, result in scenarios.items():
            # Create row for comparison
            row = {
                "scenario_id": scenario_id,
                "factual_outcome": result["factual_outcome"],
                "counterfactual_outcome": result["counterfactual_outcome"],
                "outcome_change": result["outcome_change"],
                "outcome_change_percent": result["outcome_change_percent"]
            }
            
            # Add interventions
            for feature, value in result["interventions"].items():
                row[f"intervention_{feature}"] = value
                
                # Add change from factual
                factual_value = result["factual_values"].get(feature)
                if factual_value is not None:
                    row[f"change_{feature}"] = value - factual_value
                    
                    # Add percent change
                    if factual_value != 0:
                        row[f"change_percent_{feature}"] = (value - factual_value) / factual_value * 100
            
            comparison_data.append(row)
        
        # Create DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by specified column
        if sort_by in comparison_df.columns:
            comparison_df = comparison_df.sort_values(sort_by, ascending=ascending)
        
        return comparison_df
    
    def visualize_counterfactual_effects(
        self,
        feature: str,
        outcome_var: str,
        data: pd.DataFrame,
        reference_row: Optional[int] = None,
        reference_values: Optional[Dict[str, float]] = None,
        num_points: int = 20,
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        Visualize the effect of varying a feature on the outcome.
        
        Args:
            feature: Feature to vary
            outcome_var: Name of the outcome variable
            data: DataFrame containing original data
            reference_row: Index of reference row in data
            reference_values: Dictionary with reference values
            num_points: Number of intervention points to generate
            figsize: Figure size
        """
        # Validate inputs
        if reference_row is None and reference_values is None:
            raise ValueError("Either reference_row or reference_values must be provided")
            
        # Get reference values
        if reference_values is None:
            if reference_row not in data.index:
                raise ValueError(f"Reference row {reference_row} not found in data")
                
            reference_values = data.loc[reference_row].to_dict()
        
        # Get feature range
        if feature in self.feature_ranges:
            min_val, max_val = self.feature_ranges[feature]
        else:
            min_val = data[feature].min()
            max_val = data[feature].max()
        
        # Generate intervention values
        intervention_values = np.linspace(min_val, max_val, num_points)
        
        # Generate counterfactuals
        outcomes = []
        
        for value in intervention_values:
            interventions = {feature: value}
            result = self.generate_counterfactual(
                data, interventions, outcome_var, reference_row, reference_values
            )
            outcomes.append(result["counterfactual_outcome"])
        
        # Create plot
        plt.figure(figsize=figsize)
        plt.plot(intervention_values, outcomes, marker='o', linestyle='-')
        
        # Add reference point
        factual_value = reference_values.get(feature)
        factual_outcome = reference_values.get(outcome_var)
        
        if factual_value is not None and factual_outcome is not None:
            plt.axvline(x=factual_value, color='r', linestyle='--', alpha=0.7, label='Factual Value')
            plt.axhline(y=factual_outcome, color='r', linestyle='--', alpha=0.7, label='Factual Outcome')
            plt.plot(factual_value, factual_outcome, 'ro', ms=10, label='Factual Point')
        
        # Add labels and title
        plt.xlabel(feature)
        plt.ylabel(outcome_var)
        plt.title(f'Effect of {feature} on {outcome_var}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Show plot
        plt.tight_layout()
        plt.show()
    
    def visualize_multiple_feature_effects(
        self,
        features: List[str],
        outcome_var: str,
        data: pd.DataFrame,
        reference_row: Optional[int] = None,
        reference_values: Optional[Dict[str, float]] = None,
        num_points: int = 10,
        figsize: Tuple[int, int] = (15, 10)
    ) -> None:
        """
        Visualize effects of multiple features on the outcome in separate subplots.
        
        Args:
            features: List of features to vary
            outcome_var: Name of the outcome variable
            data: DataFrame containing original data
            reference_row: Index of reference row in data
            reference_values: Dictionary with reference values
            num_points: Number of intervention points per feature
            figsize: Figure size
        """
        # Validate inputs
        if reference_row is None and reference_values is None:
            raise ValueError("Either reference_row or reference_values must be provided")
            
        # Get reference values
        if reference_values is None:
            if reference_row not in data.index:
                raise ValueError(f"Reference row {reference_row} not found in data")
                
            reference_values = data.loc[reference_row].to_dict()
        
        # Create subplots
        num_cols = min(3, len(features))
        num_rows = (len(features) + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        # Generate counterfactuals for each feature
        for i, feature in enumerate(features):
            # Get feature range
            if feature in self.feature_ranges:
                min_val, max_val = self.feature_ranges[feature]
            else:
                min_val = data[feature].min()
                max_val = data[feature].max()
            
            # Generate intervention values
            intervention_values = np.linspace(min_val, max_val, num_points)
            
            # Generate counterfactuals
            outcomes = []
            
            for value in intervention_values:
                interventions = {feature: value}
                result = self.generate_counterfactual(
                    data, interventions, outcome_var, reference_row, reference_values
                )
                outcomes.append(result["counterfactual_outcome"])
            
            # Plot on subplot
            ax = axes[i]
            ax.plot(intervention_values, outcomes, marker='o', linestyle='-')
            
            # Add reference point
            factual_value = reference_values.get(feature)
            factual_outcome = reference_values.get(outcome_var)
            
            if factual_value is not None and factual_outcome is not None:
                ax.axvline(x=factual_value, color='r', linestyle='--', alpha=0.7)
                ax.axhline(y=factual_outcome, color='r', linestyle='--', alpha=0.7)
                ax.plot(factual_value, factual_outcome, 'ro', ms=10)
            
            # Add labels
            ax.set_xlabel(feature)
            ax.set_ylabel(outcome_var)
            ax.set_title(f'Effect of {feature}')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        # Add legend for the figure
        fig.legend(['Counterfactual', 'Factual Value', 'Factual Outcome', 'Factual Point'], 
                 loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=4)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(f'Effects of Features on {outcome_var}', fontsize=16)
        plt.show()
    
    def find_optimal_intervention(
        self,
        data: pd.DataFrame,
        outcome_var: str,
        target_outcome: float,
        candidate_features: Optional[List[str]] = None,
        reference_row: Optional[int] = None,
        reference_values: Optional[Dict[str, float]] = None,
        max_features: int = 3,
        constraints: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        """
        Find the optimal intervention to achieve a target outcome.
        
        Args:
            data: DataFrame containing original data
            outcome_var: Name of the outcome variable
            target_outcome: Target value for the outcome variable
            candidate_features: List of features to consider for intervention
            reference_row: Index of reference row in data
            reference_values: Dictionary with reference values
            max_features: Maximum number of features to modify
            constraints: Dictionary mapping features to (min, max) constraints
            
        Returns:
            Dictionary with optimal intervention details
        """
        # Validate inputs
        if reference_row is None and reference_values is None:
            raise ValueError("Either reference_row or reference_values must be provided")
            
        # Get reference values
        if reference_values is None:
            if reference_row not in data.index:
                raise ValueError(f"Reference row {reference_row} not found in data")
                
            reference_values = data.loc[reference_row].to_dict()
        
        # Get candidate features
        if candidate_features is None:
            # Use all features with causal effects
            candidate_features = list(self.causal_model.causal_effects.keys())
        
        # Apply feature constraints
        feature_constraints = constraints or {}
        
        # Add default constraints from feature_ranges
        for feature in candidate_features:
            if feature not in feature_constraints and feature in self.feature_ranges:
                feature_constraints[feature] = self.feature_ranges[feature]
        
        # Use causal model to find optimal intervention
        optimal_result = self.causal_model.identify_optimal_intervention(
            data,
            candidate_features,
            outcome_var,
            maximize=target_outcome > reference_values.get(outcome_var, 0),
            n_points=10
        )
        
        # Get optimal intervention
        optimal_intervention = optimal_result["optimal_intervention"]
        
        # Generate counterfactual with optimal intervention
        counterfactual = self.generate_counterfactual(
            data, optimal_intervention, outcome_var, reference_row, reference_values
        )
        
        # Combine results
        result = {
            "optimal_intervention": optimal_intervention,
            "predicted_outcome": optimal_result["predicted_outcome"],
            "target_outcome": target_outcome,
            "outcome_gap": optimal_result["predicted_outcome"] - target_outcome,
            "outcome_gap_percent": (optimal_result["predicted_outcome"] - target_outcome) / target_outcome * 100 if target_outcome != 0 else None,
            "baseline_outcome": optimal_result["baseline"],
            "improvement": optimal_result["improvement"],
            "percent_improvement": optimal_result["percent_improvement"],
            "counterfactual_details": counterfactual
        }
        
        return result
    
    def analyze_sensitivity(
        self,
        data: pd.DataFrame,
        outcome_var: str,
        feature: str,
        reference_row: Optional[int] = None,
        reference_values: Optional[Dict[str, float]] = None,
        perturbation_range: float = 0.1,
        num_points: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze sensitivity of outcome to perturbations in a feature.
        
        Args:
            data: DataFrame containing original data
            outcome_var: Name of the outcome variable
            feature: Feature to analyze sensitivity for
            reference_row: Index of reference row in data
            reference_values: Dictionary with reference values
            perturbation_range: Range of perturbation as proportion of feature value
            num_points: Number of perturbation points
            
        Returns:
            Dictionary with sensitivity analysis results
        """
        # Validate inputs
        if reference_row is None and reference_values is None:
            raise ValueError("Either reference_row or reference_values must be provided")
            
        # Get reference values
        if reference_values is None:
            if reference_row not in data.index:
                raise ValueError(f"Reference row {reference_row} not found in data")
                
            reference_values = data.loc[reference_row].to_dict()
        
        # Get reference feature value
        reference_value = reference_values.get(feature)
        if reference_value is None:
            raise ValueError(f"Reference value for {feature} not found")
        
        # Generate perturbation values
        perturbation_factor = np.linspace(
            1 - perturbation_range,
            1 + perturbation_range,
            num_points
        )
        
        perturbation_values = reference_value * perturbation_factor
        
        # Generate counterfactuals for each perturbation
        outcomes = []
        
        for value in perturbation_values:
            interventions = {feature: value}
            result = self.generate_counterfactual(
                data, interventions, outcome_var, reference_row, reference_values
            )
            outcomes.append(result["counterfactual_outcome"])
        
        # Calculate sensitivity metrics
        perturbation_deltas = perturbation_values - reference_value
        outcome_deltas = np.array(outcomes) - reference_values.get(outcome_var, 0)
        
        # Calculate elasticity (% change in outcome / % change in feature)
        elasticities = []
        for i in range(len(perturbation_values)):
            if perturbation_deltas[i] == 0:
                elasticities.append(np.nan)
            else:
                percent_change_feature = perturbation_deltas[i] / reference_value * 100
                percent_change_outcome = outcome_deltas[i] / reference_values.get(outcome_var, 1) * 100
                elasticities.append(percent_change_outcome / percent_change_feature)
        
        # Average elasticity (excluding NaN values)
        avg_elasticity = np.nanmean(elasticities)
        
        # Prepare result
        result = {
            "feature": feature,
            "reference_value": reference_value,
            "perturbation_values": perturbation_values.tolist(),
            "outcomes": outcomes,
            "outcome_deltas": outcome_deltas.tolist(),
            "elasticities": elasticities,
            "average_elasticity": avg_elasticity,
            "perturbation_range": perturbation_range,
            "sensitivity_score": abs(avg_elasticity)
        }
        
        return result
    
    def analyze_feature_sensitivities(
        self,
        data: pd.DataFrame,
        outcome_var: str,
        features: Optional[List[str]] = None,
        reference_row: Optional[int] = None,
        reference_values: Optional[Dict[str, float]] = None,
        perturbation_range: float = 0.1,
        num_points: int = 5
    ) -> pd.DataFrame:
        """
        Analyze sensitivities for multiple features.
        
        Args:
            data: DataFrame containing original data
            outcome_var: Name of the outcome variable
            features: List of features to analyze (None for all causal features)
            reference_row: Index of reference row in data
            reference_values: Dictionary with reference values
            perturbation_range: Range of perturbation as proportion of feature value
            num_points: Number of perturbation points
            
        Returns:
            DataFrame with sensitivity analysis for all features
        """
        # Get features to analyze
        if features is None:
            features = list(self.causal_model.causal_effects.keys())
        
        # Analyze sensitivity for each feature
        sensitivities = []
        
        for feature in features:
            try:
                sensitivity = self.analyze_sensitivity(
                    data, outcome_var, feature, 
                    reference_row, reference_values,
                    perturbation_range, num_points
                )
                
                sensitivities.append({
                    "feature": feature,
                    "sensitivity_score": sensitivity["sensitivity_score"],
                    "average_elasticity": sensitivity["average_elasticity"],
                    "reference_value": sensitivity["reference_value"]
                })
            except ValueError:
                # Skip features with missing reference values
                continue
        
        # Create DataFrame and sort by sensitivity score
        sensitivity_df = pd.DataFrame(sensitivities)
        sensitivity_df = sensitivity_df.sort_values("sensitivity_score", ascending=False)
        
        return sensitivity_df
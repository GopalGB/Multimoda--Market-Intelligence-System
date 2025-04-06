# causal/causal_features.py
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Set, Any, Optional, Union
import matplotlib.pyplot as plt
import networkx as nx
from .structural_model import CausalGraph, StructuralCausalModel

class CausalFeatureSelector:
    """
    A class to identify causal features that influence audience engagement.
    
    This class provides methods to analyze content features and determine
    which ones have causal effects on audience engagement metrics, helping
    to distinguish causation from correlation in audience behavior data.
    """
    def __init__(
        self,
        causal_model: Optional[StructuralCausalModel] = None,
        alpha: float = 0.05,
        min_effect_size: float = 0.1
    ):
        """
        Initialize the causal feature selector.
        
        Args:
            causal_model: Optional pre-trained causal model
            alpha: Significance level for causal tests
            min_effect_size: Minimum causal effect size to consider
        """
        self.causal_model = causal_model
        self.alpha = alpha
        self.min_effect_size = min_effect_size
        self.causal_features = {}
        self.feature_effects = {}
    
    def fit(
        self,
        data: pd.DataFrame,
        outcome_var: str,
        feature_vars: Optional[List[str]] = None,
        discovery_method: str = 'pc',
        exclude_vars: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Discover causal features from data.
        
        Args:
            data: DataFrame containing feature and outcome data
            outcome_var: Name of the outcome variable (e.g., 'engagement')
            feature_vars: List of feature variables to consider
            discovery_method: Method for causal discovery ('pc', 'notears', 'manual')
            exclude_vars: Variables to exclude from analysis
            
        Returns:
            Dictionary mapping causal features to their effect sizes
        """
        # Validate inputs
        if outcome_var not in data.columns:
            raise ValueError(f"Outcome variable '{outcome_var}' not found in data")
        
        # Prepare feature variables
        if feature_vars is None:
            feature_vars = [col for col in data.columns if col != outcome_var]
            
            # Exclude variables if specified
            if exclude_vars:
                feature_vars = [var for var in feature_vars if var not in exclude_vars]
        
        # Initialize causal model if not provided
        if self.causal_model is None:
            self.causal_model = StructuralCausalModel(
                discovery_method=discovery_method,
                alpha=self.alpha,
                feature_names=feature_vars + [outcome_var]
            )
        
        # Discover causal graph
        self.causal_model.discover_graph(data, outcome_var)
        
        # Estimate causal effects
        causal_effects = self.causal_model.estimate_all_effects(
            data, outcome_var, min_effect=self.min_effect_size
        )
        
        # Store causal features and effects
        self.causal_features = {
            feature: effect["causal_effect"]
            for feature, effect in causal_effects.items()
        }
        
        self.feature_effects = causal_effects
        
        return self.causal_features
    
    def get_top_features(self, n: int = 5) -> Dict[str, float]:
        """
        Get the top n causal features.
        
        Args:
            n: Number of top features to return
            
        Returns:
            Dictionary with top causal features and their effects
        """
        if not self.causal_features:
            raise ValueError("No causal features identified. Run fit() first.")
        
        # Sort features by absolute effect size
        sorted_features = sorted(
            self.causal_features.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Get top n features
        top_features = dict(sorted_features[:n])
        
        return top_features
    
    def visualize_causal_features(
        self,
        figsize: Tuple[int, int] = (12, 8),
        show_effect_size: bool = True
    ) -> None:
        """
        Visualize causal features and their effects.
        
        Args:
            figsize: Figure size
            show_effect_size: Whether to show effect sizes on edges
        """
        if not self.causal_features or not self.causal_model:
            raise ValueError("No causal features identified. Run fit() first.")
        
        plt.figure(figsize=figsize)
        
        # Get the causal graph
        G = self.causal_model.causal_graph.graph
        
        # Create layout
        pos = nx.spring_layout(G, seed=42)
        
        # Get outcome variable
        outcome_nodes = [node for node, attrs in G.nodes(data=True) 
                        if attrs.get('type') == 'outcome']
        outcome_var = outcome_nodes[0] if outcome_nodes else None
        
        # Get causal features
        causal_feature_nodes = list(self.causal_features.keys())
        
        # Get other nodes
        other_nodes = [node for node in G.nodes() 
                      if node != outcome_var and node not in causal_feature_nodes]
        
        # Draw nodes
        if outcome_var:
            nx.draw_networkx_nodes(G, pos, nodelist=[outcome_var], 
                                node_color='lightcoral', node_size=2000, alpha=0.8)
        
        nx.draw_networkx_nodes(G, pos, nodelist=causal_feature_nodes, 
                             node_color='lightgreen', node_size=1500, alpha=0.8)
        
        nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, 
                             node_color='lightblue', node_size=1000, alpha=0.8)
        
        # Draw edges
        causal_edges = []
        other_edges = []
        
        for u, v in G.edges():
            if u in causal_feature_nodes and v == outcome_var:
                causal_edges.append((u, v))
            else:
                other_edges.append((u, v))
        
        nx.draw_networkx_edges(G, pos, edgelist=causal_edges, width=2, alpha=0.8, 
                             edge_color='darkgreen', arrows=True, arrowstyle='->', arrowsize=20)
        
        nx.draw_networkx_edges(G, pos, edgelist=other_edges, width=1, alpha=0.5, 
                             edge_color='gray', arrows=True, arrowstyle='->')
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        # Draw edge labels if requested
        if show_effect_size and outcome_var:
            edge_labels = {}
            for u, v in causal_edges:
                if u in self.causal_features:
                    effect = self.causal_features[u]
                    edge_labels[(u, v)] = f"{effect:.3f}"
            
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
        
        # Add legend
        plt.legend([
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', markersize=15),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=15),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=15),
            plt.Line2D([0], [0], color='darkgreen', lw=2),
            plt.Line2D([0], [0], color='gray', lw=1)
        ], [
            'Outcome Variable',
            'Causal Features',
            'Other Variables',
            'Causal Effect',
            'Other Relationship'
        ], loc='best')
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def feature_effect_summary(self) -> pd.DataFrame:
        """
        Generate a summary of causal feature effects.
        
        Returns:
            DataFrame with feature effect summary
        """
        if not self.feature_effects:
            raise ValueError("No causal effects estimated. Run fit() first.")
        
        # Create summary DataFrame
        summary_data = []
        
        for feature, effect_info in self.feature_effects.items():
            summary_data.append({
                'feature': feature,
                'causal_effect': effect_info['causal_effect'],
                'standard_error': effect_info.get('standard_error', np.nan),
                'p_value': effect_info.get('p_value', np.nan),
                'significant': effect_info.get('p_value', 1.0) < self.alpha
            })
        
        # Create DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by absolute causal effect
        summary_df = summary_df.assign(
            abs_effect=summary_df['causal_effect'].abs()
        ).sort_values('abs_effect', ascending=False).drop('abs_effect', axis=1)
        
        return summary_df
    
    def identify_feature_interactions(
        self,
        data: pd.DataFrame,
        outcome_var: str,
        top_n: int = 5
    ) -> Dict[Tuple[str, str], float]:
        """
        Identify interactions between causal features.
        
        Args:
            data: DataFrame containing feature and outcome data
            outcome_var: Name of the outcome variable
            top_n: Number of top interactions to return
            
        Returns:
            Dictionary mapping feature interactions to effect sizes
        """
        if not self.causal_features:
            raise ValueError("No causal features identified. Run fit() first.")
        
        # Get top causal features
        top_features = list(self.get_top_features(top_n).keys())
        
        # Create interaction features
        interactions = {}
        
        for i in range(len(top_features)):
            for j in range(i+1, len(top_features)):
                feat1, feat2 = top_features[i], top_features[j]
                
                # Create interaction column
                interaction_name = f"{feat1}_x_{feat2}"
                data[interaction_name] = data[feat1] * data[feat2]
                
                # Estimate causal effect of interaction
                effect = self.causal_model.estimate_causal_effect(
                    data, interaction_name, outcome_var
                )
                
                # Store if significant
                if effect['p_value'] < self.alpha and abs(effect['causal_effect']) > self.min_effect_size:
                    interactions[(feat1, feat2)] = effect['causal_effect']
                
                # Remove interaction column
                data.drop(interaction_name, axis=1, inplace=True)
        
        # Sort by absolute effect size
        sorted_interactions = dict(sorted(
            interactions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ))
        
        return sorted_interactions
    
    def calculate_feature_importance(self) -> Dict[str, float]:
        """
        Calculate importance scores for causal features.
        
        Importance considers effect size, significance, and
        centrality in the causal graph.
        
        Returns:
            Dictionary mapping features to importance scores
        """
        if not self.feature_effects or not self.causal_model:
            raise ValueError("No causal effects estimated. Run fit() first.")
        
        # Get causal graph
        G = self.causal_model.causal_graph.graph
        
        # Calculate centrality measures
        centrality = nx.betweenness_centrality(G)
        
        # Calculate importance scores
        importance_scores = {}
        
        for feature, effect_info in self.feature_effects.items():
            # Get effect size and p-value
            effect_size = abs(effect_info['causal_effect'])
            p_value = effect_info.get('p_value', 1.0)
            
            # Get centrality
            feature_centrality = centrality.get(feature, 0)
            
            # Calculate importance score
            # Weight by effect size, significance, and centrality
            significance_factor = -np.log10(max(p_value, 1e-10)) / 10
            importance = effect_size * (1 + significance_factor) * (1 + feature_centrality)
            
            importance_scores[feature] = importance
        
        # Normalize importance scores
        max_importance = max(importance_scores.values()) if importance_scores else 1
        importance_scores = {k: v / max_importance for k, v in importance_scores.items()}
        
        # Sort by importance
        importance_scores = dict(sorted(
            importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return importance_scores
    
    def generate_feature_recommendations(
        self,
        target_outcome: float,
        current_values: Dict[str, float],
        constraints: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate recommendations for feature values to achieve target outcome.
        
        Args:
            target_outcome: Target value for the outcome variable
            current_values: Current values of features
            constraints: Dictionary mapping features to (min, max) constraints
            
        Returns:
            Dictionary with feature recommendations
        """
        if not self.causal_model or not self.causal_features:
            raise ValueError("No causal model available. Run fit() first.")
        
        # Get top causal features
        causal_features = self.causal_features
        
        # Apply constraints if provided
        if constraints:
            for feature, (min_val, max_val) in constraints.items():
                if feature in current_values:
                    current_values[feature] = min(max(current_values[feature], min_val), max_val)
        
        # Calculate current outcome prediction
        current_outcome = self._predict_outcome(current_values)
        
        # Calculate target change
        target_change = target_outcome - current_outcome
        
        # Generate recommendations
        recommendations = {}
        
        for feature, effect in causal_features.items():
            if feature not in current_values:
                continue
            
            # Calculate required change
            if effect != 0:
                required_change = target_change / effect
                new_value = current_values[feature] + required_change
                
                # Apply constraints if provided
                if constraints and feature in constraints:
                    min_val, max_val = constraints[feature]
                    new_value = min(max(new_value, min_val), max_val)
                    actual_change = new_value - current_values[feature]
                    impact = actual_change * effect
                else:
                    actual_change = required_change
                    impact = target_change
                
                # Add to recommendations
                recommendations[feature] = {
                    'current_value': current_values[feature],
                    'recommended_value': new_value,
                    'change': actual_change,
                    'impact': impact
                }
        
        # Sort by absolute impact
        recommendations = dict(sorted(
            recommendations.items(),
            key=lambda x: abs(x[1]['impact']),
            reverse=True
        ))
        
        return recommendations
    
    def _predict_outcome(self, feature_values: Dict[str, float]) -> float:
        """
        Predict outcome based on feature values using causal model.
        
        Args:
            feature_values: Dictionary mapping features to values
            
        Returns:
            Predicted outcome value
        """
        # Calculate baseline outcome
        baseline = 0.0
        
        # Add causal effects
        outcome = baseline
        
        for feature, value in feature_values.items():
            if feature in self.causal_features:
                effect = self.causal_features[feature]
                outcome += value * effect
        
        return outcome
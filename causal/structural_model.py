# causal/structural_model.py
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Set, Optional, Union, Any
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import logging
import warnings
from tqdm import tqdm
from pathlib import Path
import pickle
import dowhy
from dowhy import CausalModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)

class CausalGraph:
    """
    Graph representation of causal relationships between variables.
    
    This class represents the directed acyclic graph (DAG) structure of
    causal relationships, supporting operations like adding/removing edges,
    checking for cycles, and identifying paths.
    """
    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        Initialize a causal graph.
        
        Args:
            feature_names: List of feature names to initialize the graph with
        """
        # Initialize directed graph
        self.graph = nx.DiGraph()
        
        # Add nodes if feature names provided
        if feature_names:
            for name in feature_names:
                self.add_node(name)
                
        # Track graph properties
        self.has_cycles = False
        self.backdoor_paths = {}
    
    def add_node(self, node_name: str, **attr) -> None:
        """
        Add a node to the graph.
        
        Args:
            node_name: Name of the node to add
            **attr: Optional node attributes
        """
        self.graph.add_node(node_name, **attr)
    
    def add_edge(self, source: str, target: str, **attr) -> None:
        """
        Add a directed edge to the graph.
        
        Args:
            source: Source node name
            target: Target node name
            **attr: Optional edge attributes like weight
        """
        # Add edge
        self.graph.add_edge(source, target, **attr)
        
        # Check for cycles
        self.has_cycles = not nx.is_directed_acyclic_graph(self.graph)
        if self.has_cycles:
            logger.warning(f"Adding edge {source} -> {target} created a cycle in the graph")
            
        # Reset cached paths since graph structure changed
        self.backdoor_paths = {}
    
    def remove_edge(self, source: str, target: str) -> None:
        """
        Remove an edge from the graph.
        
        Args:
            source: Source node name
            target: Target node name
        """
        if self.graph.has_edge(source, target):
            self.graph.remove_edge(source, target)
            # Update cycle status
            self.has_cycles = not nx.is_directed_acyclic_graph(self.graph)
            # Reset cached paths
            self.backdoor_paths = {}
    
    def get_parents(self, node: str) -> List[str]:
        """
        Get parent nodes of a given node.
        
        Args:
            node: Node name
            
        Returns:
            List of parent node names
        """
        return list(self.graph.predecessors(node))
    
    def get_children(self, node: str) -> List[str]:
        """
        Get child nodes of a given node.
        
        Args:
            node: Node name
            
        Returns:
            List of child node names
        """
        return list(self.graph.successors(node))
    
    def get_ancestors(self, node: str) -> Set[str]:
        """
        Get all ancestor nodes of a given node.
        
        Args:
            node: Node name
            
        Returns:
            Set of ancestor node names
        """
        return nx.ancestors(self.graph, node)
    
    def get_descendants(self, node: str) -> Set[str]:
        """
        Get all descendant nodes of a given node.
        
        Args:
            node: Node name
            
        Returns:
            Set of descendant node names
        """
        return nx.descendants(self.graph, node)
    
    def is_collider(self, node: str) -> bool:
        """
        Check if a node is a collider (has multiple parents).
        
        Args:
            node: Node name
            
        Returns:
            True if node is a collider
        """
        return len(list(self.graph.predecessors(node))) >= 2
    
    def find_backdoor_paths(self, treatment: str, outcome: str) -> List[List[str]]:
        """
        Find all backdoor paths between treatment and outcome.
        
        Args:
            treatment: Treatment node name
            outcome: Outcome node name
            
        Returns:
            List of backdoor paths (each path is a list of nodes)
        """
        # Check if we've already computed this
        key = (treatment, outcome)
        if key in self.backdoor_paths:
            return self.backdoor_paths[key]
            
        # Get ancestors of outcome
        outcome_ancestors = self.get_ancestors(outcome)
        
        # Find all paths from treatment parents to outcome or its ancestors
        backdoor_paths = []
        for parent in self.get_parents(treatment):
            # Temporary remove the edge from parent to treatment
            self.graph.remove_edge(parent, treatment)
            
            # Find paths from parent to outcome or its ancestors
            for target in outcome_ancestors.union({outcome}):
                try:
                    for path in nx.all_simple_paths(self.graph, parent, target):
                        # Add the treatment node to complete the backdoor path
                        complete_path = [treatment] + path
                        backdoor_paths.append(complete_path)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
            
            # Restore the edge
            self.graph.add_edge(parent, treatment)
                
        # Cache result
        self.backdoor_paths[key] = backdoor_paths
        
        return backdoor_paths
    
    def get_minimal_adjustment_set(self, treatment: str, outcome: str) -> Set[str]:
        """
        Get a minimal adjustment set to block all backdoor paths.
        
        This identifies a set of variables that, when conditioned on,
        blocks all backdoor paths from treatment to outcome.
        
        Args:
            treatment: Treatment node name
            outcome: Outcome node name
            
        Returns:
            Set of nodes forming a minimal adjustment set
        """
        # Using the backdoor criterion
        adjustment_set = set()
        
        # Add all parents of treatment
        adjustment_set.update(self.get_parents(treatment))
        
        # Remove descendants of treatment
        treatment_descendants = self.get_descendants(treatment)
        adjustment_set = adjustment_set - treatment_descendants
        
        return adjustment_set
    
    def is_valid_adjustment_set(self, adjustment_set: Set[str], treatment: str, outcome: str) -> bool:
        """
        Check if a set of variables is a valid adjustment set.
        
        Args:
            adjustment_set: Set of variables to adjust for
            treatment: Treatment node name
            outcome: Outcome node name
            
        Returns:
            True if the adjustment set is valid
        """
        # Cannot include descendants of treatment
        treatment_descendants = self.get_descendants(treatment)
        if any(node in treatment_descendants for node in adjustment_set):
            return False
            
        # Check if all backdoor paths are blocked
        backdoor_paths = self.find_backdoor_paths(treatment, outcome)
        
        for path in backdoor_paths:
            # Path is blocked if it contains a node in the adjustment set
            if not any(node in adjustment_set for node in path):
                return False
                
        return True
    
    def visualize(
        self,
        highlight_nodes: Optional[List[str]] = None,
        highlight_edges: Optional[List[Tuple[str, str]]] = None,
        figsize: Tuple[int, int] = (10, 8),
        node_size: int = 2000,
        node_colors: Optional[Dict[str, str]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize the causal graph.
        
        Args:
            highlight_nodes: List of nodes to highlight
            highlight_edges: List of edges to highlight
            figsize: Figure size
            node_size: Size of nodes
            node_colors: Dictionary mapping node types to colors
            save_path: Path to save figure (None to display)
        """
        plt.figure(figsize=figsize)
        
        # Set up layout
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Default colors
        if node_colors is None:
            node_colors = {
                "default": "skyblue",
                "highlight": "lightcoral",
                "treatment": "lightgreen",
                "outcome": "gold"
            }
        
        # Draw nodes
        default_nodes = set(self.graph.nodes())
        highlight_nodes = highlight_nodes or []
        
        # Draw non-highlighted nodes
        remaining_nodes = default_nodes - set(highlight_nodes)
        if remaining_nodes:
            nx.draw_networkx_nodes(
                self.graph, pos,
                nodelist=list(remaining_nodes),
                node_color=node_colors["default"],
                node_size=node_size,
                alpha=0.8
            )
        
        # Draw highlighted nodes
        if highlight_nodes:
            nx.draw_networkx_nodes(
                self.graph, pos,
                nodelist=highlight_nodes,
                node_color=node_colors["highlight"],
                node_size=node_size,
                alpha=0.8
            )
        
        # Draw edges
        default_edges = list(self.graph.edges())
        highlight_edges = highlight_edges or []
        
        # Draw non-highlighted edges
        remaining_edges = [e for e in default_edges if e not in highlight_edges]
        if remaining_edges:
            nx.draw_networkx_edges(
                self.graph, pos,
                edgelist=remaining_edges,
                width=1.5, alpha=0.6, arrows=True,
                arrowsize=20, arrowstyle='->'
            )
        
        # Draw highlighted edges
        if highlight_edges:
            nx.draw_networkx_edges(
                self.graph, pos,
                edgelist=highlight_edges,
                width=2.5, edge_color='red', alpha=0.8,
                arrows=True, arrowstyle='->', arrowsize=25
            )
        
        # Draw labels
        nx.draw_networkx_labels(
            self.graph, pos,
            font_size=12, font_family='sans-serif'
        )
        
        plt.axis('off')
        plt.tight_layout()
        
        # Save or display
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()


class StructuralCausalModel:
    """
    A class for performing structural causal inference using the DoWhy framework.
    
    This class implements methods for:
    1. Causal effect estimation
    2. Heterogeneous treatment effect analysis
    3. Counterfactual predictions
    4. Robustness checks
    5. Causal graph learning
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        outcome_var: str,
        treatment_vars: List[str],
        confounders: Optional[List[str]] = None,
        instruments: Optional[List[str]] = None
    ):
        """
        Initialize the structural causal model.
        
        Args:
            data: DataFrame containing the observational data
            outcome_var: Name of the outcome variable
            treatment_vars: List of treatment variable names
            confounders: Optional list of confounder variable names
            instruments: Optional list of instrumental variable names
        """
        self.data = data
        self.outcome_var = outcome_var
        self.treatment_vars = treatment_vars
        self.confounders = confounders or []
        self.instruments = instruments or []
        
        # Validate inputs
        self._validate_inputs()
        
        # Initialize DoWhy model
        self._initialize_model()
    
    def _validate_inputs(self) -> None:
        """Validate input data and parameters."""
        if self.data.empty:
            raise ValueError("Data cannot be empty")
        
        if self.outcome_var not in self.data.columns:
            raise ValueError(f"Outcome variable {self.outcome_var} not found in data")
        
        for var in self.treatment_vars:
            if var not in self.data.columns:
                raise ValueError(f"Treatment variable {var} not found in data")
        
        for var in self.confounders:
            if var not in self.data.columns:
                raise ValueError(f"Confounder {var} not found in data")
        
        for var in self.instruments:
            if var not in self.data.columns:
                raise ValueError(f"Instrument {var} not found in data")
    
    def _initialize_model(self) -> None:
        """Initialize the DoWhy causal model."""
        # Create treatment string for DoWhy
        treatment = " + ".join(self.treatment_vars)
        
        # Create confounder string
        confounder_str = " + ".join(self.confounders) if self.confounders else "w"
        
        # Create instrument string
        instrument_str = " + ".join(self.instruments) if self.instruments else None
        
        # Create graph
        graph = self._create_graph()
        
        # Initialize DoWhy model
        self.model = CausalModel(
            data=self.data,
            treatment=treatment,
            outcome=self.outcome_var,
            graph=graph,
            instruments=instrument_str
        )
    
    def _create_graph(self) -> str:
        """Create a graph string for DoWhy."""
        # Start with confounders affecting treatments and outcome
        edges = []
        for confounder in self.confounders:
            for treatment in self.treatment_vars:
                edges.append(f"{confounder}->{treatment}")
            edges.append(f"{confounder}->{self.outcome_var}")
        
        # Add treatment to outcome edges
        for treatment in self.treatment_vars:
            edges.append(f"{treatment}->{self.outcome_var}")
        
        # Add instrument edges if present
        if self.instruments:
            for instrument in self.instruments:
                for treatment in self.treatment_vars:
                    edges.append(f"{instrument}->{treatment}")
        
        return "\n".join(edges)
    
    def estimate_causal_effects(self) -> Dict[str, float]:
        """
        Estimate average causal effects for each treatment variable.
        
        Returns:
            Dictionary mapping treatment variables to their estimated causal effects
        """
        effects = {}
        
        for treatment in self.treatment_vars:
            # Identify causal effect
            identified_estimand = self.model.identify_effect()
            
            # Estimate effect
            estimate = self.model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression"
            )
            
            effects[treatment] = estimate.value
            
        return effects
    
    def estimate_heterogeneous_effects(self) -> Dict[str, pd.Series]:
        """
        Estimate heterogeneous treatment effects using random forests.
        
        Returns:
            Dictionary mapping treatment variables to Series of individual effects
        """
        het_effects = {}
        
        for treatment in self.treatment_vars:
            # Prepare features
            X = self.data[self.confounders].copy()
            X[treatment] = self.data[treatment]
            
            # Fit model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, self.data[self.outcome_var])
            
            # Calculate individual effects
            base_pred = model.predict(X)
            X[treatment] = X[treatment] + 1  # Unit increase
            counterfactual_pred = model.predict(X)
            
            het_effects[treatment] = pd.Series(
                counterfactual_pred - base_pred,
                index=self.data.index
            )
            
        return het_effects
    
    def perform_robustness_checks(self) -> Dict[str, Dict]:
        """
        Perform robustness checks on causal estimates.
        
        Returns:
            Dictionary containing results of various robustness checks
        """
        results = {}
        
        # Sensitivity analysis
        sensitivity = self.model.refute_estimate(
            self.model.identify_effect(),
            self.model.estimate_effect(
                self.model.identify_effect(),
                method_name="backdoor.linear_regression"
            ),
            method_name="random_common_cause"
        )
        
        results['sensitivity_analysis'] = {
            'refutation_result': str(sensitivity),
            'is_robust': sensitivity.refutation_result
        }
        
        # Placebo tests
        placebo_results = {}
        for treatment in self.treatment_vars:
            # Create placebo treatment
            placebo_data = self.data.copy()
            placebo_data[treatment] = np.random.permutation(placebo_data[treatment])
            
            # Estimate effect with placebo
            placebo_model = CausalModel(
                data=placebo_data,
                treatment=treatment,
                outcome=self.outcome_var,
                graph=self._create_graph()
            )
            
            placebo_effect = placebo_model.estimate_effect(
                placebo_model.identify_effect(),
                method_name="backdoor.linear_regression"
            )
            
            placebo_results[treatment] = {
                'effect': placebo_effect.value,
                'p_value': placebo_effect.get_significance_test_results()[0]
            }
        
        results['placebo_tests'] = placebo_results
        
        return results
    
    def learn_causal_graph(self) -> nx.DiGraph:
        """
        Learn the causal graph from data using constraint-based methods.
        
        Returns:
            NetworkX DiGraph representing the learned causal structure
        """
        # Use PC algorithm from DoWhy
        graph = self.model.get_graph()
        
        # Convert to NetworkX graph
        nx_graph = nx.DiGraph()
        for edge in graph.split('\n'):
            if '->' in edge:
                source, target = edge.split('->')
                nx_graph.add_edge(source.strip(), target.strip())
        
        return nx_graph
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'StructuralCausalModel':
        """
        Load a saved model from disk.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded StructuralCausalModel instance
        """
        path = Path(path)
        
        with open(path, 'rb') as f:
            return pickle.load(f)
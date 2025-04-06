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
    Structural Causal Model for causal inference and intervention analysis.
    
    This class implements a structural causal model (SCM) for estimating
    causal effects, performing interventions, and counterfactual analysis.
    It supports various methods for causal discovery, estimation, and validation.
    """
    def __init__(
        self,
        causal_graph: Optional[CausalGraph] = None,
        discovery_method: str = 'pc',
        alpha: float = 0.05,
        feature_names: Optional[List[str]] = None,
        models: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the structural causal model.
        
        Args:
            causal_graph: Optional pre-defined causal graph
            discovery_method: Method for causal discovery
            alpha: Significance level for independence tests
            feature_names: Names of features in the model
            models: Dictionary mapping outcome variables to fitted models
        """
        self.discovery_method = discovery_method
        self.alpha = alpha
        self.feature_names = feature_names or []
        
        # Initialize or use provided causal graph
        self.causal_graph = causal_graph or CausalGraph(self.feature_names)
        
        # Storage for node-specific models
        self.models = models or {}
        
        # Store causal effects
        self.causal_effects = {}
        
        # Track fitted state
        self.is_fitted = False
    
    def discover_graph(
        self,
        data: pd.DataFrame,
        outcome_var: Optional[str] = None,
        treatment_vars: Optional[List[str]] = None
    ) -> CausalGraph:
        """
        Discover the causal graph structure from data.
        
        Args:
            data: DataFrame with variables
            outcome_var: Outcome variable name
            treatment_vars: List of potential treatment variables
            
        Returns:
            Discovered causal graph
        """
        # Update feature names if not set
        if not self.feature_names:
            self.feature_names = list(data.columns)
            self.causal_graph = CausalGraph(self.feature_names)
        
        # Apply different causal discovery methods
        if self.discovery_method == 'pc':
            self._discover_pc_algorithm(data, outcome_var, treatment_vars)
        elif self.discovery_method == 'notears':
            self._discover_notears(data, outcome_var, treatment_vars)
        elif self.discovery_method == 'manual':
            # For manual graph specification, don't run discovery
            logger.info("Using manually specified causal graph")
        else:
            raise ValueError(f"Unsupported discovery method: {self.discovery_method}")
        
        # Ensure outcome has the right node type
        if outcome_var:
            nx.set_node_attributes(
                self.causal_graph.graph,
                {outcome_var: {'type': 'outcome'}}
            )
        
        # Mark treatment variables
        if treatment_vars:
            for var in treatment_vars:
                nx.set_node_attributes(
                    self.causal_graph.graph,
                    {var: {'type': 'treatment'}}
                )
        
        return self.causal_graph
    
    def _discover_pc_algorithm(
        self,
        data: pd.DataFrame,
        outcome_var: Optional[str] = None,
        treatment_vars: Optional[List[str]] = None
    ) -> None:
        """
        Discover causal graph using PC algorithm.
        
        Args:
            data: DataFrame with variables
            outcome_var: Outcome variable name
            treatment_vars: List of potential treatment variables
        """
        try:
            from causallearn.search.ConstraintBased.PC import pc
            from causallearn.utils.PCUtils import SkeletonDiscovery
        except ImportError:
            raise ImportError("causallearn package is required for PC algorithm. "
                             "Install with 'pip install causallearn'")
        
        logger.info("Discovering causal graph with PC algorithm")
        
        # Convert data to numpy array
        X = data.values
        
        # Run PC algorithm
        cg = pc(X, alpha=self.alpha)
        
        # Extract the graph
        graph = cg.G.graph
        
        # Create our causal graph
        self.causal_graph = CausalGraph(list(data.columns))
        
        # Add edges from the discovered graph
        for i, j in zip(*np.where(graph != 0)):
            # PC algorithm outputs a skeleton with unknown directions
            # We need to determine the direction
            
            # Heuristic 1: If j is the outcome, i -> j
            if data.columns[j] == outcome_var:
                self.causal_graph.add_edge(data.columns[i], data.columns[j])
            
            # Heuristic 2: If i is a treatment, i -> j
            elif treatment_vars and data.columns[i] in treatment_vars:
                self.causal_graph.add_edge(data.columns[i], data.columns[j])
            
            # Heuristic 3: Use temporal knowledge if available
            # This would require additional domain knowledge
            
            # Default: add edge in both directions and let domain experts refine
            else:
                # For undirected edges, tentatively add both directions
                self.causal_graph.add_edge(data.columns[i], data.columns[j])
    
    def _discover_notears(
        self,
        data: pd.DataFrame,
        outcome_var: Optional[str] = None,
        treatment_vars: Optional[List[str]] = None
    ) -> None:
        """
        Discover causal graph using NOTEARS algorithm.
        
        Args:
            data: DataFrame with variables
            outcome_var: Outcome variable name
            treatment_vars: List of potential treatment variables
        """
        try:
            import notears
        except ImportError:
            raise ImportError("notears package is required for NOTEARS algorithm. "
                             "Install with 'pip install notears'")
        
        logger.info("Discovering causal graph with NOTEARS algorithm")
        
        # Normalize data
        X = (data - data.mean()) / data.std()
        
        # Run NOTEARS algorithm
        W = notears.linear.notears_linear(X.values, lambda1=0.1, loss_type='l2')
        
        # Create our causal graph
        self.causal_graph = CausalGraph(list(data.columns))
        
        # Add edges from the discovered graph (W[i,j] means i->j)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                if abs(W[i, j]) > 0.01:  # Threshold for edge detection
                    self.causal_graph.add_edge(
                        data.columns[i],
                        data.columns[j],
                        weight=float(W[i, j])
                    )
    
    def fit_models(
        self,
        data: pd.DataFrame,
        model_type: str = 'linear',
        outcome_vars: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Fit structural models for each node with parents.
        
        Args:
            data: DataFrame with variables
            model_type: Type of model to fit ('linear', 'random_forest', etc.)
            outcome_vars: List of variables to model (None for all)
            
        Returns:
            Dictionary of fitted models
        """
        # Determine which variables to model
        if outcome_vars is None:
            outcome_vars = list(self.causal_graph.graph.nodes())
        
        self.models = {}
        
        # Fit model for each outcome variable
        for outcome in outcome_vars:
            # Get parents (causes) of the outcome
            parents = list(self.causal_graph.get_parents(outcome))
            
            if not parents:
                logger.info(f"Skipping {outcome} as it has no parents in the graph")
                continue
            
            # Create input features and target
            X = data[parents]
            y = data[outcome]
            
            # Select and fit model
            model = self._create_model(model_type, outcome)
            
            try:
                model.fit(X, y)
                self.models[outcome] = {
                    'model': model,
                    'parents': parents
                }
                logger.info(f"Fitted {model_type} model for {outcome}")
            except Exception as e:
                logger.error(f"Error fitting model for {outcome}: {str(e)}")
        
        self.is_fitted = True
        return self.models
    
    def _create_model(self, model_type: str, outcome: str) -> Any:
        """
        Create a model of the specified type.
        
        Args:
            model_type: Type of model to create
            outcome: Name of outcome variable
            
        Returns:
            Initialized model
        """
        if model_type == 'linear':
            return LinearRegression()
        elif model_type == 'logistic':
            return LogisticRegression(max_iter=1000)
        elif model_type == 'random_forest':
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'random_forest_classifier':
            return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def predict(
        self,
        data: pd.DataFrame,
        outcome_var: str
    ) -> np.ndarray:
        """
        Predict outcome using the fitted structural model.
        
        Args:
            data: DataFrame with features
            outcome_var: Outcome variable to predict
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Models have not been fitted yet")
        
        if outcome_var not in self.models:
            raise ValueError(f"No model has been fitted for {outcome_var}")
        
        # Get model and parents
        model_info = self.models[outcome_var]
        model = model_info['model']
        parents = model_info['parents']
        
        # Create input features
        X = data[parents]
        
        # Make prediction
        return model.predict(X)
    
    def do_intervention(
        self,
        data: pd.DataFrame,
        interventions: Dict[str, float],
        outcome_var: str
    ) -> float:
        """
        Perform a do-intervention: P(outcome | do(intervention)).
        
        Evaluates the effect of setting variables to specific values.
        
        Args:
            data: DataFrame with variables
            interventions: Dictionary mapping variables to intervention values
            outcome_var: Outcome variable to predict
            
        Returns:
            Predicted outcome under intervention
        """
        if not self.is_fitted:
            raise ValueError("Models have not been fitted yet")
        
        # Create a copy of the data
        data_copy = data.copy()
        
        # Apply interventions
        for var, value in interventions.items():
            data_copy[var] = value
        
        # Make prediction using the modified data
        if outcome_var in interventions:
            # If we're intervening directly on the outcome, return that value
            return interventions[outcome_var]
        elif outcome_var in self.models:
            # Otherwise predict using the structural model
            model_info = self.models[outcome_var]
            model = model_info['model']
            parents = model_info['parents']
            
            # Make prediction using parents
            X = data_copy[parents]
            predictions = model.predict(X)
            
            # Return average prediction
            return float(np.mean(predictions))
        else:
            # If no model exists for the outcome, use the mean value
            if outcome_var in data_copy.columns:
                return float(data_copy[outcome_var].mean())
            else:
                raise ValueError(f"No model or data available for {outcome_var}")
    
    def estimate_causal_effect(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        control_value: Optional[float] = None,
        treatment_value: Optional[float] = None,
        adjustment_set: Optional[List[str]] = None,
        method: str = 'backdoor'
    ) -> Dict[str, float]:
        """
        Estimate the causal effect of treatment on outcome.
        
        Args:
            data: DataFrame with variables
            treatment: Treatment variable
            outcome: Outcome variable
            control_value: Value for control (None for data minimum)
            treatment_value: Value for treatment (None for data maximum)
            adjustment_set: Variables to adjust for (None for automatic)
            method: Estimation method ('backdoor', 'iv', 'frontdoor')
            
        Returns:
            Dictionary with causal effect information
        """
        # Set default control and treatment values
        if control_value is None:
            control_value = data[treatment].min()
            
        if treatment_value is None:
            treatment_value = data[treatment].max()
            
        # Ensure the treatment value is different from control
        if treatment_value == control_value:
            treatment_value = control_value + 1.0
        
        # Apply estimation method
        if method == 'backdoor':
            effect = self._backdoor_adjustment(
                data, treatment, outcome,
                control_value, treatment_value,
                adjustment_set
            )
        elif method == 'regression':
            effect = self._regression_adjustment(
                data, treatment, outcome
            )
        else:
            raise ValueError(f"Unsupported estimation method: {method}")
        
        # Store effect
        self.causal_effects[treatment] = effect
        
        return effect
    
    def _backdoor_adjustment(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        control_value: float,
        treatment_value: float,
        adjustment_set: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Estimate causal effect using backdoor adjustment.
        
        Args:
            data: DataFrame with variables
            treatment: Treatment variable
            outcome: Outcome variable
            control_value: Value for control condition
            treatment_value: Value for treatment condition
            adjustment_set: Variables to adjust for
            
        Returns:
            Dictionary with causal effect information
        """
        # Get adjustment set if not provided
        if adjustment_set is None:
            adjustment_set = list(self.causal_graph.get_minimal_adjustment_set(treatment, outcome))
            if not adjustment_set:
                logger.info(f"No adjustment set needed for {treatment} -> {outcome}")
                adjustment_set = []
        
        # Log the adjustment set being used
        logger.info(f"Using adjustment set for {treatment} -> {outcome}: {adjustment_set}")
        
        # Create interventional datasets
        control_data = data.copy()
        control_data[treatment] = control_value
        
        treatment_data = data.copy()
        treatment_data[treatment] = treatment_value
        
        # Predict outcomes under both interventions
        if outcome in self.models:
            # Get model info
            model_info = self.models[outcome]
            model = model_info['model']
            parents = model_info['parents']
            
            # Predict using the structural model
            if treatment in parents:
                # Predict control outcome
                control_X = control_data[parents]
                control_outcome = model.predict(control_X)
                
                # Predict treatment outcome
                treatment_X = treatment_data[parents]
                treatment_outcome = model.predict(treatment_X)
                
                # Calculate average causal effect
                ace = float(np.mean(treatment_outcome - control_outcome))
                
                # Calculate standard error
                se = float(np.std(treatment_outcome - control_outcome) / np.sqrt(len(data)))
                
                # Calculate p-value
                from scipy import stats
                t_stat = ace / se
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(data) - 1))
                
                # Calculate relative effect
                baseline = float(np.mean(control_outcome))
                if baseline != 0:
                    relative_effect = ace / baseline
                else:
                    relative_effect = float('inf') if ace > 0 else float('-inf')
                
                return {
                    'causal_effect': ace,
                    'standard_error': se,
                    'p_value': p_value,
                    'baseline': baseline,
                    'treatment_outcome': float(np.mean(treatment_outcome)),
                    'control_outcome': float(np.mean(control_outcome)),
                    'relative_effect': relative_effect
                }
            else:
                # If treatment is not a parent, it has no causal effect
                return {
                    'causal_effect': 0.0,
                    'standard_error': 0.0,
                    'p_value': 1.0,
                    'baseline': float(data[outcome].mean()),
                    'treatment_outcome': float(data[outcome].mean()),
                    'control_outcome': float(data[outcome].mean()),
                    'relative_effect': 0.0
                }
        else:
            # If no model exists, use do-intervention
            control_outcome = self.do_intervention(data, {treatment: control_value}, outcome)
            treatment_outcome = self.do_intervention(data, {treatment: treatment_value}, outcome)
            
            # Calculate causal effect
            ace = treatment_outcome - control_outcome
            
            # We don't have standard error without a model
            return {
                'causal_effect': ace,
                'baseline': control_outcome,
                'treatment_outcome': treatment_outcome,
                'control_outcome': control_outcome,
                'relative_effect': ace / control_outcome if control_outcome != 0 else float('inf')
            }
    
    def _regression_adjustment(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str
    ) -> Dict[str, float]:
        """
        Estimate causal effect using regression adjustment.
        
        This method fits a regression model with the treatment and potential
        confounders as predictors to estimate the causal effect.
        
        Args:
            data: DataFrame with variables
            treatment: Treatment variable
            outcome: Outcome variable
            
        Returns:
            Dictionary with causal effect information
        """
        # Get minimal adjustment set
        adjustment_set = list(self.causal_graph.get_minimal_adjustment_set(treatment, outcome))
        
        # Create predictors including treatment and adjustment variables
        predictors = [treatment] + adjustment_set
        X = data[predictors]
        y = data[outcome]
        
        # Fit linear regression model
        model = LinearRegression()
        model.fit(X, y)
        
        # Get coefficient for treatment
        treatment_idx = predictors.index(treatment)
        causal_effect = float(model.coef_[treatment_idx])
        
        # Calculate baseline (outcome value when treatment=0)
        baseline = float(model.intercept_ + sum(model.coef_[i] * data[predictors[i]].mean() 
                                         for i in range(len(predictors)) if i != treatment_idx))
        
        # Calculate standard error
        from sklearn.metrics import mean_squared_error
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        
        # Standard error formula for regression coefficient
        X_array = X.values
        n = X_array.shape[0]
        X_t_X_inv = np.linalg.inv(X_array.T @ X_array)
        se = float(np.sqrt(mse * X_t_X_inv[treatment_idx, treatment_idx]))
        
        # Calculate p-value
        from scipy import stats
        t_stat = causal_effect / se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - len(predictors) - 1))
        
        # Calculate relative effect
        if baseline != 0:
            relative_effect = causal_effect / baseline
        else:
            relative_effect = float('inf') if causal_effect > 0 else float('-inf')
        
        return {
            'causal_effect': causal_effect,
            'standard_error': se,
            'p_value': p_value,
            'baseline': baseline,
            'relative_effect': relative_effect,
            'adjustment_set': adjustment_set
        }
    
    def estimate_all_effects(
        self,
        data: pd.DataFrame,
        outcome: str,
        potential_causes: Optional[List[str]] = None,
        method: str = 'backdoor',
        min_effect: float = 0.0
    ) -> Dict[str, Dict[str, float]]:
        """
        Estimate causal effects of multiple variables on an outcome.
        
        Args:
            data: DataFrame with variables
            outcome: Outcome variable
            potential_causes: List of potential causal variables (None for all)
            method: Estimation method ('backdoor', 'regression')
            min_effect: Minimum absolute effect size to include
            
        Returns:
            Dictionary mapping variables to their causal effects
        """
        # Determine potential causes if not provided
        if potential_causes is None:
            # Use all nodes except outcome and its descendants
            outcome_descendants = self.causal_graph.get_descendants(outcome)
            potential_causes = [node for node in self.causal_graph.graph.nodes()
                               if node != outcome and node not in outcome_descendants]
        
        # Estimate causal effect for each potential cause
        effects = {}
        
        for treatment in tqdm(potential_causes, desc="Estimating causal effects"):
            try:
                # Check if there's a path from treatment to outcome
                if nx.has_path(self.causal_graph.graph, treatment, outcome):
                    # Estimate causal effect
                    effect = self.estimate_causal_effect(
                        data=data,
                        treatment=treatment,
                        outcome=outcome,
                        method=method
                    )
                    
                    # Store if effect size exceeds minimum
                    if abs(effect['causal_effect']) >= min_effect:
                        effects[treatment] = effect
                else:
                    logger.info(f"No path from {treatment} to {outcome}, skipping")
            except Exception as e:
                logger.warning(f"Error estimating effect of {treatment} on {outcome}: {str(e)}")
        
        # Update stored causal effects
        self.causal_effects.update(effects)
        
        return effects
    
    def identify_optimal_intervention(
        self,
        data: pd.DataFrame,
        candidate_features: List[str],
        outcome: str,
        maximize: bool = True,
        n_points: int = 10,
        max_interventions: int = 3
    ) -> Dict[str, Any]:
        """
        Identify the optimal intervention to maximize/minimize outcome.
        
        Args:
            data: DataFrame with variables
            candidate_features: List of features to consider for intervention
            outcome: Outcome variable to optimize
            maximize: Whether to maximize (True) or minimize (False) outcome
            n_points: Number of intervention points to test per feature
            max_interventions: Maximum number of features to modify
            
        Returns:
            Dictionary with optimal intervention information
        """
        if not self.is_fitted:
            raise ValueError("Models have not been fitted yet")
        
        # Get baseline outcome value
        baseline = float(data[outcome].mean())
        
        # Test single interventions first
        best_interventions = {}
        best_outcome = baseline
        
        # Evaluate each feature at different intervention values
        for feature in candidate_features:
            # Get feature range
            feature_min = data[feature].min()
            feature_max = data[feature].max()
            
            # Generate intervention values
            intervention_values = np.linspace(feature_min, feature_max, n_points)
            
            # Evaluate each intervention value
            for value in intervention_values:
                # Create intervention
                intervention = {feature: value}
                
                # Predict outcome
                predicted_outcome = self.do_intervention(data, intervention, outcome)
                
                # Check if this is better than current best
                if ((maximize and predicted_outcome > best_outcome) or
                    (not maximize and predicted_outcome < best_outcome)):
                    best_outcome = predicted_outcome
                    best_interventions = intervention
        
        # Calculate improvement
        improvement = best_outcome - baseline
        percent_improvement = (improvement / abs(baseline)) * 100 if baseline != 0 else float('inf')
        
        # Format results
        result = {
            'optimal_intervention': best_interventions,
            'predicted_outcome': best_outcome,
            'baseline': baseline,
            'improvement': improvement,
            'percent_improvement': percent_improvement
        }
        
        return result
    
    def counterfactual_analysis(
        self,
        instance: pd.Series,
        interventions: Dict[str, float],
        outcome: str
    ) -> Dict[str, Any]:
        """
        Perform counterfactual analysis on a single instance.
        
        Args:
            instance: Series with feature values for one instance
            interventions: Dictionary mapping features to counterfactual values
            outcome: Outcome variable of interest
            
        Returns:
            Dictionary with counterfactual analysis results
        """
        if not self.is_fitted:
            raise ValueError("Models have not been fitted yet")
        
        # Convert instance to DataFrame
        instance_df = pd.DataFrame([instance])
        
        # Get factual outcome
        factual_outcome = float(instance[outcome]) if outcome in instance else None
        if factual_outcome is None and outcome in self.models:
            # Predict factual outcome if not available
            factual_outcome = float(self.predict(instance_df, outcome)[0])
        
        # Predict counterfactual outcome
        counterfactual_outcome = self.do_intervention(instance_df, interventions, outcome)
        
        # Calculate difference
        difference = None
        if factual_outcome is not None:
            difference = counterfactual_outcome - factual_outcome
        
        # Create result
        result = {
            'factual_outcome': factual_outcome,
            'counterfactual_outcome': counterfactual_outcome,
            'difference': difference,
            'interventions': interventions
        }
        
        return result
    
    def feature_importance(
        self,
        data: pd.DataFrame,
        outcome: str,
        method: str = 'causal_effect',
        normalize: bool = True
    ) -> Dict[str, float]:
        """
        Calculate feature importance based on causal effects.
        
        Args:
            data: DataFrame with variables
            outcome: Outcome variable
            method: Method for calculating importance ('causal_effect', 'variance')
            normalize: Whether to normalize importance scores
            
        Returns:
            Dictionary mapping features to importance scores
        """
        # Ensure we have causal effects
        if not self.causal_effects and method == 'causal_effect':
            self.estimate_all_effects(data, outcome)
        
        importance_scores = {}
        
        if method == 'causal_effect':
            # Use absolute causal effects as importance
            for feature, effect in self.causal_effects.items():
                importance_scores[feature] = abs(effect['causal_effect'])
        
        elif method == 'variance':
            # Calculate importance based on outcome variance explained
            if not self.is_fitted:
                raise ValueError("Models have not been fitted yet")
                
            if outcome not in self.models:
                raise ValueError(f"No model fitted for {outcome}")
                
            # Get model and parents
            model_info = self.models[outcome]
            model = model_info['model']
            parents = model_info['parents']
            
            # Calculate baseline variance
            baseline_variance = float(data[outcome].var())
            
            # Calculate importance for each parent
            for feature in parents:
                # Create copy of data with feature randomized
                shuffled_data = data.copy()
                shuffled_data[feature] = np.random.permutation(shuffled_data[feature].values)
                
                # Predict with shuffled feature
                X = shuffled_data[parents]
                shuffled_predictions = model.predict(X)
                
                # Calculate increase in prediction error
                original_predictions = model.predict(data[parents])
                importance = np.mean((original_predictions - shuffled_predictions) ** 2)
                
                # Normalize by baseline variance
                if normalize and baseline_variance > 0:
                    importance /= baseline_variance
                
                importance_scores[feature] = float(importance)
        
        else:
            raise ValueError(f"Unsupported importance method: {method}")
        
        # Normalize importance scores if requested
        if normalize and importance_scores:
            max_importance = max(importance_scores.values())
            if max_importance > 0:
                importance_scores = {k: v / max_importance for k, v in importance_scores.items()}
        
        # Sort by importance
        importance_scores = dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))
        
        return importance_scores
    
    def save(self, path: str) -> None:
        """
        Save the structural causal model to a file.
        
        Args:
            path: Path to save the model
        """
        import pickle
        
        # Prepare model for serialization
        serializable_model = {
            'feature_names': self.feature_names,
            'discovery_method': self.discovery_method,
            'alpha': self.alpha,
            'causal_effects': self.causal_effects,
            'is_fitted': self.is_fitted
        }
        
        # Save graph structure
        serializable_model['graph'] = nx.to_dict_of_dicts(self.causal_graph.graph)
        
        # Save model
        with open(path, 'wb') as f:
            pickle.dump(serializable_model, f)
    
    @classmethod
    def load(cls, path: str) -> 'StructuralCausalModel':
        """
        Load a structural causal model from a file.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded StructuralCausalModel
        """
        import pickle
        
        # Load serialized model
        with open(path, 'rb') as f:
            serialized_model = pickle.load(f)
        
        # Create causal graph
        causal_graph = CausalGraph(serialized_model['feature_names'])
        
        # Restore graph structure
        graph_dict = serialized_model['graph']
        for source, targets in graph_dict.items():
            for target, attrs in targets.items():
                causal_graph.add_edge(source, target, **attrs)
        
        # Create model instance
        model = cls(
            causal_graph=causal_graph,
            discovery_method=serialized_model['discovery_method'],
            alpha=serialized_model['alpha'],
            feature_names=serialized_model['feature_names']
        )
        
        # Restore causal effects
        model.causal_effects = serialized_model['causal_effects']
        model.is_fitted = serialized_model['is_fitted']
        
        return model
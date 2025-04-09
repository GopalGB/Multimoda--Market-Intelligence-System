import numpy as np
import pandas as pd
import networkx as nx
from typing import Optional, Dict, List, Tuple
from pathlib import Path

class SyntheticCausalDataGenerator:
    """Generates synthetic data with known causal structures for testing causal inference."""
    
    def __init__(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        seed: int = 42,
        noise_level: float = 0.1,
        edge_probability: float = 0.3
    ):
        """
        Initialize the synthetic data generator.
        
        Args:
            n_samples: Number of samples to generate
            n_features: Number of features (excluding outcome)
            seed: Random seed for reproducibility
            noise_level: Standard deviation of noise in structural equations
            edge_probability: Probability of edge existence in causal graph
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.seed = seed
        self.noise_level = noise_level
        self.edge_probability = edge_probability
        
        # Set random seed
        np.random.seed(seed)
        
        # Initialize causal graph
        self.causal_graph = None
        self.true_effects = {}
        
    def generate_random_dag(self) -> nx.DiGraph:
        """
        Generate a random directed acyclic graph.
        
        Returns:
            NetworkX DiGraph representing the causal structure
        """
        # Create empty graph
        G = nx.DiGraph()
        
        # Add nodes
        for i in range(self.n_features):
            G.add_node(f'X{i}')
        G.add_node('Y')  # Outcome variable
        
        # Add edges (ensuring acyclicity)
        for i in range(self.n_features):
            # Edges between features
            for j in range(i+1, self.n_features):
                if np.random.random() < self.edge_probability:
                    G.add_edge(f'X{i}', f'X{j}')
            
            # Edges to outcome
            if np.random.random() < self.edge_probability * 2:  # Higher probability for edges to outcome
                G.add_edge(f'X{i}', 'Y')
        
        self.causal_graph = G
        return G
    
    def generate_linear_data(self) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Generate synthetic data based on linear structural equations.
        
        Returns:
            Tuple of (DataFrame with generated data, Dictionary of true causal effects)
        """
        if self.causal_graph is None:
            self.generate_random_dag()
            
        # Topologically sort nodes to ensure correct generation order
        node_order = list(nx.topological_sort(self.causal_graph))
        
        # Initialize data dictionary
        data = {}
        
        # Generate data for each node
        for node in node_order:
            parents = list(self.causal_graph.predecessors(node))
            
            if not parents:  # Root node (no parents)
                data[node] = np.random.normal(0, 1, self.n_samples)
            else:
                # Generate as linear combination of parents plus noise
                node_data = np.random.normal(0, self.noise_level, self.n_samples)
                
                # Add effect from each parent
                for parent in parents:
                    # Random coefficient between 0.5 and 1.5
                    coef = np.random.uniform(0.5, 1.5)
                    node_data += coef * data[parent]
                    if node == 'Y':  # Store true causal effects
                        self.true_effects[parent] = coef
                
                data[node] = node_data
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        return df, self.true_effects
    
    def generate_nonlinear_data(
        self,
        nonlinearity: str = 'quadratic'
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Generate synthetic data with nonlinear relationships.
        
        Args:
            nonlinearity: Type of nonlinearity ('quadratic', 'exponential', 'sigmoid')
            
        Returns:
            Tuple of (DataFrame with generated data, Dictionary of true causal effects)
        """
        if self.causal_graph is None:
            self.generate_random_dag()
            
        node_order = list(nx.topological_sort(self.causal_graph))
        data = {}
        
        for node in node_order:
            parents = list(self.causal_graph.predecessors(node))
            
            if not parents:
                data[node] = np.random.normal(0, 1, self.n_samples)
            else:
                node_data = np.random.normal(0, self.noise_level, self.n_samples)
                
                for parent in parents:
                    coef = np.random.uniform(0.5, 1.5)
                    parent_data = data[parent]
                    
                    if nonlinearity == 'quadratic':
                        effect = coef * parent_data ** 2
                    elif nonlinearity == 'exponential':
                        effect = coef * np.exp(parent_data)
                    elif nonlinearity == 'sigmoid':
                        effect = coef * (1 / (1 + np.exp(-parent_data)))
                    else:
                        raise ValueError(f"Unknown nonlinearity: {nonlinearity}")
                    
                    node_data += effect
                    if node == 'Y':
                        self.true_effects[parent] = coef
                
                data[node] = node_data
        
        return pd.DataFrame(data), self.true_effects
    
    def save_data(
        self,
        data: pd.DataFrame,
        effects: Dict[str, float],
        output_dir: str = "data/synthetic"
    ) -> Tuple[Path, Path]:
        """
        Save generated data and true effects to files.
        
        Args:
            data: DataFrame with generated data
            effects: Dictionary of true causal effects
            output_dir: Directory to save files
            
        Returns:
            Tuple of (Path to data file, Path to effects file)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data
        data_path = output_dir / "synthetic_data.csv"
        data.to_csv(data_path, index=False)
        
        # Save effects
        effects_path = output_dir / "true_effects.json"
        with open(effects_path, 'w') as f:
            import json
            json.dump(effects, f, indent=2)
        
        return data_path, effects_path
    
    def get_true_causal_graph(self) -> nx.DiGraph:
        """Return the ground truth causal graph."""
        return self.causal_graph
    
    def get_true_effects(self) -> Dict[str, float]:
        """Return the true causal effects."""
        return self.true_effects 
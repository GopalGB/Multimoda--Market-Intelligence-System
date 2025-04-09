import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

class CausalValidation:
    """Framework for validating causal discovery and effect estimation."""
    
    def __init__(self, true_graph=None, estimated_graph=None):
        self.true_graph = true_graph
        self.estimated_graph = estimated_graph
        
    def compare_graph_structure(self):
        """Compare the structure of the true and estimated causal graphs."""
        if self.true_graph is None or self.estimated_graph is None:
            raise ValueError("Both true and estimated graphs must be provided")
        
        # Get adjacency matrices
        true_adj = nx.adjacency_matrix(self.true_graph).todense()
        est_adj = nx.adjacency_matrix(self.estimated_graph).todense()
        
        # Ensure same node ordering
        true_nodes = list(self.true_graph.nodes())
        est_nodes = list(self.estimated_graph.nodes())
        
        if set(true_nodes) != set(est_nodes):
            raise ValueError("Graph nodes do not match")
        
        # Reorder estimated adjacency if needed
        if true_nodes != est_nodes:
            node_idx = {node: i for i, node in enumerate(est_nodes)}
            idx_map = [node_idx[node] for node in true_nodes]
            est_adj = est_adj[idx_map, :][:, idx_map]
        
        # Convert to binary (0/1) matrices
        true_adj = (true_adj > 0).astype(int)
        est_adj = (est_adj > 0).astype(int)
        
        # Calculate metrics
        y_true = true_adj.flatten()
        y_pred = est_adj.flatten()
        
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positive": np.sum((y_true == 1) & (y_pred == 1)),
            "false_positive": np.sum((y_true == 0) & (y_pred == 1)),
            "false_negative": np.sum((y_true == 1) & (y_pred == 0)),
            "true_negative": np.sum((y_true == 0) & (y_pred == 0)),
        }
    
    def compare_causal_effects(self, true_effects, estimated_effects):
        """Compare true causal effects with estimated effects."""
        common_vars = set(true_effects.keys()) & set(estimated_effects.keys())
        
        if not common_vars:
            raise ValueError("No common variables found between true and estimated effects")
        
        results = {}
        all_errors = []
        
        for var in common_vars:
            true_effect = true_effects[var]
            est_effect = estimated_effects[var]
            abs_error = abs(true_effect - est_effect)
            rel_error = abs_error / max(abs(true_effect), 1e-10)
            
            results[var] = {
                "true_effect": true_effect,
                "estimated_effect": est_effect,
                "absolute_error": abs_error,
                "relative_error": rel_error
            }
            all_errors.append(abs_error)
        
        # Overall metrics
        results["overall"] = {
            "mean_absolute_error": np.mean(all_errors),
            "median_absolute_error": np.median(all_errors),
            "max_absolute_error": np.max(all_errors),
            "variables_evaluated": len(common_vars)
        }
        
        return results
    
    def visualize_comparison(self, output_path=None):
        """Visualize comparison between true and estimated graphs."""
        if self.true_graph is None or self.estimated_graph is None:
            raise ValueError("Both true and estimated graphs must be provided")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Draw true graph
        pos = nx.spring_layout(self.true_graph, seed=42)  # Use same layout for both
        nx.draw(self.true_graph, pos, ax=ax1, with_labels=True, node_color='lightblue', 
                node_size=1500, arrows=True, arrowsize=20)
        ax1.set_title("True Causal Graph")
        
        # Draw estimated graph
        nx.draw(self.estimated_graph, pos, ax=ax2, with_labels=True, node_color='lightgreen',
                node_size=1500, arrows=True, arrowsize=20)
        ax2.set_title("Estimated Causal Graph")
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
        return fig 
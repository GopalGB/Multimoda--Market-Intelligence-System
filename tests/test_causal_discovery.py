import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import networkx as nx

# Import our modules
from tests.synthetic_data.causal_data_generator import SyntheticCausalDataGenerator
from tests.causal.validation import CausalValidation
from causal.structural_model import StructuralCausalModel, CausalGraph

class TestCausalDiscovery(unittest.TestCase):
    """Test suite for causal discovery and inference."""
    
    def setUp(self):
        """Set up test environment."""
        # Create output directory if it doesn't exist
        self.output_dir = Path("./test_results/causal")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data generator
        self.data_generator = SyntheticCausalDataGenerator(
            n_samples=1000, 
            n_features=5,
            seed=42
        )
        
        # Generate synthetic data with known causal structure
        self.data_generator.generate_random_dag()
        self.data, self.true_effects = self.data_generator.generate_linear_data()
        
        # Get true causal graph and effects
        self.true_graph = self.data_generator.get_true_causal_graph()
        
    def test_pc_algorithm_discovery(self):
        """Test PC algorithm for causal discovery."""
        # Initialize causal model with PC algorithm
        causal_model = StructuralCausalModel(
            discovery_method='pc',
            alpha=0.05,
            feature_names=self.data.columns.tolist()
        )
        
        # Discover causal graph
        causal_graph = causal_model.discover_graph(
            data=self.data,
            outcome_var='Y'
        )
        
        # Validate discovered graph
        validator = CausalValidation(
            true_graph=self.true_graph,
            estimated_graph=causal_graph.graph
        )
        
        # Compare graph structure
        structure_metrics = validator.compare_graph_structure()
        
        # Visualize comparison
        validator.visualize_comparison(
            output_path=self.output_dir / "pc_algorithm_comparison.png"
        )
        
        # Assert minimum performance
        self.assertGreaterEqual(structure_metrics['precision'], 0.5, 
                               "Precision should be at least 0.5")
        self.assertGreaterEqual(structure_metrics['recall'], 0.5, 
                               "Recall should be at least 0.5")
        
        # Print detailed metrics
        print(f"PC Algorithm Structure Metrics: {structure_metrics}")
        
    def test_causal_effect_estimation(self):
        """Test causal effect estimation."""
        # Initialize causal model
        causal_model = StructuralCausalModel(
            discovery_method='manual',  # Use manual to provide the true graph
            alpha=0.05,
            feature_names=self.data.columns.tolist()
        )
        
        # Set the true graph directly
        causal_model.causal_graph = CausalGraph()
        for node in self.true_graph.nodes():
            causal_model.causal_graph.add_node(node)
        for edge in self.true_graph.edges():
            causal_model.causal_graph.add_edge(edge[0], edge[1])
        
        # Fit models for causal effect estimation
        causal_model.fit_models(self.data, model_type='linear')
        
        # Estimate causal effects
        estimated_effects = {}
        for feature in self.true_effects.keys():
            effect = causal_model.estimate_causal_effect(
                self.data, feature, 'Y'
            )
            estimated_effects[feature] = effect['causal_effect']
        
        # Validate effect estimation
        validator = CausalValidation()
        effect_metrics = validator.compare_causal_effects(
            self.true_effects, estimated_effects
        )
        
        # Print effect comparison
        print(f"Causal Effect Estimation Metrics: {effect_metrics['overall']}")
        
        # Assert reasonable error bounds
        self.assertLessEqual(effect_metrics['overall']['mean_absolute_error'], 0.3,
                           "Mean absolute error should be less than 0.3")
        
    def test_counterfactual_analysis(self):
        """Test counterfactual analysis."""
        from causal.counterfactual import CounterfactualAnalyzer
        
        # Initialize causal model with true graph
        causal_model = StructuralCausalModel(
            discovery_method='manual',
            alpha=0.05,
            feature_names=self.data.columns.tolist()
        )
        
        # Set the true graph directly
        causal_model.causal_graph = CausalGraph()
        for node in self.true_graph.nodes():
            causal_model.causal_graph.add_node(node)
        for edge in self.true_graph.edges():
            causal_model.causal_graph.add_edge(edge[0], edge[1])
        
        # Fit models
        causal_model.fit_models(self.data, model_type='linear')
        
        # Create counterfactual analyzer
        cf_analyzer = CounterfactualAnalyzer(causal_model)
        
        # Select a feature with known causal effect
        test_feature = list(self.true_effects.keys())[0]
        
        # Get sample instance
        sample_instance = self.data.iloc[0].copy()
        original_value = sample_instance[test_feature]
        original_outcome = sample_instance['Y']
        
        # Create counterfactual with doubled feature value
        intervention = {test_feature: original_value * 2}
        
        # Generate counterfactual
        cf_result = cf_analyzer.generate_counterfactual(
            data=self.data,
            interventions=intervention,
            outcome_var='Y',
            reference_values=sample_instance.to_dict()
        )
        
        # Verify counterfactual makes sense
        # If we increase a feature with positive effect, outcome should increase
        true_effect_sign = np.sign(self.true_effects[test_feature])
        intervention_change = np.sign(intervention[test_feature] - original_value)
        expected_outcome_change = true_effect_sign * intervention_change
        actual_outcome_change = np.sign(cf_result['outcome_change'])
        
        self.assertEqual(expected_outcome_change, actual_outcome_change,
                        f"Counterfactual outcome change should match expected direction for {test_feature}")
        
        print(f"Counterfactual result: {cf_result}")


if __name__ == '__main__':
    unittest.main() 
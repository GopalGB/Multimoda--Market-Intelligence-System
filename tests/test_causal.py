# tests/test_causal.py
import unittest
import pandas as pd
import numpy as np
import networkx as nx
import os
import tempfile
from unittest.mock import patch, MagicMock
import pytest
from pathlib import Path
from typing import Dict, Tuple

from causal.structural_model import CausalGraph, StructuralCausalModel
from causal.causal_features import CausalFeatureSelector
from causal.counterfactual import CounterfactualAnalyzer
from tests.synthetic_data.causal_data_generator import SyntheticCausalDataGenerator


class TestCausalGraph(unittest.TestCase):
    """Test the CausalGraph class."""
    
    def setUp(self):
        """Set up test causal graph."""
        self.graph = CausalGraph(feature_names=["A", "B", "C", "D", "outcome"])
        
        # Create a simple causal graph:
        # A -> B -> outcome
        # C -> outcome
        # A -> D -> outcome
        self.graph.add_edge("A", "B")
        self.graph.add_edge("B", "outcome")
        self.graph.add_edge("C", "outcome")
        self.graph.add_edge("A", "D")
        self.graph.add_edge("D", "outcome")
    
    def test_graph_structure(self):
        """Test that the graph structure is correct."""
        # Test nodes
        self.assertEqual(set(self.graph.graph.nodes()), set(["A", "B", "C", "D", "outcome"]))
        
        # Test edges
        expected_edges = [("A", "B"), ("B", "outcome"), ("C", "outcome"), ("A", "D"), ("D", "outcome")]
        for src, tgt in expected_edges:
            self.assertTrue(self.graph.graph.has_edge(src, tgt))
    
    def test_get_parents(self):
        """Test getting parents of a node."""
        self.assertEqual(set(self.graph.get_parents("outcome")), set(["B", "C", "D"]))
        self.assertEqual(set(self.graph.get_parents("B")), set(["A"]))
        self.assertEqual(set(self.graph.get_parents("A")), set())
    
    def test_get_ancestors(self):
        """Test getting ancestors of a node."""
        self.assertEqual(self.graph.get_ancestors("outcome"), set(["A", "B", "C", "D"]))
        self.assertEqual(self.graph.get_ancestors("B"), set(["A"]))
        self.assertEqual(self.graph.get_ancestors("A"), set())
    
    def test_find_backdoor_paths(self):
        """Test finding backdoor paths between treatment and outcome."""
        # Test backdoor paths from C to outcome
        backdoor_paths = self.graph.find_backdoor_paths("C", "outcome")
        self.assertEqual(len(backdoor_paths), 0)  # No backdoor paths
        
        # Test backdoor paths from B to outcome with A as a confounder
        # (need to modify the graph to add A->outcome for this test)
        self.graph.add_edge("A", "outcome")
        backdoor_paths = self.graph.find_backdoor_paths("B", "outcome")
        self.assertGreater(len(backdoor_paths), 0)
    
    def test_get_minimal_adjustment_set(self):
        """Test getting minimal adjustment set."""
        # For B->outcome, we need to adjust for A
        adjustment_set = self.graph.get_minimal_adjustment_set("B", "outcome")
        self.assertEqual(adjustment_set, set(["A"]))
        
        # For C->outcome, no adjustment needed
        adjustment_set = self.graph.get_minimal_adjustment_set("C", "outcome")
        self.assertEqual(adjustment_set, set())


class TestStructuralCausalModel(unittest.TestCase):
    """Test the StructuralCausalModel class."""
    
    def setUp(self):
        """Set up test data and model."""
        # Create synthetic causal data
        np.random.seed(42)
        n = 1000
        
        # A is exogenous
        A = np.random.normal(0, 1, n)
        
        # B is caused by A
        B = 0.8 * A + np.random.normal(0, 0.1, n)
        
        # C is exogenous
        C = np.random.normal(0, 1, n)
        
        # Outcome is caused by B and C
        outcome = 0.5 * B + 0.3 * C + np.random.normal(0, 0.1, n)
        
        # Create DataFrame
        self.data = pd.DataFrame({
            "A": A,
            "B": B,
            "C": C,
            "outcome": outcome
        })
        
        # Create a causal graph
        self.graph = CausalGraph(feature_names=["A", "B", "C", "outcome"])
        self.graph.add_edge("A", "B")
        self.graph.add_edge("B", "outcome")
        self.graph.add_edge("C", "outcome")
        
        # Create a causal model
        self.model = StructuralCausalModel(causal_graph=self.graph)
    
    def test_fit_models(self):
        """Test fitting structural models."""
        # Fit models
        fitted_models = self.model.fit_models(self.data, model_type="linear")
        
        # Check that models were fitted for nodes with parents
        self.assertIn("B", fitted_models)
        self.assertIn("outcome", fitted_models)
        self.assertNotIn("A", fitted_models)
        self.assertNotIn("C", fitted_models)
        
        # Check that the models have parents
        self.assertEqual(set(fitted_models["B"]["parents"]), set(["A"]))
        self.assertEqual(set(fitted_models["outcome"]["parents"]), set(["B", "C"]))
    
    def test_predict(self):
        """Test prediction with fitted models."""
        # Fit models first
        self.model.fit_models(self.data, model_type="linear")
        
        # Predict outcome
        predictions = self.model.predict(self.data, "outcome")
        
        # Check predictions
        self.assertEqual(len(predictions), len(self.data))
        
        # Calculate R²
        from sklearn.metrics import r2_score
        r2 = r2_score(self.data["outcome"], predictions)
        
        # Should have good predictive power
        self.assertGreater(r2, 0.7)
    
    def test_do_intervention(self):
        """Test do-intervention."""
        # Fit models first
        self.model.fit_models(self.data, model_type="linear")
        
        # Intervene on B
        intervened_outcome = self.model.do_intervention(self.data, {"B": 0}, "outcome")
        
        # Should be different from original outcome
        self.assertNotAlmostEqual(intervened_outcome, self.data["outcome"].mean(), places=1)
    
    def test_estimate_causal_effect(self):
        """Test causal effect estimation."""
        # Fit models first
        self.model.fit_models(self.data, model_type="linear")
        
        # Estimate effect of B on outcome
        effect = self.model.estimate_causal_effect(
            self.data, "B", "outcome", control_value=0, treatment_value=1
        )
        
        # Check effect structure
        self.assertIn("causal_effect", effect)
        self.assertIn("standard_error", effect)
        
        # Effect should be close to the true coefficient (0.5)
        self.assertAlmostEqual(effect["causal_effect"], 0.5, delta=0.1)
    
    def test_save_and_load(self):
        """Test model saving and loading."""
        # Fit models first
        self.model.fit_models(self.data, model_type="linear")
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl') as tmp:
            # Save the model
            self.model.save(tmp.name)
            
            # Load the model
            loaded_model = StructuralCausalModel.load(tmp.name)
            
            # Check that the loaded model has the same structure
            self.assertEqual(
                set(loaded_model.feature_names), 
                set(self.model.feature_names)
            )


class TestCausalFeatureSelector(unittest.TestCase):
    """Test the CausalFeatureSelector class."""
    
    def setUp(self):
        """Set up test data and model."""
        # Create synthetic causal data
        np.random.seed(42)
        n = 1000
        
        # Generate 5 features
        X1 = np.random.normal(0, 1, n)
        X2 = np.random.normal(0, 1, n)
        X3 = X1 * 0.7 + np.random.normal(0, 0.1, n)  # X3 depends on X1
        X4 = np.random.normal(0, 1, n)
        X5 = X2 * 0.5 + np.random.normal(0, 0.1, n)  # X5 depends on X2
        
        # Outcome depends on X2, X4
        Y = 0.3 * X2 + 0.6 * X4 + np.random.normal(0, 0.1, n)
        
        # Create DataFrame
        self.data = pd.DataFrame({
            "X1": X1,
            "X2": X2,
            "X3": X3,
            "X4": X4,
            "X5": X5,
            "Y": Y
        })
        
        # Create a causal model with a manually specified graph
        self.graph = CausalGraph(feature_names=["X1", "X2", "X3", "X4", "X5", "Y"])
        self.graph.add_edge("X1", "X3")
        self.graph.add_edge("X2", "X5")
        self.graph.add_edge("X2", "Y")
        self.graph.add_edge("X4", "Y")
        
        self.causal_model = StructuralCausalModel(
            causal_graph=self.graph,
            discovery_method="manual"
        )
        
        # Fit the model
        self.causal_model.fit_models(self.data, model_type="linear")
        
        # Create feature selector
        self.selector = CausalFeatureSelector(self.causal_model)
    
    def test_fit(self):
        """Test fitting the feature selector."""
        # Fit the selector
        causal_features = self.selector.fit(
            self.data,
            outcome_var="Y",
            feature_vars=["X1", "X2", "X3", "X4", "X5"],
            discovery_method="manual"
        )
        
        # Should have identified X2 and X4 as causal features
        self.assertIn("X2", causal_features)
        self.assertIn("X4", causal_features)
        
        # X1, X3, X5 should have weaker or no causal effect
        if "X1" in causal_features:
            self.assertLess(abs(causal_features["X1"]), abs(causal_features["X4"]))
        if "X3" in causal_features:
            self.assertLess(abs(causal_features["X3"]), abs(causal_features["X4"]))
        if "X5" in causal_features:
            self.assertLess(abs(causal_features["X5"]), abs(causal_features["X4"]))
    
    def test_get_top_features(self):
        """Test getting top causal features."""
        # Fit the selector first
        self.selector.fit(
            self.data,
            outcome_var="Y",
            feature_vars=["X1", "X2", "X3", "X4", "X5"],
            discovery_method="manual"
        )
        
        # Get top 2 features
        top_features = self.selector.get_top_features(n=2)
        
        # Should have 2 features
        self.assertEqual(len(top_features), 2)
        
        # X4 and X2 should be the top features
        self.assertTrue(set(top_features.keys()).issubset({"X2", "X4"}))
    
    def test_generate_feature_recommendations(self):
        """Test generating feature recommendations."""
        # Fit the selector first
        self.selector.fit(
            self.data,
            outcome_var="Y",
            feature_vars=["X1", "X2", "X3", "X4", "X5"],
            discovery_method="manual"
        )
        
        # Create current feature values
        current_values = {
            "X1": 0.0,
            "X2": 0.0,
            "X3": 0.0,
            "X4": 0.0,
            "X5": 0.0
        }
        
        # Set a target outcome higher than baseline
        baseline = self.selector._predict_outcome(current_values)
        target = baseline + 1.0
        
        # Generate recommendations
        recommendations = self.selector.generate_feature_recommendations(
            target_outcome=target,
            current_values=current_values
        )
        
        # Should have recommendations for at least X2 and X4
        self.assertGreater(len(recommendations), 0)
        self.assertTrue(set(recommendations.keys()).intersection({"X2", "X4"}))


class TestCounterfactualAnalyzer(unittest.TestCase):
    """Test the CounterfactualAnalyzer class."""
    
    def setUp(self):
        """Set up test data and model."""
        # Create synthetic causal data
        np.random.seed(42)
        n = 1000
        
        # Generate 3 features
        X1 = np.random.normal(0, 1, n)
        X2 = np.random.normal(0, 1, n)
        X3 = X1 * 0.5 + X2 * 0.3 + np.random.normal(0, 0.1, n)
        
        # Outcome depends on all features
        Y = 0.2 * X1 + 0.4 * X2 + 0.6 * X3 + np.random.normal(0, 0.1, n)
        
        # Create DataFrame
        self.data = pd.DataFrame({
            "X1": X1,
            "X2": X2,
            "X3": X3,
            "Y": Y
        })
        
        # Create a causal model with a manually specified graph
        self.graph = CausalGraph(feature_names=["X1", "X2", "X3", "Y"])
        self.graph.add_edge("X1", "X3")
        self.graph.add_edge("X2", "X3")
        self.graph.add_edge("X1", "Y")
        self.graph.add_edge("X2", "Y")
        self.graph.add_edge("X3", "Y")
        
        self.causal_model = StructuralCausalModel(
            causal_graph=self.graph,
            discovery_method="manual"
        )
        
        # Fit the model
        self.causal_model.fit_models(self.data, model_type="linear")
        
        # Create counterfactual analyzer
        self.analyzer = CounterfactualAnalyzer(self.causal_model)
    
    def test_generate_counterfactual(self):
        """Test generating counterfactuals."""
        # Create a reference instance
        instance = self.data.iloc[0]
        
        # Define interventions
        interventions = {"X1": 2.0}
        
        # Generate counterfactual
        result = self.analyzer.generate_counterfactual(
            self.data,
            interventions,
            "Y",
            reference_values=instance
        )
        
        # Check result structure
        self.assertIn("factual_outcome", result)
        self.assertIn("counterfactual_outcome", result)
        self.assertIn("interventions", result)
        
        # The counterfactual outcome should be different from the factual outcome
        self.assertNotEqual(result["counterfactual_outcome"], result["factual_outcome"])
        
        # The difference should reflect the causal effect of X1 on Y
        # X1 has a direct effect (0.2) and indirect effect through X3 (0.5 * 0.6 = 0.3)
        # So total effect is approximately 0.5 per unit change in X1
        # The intervention changes X1 by 2.0, so expected effect is around 1.0
        intervention_size = interventions["X1"] - instance["X1"]
        expected_effect = intervention_size * 0.5  # approximate effect
        actual_effect = result["counterfactual_outcome"] - result["factual_outcome"]
        
        # Check that the effect is in the expected range
        self.assertAlmostEqual(actual_effect, expected_effect, delta=0.5)
    
    def test_generate_multiple_counterfactuals(self):
        """Test generating multiple counterfactuals."""
        # Create a reference instance
        instance = self.data.iloc[0]
        
        # Define multiple interventions
        feature_interventions = {
            "X1": [0.0, 1.0],
            "X2": [0.0, 1.0]
        }
        
        # Generate multiple counterfactuals
        results = self.analyzer.generate_multiple_counterfactuals(
            self.data,
            feature_interventions,
            "Y",
            reference_values=instance
        )
        
        # Should have 4 counterfactuals (2 values for X1 × 2 values for X2)
        self.assertEqual(len(results), 4)
        
        # Each counterfactual should have the structure from generate_counterfactual
        for cf in results.values():
            self.assertIn("factual_outcome", cf)
            self.assertIn("counterfactual_outcome", cf)
            self.assertIn("interventions", cf)


class TestCausalInference:
    """Test suite for causal inference components."""
    
    @pytest.fixture
    def synthetic_data(self) -> Tuple[pd.DataFrame, Dict[str, float], nx.DiGraph]:
        """Generate synthetic data for testing."""
        generator = SyntheticCausalDataGenerator(
            n_samples=1000,
            n_features=5,
            seed=42,
            noise_level=0.1
        )
        
        # Generate both linear and nonlinear data
        linear_data, linear_effects = generator.generate_linear_data()
        nonlinear_data, nonlinear_effects = generator.generate_nonlinear_data()
        
        # Combine datasets
        combined_data = pd.concat([linear_data, nonlinear_data], axis=0)
        combined_effects = {**linear_effects, **nonlinear_effects}
        
        return combined_data, combined_effects, generator.get_true_causal_graph()
    
    def test_structural_model_initialization(self, synthetic_data):
        """Test initialization of structural causal model."""
        data, effects, true_graph = synthetic_data
        
        model = StructuralCausalModel(
            data=data,
            outcome_var='Y',
            treatment_vars=list(effects.keys())
        )
        
        assert model.data is not None
        assert model.outcome_var == 'Y'
        assert set(model.treatment_vars) == set(effects.keys())
    
    def test_causal_effect_estimation(self, synthetic_data):
        """Test causal effect estimation accuracy."""
        data, true_effects, true_graph = synthetic_data
        
        model = StructuralCausalModel(
            data=data,
            outcome_var='Y',
            treatment_vars=list(true_effects.keys())
        )
        
        # Estimate causal effects
        estimated_effects = model.estimate_causal_effects()
        
        # Compare with true effects
        for var, true_effect in true_effects.items():
            assert var in estimated_effects
            # Allow for some estimation error
            assert abs(estimated_effects[var] - true_effect) < 0.5
    
    def test_counterfactual_analysis(self, synthetic_data):
        """Test counterfactual analysis functionality."""
        data, effects, true_graph = synthetic_data
        
        model = StructuralCausalModel(
            data=data,
            outcome_var='Y',
            treatment_vars=list(effects.keys())
        )
        
        analyzer = CounterfactualAnalyzer(model)
        
        # Test counterfactual prediction
        test_instance = data.iloc[0]
        intervention = {var: 1.0 for var in effects.keys()}
        
        counterfactual_outcome = analyzer.predict_counterfactual(
            instance=test_instance,
            intervention=intervention
        )
        
        assert isinstance(counterfactual_outcome, float)
        assert not np.isnan(counterfactual_outcome)
    
    def test_heterogeneous_effects(self, synthetic_data):
        """Test estimation of heterogeneous treatment effects."""
        data, effects, true_graph = synthetic_data
        
        model = StructuralCausalModel(
            data=data,
            outcome_var='Y',
            treatment_vars=list(effects.keys())
        )
        
        # Estimate heterogeneous effects
        het_effects = model.estimate_heterogeneous_effects()
        
        assert isinstance(het_effects, dict)
        for var in effects.keys():
            assert var in het_effects
            assert isinstance(het_effects[var], pd.Series)
    
    def test_robustness_checks(self, synthetic_data):
        """Test robustness of causal inference results."""
        data, effects, true_graph = synthetic_data
        
        model = StructuralCausalModel(
            data=data,
            outcome_var='Y',
            treatment_vars=list(effects.keys())
        )
        
        # Perform robustness checks
        robustness_results = model.perform_robustness_checks()
        
        assert isinstance(robustness_results, dict)
        assert 'sensitivity_analysis' in robustness_results
        assert 'placebo_tests' in robustness_results
    
    def test_graph_learning(self, synthetic_data):
        """Test causal graph learning from data."""
        data, effects, true_graph = synthetic_data
        
        model = StructuralCausalModel(
            data=data,
            outcome_var='Y',
            treatment_vars=list(effects.keys())
        )
        
        # Learn causal graph
        learned_graph = model.learn_causal_graph()
        
        assert isinstance(learned_graph, nx.DiGraph)
        # Check if key edges are recovered
        for edge in true_graph.edges():
            if edge[1] == 'Y':  # Focus on edges to outcome
                assert learned_graph.has_edge(*edge)
    
    def test_model_persistence(self, synthetic_data, tmp_path):
        """Test saving and loading of causal model."""
        data, effects, true_graph = synthetic_data
        
        model = StructuralCausalModel(
            data=data,
            outcome_var='Y',
            treatment_vars=list(effects.keys())
        )
        
        # Save model
        model_path = tmp_path / "causal_model.pkl"
        model.save(model_path)
        
        # Load model
        loaded_model = StructuralCausalModel.load(model_path)
        
        assert loaded_model.outcome_var == model.outcome_var
        assert set(loaded_model.treatment_vars) == set(model.treatment_vars)
        
        # Compare causal effects
        original_effects = model.estimate_causal_effects()
        loaded_effects = loaded_model.estimate_causal_effects()
        
        for var in effects.keys():
            assert abs(original_effects[var] - loaded_effects[var]) < 1e-6
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Test with empty data
        with pytest.raises(ValueError):
            StructuralCausalModel(
                data=pd.DataFrame(),
                outcome_var='Y',
                treatment_vars=['X1']
            )
        
        # Test with missing outcome variable
        data = pd.DataFrame({'X1': [1, 2, 3]})
        with pytest.raises(ValueError):
            StructuralCausalModel(
                data=data,
                outcome_var='Y',
                treatment_vars=['X1']
            )
        
        # Test with invalid treatment variables
        data = pd.DataFrame({'X1': [1, 2, 3], 'Y': [1, 2, 3]})
        with pytest.raises(ValueError):
            StructuralCausalModel(
                data=data,
                outcome_var='Y',
                treatment_vars=['Z1']  # Non-existent variable
            )


if __name__ == '__main__':
    unittest.main()
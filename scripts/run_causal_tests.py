#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

from tests.synthetic_data.causal_data_generator import SyntheticCausalDataGenerator
from tests.causal.validation import CausalValidation
from causal.structural_model import StructuralCausalModel

def run_tests(configs, output_dir):
    """Run tests with different configurations."""
    results = {}
    
    for config_name, config in configs.items():
        print(f"Running tests for configuration: {config_name}")
        config_results = []
        
        # Run multiple trials
        for trial in range(config['n_trials']):
            print(f"  Trial {trial+1}/{config['n_trials']}")
            
            # Generate data
            generator = SyntheticCausalDataGenerator(
                n_samples=config['n_samples'],
                n_features=config['n_features'],
                seed=42 + trial
            )
            generator.generate_random_dag()
            data, true_effects = generator.generate_linear_data()
            
            # Get true graph and effects
            true_graph = generator.get_true_causal_graph()
            
            # Initialize causal model
            causal_model = StructuralCausalModel(
                discovery_method=config['discovery_method'],
                alpha=config['alpha'],
                feature_names=data.columns.tolist()
            )
            
            # Time the discovery process
            start_time = time.time()
            causal_graph = causal_model.discover_graph(
                data=data,
                outcome_var='Y'
            )
            discovery_time = time.time() - start_time
            
            # Validate graph structure
            validator = CausalValidation(
                true_graph=true_graph,
                estimated_graph=causal_graph.graph
            )
            structure_metrics = validator.compare_graph_structure()
            
            # Save visualization
            vis_path = output_dir / f"{config_name}_trial{trial+1}.png"
            validator.visualize_comparison(output_path=vis_path)
            
            # Fit models and estimate effects
            causal_model.fit_models(data, model_type=config['model_type'])
            
            # Estimate effects
            estimated_effects = {}
            for feature in true_effects.keys():
                effect = causal_model.estimate_causal_effect(
                    data, feature, 'Y'
                )
                estimated_effects[feature] = effect['causal_effect']
            
            # Validate effect estimation
            effect_metrics = validator.compare_causal_effects(
                true_effects, estimated_effects
            )
            
            # Record results
            trial_result = {
                "structure_metrics": structure_metrics,
                "effect_metrics": effect_metrics['overall'],
                "discovery_time": discovery_time
            }
            config_results.append(trial_result)
        
        # Aggregate results across trials
        agg_results = {
            "precision_mean": np.mean([r['structure_metrics']['precision'] for r in config_results]),
            "recall_mean": np.mean([r['structure_metrics']['recall'] for r in config_results]),
            "f1_mean": np.mean([r['structure_metrics']['f1'] for r in config_results]),
            "effect_error_mean": np.mean([r['effect_metrics']['mean_absolute_error'] for r in config_results]),
            "discovery_time_mean": np.mean([r['discovery_time'] for r in config_results]),
            "trials": config_results
        }
        
        results[config_name] = agg_results
    
    # Save results
    with open(output_dir / "causal_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    # Create output directory
    output_dir = Path("./test_results/causal_configs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define test configurations
    configs = {
        "small_data": {
            "n_samples": 500,
            "n_features": 5,
            "edge_probability": 0.3,
            "noise_level": 0.1,
            "discovery_method": "pc",
            "alpha": 0.05,
            "model_type": "linear",
            "n_trials": 5
        },
        "large_data": {
            "n_samples": 2000,
            "n_features": 10,
            "edge_probability": 0.3,
            "noise_level": 0.1,
            "discovery_method": "pc",
            "alpha": 0.05,
            "model_type": "linear",
            "n_trials": 5
        },
        "high_noise": {
            "n_samples": 1000,
            "n_features": 5,
            "edge_probability": 0.3,
            "noise_level": 0.5,
            "discovery_method": "pc",
            "alpha": 0.05,
            "model_type": "linear",
            "n_trials": 5
        },
        "nonlinear": {
            "n_samples": 1000,
            "n_features": 5,
            "edge_probability": 0.3,
            "noise_level": 0.1,
            "discovery_method": "pc",
            "alpha": 0.05,
            "model_type": "random_forest",
            "n_trials": 5
        }
    }
    
    # Run tests
    results = run_tests(configs, output_dir)
    
    # Print summary
    print("\nTest Results Summary:")
    for config_name, config_results in results.items():
        print(f"\n{config_name}:")
        print(f"  Precision: {config_results['precision_mean']:.3f}")
        print(f"  Recall: {config_results['recall_mean']:.3f}")
        print(f"  F1 Score: {config_results['f1_mean']:.3f}")
        print(f"  Effect Error: {config_results['effect_error_mean']:.3f}")
        print(f"  Discovery Time: {config_results['discovery_time_mean']:.3f} seconds") 
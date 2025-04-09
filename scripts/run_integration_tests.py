#!/usr/bin/env python
"""
Run integration tests for the Cross-Modal Audience Intelligence Platform.

This script runs the end-to-end integration tests with different configurations
and collects test results for reporting.
"""
import os
import sys
import unittest
import argparse
import logging
import json
from datetime import datetime
from pathlib import Path
import tempfile
import shutil
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("integration_test_results.log")
    ]
)
logger = logging.getLogger("integration-test-runner")

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

def run_tests(test_modules, output_dir=None, skip_slow=False, device=None):
    """
    Run integration tests and collect results.
    
    Args:
        test_modules: List of test modules to run
        output_dir: Directory to store test results
        skip_slow: Skip slow tests
        device: Device to use for testing (cpu, cuda)
    
    Returns:
        Dictionary with test results
    """
    # Set environment variables for tests
    if skip_slow:
        os.environ["SKIP_SLOW_TESTS"] = "1"
    else:
        os.environ.pop("SKIP_SLOW_TESTS", None)
        
    if device:
        os.environ["TEST_DEVICE"] = device
    
    if output_dir:
        os.environ["TEST_OUTPUT_DIR"] = str(output_dir)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test modules to suite
    for module in test_modules:
        try:
            tests = loader.loadTestsFromName(module)
            suite.addTest(tests)
        except Exception as e:
            logger.error(f"Error loading tests from {module}: {e}")
    
    # Run tests
    start_time = datetime.now()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    end_time = datetime.now()
    
    # Collect results
    duration = (end_time - start_time).total_seconds()
    
    test_results = {
        "timestamp": start_time.isoformat(),
        "duration": duration,
        "total": result.testsRun,
        "passed": result.testsRun - len(result.errors) - len(result.failures),
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "details": {
            "failures": [
                {
                    "test": str(test),
                    "message": err
                }
                for test, err in result.failures
            ],
            "errors": [
                {
                    "test": str(test),
                    "message": err
                }
                for test, err in result.errors
            ],
            "skipped": [
                {
                    "test": str(test),
                    "reason": reason
                }
                for test, reason in result.skipped
            ]
        }
    }
    
    # Save results to file if output directory is provided
    if output_dir:
        output_path = Path(output_dir) / "test_results.json"
        with open(output_path, "w") as f:
            json.dump(test_results, f, indent=2)
        
        logger.info(f"Test results saved to {output_path}")
    
    return test_results

def print_test_summary(results):
    """Print a summary of test results."""
    print("\n" + "="*80)
    print(f"INTEGRATION TEST SUMMARY")
    print("="*80)
    print(f"Date: {results['timestamp']}")
    print(f"Duration: {results['duration']:.2f} seconds")
    print(f"Tests Run: {results['total']}")
    print(f"Passed: {results['passed']} ({results['passed']/results['total']*100:.2f}%)")
    print(f"Failed: {results['failures']}")
    print(f"Errors: {results['errors']}")
    print(f"Skipped: {results['skipped']}")
    print("="*80)
    
    if results["failures"] > 0 or results["errors"] > 0:
        print("\nFAILURES AND ERRORS:")
        for failure in results["details"]["failures"]:
            print(f"\nFAILURE: {failure['test']}")
            print("-"*40)
            print(failure["message"][:500] + "..." if len(failure["message"]) > 500 else failure["message"])
        
        for error in results["details"]["errors"]:
            print(f"\nERROR: {error['test']}")
            print("-"*40)
            print(error["message"][:500] + "..." if len(error["message"]) > 500 else error["message"])
    
    print("\n")

def main():
    """Run integration tests."""
    parser = argparse.ArgumentParser(description="Run integration tests")
    parser.add_argument("--output-dir", help="Directory to store test results")
    parser.add_argument("--skip-slow", action="store_true", help="Skip slow tests")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Device to use")
    parser.add_argument("--test-modules", nargs="+", default=["tests.integration.test_end_to_end"],
                        help="Test modules to run (default: tests.integration.test_end_to_end)")
    args = parser.parse_args()
    
    # Create output directory if provided
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run tests
        logger.info(f"Running integration tests from modules: {', '.join(args.test_modules)}")
        results = run_tests(
            test_modules=args.test_modules,
            output_dir=output_dir,
            skip_slow=args.skip_slow,
            device=args.device
        )
        
        # Print results
        print_test_summary(results)
        
        # Return exit code based on test results
        return 1 if results["failures"] > 0 or results["errors"] > 0 else 0
        
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
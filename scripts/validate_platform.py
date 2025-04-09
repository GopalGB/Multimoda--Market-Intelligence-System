#!/usr/bin/env python
"""
Validate the Cross-Modal Audience Intelligence Platform.

This script performs a complete validation of the platform by:
1. Running unit tests for individual components
2. Running integration tests for end-to-end functionality
3. Starting the API server and verifying its health
4. Testing model inference via the API
5. Generating a validation report

Usage:
    python validate_platform.py [--output-dir OUTPUT_DIR] [--skip-api] [--skip-slow]
"""
import os
import sys
import subprocess
import argparse
import logging
import json
import time
import requests
from datetime import datetime
from pathlib import Path
import unittest
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("platform_validation.log")
    ]
)
logger = logging.getLogger("platform-validator")

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

def run_unit_tests(test_modules, output_dir=None):
    """
    Run unit tests for individual components.
    
    Args:
        test_modules: List of test modules to run
        output_dir: Directory to store test results
    
    Returns:
        Dictionary with test results
    """
    logger.info("Running unit tests...")
    
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
        output_path = Path(output_dir) / "unit_test_results.json"
        with open(output_path, "w") as f:
            json.dump(test_results, f, indent=2)
        
        logger.info(f"Unit test results saved to {output_path}")
    
    return test_results

def run_integration_tests(skip_slow=False, device=None, output_dir=None):
    """
    Run integration tests using the run_integration_tests.py script.
    
    Args:
        skip_slow: Skip slow tests
        device: Device to use for testing (cpu, cuda)
        output_dir: Directory to store test results
    
    Returns:
        Dictionary with test results
    """
    logger.info("Running integration tests...")
    
    cmd = [sys.executable, "scripts/run_integration_tests.py"]
    
    if skip_slow:
        cmd.append("--skip-slow")
    
    if device:
        cmd.extend(["--device", device])
    
    if output_dir:
        cmd.extend(["--output-dir", str(output_dir)])
    
    # Run integration tests
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("Integration tests completed successfully")
        
        # Try to parse results from output_dir
        if output_dir:
            try:
                with open(Path(output_dir) / "test_results.json", "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading integration test results: {e}")
                return {"success": True, "raw_output": result.stdout}
        else:
            return {"success": True, "raw_output": result.stdout}
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Integration tests failed with exit code {e.returncode}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return {"success": False, "error": str(e), "stdout": e.stdout, "stderr": e.stderr}

def start_api_server(timeout=30):
    """
    Start the API server in a separate process.
    
    Args:
        timeout: Timeout in seconds to wait for the server to start
    
    Returns:
        Tuple of (process, success, error_message)
    """
    logger.info("Starting API server...")
    
    # Use the start_platform.sh script to start the API server
    cmd = ["bash", "scripts/start_platform.sh", "api"]
    
    # Start server process
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    # Wait for server to start
    start_time = time.time()
    server_started = False
    error_message = None
    
    while time.time() - start_time < timeout:
        try:
            # Check if server is running
            response = requests.get("http://localhost:5000/")
            if response.status_code == 200:
                server_started = True
                logger.info("API server started successfully")
                break
        except requests.RequestException:
            # Read line from stdout/stderr to check progress
            stdout_line = process.stdout.readline() if process.stdout else ""
            stderr_line = process.stderr.readline() if process.stderr else ""
            
            if stdout_line:
                logger.debug(f"Server stdout: {stdout_line.strip()}")
            
            if stderr_line:
                logger.debug(f"Server stderr: {stderr_line.strip()}")
                if "error" in stderr_line.lower():
                    error_message = stderr_line.strip()
            
            # Sleep briefly before retrying
            time.sleep(1)
    
    if not server_started:
        logger.error("Failed to start API server within timeout")
        if error_message:
            logger.error(f"Error: {error_message}")
        return process, False, error_message
    
    return process, True, None

def test_api_endpoints(output_dir=None):
    """
    Test API endpoints.
    
    Args:
        output_dir: Directory to store test results
    
    Returns:
        Dictionary with API test results
    """
    logger.info("Testing API endpoints...")
    
    api_results = {
        "timestamp": datetime.now().isoformat(),
        "endpoints": {},
        "success_rate": 0
    }
    
    # List of endpoints to test
    endpoints = [
        {"name": "Health Check", "url": "http://localhost:5000/", "method": "GET"},
        {"name": "API Status", "url": "http://localhost:5000/api/v1/status", "method": "GET"},
        {"name": "Content Analysis", "url": "http://localhost:5000/api/v1/analyze", "method": "POST", 
         "payload": {
             "url": "https://example.com/test",
             "title": "Test Content",
             "text": "This is a test content with engaging elements that should appeal to the audience.",
             "primary_image": None
         }}
    ]
    
    # Test each endpoint
    success_count = 0
    
    for endpoint in endpoints:
        try:
            logger.info(f"Testing endpoint: {endpoint['name']}")
            
            if endpoint["method"] == "GET":
                response = requests.get(endpoint["url"], timeout=10)
            elif endpoint["method"] == "POST":
                response = requests.post(endpoint["url"], json=endpoint.get("payload", {}), timeout=30)
            else:
                raise ValueError(f"Unsupported method: {endpoint['method']}")
            
            # Check response
            response_json = None
            try:
                response_json = response.json()
            except:
                response_json = {"error": "Failed to parse JSON response"}
            
            success = 200 <= response.status_code < 300
            
            api_results["endpoints"][endpoint["name"]] = {
                "url": endpoint["url"],
                "method": endpoint["method"],
                "status_code": response.status_code,
                "success": success,
                "response": response_json
            }
            
            if success:
                success_count += 1
                logger.info(f"Endpoint {endpoint['name']} test passed")
            else:
                logger.error(f"Endpoint {endpoint['name']} test failed with status code {response.status_code}")
        
        except Exception as e:
            logger.error(f"Error testing endpoint {endpoint['name']}: {e}")
            api_results["endpoints"][endpoint["name"]] = {
                "url": endpoint["url"],
                "method": endpoint["method"],
                "success": False,
                "error": str(e)
            }
    
    # Calculate success rate
    api_results["success_rate"] = (success_count / len(endpoints)) * 100 if endpoints else 0
    
    # Save results to file if output directory is provided
    if output_dir:
        output_path = Path(output_dir) / "api_test_results.json"
        with open(output_path, "w") as f:
            json.dump(api_results, f, indent=2)
        
        logger.info(f"API test results saved to {output_path}")
    
    return api_results

def generate_validation_report(unit_results, integration_results, api_results, output_dir):
    """
    Generate a comprehensive validation report.
    
    Args:
        unit_results: Results from unit tests
        integration_results: Results from integration tests
        api_results: Results from API tests
        output_dir: Directory to store the report
    """
    logger.info("Generating validation report...")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "platform_info": {
            "version": "1.0.0",  # This could be extracted from a version file
            "platform": sys.platform,
            "python_version": sys.version
        },
        "unit_tests": unit_results,
        "integration_tests": integration_results,
        "api_tests": api_results,
        "overall_status": "PASSED"
    }
    
    # Determine overall status
    if (unit_results.get("failures", 0) > 0 or 
        unit_results.get("errors", 0) > 0 or 
        not integration_results.get("success", False) or
        api_results.get("success_rate", 0) < 70):
        report["overall_status"] = "FAILED"
    
    # Save report to file
    output_path = Path(output_dir) / "validation_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    # Generate HTML report
    html_report = generate_html_report(report)
    html_path = Path(output_dir) / "validation_report.html"
    with open(html_path, "w") as f:
        f.write(html_report)
    
    logger.info(f"Validation report saved to {output_path} and {html_path}")
    return report

def generate_html_report(report):
    """
    Generate an HTML report from the validation results.
    
    Args:
        report: Validation report dictionary
    
    Returns:
        HTML string
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CAIP Platform Validation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .passed {{ background-color: #dff0d8; }}
            .failed {{ background-color: #f2dede; }}
            .metric {{ margin: 10px 0; }}
            .metric span {{ font-weight: bold; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>Cross-Modal Audience Intelligence Platform Validation Report</h1>
        <p>Generated on: {report["timestamp"]}</p>
        
        <div class="section {report["overall_status"].lower()}">
            <h2>Overall Status: {report["overall_status"]}</h2>
            <div class="metric">
                <span>Platform Version:</span> {report["platform_info"]["version"]}
            </div>
            <div class="metric">
                <span>Platform:</span> {report["platform_info"]["platform"]}
            </div>
            <div class="metric">
                <span>Python Version:</span> {report["platform_info"]["python_version"]}
            </div>
        </div>
        
        <div class="section {('passed' if report['unit_tests'].get('failures', 0) == 0 and report['unit_tests'].get('errors', 0) == 0 else 'failed')}">
            <h2>Unit Tests</h2>
            <div class="metric">
                <span>Total Tests:</span> {report["unit_tests"].get("total", "N/A")}
            </div>
            <div class="metric">
                <span>Passed:</span> {report["unit_tests"].get("passed", "N/A")}
            </div>
            <div class="metric">
                <span>Failures:</span> {report["unit_tests"].get("failures", "N/A")}
            </div>
            <div class="metric">
                <span>Errors:</span> {report["unit_tests"].get("errors", "N/A")}
            </div>
            <div class="metric">
                <span>Duration:</span> {report["unit_tests"].get("duration", "N/A")} seconds
            </div>
        </div>
        
        <div class="section {('passed' if report['integration_tests'].get('success', False) else 'failed')}">
            <h2>Integration Tests</h2>
            <div class="metric">
                <span>Status:</span> {("PASSED" if report["integration_tests"].get("success", False) else "FAILED")}
            </div>
        </div>
        
        <div class="section {('passed' if report['api_tests'].get('success_rate', 0) >= 70 else 'failed')}">
            <h2>API Tests</h2>
            <div class="metric">
                <span>Success Rate:</span> {report["api_tests"].get("success_rate", "N/A")}%
            </div>
            
            <h3>Endpoint Results</h3>
            <table>
                <tr>
                    <th>Endpoint</th>
                    <th>Method</th>
                    <th>URL</th>
                    <th>Status</th>
                    <th>Status Code</th>
                </tr>
    """
    
    # Add API test results
    for name, endpoint in report["api_tests"].get("endpoints", {}).items():
        status = "✅ PASSED" if endpoint.get("success", False) else "❌ FAILED"
        status_code = endpoint.get("status_code", "N/A")
        
        html += f"""
                <tr>
                    <td>{name}</td>
                    <td>{endpoint.get("method", "N/A")}</td>
                    <td>{endpoint.get("url", "N/A")}</td>
                    <td>{status}</td>
                    <td>{status_code}</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
        
        <div class="section">
            <h2>Recommendations</h2>
            <ul>
    """
    
    # Add recommendations based on test results
    if report["unit_tests"].get("failures", 0) > 0:
        html += f"""
                <li>Fix the {report["unit_tests"]["failures"]} unit test failures</li>
        """
    
    if report["unit_tests"].get("errors", 0) > 0:
        html += f"""
                <li>Fix the {report["unit_tests"]["errors"]} unit test errors</li>
        """
    
    if not report["integration_tests"].get("success", False):
        html += """
                <li>Investigate integration test failures</li>
        """
    
    if report["api_tests"].get("success_rate", 0) < 100:
        failed_endpoints = [name for name, endpoint in report["api_tests"].get("endpoints", {}).items() if not endpoint.get("success", False)]
        if failed_endpoints:
            html += f"""
                <li>Fix API endpoint issues: {', '.join(failed_endpoints)}</li>
            """
    
    if report["overall_status"] == "PASSED":
        html += """
                <li>All validation tests passed. The platform is ready for deployment.</li>
        """
    
    html += """
            </ul>
        </div>
    </body>
    </html>
    """
    
    return html

def main():
    """Run validation tests."""
    parser = argparse.ArgumentParser(description="Validate CAIP platform")
    parser.add_argument("--output-dir", default="validation_results", help="Directory to store validation results")
    parser.add_argument("--skip-api", action="store_true", help="Skip API testing")
    parser.add_argument("--skip-slow", action="store_true", help="Skip slow tests")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device to use")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Run unit tests
        unit_test_modules = [
            "tests.test_data",
            "tests.test_models",
            "tests.test_rag",
            "tests.test_causal",
            "tests.test_serving"
        ]
        unit_results = run_unit_tests(unit_test_modules, output_dir)
        
        # Step 2: Run integration tests
        integration_results = run_integration_tests(
            skip_slow=args.skip_slow,
            device=args.device,
            output_dir=output_dir
        )
        
        # Step 3: Test API (if not skipped)
        api_results = {"success_rate": 0, "endpoints": {}}
        server_process = None
        
        if not args.skip_api:
            try:
                # Start API server
                server_process, server_started, error = start_api_server()
                
                if server_started:
                    # Test API endpoints
                    api_results = test_api_endpoints(output_dir)
                else:
                    logger.error(f"Failed to start API server: {error}")
                    api_results = {
                        "success_rate": 0,
                        "error": f"Failed to start API server: {error}",
                        "endpoints": {}
                    }
            finally:
                # Terminate API server process if it was started
                if server_process:
                    logger.info("Terminating API server...")
                    server_process.terminate()
                    try:
                        server_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        server_process.kill()
        
        # Step 4: Generate validation report
        report = generate_validation_report(
            unit_results=unit_results,
            integration_results=integration_results,
            api_results=api_results,
            output_dir=output_dir
        )
        
        # Print summary
        print("\n" + "="*80)
        print(f"VALIDATION REPORT SUMMARY")
        print("="*80)
        print(f"Overall Status: {report['overall_status']}")
        print(f"Unit Tests: {unit_results['passed']}/{unit_results['total']} passed")
        print(f"Integration Tests: {'PASSED' if integration_results.get('success', False) else 'FAILED'}")
        print(f"API Tests: {api_results.get('success_rate', 0)}% success rate")
        print(f"Report saved to: {output_dir / 'validation_report.html'}")
        print("="*80 + "\n")
        
        # Return exit code based on validation status
        return 0 if report["overall_status"] == "PASSED" else 1
        
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
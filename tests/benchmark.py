"""
Regression testing script for Latent Forecasting Network.
Compares current performance against a known baseline.
"""

import argparse
import json
from pathlib import Path

# Known benchmark values for a tiny integration test run
# (These would normally correspond to a full run on a known dataset)
BASELINE_METRICS = {
    "perplexity": 100.0,
    "token_loss": 4.6,
    "latent_entropy": 0.5,
    "latent_variance": 0.1,
    "training_time_s": 300,  # 5 minutes for a short run
}

# Allowable degradation thresholds (percent)
THRESHOLDS = {
    "perplexity": 5.0,  # max 5% increase
    "token_loss": 5.0,  # max 5% increase
    "latent_entropy": -10.0,  # max 10% decrease
    "latent_variance": -10.0,  # max 10% decrease
    "training_time_s": 15.0,  # max 15% increase
}


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def run_regression_test(current_results: dict):
    print("========================================")
    print(" LFN Performance Regression Report")
    print("========================================")

    passed = True
    for metric, baseline_val in BASELINE_METRICS.items():
        if metric not in current_results:
            print(f"⚠️  Metric '{metric}' missing from current results. Skipping.")
            continue

        current_val = current_results[metric]
        threshold_pct = THRESHOLDS[metric]

        diff = current_val - baseline_val
        diff_pct = (diff / baseline_val) * 100

        # Check if degradation
        if threshold_pct >= 0:
            # Lower is better (e.g., perplexity, time)
            is_degradation = diff_pct > threshold_pct
        else:
            # Higher is better (e.g., entropy)
            is_degradation = diff_pct < threshold_pct

        status = "❌ FAIL" if is_degradation else "✅ PASS"
        if is_degradation:
            passed = False

        print(
            f"{status} | {metric:<16}: Base {baseline_val:.4f} -> Curr {current_val:.4f} ({diff_pct:+.2f}%)"
        )

    print("========================================")
    if passed:
        print("🎉 All regression tests passed!")
        return 0
    else:
        print("💥 Regression detected! Performance degraded beyond thresholds.")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser("LFN Regression Testing")
    parser.add_argument(
        "results_file", help="Path to evaluation_results.json from current run"
    )
    args = parser.parse_args()

    try:
        results = load_results(args.results_file)

        # Flatten representation metrics if present
        if "representation_metrics" in results:
            for k, v in results["representation_metrics"].items():
                results[k] = v

        exit_code = run_regression_test(results)
        exit(exit_code)
    except Exception as e:
        print(f"Error running regression test: {e}")
        exit(1)

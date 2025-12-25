#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Convenience script to run Falcon3-1B LoRA training experiments.

This script provides an easy interface to run:
1. TT-N150 training
2. CPU baseline training  
3. Both and compare results

Usage:
    # Run TT-N150 training only
    python run_experiments.py --mode tt

    # Run CPU baseline only
    python run_experiments.py --mode cpu

    # Run both and compare
    python run_experiments.py --mode both

    # Quick test run (limited steps)
    python run_experiments.py --mode tt --quick
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


def get_script_dir() -> Path:
    """Get the directory containing this script."""
    return Path(__file__).parent.resolve()


def run_training(config_path: str, extra_args: list = None) -> int:
    """Run the training script with specified config."""
    script_dir = get_script_dir()
    train_script = script_dir / "train_falcon3_lora.py"
    
    cmd = [
        sys.executable,
        str(train_script),
        "--config", config_path,
    ]
    
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, cwd=str(script_dir.parent.parent.parent.parent))
    return result.returncode


def run_comparison(cpu_metrics: str, tt_metrics: str, output_dir: str) -> int:
    """Run the comparison script."""
    script_dir = get_script_dir()
    compare_script = script_dir / "compare_results.py"
    
    cmd = [
        sys.executable,
        str(compare_script),
        "--cpu-metrics", cpu_metrics,
        "--tt-metrics", tt_metrics,
        "--output-dir", output_dir,
    ]
    
    print(f"\n{'='*60}")
    print(f"Running comparison: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd)
    return result.returncode


def create_quick_config(base_config: str, output_path: str) -> str:
    """Create a quick-test config with limited steps."""
    import yaml
    
    with open(base_config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override for quick testing
    config['max_steps'] = 50
    config['eval_steps'] = 10
    config['steps_freq'] = 5
    config['num_epochs'] = 1
    config['wandb_project'] = config.get('wandb_project', 'falcon3-test') + '-quick'
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Run Falcon3-1B LoRA training experiments"
    )
    parser.add_argument(
        "--mode",
        choices=["tt", "cpu", "both", "compare"],
        default="both",
        help="Training mode: 'tt' for TT-N150, 'cpu' for baseline, 'both' for both, 'compare' to compare existing results"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode with limited steps"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for TT execution verification"
    )
    parser.add_argument(
        "--cpu-metrics",
        type=str,
        default=None,
        help="Path to CPU metrics file (for compare mode)"
    )
    parser.add_argument(
        "--tt-metrics",
        type=str,
        default=None,
        help="Path to TT metrics file (for compare mode)"
    )
    
    args = parser.parse_args()
    
    script_dir = get_script_dir()
    configs_dir = script_dir / "configs"
    results_dir = script_dir / "results"
    
    # Setup environment
    if args.debug:
        os.environ["LOGGER_LEVEL"] = "DEBUG"
        print("Debug logging enabled - watch for TTIR graphs to verify TT execution")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Handle compare-only mode
    if args.mode == "compare":
        cpu_metrics = args.cpu_metrics or str(results_dir / "metrics_history_cpu.json")
        tt_metrics = args.tt_metrics or str(results_dir / "metrics_history_tt.json")
        output_dir = str(results_dir / "comparison")
        
        return run_comparison(cpu_metrics, tt_metrics, output_dir)
    
    # Prepare configs
    tt_config = str(configs_dir / "tt_n150.yaml")
    cpu_config = str(configs_dir / "cpu_baseline.yaml")
    
    if args.quick:
        print("Quick test mode - creating limited configs...")
        quick_dir = script_dir / "configs" / "quick"
        quick_dir.mkdir(parents=True, exist_ok=True)
        
        tt_config = create_quick_config(tt_config, str(quick_dir / "tt_n150_quick.yaml"))
        cpu_config = create_quick_config(cpu_config, str(quick_dir / "cpu_baseline_quick.yaml"))
    
    # Run experiments
    tt_success = True
    cpu_success = True
    
    if args.mode in ["tt", "both"]:
        print(f"\n{'#'*60}")
        print("# Starting TT-N150 Training")
        print(f"{'#'*60}")
        
        ret = run_training(tt_config)
        tt_success = (ret == 0)
        
        if not tt_success:
            print(f"\n⚠️  TT-N150 training failed with return code {ret}")
    
    if args.mode in ["cpu", "both"]:
        print(f"\n{'#'*60}")
        print("# Starting CPU Baseline Training")
        print(f"{'#'*60}")
        
        ret = run_training(cpu_config)
        cpu_success = (ret == 0)
        
        if not cpu_success:
            print(f"\n⚠️  CPU training failed with return code {ret}")
    
    # Run comparison if both succeeded
    if args.mode == "both" and tt_success and cpu_success:
        print(f"\n{'#'*60}")
        print("# Comparing Results")
        print(f"{'#'*60}")
        
        cpu_metrics = str(results_dir / "metrics_history_cpu.json")
        tt_metrics = str(results_dir / "metrics_history_tt.json")
        output_dir = str(results_dir / f"comparison_{timestamp}")
        
        run_comparison(cpu_metrics, tt_metrics, output_dir)
    
    # Summary
    print(f"\n{'='*60}")
    print("Experiment Summary")
    print(f"{'='*60}")
    
    if args.mode in ["tt", "both"]:
        status = "✅ Success" if tt_success else "❌ Failed"
        print(f"TT-N150 Training: {status}")
    
    if args.mode in ["cpu", "both"]:
        status = "✅ Success" if cpu_success else "❌ Failed"
        print(f"CPU Baseline:     {status}")
    
    if args.mode == "both" and tt_success and cpu_success:
        print(f"\nComparison results saved to: {results_dir}/comparison_{timestamp}")
    
    print(f"{'='*60}")
    
    return 0 if (tt_success and cpu_success) else 1


if __name__ == "__main__":
    sys.exit(main())


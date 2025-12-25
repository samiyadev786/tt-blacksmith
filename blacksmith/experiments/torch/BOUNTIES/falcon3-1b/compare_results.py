# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Comparison Script for CPU vs TT-N150 Training Results.

This script loads metrics from both CPU and TT-N150 training runs
and generates comparison plots and parity reports.

Usage:
    python compare_results.py \
        --cpu-metrics results/metrics_history_cpu.json \
        --tt-metrics results/metrics_history_tt.json \
        --output-dir results/comparison
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path for imports (handles hyphenated directory name)
_current_dir = Path(__file__).parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from utils import (
    compare_metrics,
    compute_metric_parity,
    print_parity_report,
    MATPLOTLIB_AVAILABLE,
)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    pass


def load_metrics(metrics_path: str) -> Dict[str, List[float]]:
    """Load metrics from JSON file."""
    with open(metrics_path, 'r') as f:
        return json.load(f)


def generate_summary_report(
    cpu_metrics: Dict[str, List[float]],
    tt_metrics: Dict[str, List[float]],
    parity: Dict[str, Dict[str, float]],
    output_path: str,
):
    """Generate a markdown summary report."""
    report = []
    report.append("# Falcon3-1B LoRA Training: CPU vs TT-N150 Comparison Report")
    report.append("")
    report.append("## Summary")
    report.append("")
    report.append("This report compares the training results between CPU baseline and TT-N150 hardware.")
    report.append("")

    # Training summary
    report.append("## Training Configuration")
    report.append("")
    report.append("| Parameter | Value |")
    report.append("|-----------|-------|")
    report.append("| Model | tiiuae/Falcon3-1B-Base |")
    report.append("| Dataset | Wikitext-2 |")
    report.append("| Training Type | LoRA |")
    report.append("| LoRA Rank | 16 |")
    report.append("| LoRA Alpha | 32 |")
    report.append("")

    # Metric parity
    report.append("## Metric Parity")
    report.append("")
    report.append("| Metric | Final CPU | Final TT-N150 | Mean Abs Diff | Mean Rel Diff |")
    report.append("|--------|-----------|---------------|---------------|---------------|")

    for metric_name, stats in parity.items():
        cpu_val = f"{stats['final_cpu']:.4f}" if stats['final_cpu'] is not None else "N/A"
        tt_val = f"{stats['final_tt']:.4f}" if stats['final_tt'] is not None else "N/A"
        mean_abs = f"{stats['mean_abs_diff']:.6f}"
        mean_rel = f"{stats['mean_rel_diff']:.2%}"
        report.append(f"| {metric_name} | {cpu_val} | {tt_val} | {mean_abs} | {mean_rel} |")

    report.append("")

    # Convergence analysis
    report.append("## Convergence Analysis")
    report.append("")

    if "train_loss" in cpu_metrics and "train_loss" in tt_metrics:
        cpu_final_loss = cpu_metrics["train_loss"][-1] if cpu_metrics["train_loss"] else None
        tt_final_loss = tt_metrics["train_loss"][-1] if tt_metrics["train_loss"] else None

        if cpu_final_loss and tt_final_loss:
            loss_diff = abs(cpu_final_loss - tt_final_loss)
            loss_diff_pct = loss_diff / cpu_final_loss * 100

            report.append(f"- **Final Training Loss Difference**: {loss_diff:.6f} ({loss_diff_pct:.2f}%)")

    if "val_loss" in cpu_metrics and "val_loss" in tt_metrics:
        cpu_final_val = cpu_metrics["val_loss"][-1] if cpu_metrics["val_loss"] else None
        tt_final_val = tt_metrics["val_loss"][-1] if tt_metrics["val_loss"] else None

        if cpu_final_val and tt_final_val:
            val_diff = abs(cpu_final_val - tt_final_val)
            val_diff_pct = val_diff / cpu_final_val * 100

            report.append(f"- **Final Validation Loss Difference**: {val_diff:.6f} ({val_diff_pct:.2f}%)")

    if "val_ppl" in cpu_metrics and "val_ppl" in tt_metrics:
        cpu_final_ppl = cpu_metrics["val_ppl"][-1] if cpu_metrics["val_ppl"] else None
        tt_final_ppl = tt_metrics["val_ppl"][-1] if tt_metrics["val_ppl"] else None

        if cpu_final_ppl and tt_final_ppl:
            ppl_diff = abs(cpu_final_ppl - tt_final_ppl)
            ppl_diff_pct = ppl_diff / cpu_final_ppl * 100

            report.append(f"- **Final Perplexity Difference**: {ppl_diff:.4f} ({ppl_diff_pct:.2f}%)")

    report.append("")
    report.append("## Conclusion")
    report.append("")

    # Determine if parity is acceptable
    acceptable = True
    for metric_name, stats in parity.items():
        if "loss" in metric_name.lower() and stats['mean_rel_diff'] > 0.05:  # 5% threshold
            acceptable = False
            break

    if acceptable:
        report.append("✅ **Training results show acceptable parity between CPU and TT-N150.**")
        report.append("")
        report.append("The loss curves and perplexity metrics converge similarly on both platforms,")
        report.append("indicating successful LoRA fine-tuning on TT hardware.")
    else:
        report.append("⚠️ **Some metrics show divergence between CPU and TT-N150.**")
        report.append("")
        report.append("Please review the comparison plots and investigate any fallback operations")
        report.append("that may have affected training.")

    report.append("")
    report.append("## Plots")
    report.append("")
    report.append("See the following plots in the output directory:")
    report.append("- `loss_comparison.png` - Training and validation loss comparison")
    report.append("- `perplexity_comparison.png` - Perplexity comparison")

    # Write report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"Generated comparison report: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare CPU and TT-N150 training results"
    )
    parser.add_argument(
        "--cpu-metrics",
        type=str,
        required=True,
        help="Path to CPU metrics JSON file"
    )
    parser.add_argument(
        "--tt-metrics",
        type=str,
        required=True,
        help="Path to TT-N150 metrics JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/comparison",
        help="Directory to save comparison outputs"
    )

    args = parser.parse_args()

    # Load metrics
    print(f"Loading CPU metrics from: {args.cpu_metrics}")
    cpu_metrics = load_metrics(args.cpu_metrics)

    print(f"Loading TT-N150 metrics from: {args.tt_metrics}")
    tt_metrics = load_metrics(args.tt_metrics)

    # Get steps for plotting
    cpu_steps = cpu_metrics.get("steps", [])
    tt_steps = tt_metrics.get("steps", [])

    # Use the shorter steps array for alignment
    min_steps = min(len(cpu_steps), len(tt_steps))
    steps = cpu_steps[:min_steps]

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate comparison plots
    print("Generating comparison plots...")
    compare_metrics(cpu_metrics, tt_metrics, steps, args.output_dir)

    # Compute parity statistics
    print("Computing metric parity...")
    parity = compute_metric_parity(cpu_metrics, tt_metrics)
    print_parity_report(parity)

    # Generate summary report
    report_path = os.path.join(args.output_dir, "comparison_report.md")
    generate_summary_report(cpu_metrics, tt_metrics, parity, report_path)

    # Save parity data
    parity_path = os.path.join(args.output_dir, "parity_statistics.json")
    with open(parity_path, 'w') as f:
        json.dump(parity, f, indent=2)
    print(f"Saved parity statistics to: {parity_path}")

    print("\nComparison complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()


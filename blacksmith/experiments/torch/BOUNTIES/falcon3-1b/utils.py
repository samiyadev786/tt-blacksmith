# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Utility functions for Falcon3-1B LoRA Training.

This module provides helper functions including:
- CPU fallback mechanisms for operations that fail on TT-N150
- Custom loss functions
- Metrics computation (perplexity, accuracy)
- Plotting utilities for loss curves comparison
"""
import logging
import math
import os
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for server environments
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


logger = logging.getLogger(__name__)


# ============================================================================
# CPU Fallback Mechanism
# ============================================================================


class FallbackRegistry:
    """
    Registry for tracking CPU fallback operations.

    This class maintains a record of operations that have been
    redirected to CPU due to TT-N150 compilation or runtime issues.
    """

    def __init__(self):
        self._fallback_ops: List[Dict[str, Any]] = []
        self._enabled: bool = True

    def register_fallback(
        self,
        operation_name: str,
        reason: str,
        issue_url: Optional[str] = None,
    ):
        """Register a fallback operation."""
        self._fallback_ops.append(
            {
                "operation": operation_name,
                "reason": reason,
                "issue_url": issue_url,
            }
        )
        logger.warning(
            f"CPU Fallback: {operation_name} - {reason}"
            + (f" (Issue: {issue_url})" if issue_url else "")
        )

    def get_fallbacks(self) -> List[Dict[str, Any]]:
        """Get all registered fallback operations."""
        return self._fallback_ops.copy()

    def print_summary(self):
        """Print a summary of all fallback operations."""
        if not self._fallback_ops:
            logger.info("No CPU fallback operations were required.")
            return

        logger.info("=" * 60)
        logger.info("CPU Fallback Operations Summary")
        logger.info("=" * 60)
        for i, op in enumerate(self._fallback_ops, 1):
            logger.info(f"{i}. {op['operation']}")
            logger.info(f"   Reason: {op['reason']}")
            if op.get("issue_url"):
                logger.info(f"   Issue: {op['issue_url']}")
        logger.info("=" * 60)


# Global fallback registry
fallback_registry = FallbackRegistry()


def cpu_fallback(
    operation_name: str,
    reason: str = "Compilation/runtime issue on TT-N150",
    issue_url: Optional[str] = None,
):
    """
    Decorator for CPU fallback on specific operations.

    Use this decorator to wrap functions that may fail on TT hardware.
    When the decorated function fails, it will automatically retry
    on CPU with the same inputs.

    Args:
        operation_name: Name of the operation (for logging)
        reason: Reason for potential fallback
        issue_url: URL to the issue tracker for this problem

    Example:
        @cpu_fallback("custom_attention", "Attention pattern not supported")
        def custom_attention(q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Register the fallback
                fallback_registry.register_fallback(
                    operation_name=operation_name,
                    reason=f"{reason}: {str(e)}",
                    issue_url=issue_url,
                )

                # Move tensors to CPU and retry
                cpu_args = tuple(
                    arg.cpu() if isinstance(arg, torch.Tensor) else arg for arg in args
                )
                cpu_kwargs = {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in kwargs.items()
                }

                result = func(*cpu_args, **cpu_kwargs)

                # Move result back to original device if needed
                if args and isinstance(args[0], torch.Tensor):
                    original_device = args[0].device
                    if isinstance(result, torch.Tensor):
                        result = result.to(original_device)
                    elif isinstance(result, tuple):
                        result = tuple(
                            r.to(original_device) if isinstance(r, torch.Tensor) else r
                            for r in result
                        )

                return result

        return wrapper

    return decorator


class CPUFallbackModule(nn.Module):
    """
    Wrapper module that executes operations on CPU when TT execution fails.

    Use this to wrap specific layers or operations that need fallback support.
    """

    def __init__(
        self,
        module: nn.Module,
        operation_name: str,
        reason: str = "Module execution failed on TT",
        issue_url: Optional[str] = None,
    ):
        super().__init__()
        self.module = module
        self.operation_name = operation_name
        self.reason = reason
        self.issue_url = issue_url
        self._fallback_triggered = False

    def forward(self, *args, **kwargs):
        if self._fallback_triggered:
            # Already in fallback mode, execute on CPU directly
            return self._execute_on_cpu(*args, **kwargs)

        try:
            return self.module(*args, **kwargs)
        except Exception as e:
            self._fallback_triggered = True
            fallback_registry.register_fallback(
                operation_name=self.operation_name,
                reason=f"{self.reason}: {str(e)}",
                issue_url=self.issue_url,
            )
            return self._execute_on_cpu(*args, **kwargs)

    def _execute_on_cpu(self, *args, **kwargs):
        """Execute the module on CPU."""
        # Store original device
        original_device = None
        if args and isinstance(args[0], torch.Tensor):
            original_device = args[0].device

        # Move inputs to CPU
        cpu_args = tuple(
            arg.cpu() if isinstance(arg, torch.Tensor) else arg for arg in args
        )
        cpu_kwargs = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()
        }

        # Move module to CPU temporarily
        original_module_device = next(self.module.parameters()).device
        self.module = self.module.cpu()

        # Execute
        result = self.module(*cpu_args, **cpu_kwargs)

        # Move module back
        self.module = self.module.to(original_module_device)

        # Move result back to original device
        if original_device is not None:
            if isinstance(result, torch.Tensor):
                result = result.to(original_device)
            elif isinstance(result, tuple):
                result = tuple(
                    r.to(original_device) if isinstance(r, torch.Tensor) else r
                    for r in result
                )

        return result


# ============================================================================
# Loss Functions
# ============================================================================


def cross_entropy_loss_with_mask(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute cross-entropy loss with label masking.

    Args:
        logits: Model output logits [batch, seq_len, vocab_size]
        labels: Target labels [batch, seq_len]
        ignore_index: Index to ignore in loss computation

    Returns:
        Scalar loss tensor
    """
    # Shift for causal LM: predict next token
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)

    # Compute loss
    loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=ignore_index)

    return loss


@cpu_fallback(
    operation_name="custom_cross_entropy",
    reason="Custom cross-entropy may have compilation issues on TT",
    issue_url="https://github.com/tenstorrent/tt-xla/issues/1993",
)
def custom_cross_entropy_loss(
    shift_logits: torch.Tensor,
    expected_output: torch.Tensor,
    labels_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Custom cross-entropy loss implementation.

    This version uses log_softmax and manual reduction for better
    compatibility with TT hardware.

    Args:
        shift_logits: Shifted logits [batch, seq_len, vocab_size]
        expected_output: One-hot encoded targets [batch, seq_len, vocab_size]
        labels_mask: Boolean mask for valid positions [batch, seq_len]

    Returns:
        Scalar loss tensor
    """
    log_probs = F.log_softmax(shift_logits, dim=-1)
    ce_loss = -(expected_output * log_probs).sum(dim=-1, keepdim=True)

    labels_mask = labels_mask.unsqueeze(-1).float()
    ce_loss = ce_loss * labels_mask

    num_valid = labels_mask.sum(dim=1, keepdim=True)
    num_valid = torch.clamp(num_valid, min=1.0)
    loss_per_sample = ce_loss.sum(dim=1, keepdim=True) / num_valid
    loss = loss_per_sample.mean()

    return loss


def transform_labels_for_custom_loss(
    labels: torch.Tensor,
    ignored_index: int,
    vocab_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transform labels for custom cross-entropy loss.

    Args:
        labels: Target labels [batch, seq_len]
        ignored_index: Index to ignore
        vocab_size: Size of vocabulary

    Returns:
        Tuple of (one_hot_labels, labels_mask)
    """
    labels_mask = labels != ignored_index
    labels_clean = labels.clone()
    labels_clean[labels == ignored_index] = 0
    expected_output = F.one_hot(labels_clean, num_classes=vocab_size)

    return expected_output.float(), labels_mask


# ============================================================================
# Metrics Computation
# ============================================================================


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.

    Args:
        loss: Cross-entropy loss value

    Returns:
        Perplexity value
    """
    return math.exp(loss) if loss < 100 else float("inf")


def compute_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """
    Compute token-level accuracy.

    Args:
        predictions: Predicted token IDs [batch, seq_len]
        labels: Ground truth token IDs [batch, seq_len]
        ignore_index: Index to ignore

    Returns:
        Accuracy as a float between 0 and 1
    """
    mask = labels != ignore_index
    if mask.sum() == 0:
        return 0.0

    correct = (predictions == labels) & mask
    accuracy = correct.sum().float() / mask.sum().float()

    return accuracy.item()


# ============================================================================
# Plotting Utilities
# ============================================================================


def save_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    steps: List[int],
    output_path: str,
    title: str = "Training and Validation Loss",
):
    """
    Save training and validation loss curves to a file.

    Args:
        train_losses: List of training loss values
        val_losses: List of validation loss values
        steps: List of step numbers
        output_path: Path to save the plot
        title: Plot title
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available, skipping loss curve plot")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, label="Training Loss", color="blue", linewidth=2)
    plt.plot(steps, val_losses, label="Validation Loss", color="orange", linewidth=2)
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved loss curves to {output_path}")


def save_perplexity_curves(
    train_ppl: List[float],
    val_ppl: List[float],
    steps: List[int],
    output_path: str,
    title: str = "Training and Validation Perplexity",
):
    """
    Save perplexity curves to a file.

    Args:
        train_ppl: List of training perplexity values
        val_ppl: List of validation perplexity values
        steps: List of step numbers
        output_path: Path to save the plot
        title: Plot title
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available, skipping perplexity curve plot")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_ppl, label="Training Perplexity", color="blue", linewidth=2)
    plt.plot(steps, val_ppl, label="Validation Perplexity", color="orange", linewidth=2)
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Perplexity", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved perplexity curves to {output_path}")


def compare_metrics(
    cpu_metrics: Dict[str, List[float]],
    tt_metrics: Dict[str, List[float]],
    steps: List[int],
    output_dir: str,
    prefix: str = "",
):
    """
    Compare and plot metrics from CPU and TT-N150 runs.

    Args:
        cpu_metrics: Dictionary of CPU metrics
        tt_metrics: Dictionary of TT-N150 metrics
        steps: List of step numbers
        output_dir: Directory to save comparison plots
        prefix: Prefix for output filenames
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available, skipping metrics comparison plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Compare loss
    if "train_loss" in cpu_metrics and "train_loss" in tt_metrics:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(
            steps, cpu_metrics["train_loss"], label="CPU", color="blue", linewidth=2
        )
        plt.plot(
            steps,
            tt_metrics["train_loss"],
            label="TT-N150",
            color="green",
            linewidth=2,
            linestyle="--",
        )
        plt.xlabel("Steps")
        plt.ylabel("Training Loss")
        plt.title("Training Loss: CPU vs TT-N150")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        if "val_loss" in cpu_metrics and "val_loss" in tt_metrics:
            plt.plot(
                steps, cpu_metrics["val_loss"], label="CPU", color="blue", linewidth=2
            )
            plt.plot(
                steps,
                tt_metrics["val_loss"],
                label="TT-N150",
                color="green",
                linewidth=2,
                linestyle="--",
            )
            plt.xlabel("Steps")
            plt.ylabel("Validation Loss")
            plt.title("Validation Loss: CPU vs TT-N150")
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}loss_comparison.png"), dpi=150)
        plt.close()

    # Compare perplexity
    if "train_ppl" in cpu_metrics and "train_ppl" in tt_metrics:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(
            steps, cpu_metrics["train_ppl"], label="CPU", color="blue", linewidth=2
        )
        plt.plot(
            steps,
            tt_metrics["train_ppl"],
            label="TT-N150",
            color="green",
            linewidth=2,
            linestyle="--",
        )
        plt.xlabel("Steps")
        plt.ylabel("Training Perplexity")
        plt.title("Training Perplexity: CPU vs TT-N150")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        if "val_ppl" in cpu_metrics and "val_ppl" in tt_metrics:
            plt.plot(
                steps, cpu_metrics["val_ppl"], label="CPU", color="blue", linewidth=2
            )
            plt.plot(
                steps,
                tt_metrics["val_ppl"],
                label="TT-N150",
                color="green",
                linewidth=2,
                linestyle="--",
            )
            plt.xlabel("Steps")
            plt.ylabel("Validation Perplexity")
            plt.title("Validation Perplexity: CPU vs TT-N150")
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{prefix}perplexity_comparison.png"), dpi=150
        )
        plt.close()

    logger.info(f"Saved comparison plots to {output_dir}")


def compute_metric_parity(
    cpu_metrics: Dict[str, List[float]],
    tt_metrics: Dict[str, List[float]],
) -> Dict[str, Dict[str, float]]:
    """
    Compute parity statistics between CPU and TT-N150 metrics.

    Args:
        cpu_metrics: Dictionary of CPU metrics
        tt_metrics: Dictionary of TT-N150 metrics

    Returns:
        Dictionary containing parity statistics for each metric
    """
    parity = {}

    for metric_name in cpu_metrics.keys():
        if metric_name not in tt_metrics:
            continue

        cpu_vals = np.array(cpu_metrics[metric_name])
        tt_vals = np.array(tt_metrics[metric_name])

        # Ensure same length
        min_len = min(len(cpu_vals), len(tt_vals))
        cpu_vals = cpu_vals[:min_len]
        tt_vals = tt_vals[:min_len]

        # Compute statistics
        abs_diff = np.abs(cpu_vals - tt_vals)
        rel_diff = abs_diff / (np.abs(cpu_vals) + 1e-8)

        parity[metric_name] = {
            "mean_abs_diff": float(np.mean(abs_diff)),
            "max_abs_diff": float(np.max(abs_diff)),
            "mean_rel_diff": float(np.mean(rel_diff)),
            "max_rel_diff": float(np.max(rel_diff)),
            "final_cpu": float(cpu_vals[-1]) if len(cpu_vals) > 0 else None,
            "final_tt": float(tt_vals[-1]) if len(tt_vals) > 0 else None,
        }

    return parity


def print_parity_report(parity: Dict[str, Dict[str, float]]):
    """Print a formatted parity report."""
    logger.info("=" * 70)
    logger.info("Metric Parity Report: CPU vs TT-N150")
    logger.info("=" * 70)

    for metric_name, stats in parity.items():
        logger.info(f"\n{metric_name}:")
        logger.info(
            f"  Final CPU:         {stats['final_cpu']:.6f}"
            if stats["final_cpu"]
            else "  Final CPU:         N/A"
        )
        logger.info(
            f"  Final TT-N150:     {stats['final_tt']:.6f}"
            if stats["final_tt"]
            else "  Final TT-N150:     N/A"
        )
        logger.info(f"  Mean Abs Diff:     {stats['mean_abs_diff']:.6f}")
        logger.info(f"  Max Abs Diff:      {stats['max_abs_diff']:.6f}")
        logger.info(f"  Mean Rel Diff:     {stats['mean_rel_diff']:.2%}")
        logger.info(f"  Max Rel Diff:      {stats['max_rel_diff']:.2%}")

    logger.info("=" * 70)


# ============================================================================
# Model Utilities
# ============================================================================


def get_trainable_params(model: nn.Module) -> Tuple[int, int, float]:
    """
    Get trainable parameter statistics for a model.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (total_params, trainable_params, trainable_percentage)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = 100.0 * trainable_params / total_params if total_params > 0 else 0.0

    return total_params, trainable_params, trainable_pct


def estimate_memory_usage(
    model: nn.Module, batch_size: int, seq_length: int
) -> Dict[str, float]:
    """
    Estimate memory usage for training.

    Args:
        model: PyTorch model
        batch_size: Training batch size
        seq_length: Sequence length

    Returns:
        Dictionary with memory estimates in GB
    """
    total_params, trainable_params, _ = get_trainable_params(model)

    # Estimate memory components (assuming bfloat16)
    bytes_per_param = 2  # bfloat16
    bytes_per_grad = 2  # bfloat16
    bytes_per_optim_state = 8  # Adam maintains 2 states per param, 4 bytes each

    param_memory = total_params * bytes_per_param
    grad_memory = trainable_params * bytes_per_grad
    optim_memory = trainable_params * bytes_per_optim_state

    # Activation memory (rough estimate)
    # For Falcon3-1B: hidden_size=2048, num_layers=24
    hidden_size = 2048
    num_layers = 24
    activation_memory = (
        batch_size * seq_length * hidden_size * num_layers * bytes_per_param
    )

    total_memory = param_memory + grad_memory + optim_memory + activation_memory

    return {
        "parameters_gb": param_memory / 1e9,
        "gradients_gb": grad_memory / 1e9,
        "optimizer_gb": optim_memory / 1e9,
        "activations_gb": activation_memory / 1e9,
        "total_estimated_gb": total_memory / 1e9,
    }

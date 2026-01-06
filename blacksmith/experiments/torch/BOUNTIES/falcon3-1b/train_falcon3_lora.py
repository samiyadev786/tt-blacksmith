# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Falcon3-1B-Base LoRA Training Script for TT-N150 and CPU.

This script implements end-to-end LoRA fine-tuning of tiiuae/Falcon3-1B-Base
on the Wikitext-2 dataset. It supports both TT-N150 hardware execution
and CPU baseline for comparison.

Usage:
    # Train on TT-N150:
    LOGGER_LEVEL=DEBUG python train_falcon3_lora.py --config configs/tt_n150.yaml

    # Train on CPU baseline:
    python train_falcon3_lora.py --config configs/cpu_baseline.yaml

Verification:
    To confirm training is running on TT hardware, set:
    export LOGGER_LEVEL=DEBUG
    Look for printed TTIR graphs confirming TT execution.
"""
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports (handles hyphenated directory name)
_current_dir = Path(__file__).parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Conditional import for TT-XLA
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr

    TORCH_XLA_AVAILABLE = True
except ImportError:
    TORCH_XLA_AVAILABLE = False
    print("Warning: torch_xla not available. TT device support disabled.")

from configs import Falcon3TrainingConfig
from utils import (
    compute_perplexity,
    estimate_memory_usage,
    fallback_registry,
    get_trainable_params,
    save_loss_curves,
    save_perplexity_curves,
)
from wikitext_dataset import get_wikitext_dataset

from blacksmith.tools.checkpoints_manager import CheckpointManager
from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.reproducibility_manager import ReproducibilityManager


class Falcon3DeviceManager:
    """
    Device manager for Falcon3-1B training.

    Handles device setup for both TT-N150 and CPU training,
    with proper environment configuration.
    """

    def __init__(self, config: Falcon3TrainingConfig):
        self.config = config
        self.device = None
        self._setup()

    def _setup(self):
        """Setup device based on configuration."""
        if not self.config.use_tt:
            # CPU mode
            self.device = torch.device("cpu")
            print(f"Using device: {self.device}")
            return

        if not TORCH_XLA_AVAILABLE:
            print("Warning: torch_xla not available, falling back to CPU")
            self.config.use_tt = False
            self.device = torch.device("cpu")
            return

        # TT device setup
        self._setup_tt_environment()
        self.device = torch_xla.device()
        print(f"Using TT device: {self.device}")

    def _setup_tt_environment(self):
        """Setup TT-specific environment variables."""
        xr.set_device_type("TT")
        os.environ["PJRT_DEVICE"] = "TT"
        os.environ["XLA_STABLEHLO_COMPILE"] = "1"

        # Enable debug logging to verify TTIR graphs
        if os.environ.get("LOGGER_LEVEL", "").upper() == "DEBUG":
            os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_dump_to=/tmp/xla_dumps"

    def prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch tensors to device."""
        return {k: v.to(self.device) for k, v in batch.items()}

    def sync(self, wait: bool = True):
        """Synchronize device (for TT)."""
        if self.config.use_tt and TORCH_XLA_AVAILABLE:
            torch_xla.sync(wait=wait)


def get_falcon3_model(config: Falcon3TrainingConfig, device: torch.device):
    """
    Load Falcon3-1B model with LoRA configuration.

    Args:
        config: Training configuration
        device: Target device

    Returns:
        Model with LoRA adapters applied
    """
    print(f"Loading model: {config.model_name}")

    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=eval(config.dtype),
        trust_remote_code=True,
        use_cache=not config.gradient_checkpointing,
    )

    # Enable gradient checkpointing if configured
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Configure LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Move to device
    model = model.to(device)

    # Print trainable parameters
    total, trainable, pct = get_trainable_params(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,} ({pct:.2f}%)")

    return model


def collate_fn_causal_lm(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Collate function that shifts labels for causal LM training.

    Args:
        batch: Dictionary with input_ids, attention_mask, labels

    Returns:
        Batch with shifted labels
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    # Shift labels: we want to predict token[i+1] from token[i]
    shifted_labels = labels[:, 1:].contiguous()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": shifted_labels,
    }


def training_step(
    model,
    batch: Dict[str, torch.Tensor],
    config: Falcon3TrainingConfig,
) -> torch.Tensor:
    """
    Execute a single training step.

    This function is designed to keep large tensors (logits) scoped locally
    to prevent memory issues.

    Args:
        model: The model to train
        batch: Input batch
        config: Training configuration

    Returns:
        Loss tensor
    """
    # Forward pass
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )

    logits = outputs.logits

    # Compute loss
    # Shift logits to align with shifted labels
    shift_logits = logits[:, :-1, :].contiguous()

    # Standard cross-entropy loss
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        batch["labels"].view(-1),
        ignore_index=config.ignored_index,
    )

    return loss


def validate(
    model,
    val_dataloader: DataLoader,
    device_manager: Falcon3DeviceManager,
    config: Falcon3TrainingConfig,
    logger: TrainingLogger,
) -> Dict[str, float]:
    """
    Run validation and compute metrics.

    Args:
        model: Model to evaluate
        val_dataloader: Validation data loader
        device_manager: Device manager
        config: Training configuration
        logger: Logger instance

    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation", leave=False):
            batch = device_manager.prepare_batch(batch)

            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()

            # Compute loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                batch["labels"].view(-1),
                ignore_index=config.ignored_index,
            )

            # Compute accuracy
            predictions = shift_logits.argmax(dim=-1)
            mask = batch["labels"] != config.ignored_index
            correct = ((predictions == batch["labels"]) & mask).sum()
            total_correct += correct.item()
            total_tokens += mask.sum().item()

            total_loss += loss.item()
            num_batches += 1

            if config.use_tt:
                device_manager.sync()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    perplexity = compute_perplexity(avg_loss)

    return {
        "val_loss": avg_loss,
        "val_accuracy": accuracy,
        "val_perplexity": perplexity,
    }


def train(
    config: Falcon3TrainingConfig,
    device_manager: Falcon3DeviceManager,
    logger: TrainingLogger,
    checkpoint_manager: CheckpointManager,
) -> Dict[str, List[float]]:
    """
    Main training loop for Falcon3-1B LoRA fine-tuning.

    Args:
        config: Training configuration
        device_manager: Device manager
        logger: Training logger
        checkpoint_manager: Checkpoint manager

    Returns:
        Dictionary of training metrics history
    """
    logger.info("=" * 60)
    logger.info("Starting Falcon3-1B LoRA Training")
    logger.info(f"Device: {'TT-N150' if config.use_tt else 'CPU'}")
    logger.info("=" * 60)

    # Load model
    model = get_falcon3_model(config, device_manager.device)
    logger.info(f"Loaded model: {config.model_name}")

    # Estimate memory
    mem_estimate = estimate_memory_usage(model, config.batch_size, config.max_length)
    logger.info(f"Estimated memory usage: {mem_estimate['total_estimated_gb']:.2f} GB")

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Learning rate scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR

    scheduler = None
    if config.warmup_steps > 0:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.num_epochs * 1000,  # Approximate total steps
            eta_min=config.learning_rate * 0.1,
        )

    # Load checkpoint if resuming
    start_step = 0
    start_epoch = 0
    if config.resume_from_checkpoint:
        checkpoint_data = checkpoint_manager.load_checkpoint(model, optimizer)
        if checkpoint_data:
            start_step = checkpoint_data.get("step", 0)
            start_epoch = checkpoint_data.get("epoch", 0)
            logger.info(f"Resumed from checkpoint: step {start_step}, epoch {start_epoch}")

    # Load datasets
    logger.info("Loading Wikitext-2 dataset...")
    train_dataset = get_wikitext_dataset(config, split="train", collate_fn=collate_fn_causal_lm)
    train_dataloader = train_dataset.get_dataloader()
    logger.info(f"Train dataset size: {len(train_dataset)} examples")

    val_dataset = get_wikitext_dataset(config, split="validation", collate_fn=collate_fn_causal_lm)
    val_dataloader = val_dataset.get_dataloader()
    logger.info(f"Validation dataset size: {len(val_dataset)} examples")

    # Metrics tracking
    metrics_history = {
        "train_loss": [],
        "train_ppl": [],
        "val_loss": [],
        "val_ppl": [],
        "val_accuracy": [],
        "steps": [],
        "learning_rate": [],
    }

    global_step = start_step
    running_loss = 0.0
    best_val_loss = float("inf")

    try:
        for epoch in range(start_epoch, config.num_epochs):
            model.train()
            epoch_loss = 0.0
            epoch_steps = 0

            progress_bar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch + 1}/{config.num_epochs}",
                leave=True,
            )

            for batch_idx, batch in enumerate(progress_bar):
                # Check max_steps
                if config.max_steps > 0 and global_step >= config.max_steps:
                    logger.info(f"Reached max_steps ({config.max_steps}), stopping training")
                    break

                # Zero gradients
                optimizer.zero_grad()

                # Prepare batch
                batch = device_manager.prepare_batch(batch)

                # Training step
                loss = training_step(model, batch, config)

                # Backward pass
                loss.backward()

                # Sync for TT device
                if config.use_tt:
                    device_manager.sync()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimizer step
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                # Track metrics
                loss_value = loss.item()
                running_loss += loss_value
                epoch_loss += loss_value
                epoch_steps += 1
                global_step += 1

                # Update progress bar
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss_value:.4f}",
                        "avg_loss": f"{epoch_loss / epoch_steps:.4f}",
                    }
                )

                # Periodic logging and validation
                if global_step % config.steps_freq == 0:
                    avg_loss = running_loss / config.steps_freq
                    perplexity = compute_perplexity(avg_loss)
                    current_lr = optimizer.param_groups[0]["lr"]

                    # Log training metrics
                    logger.log_metrics(
                        {
                            "train/loss": avg_loss,
                            "train/perplexity": perplexity,
                            "train/learning_rate": current_lr,
                        },
                        step=global_step,
                        commit=False,
                    )

                    running_loss = 0.0

                    # Track metrics
                    metrics_history["train_loss"].append(avg_loss)
                    metrics_history["train_ppl"].append(perplexity)
                    metrics_history["steps"].append(global_step)
                    metrics_history["learning_rate"].append(current_lr)

                # Validation
                if config.do_validation and global_step % config.eval_steps == 0:
                    val_metrics = validate(model, val_dataloader, device_manager, config, logger)

                    logger.log_metrics(
                        {
                            "val/loss": val_metrics["val_loss"],
                            "val/perplexity": val_metrics["val_perplexity"],
                            "val/accuracy": val_metrics["val_accuracy"],
                        },
                        step=global_step,
                    )

                    # Track metrics
                    metrics_history["val_loss"].append(val_metrics["val_loss"])
                    metrics_history["val_ppl"].append(val_metrics["val_perplexity"])
                    metrics_history["val_accuracy"].append(val_metrics["val_accuracy"])

                    # Save best model
                    if val_metrics["val_loss"] < best_val_loss:
                        best_val_loss = val_metrics["val_loss"]
                        checkpoint_manager.save_checkpoint(
                            model,
                            global_step,
                            epoch,
                            optimizer,
                            metrics=val_metrics,
                            checkpoint_name="best_model.pt",
                        )

                    # Clear XLA cache if using TT
                    if config.use_tt and TORCH_XLA_AVAILABLE:
                        xr.clear_computation_cache()

                    model.train()

                # Checkpoint saving
                if checkpoint_manager.should_save_checkpoint(global_step):
                    checkpoint_manager.save_checkpoint(
                        model,
                        global_step,
                        epoch,
                        optimizer,
                        metrics={"train/loss": loss_value},
                    )

            # End of epoch logging
            epoch_avg_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0.0
            logger.info(f"Epoch {epoch + 1} completed. Average loss: {epoch_avg_loss:.4f}")

            # Epoch checkpoint
            if checkpoint_manager.should_save_checkpoint(global_step, epoch):
                checkpoint_manager.save_checkpoint(
                    model,
                    global_step,
                    epoch,
                    optimizer,
                    checkpoint_name=f"epoch_{epoch + 1}.pt",
                )

        # Final checkpoint
        final_path = checkpoint_manager.save_checkpoint(
            model,
            global_step,
            epoch,
            optimizer,
            checkpoint_name="final_model.pt",
        )
        logger.info(f"Saved final model to {final_path}")

        # Save training plots
        if config.save_plots and len(metrics_history["steps"]) > 0:
            plots_dir = os.path.join(config.project_dir, config.plots_dir)
            os.makedirs(plots_dir, exist_ok=True)

            device_suffix = "tt_n150" if config.use_tt else "cpu"

            if metrics_history["train_loss"] and metrics_history["val_loss"]:
                # Align lengths for plotting
                min_len = min(len(metrics_history["train_loss"]), len(metrics_history["val_loss"]))
                save_loss_curves(
                    metrics_history["train_loss"][:min_len],
                    metrics_history["val_loss"][:min_len],
                    metrics_history["steps"][:min_len],
                    os.path.join(plots_dir, f"loss_curves_{device_suffix}.png"),
                    title=f"Falcon3-1B LoRA Training - {'TT-N150' if config.use_tt else 'CPU'}",
                )

            if metrics_history["train_ppl"] and metrics_history["val_ppl"]:
                min_len = min(len(metrics_history["train_ppl"]), len(metrics_history["val_ppl"]))
                save_perplexity_curves(
                    metrics_history["train_ppl"][:min_len],
                    metrics_history["val_ppl"][:min_len],
                    metrics_history["steps"][:min_len],
                    os.path.join(plots_dir, f"perplexity_curves_{device_suffix}.png"),
                    title=f"Falcon3-1B Perplexity - {'TT-N150' if config.use_tt else 'CPU'}",
                )

        # Save metrics history
        metrics_path = os.path.join(
            config.project_dir,
            f"metrics_history_{'tt' if config.use_tt else 'cpu'}.json",
        )
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics_history, f, indent=2)
        logger.info(f"Saved metrics history to {metrics_path}")

        # Print fallback summary
        fallback_registry.print_summary()

        logger.info("Training completed successfully!")

    except Exception as e:
        traceback_str = traceback.format_exc()
        logger.error(f"Training failed: {str(e)}", traceback_str)
        raise

    finally:
        logger.finish()

    return metrics_history


def main():
    """Main entry point for training."""
    # Parse CLI arguments
    default_config = Path(__file__).parent / "configs" / "tt_n150.yaml"
    args = parse_cli_options(default_config=default_config)

    # Load configuration
    config: Falcon3TrainingConfig = generate_config(Falcon3TrainingConfig, args.config)

    # Setup reproducibility
    repro_manager = ReproducibilityManager(config)
    repro_manager.setup()

    # Setup logger
    logger = TrainingLogger(config)
    logger.info(f"Configuration loaded from: {args.config}")
    logger.info(f"Training type: {config.training_type}")
    logger.info(f"Use TT: {config.use_tt}")

    # Setup checkpoint manager
    checkpoint_manager = CheckpointManager(config, logger)

    # Setup device manager
    device_manager = Falcon3DeviceManager(config)
    logger.info(f"Using device: {device_manager.device}")

    # Start training
    metrics_history = train(config, device_manager, logger, checkpoint_manager)

    return metrics_history


if __name__ == "__main__":
    main()

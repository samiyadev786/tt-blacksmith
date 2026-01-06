# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Configuration for Falcon3-1B-Base LoRA Training.

This module defines the training configuration for LoRA fine-tuning
of the Falcon3-1B-Base model on Wikitext-2 dataset.
"""
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


class Falcon3TrainingConfig(BaseModel):
    """Configuration for Falcon3-1B LoRA training on TT-N150 and CPU."""

    # Dataset settings
    dataset_id: str = Field(default="wikitext-2")
    dataset_name: str = Field(default="wikitext-2-raw-v1")

    # Model settings
    model_name: str = Field(default="tiiuae/Falcon3-1B-Base")
    max_length: int = Field(default=256, gt=0)
    dtype: str = Field(default="torch.bfloat16")

    # Training hyperparameters
    training_type: str = Field(default="lora")
    learning_rate: float = Field(default=1e-4, gt=0)
    batch_size: int = Field(default=4, gt=0)
    gradient_accumulation_steps: int = Field(default=4, gt=0)
    gradient_checkpointing: bool = Field(default=False)
    weight_decay: float = Field(default=0.01, ge=0)
    num_epochs: int = Field(default=3, gt=0)
    max_steps: int = Field(default=-1)  # -1 means use num_epochs
    warmup_steps: int = Field(default=100, ge=0)
    optim: str = Field(default="adamw_torch")
    ignored_index: int = Field(default=-100)

    # LoRA configuration
    lora_r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, gt=0)
    lora_dropout: float = Field(default=0.05, ge=0, le=1)
    lora_target_modules: List[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    lora_task_type: str = Field(default="CAUSAL_LM")
    lora_bias: str = Field(default="none")

    # Logging settings
    log_level: str = Field(default="INFO")
    use_wandb: bool = Field(default=True)
    wandb_project: str = Field(default="falcon3-1b-lora-training")
    wandb_run_name: str = Field(default="falcon3-1b-lora")
    wandb_tags: List[str] = Field(default_factory=lambda: ["falcon3", "lora", "wikitext-2"])
    wandb_watch_mode: str = Field(default="all")
    wandb_log_freq: int = Field(default=100)
    model_to_wandb: bool = Field(default=False)
    steps_freq: int = Field(default=10)
    epoch_freq: int = Field(default=1)

    # Checkpoint settings
    resume_from_checkpoint: bool = Field(default=False)
    resume_option: str = Field(default="last")
    checkpoint_path: str = Field(default="")
    checkpoint_metric: str = Field(default="eval/loss")
    checkpoint_metric_mode: str = Field(default="min")
    keep_last_n: int = Field(default=3, ge=0)
    keep_best_n: int = Field(default=2, ge=0)
    save_strategy: str = Field(default="step")
    project_dir: str = Field(default="blacksmith/experiments/torch/BOUNTIES/falcon3-1b")
    save_optim: bool = Field(default=True)
    storage_backend: str = Field(default="local")
    sync_to_storage: bool = Field(default=False)
    load_from_storage: bool = Field(default=False)
    remote_path: str = Field(default="")

    # Reproducibility settings
    seed: int = Field(default=42)
    deterministic: bool = Field(default=True)

    # Device settings
    use_tt: bool = Field(default=True)
    mesh_shape: Optional[List[int]] = Field(default=None)
    mesh_axis_names: Optional[List[str]] = Field(default=None)
    model_sharding_patterns: Optional[List[Tuple[str, Tuple[Optional[str], ...]]]] = Field(default=None)

    # Fallback settings for TT-N150
    enable_fallback: bool = Field(default=True)
    fallback_operations: List[str] = Field(default_factory=list)
    log_fallback_operations: bool = Field(default=True)

    # Validation settings
    do_validation: bool = Field(default=True)
    eval_steps: int = Field(default=50)
    eval_batch_size: int = Field(default=4)

    # Other settings
    output_dir: str = Field(default="results/falcon3-1b")
    logging_steps: int = Field(default=10, gt=0)
    do_train: bool = Field(default=True)
    print_examples: bool = Field(default=True)
    framework: str = Field(default="pytorch")
    save_plots: bool = Field(default=True)
    plots_dir: str = Field(default="plots")

    # Perplexity tracking
    compute_perplexity: bool = Field(default=True)

# Falcon3-1B-Base LoRA Training on TT-N150

This directory contains the implementation for end-to-end LoRA fine-tuning of **tiiuae/Falcon3-1B-Base** on Tenstorrent N150 hardware with CPU baseline comparison.

## Overview

This bounty implementation provides:

- **LoRA Fine-tuning**: Parameter-efficient fine-tuning of Falcon3-1B-Base using Low-Rank Adaptation
- **Wikitext-2 Dataset**: Training on the standard language modeling benchmark
- **TT-N150 Support**: Single-chip training on Tenstorrent N150 hardware
- **CPU Baseline**: Identical training configuration for CPU comparison
- **Metric Parity**: Tools for comparing convergence between CPU and TT runs
- **Fallback Mechanism**: Targeted CPU fallbacks for operations not yet supported on TT

## Model Information

- **Model**: [tiiuae/Falcon3-1B-Base](https://huggingface.co/tiiuae/Falcon3-1B-Base)
- **Parameters**: ~1 billion
- **Architecture**: Decoder-only transformer with RoPE, SwiGLU activation
- **Context Length**: 8192 tokens (training uses 256 for efficiency)

## LoRA Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lora_r` | 16 | Rank of LoRA matrices |
| `lora_alpha` | 32 | Scaling factor |
| `lora_dropout` | 0.05 | Dropout for LoRA layers |
| `target_modules` | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | Modules with LoRA adapters |

With this configuration, only ~0.5% of parameters are trainable, significantly reducing memory requirements.

## Directory Structure

```
falcon3-1b/
├── __init__.py
├── README.md                    # This file
├── configs.py                   # Training configuration class
├── train_falcon3_lora.py        # Main training script
├── wikitext_dataset.py          # Wikitext-2 dataset implementation
├── utils.py                     # Utilities (fallback, loss, metrics, plotting)
├── compare_results.py           # Script for comparing CPU vs TT results
├── configs/
│   ├── tt_n150.yaml            # TT-N150 training configuration
│   └── cpu_baseline.yaml        # CPU baseline configuration
├── results/                     # Training outputs (created during training)
│   ├── checkpoints/
│   ├── plots/
│   └── metrics_history_*.json
└── examples/                    # Example outputs and configurations
```

## Setup Instructions

### Prerequisites

1. **Python 3.10+**
2. **PyTorch 2.1+** with CUDA support (for GPU baseline) or CPU
3. **Tenstorrent TT-XLA** (for TT-N150 training)
4. **HuggingFace Transformers** and **PEFT** libraries

### Installation

1. Clone the tt-blacksmith repository:
```bash
git clone https://github.com/tenstorrent/tt-blacksmith.git
cd tt-blacksmith
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install additional dependencies for this experiment:
```bash
pip install peft datasets transformers matplotlib wandb
```

4. For TT-N150 training, install TT-XLA:
```bash
# Follow Tenstorrent installation instructions
# https://github.com/tenstorrent/tt-xla
```

### Koyeb N150 Access

Training should be run on Koyeb-hosted N150 instances:

1. Request access at: https://www.koyeb.com/solutions/tenstorrent
2. Wait for the onboarding email from Koyeb
3. Follow the setup instructions provided

## Training

### TT-N150 Training

Run LoRA fine-tuning on TT-N150:

```bash
# Set debug logging to verify TT execution
export LOGGER_LEVEL=DEBUG

# Run training
cd tt-blacksmith
python blacksmith/experiments/torch/BOUNTIES/falcon3-1b/train_falcon3_lora.py \
    --config blacksmith/experiments/torch/BOUNTIES/falcon3-1b/configs/tt_n150.yaml
```

**Verification**: With `LOGGER_LEVEL=DEBUG`, look for printed TTIR graphs (e.g., `graph module @jit_training_step`) to confirm TT execution.

### CPU Baseline Training

Run the same training on CPU for comparison:

```bash
python blacksmith/experiments/torch/BOUNTIES/falcon3-1b/train_falcon3_lora.py \
    --config blacksmith/experiments/torch/BOUNTIES/falcon3-1b/configs/cpu_baseline.yaml
```

### Comparing Results

After running both trainings, compare the results:

```bash
python blacksmith/experiments/torch/BOUNTIES/falcon3-1b/compare_results.py \
    --cpu-metrics blacksmith/experiments/torch/BOUNTIES/falcon3-1b/results/metrics_history_cpu.json \
    --tt-metrics blacksmith/experiments/torch/BOUNTIES/falcon3-1b/results/metrics_history_tt.json \
    --output-dir blacksmith/experiments/torch/BOUNTIES/falcon3-1b/results/comparison
```

This generates:
- Loss comparison plots
- Perplexity comparison plots
- Metric parity statistics
- Comparison report (Markdown)

## Configuration

### Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | tiiuae/Falcon3-1B-Base | HuggingFace model ID |
| `dataset_id` | wikitext-2 | Dataset identifier |
| `max_length` | 256 | Maximum sequence length |
| `batch_size` | 4 | Training batch size |
| `learning_rate` | 1e-4 | Learning rate |
| `num_epochs` | 3 | Number of training epochs |
| `lora_r` | 16 | LoRA rank |
| `use_tt` | true/false | Enable TT device |
| `enable_fallback` | true | Enable CPU fallback |

### Modifying Configuration

Create a custom YAML config file:

```yaml
# my_config.yaml
model_name: "tiiuae/Falcon3-1B-Base"
max_length: 512
batch_size: 2
learning_rate: 5e-5
lora_r: 32
lora_alpha: 64
num_epochs: 5
use_tt: true
```

Run with custom config:
```bash
python train_falcon3_lora.py --config my_config.yaml
```

## Fallback Mechanism

When operations fail on TT-N150, the implementation provides targeted CPU fallbacks:

### How It Works

1. Operations are wrapped with the `@cpu_fallback` decorator
2. If execution fails on TT, only that specific operation runs on CPU
3. Results are moved back to TT device
4. Full training step does NOT fall back to CPU

### Example Fallback

```python
@cpu_fallback(
    operation_name="custom_attention",
    reason="Pattern not supported on TT-N150",
    issue_url="https://github.com/tenstorrent/tt-xla/issues/XXX"
)
def custom_attention(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)
```

### Tracking Fallbacks

At the end of training, a fallback summary is printed:

```
============================================================
CPU Fallback Operations Summary
============================================================
1. custom_cross_entropy
   Reason: Compilation issue on TT: [error details]
   Issue: https://github.com/tenstorrent/tt-xla/issues/1993
============================================================
```

### Filing Issues

When encountering fallback operations:

1. Create a minimal reproducer
2. File an issue at:
   - Compilation issues: https://github.com/tenstorrent/tt-xla or https://github.com/tenstorrent/tt-mlir
   - Runtime issues: https://github.com/tenstorrent/tt-metal
3. Reference the issue URL in the fallback decorator

## Dataset

### Wikitext-2

The [Wikitext-2](https://huggingface.co/datasets/wikitext) dataset contains text from verified Wikipedia articles:

- **Training**: ~2M tokens
- **Validation**: ~217K tokens
- **Test**: ~245K tokens

### Data Processing

1. Text is tokenized using Falcon3's tokenizer
2. Sequences are concatenated and chunked into fixed-length blocks
3. Labels are identical to inputs (causal LM objective)

## Outputs

### Checkpoints

Saved in `results/checkpoints/`:
- `best_model.pt` - Best validation loss
- `final_model.pt` - Final model after training
- `epoch_*.pt` - End-of-epoch checkpoints

### Metrics

Saved in `results/`:
- `metrics_history_tt.json` - TT-N150 metrics
- `metrics_history_cpu.json` - CPU metrics

### Plots

Saved in `results/plots/`:
- `loss_curves_tt_n150.png` - TT training curves
- `loss_curves_cpu.png` - CPU training curves
- `perplexity_curves_*.png` - Perplexity curves

## Expected Results

### Metric Parity

With identical seeds and configurations, CPU and TT-N150 runs should show:

| Metric | Expected Difference |
|--------|-------------------|
| Training Loss | < 5% relative |
| Validation Loss | < 5% relative |
| Perplexity | < 5% relative |
| Convergence | Similar curves |

### Sample Loss Curves

```
Training Progress (Example):
Step 100: Train Loss 3.45, Val Loss 3.52
Step 200: Train Loss 3.12, Val Loss 3.21
Step 300: Train Loss 2.89, Val Loss 2.95
...
Final:    Train Loss 2.15, Val Loss 2.28
```

## Troubleshooting

### TT Device Not Found

```bash
# Verify TT device is available
python -c "import torch_xla; print(torch_xla.device())"
```

### Out of Memory

- Reduce `batch_size` to 2 or 1
- Reduce `max_length` to 128
- Enable `gradient_checkpointing: true`

### Slow Training

- Ensure TT device is being used (check for TTIR graphs in debug output)
- Verify no excessive CPU fallbacks are occurring

### Metrics Don't Match

- Verify identical `seed` in both configs
- Check `deterministic: true` is set
- Review fallback operations summary

## Contributing

1. Fork tt-blacksmith repository
2. Create feature branch
3. Submit PR with:
   - Detailed description
   - Results and comparison plots
   - Any fallback operations documented

## References

- [Falcon3 Model Card](https://huggingface.co/tiiuae/Falcon3-1B-Base)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Library](https://github.com/huggingface/peft)
- [Wikitext-2 Dataset](https://huggingface.co/datasets/wikitext)
- [TT-XLA Repository](https://github.com/tenstorrent/tt-xla)
- [TT-Metal Repository](https://github.com/tenstorrent/tt-metal)

## License

SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC

SPDX-License-Identifier: Apache-2.0


# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Minimal Reproducer Template for TT-XLA/TT-Metal Issues.

When encountering compilation or runtime issues on TT-N150,
use this template to create a minimal reproducer for filing issues.

Instructions:
1. Copy this file and rename it (e.g., `issue_attention_pattern.py`)
2. Fill in the relevant sections
3. Test that the reproducer triggers the issue
4. File the issue at the appropriate repository:
   - Compilation issues: https://github.com/tenstorrent/tt-xla or https://github.com/tenstorrent/tt-mlir
   - Runtime issues: https://github.com/tenstorrent/tt-metal

Example Usage:
    # To reproduce the issue:
    python minimal_repro_template.py
"""
import os

import torch
import torch.nn as nn

# TT-XLA imports (if available)
try:
    import torch_xla
    import torch_xla.runtime as xr

    TORCH_XLA_AVAILABLE = True
except ImportError:
    TORCH_XLA_AVAILABLE = False


# =============================================================================
# ISSUE INFORMATION (Fill this section)
# =============================================================================

ISSUE_TITLE = "Your Issue Title Here"
ISSUE_DESCRIPTION = """
Brief description of the issue:
- What operation fails?
- What error message do you see?
- Is this a compilation or runtime error?
"""

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================


def setup_tt_device():
    """Setup TT device for testing."""
    if not TORCH_XLA_AVAILABLE:
        raise RuntimeError("torch_xla not available")

    xr.set_device_type("TT")
    os.environ["PJRT_DEVICE"] = "TT"
    os.environ["XLA_STABLEHLO_COMPILE"] = "1"

    return torch_xla.device()


# =============================================================================
# MINIMAL MODEL/OPERATION (Fill this section)
# =============================================================================


class MinimalModel(nn.Module):
    """
    Minimal model that reproduces the issue.

    Replace this with the simplest model that triggers the error.
    """

    def __init__(self):
        super().__init__()
        # Example: Simple linear layer
        self.linear = nn.Linear(256, 256)

    def forward(self, x):
        # Example: The operation that fails
        return self.linear(x)


def create_minimal_input():
    """
    Create minimal input that reproduces the issue.

    Use the smallest input shape that triggers the error.
    """
    batch_size = 1
    seq_len = 32
    hidden_dim = 256

    return torch.randn(batch_size, seq_len, hidden_dim)


# =============================================================================
# REPRODUCER
# =============================================================================


def reproduce_issue():
    """Main reproducer function."""
    print(f"Issue: {ISSUE_TITLE}")
    print("-" * 60)
    print(ISSUE_DESCRIPTION)
    print("-" * 60)

    # Setup device
    if TORCH_XLA_AVAILABLE:
        print("Setting up TT device...")
        device = setup_tt_device()
        print(f"Device: {device}")
    else:
        print("torch_xla not available, using CPU")
        device = torch.device("cpu")

    # Create model
    print("\nCreating model...")
    model = MinimalModel()
    model = model.to(torch.bfloat16)
    model = model.to(device)

    # Create input
    print("Creating input...")
    x = create_minimal_input()
    x = x.to(torch.bfloat16)
    x = x.to(device)

    # Run forward pass (this should trigger the issue)
    print("\nRunning forward pass...")
    try:
        output = model(x)

        # Sync to force execution
        if TORCH_XLA_AVAILABLE:
            torch_xla.sync(wait=True)

        print(f"Output shape: {output.shape}")
        print("\n✅ No error occurred!")

    except Exception as e:
        print(f"\n❌ Error occurred:")
        print(f"   Type: {type(e).__name__}")
        print(f"   Message: {str(e)}")

        # Print full traceback for debugging
        import traceback

        print("\nFull traceback:")
        traceback.print_exc()


# =============================================================================
# WORKAROUND (Optional)
# =============================================================================


def workaround():
    """
    Document any workaround that allows training to continue.

    This helps others while the issue is being investigated.
    """
    print("Workaround:")
    print("  1. Wrap the failing operation with @cpu_fallback decorator")
    print("  2. Or manually move tensors to CPU for this specific operation")
    print("")
    print("Example:")
    print("  @cpu_fallback('operation_name', 'Brief reason')")
    print("  def failing_operation(x):")
    print("      return some_tt_unsupported_op(x)")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    reproduce_issue()
    print("\n" + "=" * 60)
    workaround()

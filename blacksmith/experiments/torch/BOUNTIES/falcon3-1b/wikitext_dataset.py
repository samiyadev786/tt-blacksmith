# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wikitext-2 Dataset Implementation for Falcon3-1B LoRA Training.

This module provides a dataset wrapper for the Wikitext-2 dataset,
suitable for causal language model fine-tuning.
"""
import sys
from pathlib import Path
from typing import Optional, Callable, Dict, List

# Add parent directory to path for imports (handles hyphenated directory name)
_current_dir = Path(__file__).parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader, Dataset

from configs import Falcon3TrainingConfig


class WikitextDataset(Dataset):
    """
    Wikitext-2 dataset for causal language model training.
    
    Processes the Wikitext-2 dataset into tokenized chunks suitable
    for training Falcon3-1B with LoRA.
    """

    def __init__(
        self,
        config: Falcon3TrainingConfig,
        split: str = "train",
        collate_fn: Optional[Callable] = None,
    ):
        """
        Initialize the Wikitext dataset.

        Args:
            config: Training configuration
            split: Dataset split ("train", "validation", or "test")
            collate_fn: Optional custom collate function
        """
        self.config = config
        self.split = split
        self.collate_fn = collate_fn

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )

        # Set pad token if not available
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.required_columns = ["input_ids", "attention_mask", "labels"]
        self._prepare_dataset()

    def _prepare_dataset(self):
        """Load and prepare the Wikitext-2 dataset."""
        # Load the raw dataset
        raw_dataset = load_dataset("wikitext", self.config.dataset_name, split=self.split)

        # Filter out empty examples
        raw_dataset = raw_dataset.filter(
            lambda example: len(example["text"].strip()) > 0
        )

        # Tokenize and chunk the dataset
        tokenized_dataset = raw_dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=raw_dataset.column_names,
            desc=f"Tokenizing {self.split} split",
        )

        # Group texts into chunks of max_length
        self.dataset = tokenized_dataset.map(
            self._group_texts,
            batched=True,
            desc=f"Grouping texts into chunks of {self.config.max_length}",
        )

    def _tokenize_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Tokenize the text examples."""
        return self.tokenizer(
            examples["text"],
            truncation=False,
            padding=False,
            return_attention_mask=True,
        )

    def _group_texts(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        Group tokenized texts into chunks of max_length.
        
        This concatenates all texts and then splits them into fixed-length
        chunks for efficient training.
        """
        # Concatenate all input_ids and attention_masks
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}

        total_length = len(concatenated["input_ids"])
        block_size = self.config.max_length

        # Drop the remainder that doesn't fit into a full block
        total_length = (total_length // block_size) * block_size

        # Split by chunks of block_size
        result = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        for i in range(0, total_length, block_size):
            input_ids = concatenated["input_ids"][i : i + block_size]
            attention_mask = concatenated["attention_mask"][i : i + block_size]

            result["input_ids"].append(input_ids)
            result["attention_mask"].append(attention_mask)
            # Labels are same as input_ids for causal LM
            result["labels"].append(input_ids.copy())

        return result

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example from the dataset.

        Args:
            idx: Index of the example

        Returns:
            Dictionary containing input_ids, attention_mask, and labels
        """
        sample = self.dataset[idx]

        return {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(sample["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(sample["labels"], dtype=torch.long),
        }

    def get_dataloader(self) -> DataLoader:
        """
        Create and return a DataLoader for this dataset.

        Returns:
            DataLoader configured for training/evaluation
        """
        # Use DataCollatorForLanguageModeling for proper padding
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
            pad_to_multiple_of=8,  # For efficient tensor operations
        )

        if self.collate_fn is not None:
            # Wrap the data collator with custom collate function
            def combined_collate_fn(batch):
                # First apply the HuggingFace data collator
                collated = data_collator(batch)
                # Then apply the custom collate function
                return self.collate_fn(collated)

            collate_function = combined_collate_fn
        else:
            collate_function = data_collator

        batch_size = self.config.batch_size if self.split == "train" else self.config.eval_batch_size

        return DataLoader(
            self,  # Use self (WikitextDataset) which implements __getitem__ and __len__
            batch_size=batch_size,
            collate_fn=collate_function,
            shuffle=(self.split == "train"),
            drop_last=(self.split == "train"),
            num_workers=0,  # Set to 0 for compatibility with TT devices
            pin_memory=False,
        )


def get_wikitext_dataset(
    config: Falcon3TrainingConfig,
    split: str = "train",
    collate_fn: Optional[Callable] = None,
) -> WikitextDataset:
    """
    Factory function to create a WikitextDataset.

    Args:
        config: Training configuration
        split: Dataset split ("train", "validation", or "test")
        collate_fn: Optional custom collate function

    Returns:
        WikitextDataset instance
    """
    return WikitextDataset(config=config, split=split, collate_fn=collate_fn)


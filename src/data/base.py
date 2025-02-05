from typing import Optional
from pathlib import Path
import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm

LOGGER = logging

@dataclass
class BaseData(ABC):
    dataset: str
    tokenizer: str
    target_column: Optional[str]=None
    max_length: Optional[int]=1024
    batch_size: int=32
    num_proc: int=os.cpu_count()
    subset_samples: int=13_000_000

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer, trust_remote_code=True)
    
    def download_data(self, subset: Optional[str]=None) -> None:
        """Download the data from huggingface hub and split"""
        dataset = load_dataset(self.dataset, subset)
        return dataset

    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize the dataset and truncate to max"""
        remove_columns = set(dataset.column_names).intersection({self.target_column})
        tokenized_dataset = dataset.map(
            lambda examples: self.tokenizer(examples[self.target_column],
                                            truncation=self.max_length is not None,
                                            max_length=self.max_length,
                                            return_attention_mask=False),
            num_proc=self.num_proc,
            batched=True,
            batch_size=self.batch_size,
            remove_columns=remove_columns,
            desc="Tokenizing the dataset.",
        )
        return tokenized_dataset

    def subset(self,
               tokenized_dataset: Dataset,
               tokens: int=13_000_000,
               logger=LOGGER.info) -> Dataset:
        tokenized_dataset = tokenized_dataset.shuffle(48)
        total_tokens = 0
        idx = 0
        for item in tqdm(tokenized_dataset, desc=f"Keeping only {tokens:,} tokens."):
            if total_tokens >= tokens:
                logger(f"Total training tokens: {total_tokens:,}")
                return tokenized_dataset.select(list(range(idx)))
            total_tokens += len(item["input_ids"])
            idx += 1

    def sort_by_length(self, dataset: Dataset, logger=LOGGER.info) -> Dataset:
        lengths = dataset.map(lambda batch: {"length": [len(input_ids) for input_ids in batch]},
                                input_columns="input_ids",
                                batched=True,
                                batch_size=self.batch_size,
                                desc="Computing the lengths.")
        lengths = lengths.shard(1, 0, contiguous=True)
        logger("Sorting the data...")
        return lengths.sort("length", reverse=True)
    
    def split_train_test_dev(self, dataset: Dataset):
        train_remaining = dataset.train_test_split(0.03)
        train = train_remaining["train"]
        dev = train_remaining["test"]
        return train, dev
    
    @abstractmethod
    def __call__(self, output_folder: Path) -> None:
        raise NotImplementedError("You should implement a callable class.")
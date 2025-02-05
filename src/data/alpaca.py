from .base import BaseData
from .parse_args import parse_args
import logging
from dataclasses import dataclass
from datasets import Dataset

# Set up a root logger with WARNING level (this affects all loggers by default)
logging.basicConfig(
    level=logging.WARNING,  # This will affect all loggers that aren't explicitly set
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set up your specific logger with DEBUG level
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

@dataclass
class Alpaca(BaseData):
    def __call__(self) -> None:
        dataset: Dataset = self.download_data()
        dataset = dataset.remove_columns(['input', 'output', 'instruction'])
        dataset = self.tokenize_dataset(dataset=dataset)
        print(dataset)
        if self.subset_samples is not None:
            dataset = self.subset(dataset["train"], logger=LOGGER.info)
        return dataset

def main():
    args = parse_args()
    subset_samples = None if args.subset == 0 else args.subset
    dataloader = Alpaca(
        dataset=args.dataset,
        tokenizer=args.tokenizer,
        batch_size=args.batch_size,
        target_column=args.target_column,
        max_length=args.max_length,
        num_proc=args.num_proc,
        subset_samples=subset_samples
        )
    train = dataloader()
    train.save_to_disk(f"{args.output_folder}/train")

if __name__ == "__main__":
    main()
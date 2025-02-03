from .base import BaseData
from .parse_args import parse_args
from dataclasses import dataclass
from datasets import Dataset

@dataclass
class Alpaca(BaseData):
    def __call__(self) -> None:
        dataset: Dataset = self.download_data()
        dataset = dataset.remove_columns(['input', 'output', 'instruction'])
        dataset = self.tokenize_dataset(dataset=dataset)
        print(dataset)
        if self.subset_samples is not None:
            dataset = self.subset(dataset["train"])
        train, dev = self.split_train_test_dev(dataset)
        train = self.sort_by_length(train)
        dev = self.sort_by_length(dev)
        return train, dev

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
    train, dev = dataloader()
    train.save_to_disk(f"{args.output_folder}/train")
    dev.save_to_disk(f"{args.output_folder}/dev")

if __name__ == "__main__":
    main()
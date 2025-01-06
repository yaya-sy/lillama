from .base import BaseData
from .parse_args import parse_args
from pathlib import Path

class WikiText(BaseData):
    def __call__(self, output_folder: Path) -> None:
        dataset = self.download_data(subset="wikitext-2-raw-v1")
        dataset = self.tokenize_dataset(dataset=dataset,
                                        target_column=self.target_column)
        train, test, dev = dataset["train"], dataset["test"], dataset["validation"]
        train = self.sort_by_length(train)
        test = self.sort_by_length(test)
        dev = self.sort_by_length(dev)
        train.save_to_disk(f"{output_folder}/train")
        test.save_to_disk(f"{output_folder}/test")
        dev.save_to_disk(f"{output_folder}/dev")

def main():
    args = parse_args()
    dataloader = WikiText(
        dataset=args.dataset,
        tokenizer=args.tokenizer,
        batch_size=args.batch_size,
        target_column=args.target_column,
        max_length=args.max_length,
        num_proc=args.num_proc
        )
    dataloader(output_folder=args.output_folder)

if __name__ == "__main__":
    main()
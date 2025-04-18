from .base import BaseData
from .parse_args import parse_args
import logging
import os
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
class ShareGPT(BaseData):
    def merge_conversations(self, batch: list) -> None:
        conversations = []
        for dialog in batch[self.target_column]:
            text = "\n\n".join(turn["value"] for turn in dialog if turn["from"] in {"human", "gpt"})
            conversations.append(text)
        batch[self.target_column] = conversations
        return batch

    def create_conversation(self, sample):
        sample = sample["conversations"]
        chat = [{"role": "user" if data["from"] == "human" else "assistant", "content": data["value"]} for data in sample if data["from"] != "system"]
        if self.tokenizer.chat_template:
            return {"conversations": self.tokenizer.apply_chat_template(chat, tokenize=False)}
        return {"conversations": "\n\n".join(turn["content"] for turn in chat)}
    

    def merge_dialog(self,
                     dataset: Dataset,
                     batch_size: int=8,
                     num_process: int=os.cpu_count()
                     ) -> Dataset:
        remove_columns = list(dataset.features.keys() - {self.target_column})
        dataset = dataset.map(self.create_conversation,
                            batched=False,
                            batch_size=batch_size,
                            num_proc=num_process,
                            remove_columns=remove_columns,
                            desc="Merging the dialogs.")
        return dataset

    def __call__(self) -> None:
        dataset = self.download_data()
        dataset = self.merge_dialog(dataset=dataset["train"],
                                    batch_size=self.batch_size,
                                    num_process=self.num_proc)
        dataset = self.tokenize_dataset(dataset=dataset)
        dataset = self.subset(dataset, logger=LOGGER.info)
        train = self.sort_by_length(dataset, logger=LOGGER.info)
        return train

def main():
    args = parse_args()
    dataloader = ShareGPT(
        dataset=args.dataset,
        tokenizer=args.tokenizer,
        batch_size=args.batch_size,
        target_column=args.target_column,
        max_length=args.max_length,
        num_proc=args.num_proc
        )
    train = dataloader()
    train.save_to_disk(f"{args.output_folder}")
if __name__ == "__main__":
    main()
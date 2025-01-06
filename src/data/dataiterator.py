from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from copy import deepcopy

class Collator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)

        labels = [item["input_ids"] for item in batch]
        labels = pad_sequence(labels, batch_first=True, padding_value=self.pad_token_id)
        labels[labels == self.pad_token_id] = -100
        
        return {"input_ids": input_ids, "labels": labels}

class DataIterator(DataLoader):
    def __len__(self):
        return sum(len(item["input_ids"]) for item in self.dataset)
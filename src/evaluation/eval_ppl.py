"""we use the same ppl evaluation in the Wanda paper: https://github.com/locuslab/wanda/blob/main/lib/eval.py#L83"""
from tqdm import tqdm
import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data.dataset import Dataset
import argparse
from transformers import AutoModelForCausalLM

class IndexDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)

def process_data(samples, tokenizer, seq_len, field_name):
    test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
    test_ids_batch = []
    nsamples = test_ids.numel() // seq_len

    for i in range(nsamples):
        batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
        test_ids_batch.append(batch)
    test_ids_batch = torch.stack(test_ids_batch)
    return IndexDataset(tensors=test_ids_batch)

def get_loaders(tokenizer, seq_len=2048, batch_size=4):
    test_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    test_dataset = process_data(test_data, tokenizer, seq_len, 'text')

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

@torch.no_grad()
def ppl_eval(model, test_loader, device):
    nlls = []
    for batch in tqdm(test_loader):
        batch = batch.to(device)
        output = model(batch, use_cache=False, output_attentions=False)
        lm_logits = output.logits
    
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss)
    ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
    return ppl.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, required=True)
    parser.add_argument("--lowrank-weights", type=str, required=False, default=None)

    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.llm, torch_dtype=torch.bfloat16).cuda()
    model.to(args.device)

    test_loader = get_loaders(model, seq_len=1024, batch_size=4)
    ppl = ppl_eval(model, test_loader, args.device)
    print(f"PPL: {ppl}")


if __name__ == "__main__":
    pass
    # TODO
    
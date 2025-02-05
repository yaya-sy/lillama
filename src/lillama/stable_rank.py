import re
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import torch
from datasets import load_dataset
from torch.utils.data.dataset import Dataset
from pandas import DataFrame

class IndexDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)

# Loading the data and tokenizing it
def process_data(samples, tokenizer, seq_len, field_name):
    test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
    test_ids_batch = []
    nsamples = test_ids.numel() // seq_len

    for i in range(nsamples):
        batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
        test_ids_batch.append(batch)
    test_ids_batch = torch.stack(test_ids_batch)
    return IndexDataset(tensors=test_ids_batch)

def get_loaders(tokenizer, seq_len=128, batch_size = 4, max_samples=256):
    test_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
    test_data = test_data.shuffle(seed=42)
    test_data = test_data.select(range(max_samples)) # select a small subset just for testing
    test_dataset = process_data(test_data, tokenizer, seq_len, 'text')

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


class StableRankHook:
    """
    A callable class as a forward hook.\
    This will be used to compute and store\
    the statistics after each forward of a linear layer.
    """
    def __init__(self, name: str, batch_size: int) -> None:
        self.name = name # The name of the linear module
        self.total: float = 0 # number of total sequences
        self.batch_size = batch_size
        self.srank = 0.0

    def __call__(self, module, input, output) -> None:
        """Get singularvalues of output and then compute the stable rank"""
        output = output[0] if isinstance(output, tuple) else output # output=[batch_size, seq_len, embed_dim]
        output = output.view(-1, output.shape[-1])
        # get singular values of the activations
        singularvalues = torch.linalg.svdvals(output)
        # stable_rank = sum(singularvalues ** 2) / max(singularvalues) ** 2
        singularvalues = singularvalues ** 2
        srank = torch.sum(singularvalues, dim=-1) / torch.max(singularvalues, dim=-1).values
        srank = srank.mean()
        self.srank += srank
        self.total += 1

    def __hash__(self):
        return hash(self.name)

def register_forward_hooks(llm: PreTrainedModel, batch_size: int=32):
    """Register forward hooks for all transformer layers in the model."""
    hooks = set()
    handles = set()
    for name, module in llm.named_modules():
        if not re.search("layers\.\d+$", name):
            continue
        # if not isinstance(module, torch.nn.Linear) or "lm_head" in name:
            # continue
        hook = StableRankHook(name, batch_size=batch_size)
        handle = module.register_forward_hook(hook)
        handles.add(handle)
        hooks.add(hook)
    return hooks, handles

@torch.no_grad()
def forward(llm, data):
    for batch in tqdm(data):
        batch = batch.to("cuda")
        llm(input_ids=batch)

def weight_rank(llm):
    total = sum(1 for _ in llm.named_parameters())
    results = []
    for n, p in tqdm(llm.named_parameters(), total=total):
        if "layers" not in n or "weight" not in n or "norm" in n:
            continue
        # get singular values of the activations
        singularvalues = torch.linalg.svdvals(p)
        # stable_rank = sum(singularvalues ** 2) / max(singularvalues) ** 2
        singularvalues = singularvalues ** 2
        srank = (torch.sum(singularvalues, dim=-1) / torch.max(singularvalues, dim=-1).values)
        srank = srank.mean().item()
        layer = re.search("layers\.\d+", n).group().split(".")[-1]
        weight = n.split(".")[-2]
        results.append({
            "Module": weight,
            "Layer": int(layer),
            "StableRank": srank
        })
    return results

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model-path",
                        help="Path to the model to analyze",
                        required=True,
                        type=str)
    parser.add_argument("-o", "--output-path",
                        help="Path to save the results CSV",
                        required=True,
                        type=str)
    return parser.parse_args()

def main():
    args = parse_args()
    
    llm = AutoModelForCausalLM.from_pretrained(args.model_path)
    llm.cuda()
    print(llm)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                              trust_remote_code=True)
    test_loader = get_loaders(tokenizer=tokenizer)
    
    results = []
    hooks, _ = register_forward_hooks(llm)
    forward(llm, test_loader)
    
    for hook in sorted(hooks, key=lambda x: int(x.name.split(".")[-1])):
        layer = hook.name.split(".")[-1]
        srank = hook.srank / hook.total
        results.append({
            "Module": "TransformerLayer", 
            "Layer": int(layer),
            "StableRank": srank.item()
        })
    
    results.extend(weight_rank(llm))
    df = DataFrame(results)
    Path(args.output_path).parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(args.output_path, index=None)

if __name__ == "__main__":
    main()
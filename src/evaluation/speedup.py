from .eval_ppl import get_loaders, ppl_eval
from ..lillama.lowrank_llm import lowrank_llm
import json
import logging
from pathlib import Path
from timeit import default_timer as timer
import GPUtil
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model",
                        type=str)
    parser.add_argument("-d", "--distill-path",
                        type=str)
    parser.add_argument("-o", "--output-filename",
                        type=str,
                        default="20p")
    return parser.parse_args()

def load_lr_llm(distill_path, checkpoint, logger=logging.info):
    lr_llm = AutoModelForCausalLM.from_pretrained(checkpoint,
                                                  attn_implementation="flash_attention_2",
                                                  torch_dtype=torch.bfloat16,
                                                  trust_remote_code=True)
    path = Path(distill_path)
    with open(path.parent / "rank_config.json", "r") as config_file:
        config = json.load(config_file)
    lowrank_llm(llm=lr_llm, d_model=lr_llm.config.hidden_size, config=config)
    lr_modules = [path.stem for path in Path(distill_path).glob("*.pt")]
    for name, module in lr_llm.named_modules():
        if name in lr_modules:
            msg = module.load_state_dict(torch.load(f"{distill_path}/{name}.pt"))
            logger(f"{msg} for {name}")
    return None, lr_llm, config

def get_llm(args):
    if args.distill_path:
        logging.info(f"Loading distill {args.model} model from {args.distill_path}")
        _, llm, _ = load_lr_llm(args.distill_path, args.model)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"PyTorch device: {device}")
        return llm.to(device)
    logging.info(f"Loading the base model of {args.model}.")
    return AutoModelForCausalLM.from_pretrained(args.model,
                                                attn_implementation="flash_attention_2",
                                                torch_dtype=torch.bfloat16,
                                                trust_remote_code=True)

def mem_report(output_path):  
  GPUs = GPUtil.getGPUs()
  msg = 'GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Used Mem: {:.0f}MB | Utilization {:3.0f}%'
  for i, gpu in enumerate(GPUs):
    log = msg.format(i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryTotal - gpu.memoryFree, gpu.memoryUtil*100)
    with open(output_path, "w") as output_file:
       output_file.write(f"{log}\n")

@torch.no_grad()
def forward(llm: torch.nn.Module,
            test_loader: torch.utils.data.DataLoader,
            desc: str):  
    for batch in tqdm(test_loader, desc=desc):
        batch = batch.to(llm.device)
        llm(batch, use_cache=False, output_attentions=False)
      
def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    llm = get_llm(args)

    output_folder = (Path("Speed-Benchmarks") / Path(args.model).name)
    output_folder.mkdir(exist_ok=True, parents=True)
    output_path = output_folder / f"{args.output_filename}.txt"
    llm.to(device=torch.device("cuda"))
    mem_report(output_path)

    for seq_len in [512, 1024, 2048, 4096, 8192, 16_384, 32_768]:
        test_dataloader = get_loaders(tokenizer=tokenizer, seq_len=seq_len, batch_size=8)
        start = timer()
        forward(llm, test_loader=test_dataloader, desc=f"Running for sequence length {seq_len}")
        end = timer()
        elapsed = end - start
        total_tokens = sum(batch.shape[0] * batch.shape[1] for batch in test_dataloader)
        throughput = total_tokens / elapsed
        with open(output_path, "a") as output_file:
            output_file.write(f"Seq Len: {seq_len}\tElasped time: {elapsed}\tTotal tokens: {total_tokens}\tThroughput: {throughput}\n")
if __name__ == "__main__":
    main()
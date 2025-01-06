from ..lillama.lowrank_llm import lowrank_llm
from ..utils import load_llm
import logging
import json
from pathlib import Path
import torch

def load_lr_llm(distill_path, checkpoint, device=None, logger=logging.info):
    lr_llm = load_llm(checkpoint=checkpoint, device_map=device)
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

def save_lr_llm(distill_path, checkpoint, output_folder):
    _, lr_llm, lr_config = load_lr_llm(distill_path, checkpoint)
    lr_llm.generation_config.temperature = None
    lr_llm.generation_config.top_p = None
    lr_llm.config.architectures = [f"LowRank{lr_llm.config.architectures[0]}"]
    lr_llm.config.ranks = lr_config
    lr_llm.save_pretrained(output_folder)

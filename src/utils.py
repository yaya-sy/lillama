from typing import Optional
import re
import torch
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel, AutoTokenizer, AutoConfig
import gc
import inspect
import logging
from tqdm import tqdm

def load_llm(checkpoint: str,
             device_map: int=None,
             max_memory: dict=None,
             num_hidden_layers: Optional[int]=None,
             torch_dtype: torch.dtype=None,
             ) -> PreTrainedModel:
    """Load the pretrained model."""
    if num_hidden_layers is None:
        config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
        num_hidden_layers = config.num_hidden_layers
    return AutoModelForCausalLM.from_pretrained(checkpoint,
                                                num_hidden_layers=num_hidden_layers,
                                                # attn_implementation="flash_attention_2" if torch_dtype in {torch.bfloat16, torch.float16, None} else "sdpa",
                                                device_map=device_map,
                                                max_memory=max_memory,
                                                torch_dtype=torch.bfloat16 if torch_dtype is None else torch_dtype,
                                                trust_remote_code=True)

def load_tokenizer(checkepoint):
    return AutoTokenizer.from_pretrained(checkepoint, trust_remote_code=True)

def hasmodule(module: nn.Module, target_module: str):
    """Set a target module from in a given module."""
    submodules = target_module.split(".", 1)
    if len(submodules) == 1:
        try:
            getattr(module, submodules[0])
            return True
        except:
            return False
    else:
        hasmodule(getattr(module, submodules[0]), submodules[-1])

def setmodule(module: nn.Module, target_module: str, value: nn.Module):
    """Set a target module from in a given module."""
    submodules = target_module.split(".", 1)
    if len(submodules) == 1:
        if submodules[0].isdigit():
            module[int(submodules[0])] = value
        else:
            setattr(module, submodules[0], value)
    else:
        setmodule(getattr(module, submodules[0]), submodules[-1], value)

def getmodule(module: nn.Module, target_module: str):
    """Get a target module from a given module."""
    submodules = target_module.split(".", 1)
    if submodules[0].isdigit():
      next_module = module[int(submodules[0])]
    else:
      next_module = getattr(module, submodules[0])
    if len(submodules) == 1:
        return next_module
    return getmodule(next_module, submodules[-1])

def delmodule(module: nn.Module, target_module: str):
    """Set a target module from in a given module."""
    submodules = target_module.split(".", 1)
    if len(submodules) == 1:
        delattr(module, submodules[0])
    else:
        delmodule(getattr(module, submodules[0]), submodules[-1])

def linear_iterator(llm):
    for name, module in llm.named_modules():
        if not isinstance(module, nn.Linear) or "lm_head" in name:
            continue
        else:
            yield name, module

def get_layers(llm):
    for name, module in llm.named_modules():
        if re.search("layers\.\d+$", name):
            yield name, module

def freeze_llm(llm):
    """Freeze the LLM weights."""
    for w in llm.parameters():
        w.requires_grad = False

def unfreeze(module):
    """unfreeze the module weights."""
    for w in module.parameters():
        w.requires_grad = True

def crop_llm(llm, lr_llm):
    """Crop the lr_llm to llm"""
    llm_layers = llm.model.layers
    lr_llm_layers = lr_llm.model.layers
    named_modules = [name for name, _ in llm_layers.named_modules() \
                        if re.search("^\d+$", name)]
    for name in named_modules:
        if not hasattr(lr_llm_layers, name):
            # delete layers that are not in the student llm
            delattr(llm_layers, name)
    delattr(llm, "lm_head")
    delattr(lr_llm, "lm_head")

def save_distilled_llm(lr_llm, lr_llm_config, output_folder):
    lr_llm.generation_config.temperature = 1.0
    lr_llm.generation_config.top_p = 0.9
    lr_llm.config.ranks = lr_llm_config
    lr_llm.save_pretrained(output_folder)
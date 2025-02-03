from typing import Optional
from math import sqrt
from tqdm import tqdm
import torch
from torch import nn
import torch

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


class LowRankLinear(nn.Module):
    """Low Rank linear of a base Full Rank linear."""
    def __init__(self,
                 in_features: int,
                 rank: int,
                 out_features: int,
                 d_model: Optional[int]=None,
                 init_method: str="svd",
                 W: torch.Tensor=None,
                 device: torch.device=torch.device("cpu"),
                 dtype=torch.bfloat16
                 ):
        super().__init__()
        self.d_model = d_model
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.linear = torch.nn.Sequential(
            nn.Linear(in_features=in_features,
                      out_features=rank,
                      bias=False,
                      dtype=dtype),
            nn.Linear(in_features=rank,
                      out_features=out_features,
                      bias=False,
                      dtype=dtype)
                      ).to(device=device)
        self.dtype = dtype
        if W is not None and init_method == "svd":
            self._init_svd(W, rank)
        elif init_method == "random":
            assert self.d_model is not None, "You have to pass the dimension of the model (d_model) for random init."
            self.linear.apply(self._init_random)
            
    @torch.no_grad()
    def _init_random(self, module):
        """Init the weights randomly."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=sqrt(2 / (2 * self.d_model)))

    @torch.no_grad()
    def _init_svd(self, w: torch.Tensor, r: int):
        """Init weight with svd."""
        u, s, v = torch.linalg.svd(w.to(dtype=torch.float32), full_matrices=False)
        w1 = u @ torch.sqrt(torch.diag(s)[:, :r]) # u[:, :r]
        w2 = torch.sqrt(torch.diag(s)[:r, :]) @ v
        if self.dtype is not None:
            w1 = w1.to(self.dtype)
            w2 = w2.to(self.dtype)
        self.linear[0].weight = torch.nn.Parameter(w2.contiguous())
        self.linear[1].weight = torch.nn.Parameter(w1.contiguous())

def lowrank_llm_random(llm: nn.Module, d_model, config: dict, dtype=torch.bfloat16):
    """
    Low-Rank a given LLM.

    Parameters
    ----------
    - llm: nn.Module
        The LLM to lowrank.
    - config: dict
        A dictionary containing the rank value of each linear module in the LLM.
    """
    total = sum(1 for _ in llm.named_modules())
    for name, module in tqdm(llm.named_modules(), total=total):
        if name in config:
            rank = config[name]
            # because in the case of svd_init, cpu weights will be replace by low-rank cuda weights
            device = module.weight.device
            lowrank_linear = LowRankLinear(in_features=module.in_features,
                                           rank=rank,
                                           out_features=module.out_features,
                                           d_model=d_model,
                                           device=device,
                                           init_method="random",
                                           dtype=dtype)
            setmodule(llm, name, lowrank_linear.linear)


def lowrank_llm_svd(llm: nn.Module, config: dict, dtype=torch.bfloat16):
    """
    Low-Rank a given LLM.

    Parameters
    ----------
    - llm: nn.Module
        The LLM to lowrank.
    - config: dict
        A dictionary containing the rank value of each linear module in the LLM.
    """
    total = sum(1 for _ in llm.named_modules())
    for name, module in tqdm(llm.named_modules(), total=total):
        if name in config:
            rank = config[name]
            # because in the case of svd_init, cpu weights will be replace by low-rank cuda weights
            device = torch.device("cpu")
            lowrank_linear = LowRankLinear(in_features=module.in_features,
                                           rank=rank,
                                           out_features=module.out_features,
                                           init_method="svd",
                                           W=module.weight,
                                           device=device,
                                           dtype=dtype)
            setmodule(llm, name, lowrank_linear.linear)

def lowrank_llm(llm: nn.Module, d_model: int, config: dict, init_method: str="random", dtype=torch.bfloat16):
    if init_method == "random":
        lowrank_llm_random(llm, d_model, config)
    elif init_method == "svd":
        lowrank_llm_svd(llm, config, dtype)
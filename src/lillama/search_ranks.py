from typing import List
from ..utils import linear_iterator
import math
from collections import defaultdict
import re
import json
import torch
from torch import nn

class RankSearcher:
    def __init__(self,
                 llm: nn.Module,
                 min_rank: int,
                 target_weights: List[str],
                 ignore_first_layers: int=2,
                 strategy: str="bottom",
                 reduction: float=20.0):
        self.target_weights = target_weights
        self.ignore_first_layers = ignore_first_layers
        self.strategy = strategy
        self.remaining_params = self.remain_params(llm)
        self.config = self.get_initial_config(llm)
        if strategy in {"top", "bottom"}:
            self.ranks = self.sorted_ranks(llm, min_rank)
            base_num_params = sum(int(p.nelement()) for p in llm.parameters())
            params = base_num_params - (base_num_params * reduction / 100)
            self.search_ranks(max_params=params)
        elif strategy == "uniform":
            self.uniform_ranks(llm, reduction=reduction)
    
    def save_config(self, output_folder):
        with open(output_folder / f"rank_config.json", "w") as output_file:
            json.dump(self.config, output_file)

    def params_of_config(self, config) -> int:
        """Compute the number of parameters of the config of the compressed llm."""
        return sum(config[name]["params"] for name in config)

    def weight_params(self, weight: torch.Tensor) -> int:
        """The number of parameters in the weight."""
        return weight.shape[0] * weight.shape[-1] if len(weight.shape) > 1 else weight.shape[0]

    def rank_params(self, rank, shape):
        """The number of parameters when setting the given rank."""
        return rank * (shape[0] + shape[1])

    def possible_ranks(self, min_rank=1024, max_rank=4096):
        """Get possible ranks"""
        for rank in range(min_rank, max_rank, 256):
            if rank < min_rank:
                continue
            if rank >= max_rank:
                break
            yield rank

    def get_compressable_modules(self, llm):
        compressable_modules = set()
        for name, _ in linear_iterator(llm):
            compressable_modules.add(name.split(".")[-1])
        return compressable_modules

    def remain_params(self, llm):
        compressable_modules = self.get_compressable_modules(llm)
        remain_params = 0
        for np, p in llm.named_parameters():
            m = np.split(".")[-2]
            if m not in compressable_modules:
                remain_params += self.weight_params(p)
        return remain_params

    def sorted_ranks(self, llm, min_rank):
        ranks = []
        for name, module in linear_iterator(llm):
            target_module = name.split(".")[-1]
            if target_module not in self.target_weights:
                continue
            layer = re.search("layers\.\d+", name).group(0).split(".")[-1]
            if int(layer) < self.ignore_first_layers:
                print(name)
                continue
            params = self.weight_params(module.weight)
            shape = module.weight.shape
            for rank in self.possible_ranks(min_rank, min(shape)):
                rank_params = self.rank_params(rank, shape)
                if rank_params >= params:
                    # interested only in ranks that reduce the num of parameters
                    continue
                ranks.append({"name": name, "layer": int(layer), "rank": rank, "params": rank_params})
        if self.strategy == "bottom":
            key = lambda x: (x["layer"], -x["rank"]) 
        elif self.strategy == "top":
            key = lambda x: (-x["layer"], -x["rank"])
        return sorted(ranks, key=key)

    def get_initial_config(self, llm):
        config = defaultdict(dict)
        for name, module in linear_iterator(llm):
            config[name]["rank"] = min(module.in_features, module.out_features)
            config[name]["params"] = self.weight_params(module.weight)
            config[name]["low_rank"] = False
        return config

    def get_num_layers(self):
        layers = set()
        for name in self.config:
            layer = re.search("layers\.\d+", name).group(0).split(".")[-1]
            layers.add(int(layer))
        return max(layers) + 1, layers

    def round_to_nearest(self, n, m):
        return m * math.ceil(n / m)

    def get_min_rank(self, shape):
        num_params = shape[0] * shape[-1]
        return max(rank for rank in range(max(shape) + 1) if (rank * sum(shape)) < num_params)

    def uniform_ranks(self, llm, reduction=20.0):
        for name, module in linear_iterator(llm):
            target_module = name.split(".")[-1]
            if target_module not in self.target_weights:
                del self.config[name]
                continue
            r = min(module.weight.shape)
            num_params = module.weight.shape[0] * module.weight.shape[-1]
            min_rank = self.get_min_rank(module.weight.shape)
            new_rank = self.round_to_nearest(min_rank - (min_rank * reduction / 100), 8) # 64 is good but tested with 2
            new_params = new_rank * sum(module.weight.shape)
            assert new_params < num_params, f"The size of {name} is not reduced when using rank {new_rank}"
            print(f"Rank from {r} to {new_rank} for {name}")
            self.config[name] = new_rank

    def search_ranks(self, max_params):
        total_params = self.params_of_config(self.config) + self.remaining_params
        while total_params > max_params:
            next_rank = self.ranks.pop(0)
            name = next_rank["name"]
            if next_rank["params"] > self.config[name]["params"]:
                continue
            self.config[name]["rank"] = int(next_rank["rank"])
            self.config[name]["params"] = next_rank["params"]
            self.config[name]["low_rank"] = True
            total_params = self.params_of_config(self.config) + self.remaining_params
        self.config = {name: value["rank"] for name, value in self.config.items()\
                       if value["low_rank"]}

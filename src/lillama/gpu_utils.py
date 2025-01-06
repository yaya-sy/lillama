# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import time

import numpy as np
import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from tqdm import tqdm

from . import utils
from .config import config


def distribute_model(model) -> None:
    """Distribute the model across available GPUs."""
    # TODO: dipatch base llm layers first and map each layer device to the corresponding student layer
    max_memory = get_balanced_memory(
        model,
        no_split_module_classes=model.no_split_module_classes,
    )

    device_map = infer_auto_device_map(
        model, max_memory=max_memory, no_split_module_classes=model.no_split_module_classes
    )

    dispatch_model(
        model,
        device_map=device_map,
        offload_buffers=True,
        offload_dir="offload",
        state_dict=model.state_dict(),
    )

    # Run GC and cleanup GPU memory
    utils.cleanup_memory()


def benchmark(model_adapter, input_batch: torch.Tensor) -> dict:
    """Benchmark the model's latency and throughput on the given input batch."""
    model_adapter.use_cache = True

    cache = {"past": None}

    def clear_past_cache(layer_idx):
        def tmp(*_):
            if cache["past"]:
                cache["past"][layer_idx] = None

        return tmp

    layers = model_adapter.get_layers()
    for idx, layer_adapter in enumerate(layers):
        # Clear past cache after each layer gets called to get accurate timing of each forward pass.
        layer_adapter.layer.register_forward_hook(clear_past_cache(idx))

    with torch.no_grad():
        input_ids = input_batch["input_ids"]
        batch_size, input_seq_len = input_ids.shape[:2]
        attention_mask = input_batch["attention_mask"]
        time_measurements = []

        for i in tqdm(range(input_seq_len), desc="Benchmarking"):
            input_ids_i = input_ids[:, i].reshape((batch_size, 1)).to(config.device)
            attention_mask_i = attention_mask[:, : (i + 1)].to(config.device)

            sync_gpus()
            start_time = time.time()
            output = model_adapter.model(input_ids_i, past_key_values=cache["past"], attention_mask=attention_mask_i)
            sync_gpus()
            time_measurements.append(time.time() - start_time)

            cache["past"] = list(output.past_key_values)
            del output

            input_ids_i, attention_mask_i = input_ids_i.to("cpu"), attention_mask_i.to("cpu")

        median_time = np.median(time_measurements)
        throughput = batch_size / median_time

        results = {"median_time": median_time, "latency": 1 / throughput, "throughput": throughput}
        return results
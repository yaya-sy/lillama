# source: https://github.com/microsoft/TransformerCompression/blob/main/experiments/run_lm_eval.py
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse, os
import json
import logging
import pathlib
import datetime

import lm_eval
import torch
from lm_eval import tasks
from lm_eval import utils as lm_eval_utils
from lm_eval.api.registry import ALL_TASKS
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import initialize_tasks

from .load_lr_llm import load_lr_llm
from ..utils import load_llm, load_tokenizer

tiktoken_cache_dir = "/gpfsscratch/rech/knb/urc37ho/tiktoken/"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

TASK_METRIC_MAP = {
    "arc_challenge": "acc_norm,none",
    "arc_easy": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "piqa": "acc_norm,none",
    "winogrande": "acc,none",
}

def create_file_handler(log_dir: str):
    path = pathlib.Path.cwd() / log_dir / f'{datetime.datetime.now():log_%Y-%m-%d-%H-%M-%S}.log'
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)04d\t%(levelname)s\t%(name)s\t%(message)s', datefmt='%Y-%m-%dT%H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    return file_handler


def eval_arg_parser(interactive: bool = True) -> argparse.Namespace:
    initialize_tasks()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="Model to load",
    )
    parser.add_argument(
        "--distill_path",
        type=str,
        help="Path to load the model to fine-tune (sliced) and tokenizer from",
        default=None,
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluating with lm eval harness.")
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande"],
        choices=lm_eval_utils.MultiChoice(tasks.ALL_TASKS),
    )
    parser.add_argument('--num-fewshot', type=int, default=0, help="Number of fewshots for all tasks.")
    parser.add_argument("--save-dir", type=str, default=".", help="Path to save the lm eval results")
    return parser.parse_args() if interactive else parser.parse_args('')


def process_eval_args(args: argparse.Namespace):
    logging.info(f'Parsed arguments:')
    for arg, argv in vars(args).items():
        logging.info(f'{arg} = {argv}')


def calculate_avg_accuracy(task_names: str, results: dict) -> float:
    n_tasks = len(task_names)
    acc_cumul = sum(result.get(TASK_METRIC_MAP[task]) for task, result in results.items() if 'mmlu' not in task)

    questions_per_mmlu_task = {
        task_name: lm_eval.tasks.get_task_dict([task_name])[task_name].dataset["test"].num_rows
        for task_name in task_names
        if 'mmlu' in task_name
    }

    if not questions_per_mmlu_task:
        return acc_cumul / n_tasks

    # Calculate average accuracy for mmlu tasks, weighted by number of questions in each task
    acc_mmlu = sum(
        result.get(TASK_METRIC_MAP[task]) * questions_per_mmlu_task[task]
        for task, result in results.items()
        if 'mmlu' in task
    )
    acc_mmlu_avg = acc_mmlu / sum(questions_per_mmlu_task.values())

    return (acc_cumul + acc_mmlu_avg) / (n_tasks - len(questions_per_mmlu_task) + 1)


def eval_main(llm,
              tokenizer,
              save_dir,
              batch_size=32,
              tasks=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande"],
              num_few_shot=0) -> None:
    initialize_tasks()
    logging.info("Running SliceGPT LM eval experiment.")
    logging.info(f"Number of available cuda devices: {torch.cuda.device_count()}")

    # the lm eval harness ties the weights, but this should not be done for sliced models unless the lm_head was sliced
    llm.tie_weights = lambda: None

    ### LM Eval Harness ###
    hflm = HFLM(pretrained=llm, tokenizer=tokenizer, batch_size=batch_size)

    if tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = lm_eval_utils.pattern_match(tasks, ALL_TASKS)

    logging.info(f"Selected Tasks: {task_names}")

    for task in task_names:
        if task not in TASK_METRIC_MAP:
            raise NotImplementedError(
                f"Please specify the metric to use for {task} in TASK_METRIC_MAP. Available info {TASK_METRIC_MAP}"
            )

    results = lm_eval.simple_evaluate(hflm, tasks=task_names, num_fewshot=num_few_shot, batch_size=batch_size)[
        'results'
    ]
    num_params = sum(int(p.nelement()) for p in llm.parameters())
    logging.info(f"Number of parameters in the compressed LLM distill {num_params:,}")
    logging.info(results)

    with open(f"{save_dir}/full_results_{0}_shot.json", "w") as f:
        json.dump(results, f)

    metric_vals = {task: round(result.get(TASK_METRIC_MAP[task]), 4) for task, result in results.items()}
    acc_avg = calculate_avg_accuracy(task_names, results)
    metric_vals['average'] = round(acc_avg, 4)
    pathlib.Path(save_dir).mkdir(exist_ok=True, parents=True)
    with open(f"{save_dir}/{num_few_shot}_shot_task_results.json", "w") as f:
        json.dump(metric_vals, f)

    logging.info(json.dumps(metric_vals, indent=4))
    logging.info(f"Average accuracy across tasks: {acc_avg}")

def get_llm(args):
    if args.distill_path:
        logging.info(f"Loading distill {args.model} model from {args.distill_path}")
        _, llm, _ = load_lr_llm(args.distill_path, args.model)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"PyTorch device: {device}")
        return llm.to(device)
    logging.info(f"Loading the base model of {args.model}.")
    return load_llm(args.model, device_map="auto")


if __name__ == "__main__":
    # Use the logger from lm_eval, adding a file handler to write the log to file
    logger = lm_eval_utils.eval_logger
    logger.addHandler(create_file_handler(log_dir="log"))

    args = eval_arg_parser()
    process_eval_args(args)
    llm = get_llm(args)
    tokenizer = load_tokenizer(args.model)
    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    eval_main(llm=llm,
              tokenizer=tokenizer,
              batch_size=args.batch_size,
              tasks=args.tasks,
              num_few_shot=args.num_fewshot,
              save_dir=args.save_dir)
# tasks.get_task_dict(["boolq", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa", "piqa"])
# from lm_eval import tasks, evaluator
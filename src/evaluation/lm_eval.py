from ..utils import load_lr_llm, load_llm
from pathlib import Path
import json
from argparse import ArgumentParser
from transformers import AutoTokenizer
import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
import logging

# Set up a root logger with WARNING level (this affects all loggers by default)
logging.basicConfig(
    level=logging.WARNING,  # This will affect all loggers that aren't explicitly set
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set up your specific logger with DEBUG level
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

TASK_METRIC_MAP = {
    "arc_challenge": "acc_norm,none",
    "arc_easy": "acc_norm,none",
    "piqa": "acc_norm,none",
    "social_iqa": "acc,none",
    "logiqa": "acc_norm,none",
    "truthfulqa_mc2": "acc,none",
    "winogrande": "acc,none",
    "boolq": "acc,none",
    "openbookqa": "acc_norm,none",
}

def get_llm(checkpoint, distill_path=None):
    if distill_path is not None:
        _, llm, _ = load_lr_llm(distill_path, checkpoint)
        llm = llm.cuda()
    else:
        llm = load_llm(checkpoint, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    num_params = sum(int(p.nelement()) for p in llm.parameters())
    LOGGER.info(f"Number of parameters in the LLM {num_params:,}")
    return llm, tokenizer

def parse_args():
    parser = ArgumentParser(description="Evaluation of compressed or base LLMs.")
    parser.add_argument("-l", "--llm",
                        type=str,
                        required=True,
                        help="The base uncompressed LLM.")
    parser.add_argument("-d", "--distill-path",
                        type=str,
                        required=False,
                        default=None,
                        help="Where the compressed layers are located.")
    parser.add_argument("-o", "--output-folder",
                        type=str,
                        required=True,
                        help="Where to save the results.")
    
    return parser.parse_args()

def evaluate(llm, tokenizer, tasks=list(TASK_METRIC_MAP.keys())):
    lm_obj = HFLM(pretrained=llm, tokenizer=tokenizer, trust_remote_code=True)

    task_manager = lm_eval.tasks.TaskManager()

    n_tasks = len(tasks)

    results = lm_eval.simple_evaluate( # call simple_evaluate
        model=lm_obj,
        tasks= tasks,
        task_manager=task_manager)

    results = results["results"]

    tasks_score = {task: round(result.get(TASK_METRIC_MAP[task]), 4) for task, result in results.items() if task in tasks}

    # average
    tasks_score["average"] = sum(result.get(TASK_METRIC_MAP[task]) for task, result in results.items() if task in tasks) / n_tasks

    LOGGER.info(tasks_score)

    return results, tasks_score


def main():
    args = parse_args()

    llm, tokenizer = get_llm(checkpoint=args.llm,
                             distill_path=args.distill_path)
    results, tasks_score = evaluate(llm=llm, tokenizer=tokenizer)

    LOGGER.info(f"Saving results to {args.output_folder}.")
    Path(args.output_folder).mkdir(exist_ok=True, parents=True)

    with open(f"{args.output_folder}/full_results_0_shot.json", "w") as f:
        json.dump(results, f)
    with open(f"{args.output_folder}/0_shot_task_results.json", "w") as f:
        json.dump(tasks_score, f)

if __name__ == "__main__":
    main()
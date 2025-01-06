from .load_lr_llm import load_lr_llm
from ..utils import load_llm
from pathlib import Path
import json
from argparse import ArgumentParser
from transformers import AutoTokenizer
import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
import logging

TASK_METRIC_MAP = {
    "arc_challenge": "acc_norm,none",
    "arc_easy": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "piqa": "acc_norm,none",
    "social_iqa": "acc,none",
    "logiqa": "acc_norm,none",
    "truthfulqa_mc2": "acc,none",
    "winogrande": "acc,none",
    "boolq": "acc,none",
    "openbookqa": "acc_norm,none",
    "mmlu": "acc,none"
}

# TASK_METRIC_MAP = {
#     "mmlu": "acc,none"
# }

# "mmlu": "acc_norm,none", # gms8k too long because it's a generate_unitl task

def get_llm(checkpoint, distill_path=None):
    if distill_path is not None:
        _, llm, _ = load_lr_llm(distill_path, checkpoint)
        llm = llm.cuda()
    else:
        llm = load_llm(checkpoint, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    num_params = sum(int(p.nelement()) for p in llm.parameters())
    logging.info(f"Number of parameters in the LLM {num_params:,}")
    return llm, tokenizer

def parse_args():
    parser = ArgumentParser(description="Evaluation of compressed or base LLMs.")
    parser.add_argument("-c", "--checkpoint",
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
        tasks= tasks, # ["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "boolq", "openbookqa"],
        num_fewshot=0,
        task_manager=task_manager)
    
    logging.info(f"Evaluation finished. Results:")
    print(make_table(result_dict=results))

    results = results["results"]

    tasks_score = {task: round(result.get(TASK_METRIC_MAP[task]), 4) for task, result in results.items() if task in tasks}

    # average
    tasks_score["average"] = sum(result.get(TASK_METRIC_MAP[task]) for task, result in results.items() if task in tasks) / n_tasks

    logging.info(tasks_score)

    return results, tasks_score


def main():
    args = parse_args()

    llm, tokenizer = get_llm(checkpoint=args.checkpoint,
                             distill_path=args.distill_path)
    results, tasks_score = evaluate(llm=llm, tokenizer=tokenizer)

    logging.info(f"Saving results to {args.output_folder}.")
    Path(args.output_folder).mkdir(exist_ok=True, parents=True)

    with open(f"{args.output_folder}/full_results_0_shot.json", "w") as f:
        json.dump(results, f)
    with open(f"{args.output_folder}/0_shot_task_results.json", "w") as f:
        json.dump(tasks_score, f)

if __name__ == "__main__":
    main()


# phi-1.7b: 56.54
# smolm-1.7b: 61.47
# stablelm-2-1.6: 61.42
# Qwen-1.8b: 
# gemma-2.5b: 
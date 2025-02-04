from .lowrank_llm import lowrank_llm
from .distiller import prepare_for_distillation, unset_distillers, set_distilled_layers_to_llm, Distiller
from .search_ranks import RankSearcher
from ..evaluation.lm_eval import evaluate
from ..utils import load_llm, freeze_llm, crop_llm, save_distilled_llm, linear_iterator
from .train import train
from .config import DistillationParams
from ..data.dataiterator import DataIterator, Collator
import json
import logging
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_from_disk

import os

tiktoken_cache_dir = "/gpfsscratch/rech/knb/urc37ho/tiktoken/"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

# validate
# assert os.path.exists(os.path.join(tiktoken_cache_dir,"9b5ad71b2ce5302211f9c61530b329a4922fc6a4"))

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-t", "--train-data",
                        help="Training dataset.",
                        default=None,
                        required=True,
                        type=str)
    parser.add_argument("-l", "--llm",
                        help="The LLM to distill.",
                        default="None",
                        required=True,
                        type=str)
    parser.add_argument("-b", "--batch-size",
                        help="The batch size for training and evaluation.",
                        default=8,
                        type=int)
    parser.add_argument("-r", "--lr",
                        help="The learning rate for the training.",
                        default=0.000086,
                        type=int)
    parser.add_argument("-g", "--log-interval",
                        help="The interval of updates for logging the progress.",
                        default=32,
                        type=int)
    parser.add_argument("--evaluate",
                        help="Whether evaluate on benchmarks at the end of distillation or not.",
                        action=BooleanOptionalAction,
                        default=True)
    parser.add_argument("-s", "--strategy",
                        help="The strategy of the layers compression: top, bottom or uniform.",
                        default='bottom',
                        const='bottom',
                        nargs='?',
                        choices=['bottom', 'top', 'uniform'])
    parser.add_argument("-d", "--distillation-loss",
                        help="The mode of distillation: whether distill using the student activations only,\
                            the teacher actiavations only or both.",
                        choices=["teacher", "student", "teacher+student"],
                        default="teacher+student",
                        type=str)
    parser.add_argument("-m", "--target-weights",
                        help="Only compress these weights.",
                        nargs="+",
                        default="all-linear"
                        )
    parser.add_argument("-k", "--min-rank",
                        help="The minimum rank for possible ranks of the weights of the matrices in the LLM.",
                        default=1024,
                        type=int)
    parser.add_argument("-p", "--reduction",
                        help="The compression ratio of the LLM.",
                        default=20.0,
                        type=float)
    parser.add_argument("-i", "--ignore-first-layers",
                        help="The n first layers to ignore.",
                        default=0,
                        type=int)
    parser.add_argument("-w", "--init-method",
                        help="The strategy of the layers compression: top, bottom or uniform.",
                        default='svd',
                        const='svd',
                        nargs='?',
                        choices=['svd', 'random'])
    parser.add_argument("-o", "--output-folder",
                        help="The output folder where the low-rank modules will be saved.",
                        required=True,
                        type=str)
    return parser.parse_args()

def load_best_model(lr_llm):
    for _, module in lr_llm.named_modules():
        if isinstance(module, Distiller):
            module.load_best_student()

def get_distillation_params(args) -> DistillationParams:
    return DistillationParams(
        batch_size=args.batch_size,
        lr=args.lr,
        evaluate=args.evaluate,
        log_interval=args.log_interval,
        distillation_loss=args.distillation_loss,
        min_rank=args.min_rank,
        output_folder=args.output_folder
        )

def main():
    args = parse_args()
    distill_params = get_distillation_params(args)
    
    # setup checkpoint directory
    checkpoints_dir = Path(args.output_folder) / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True, parents=True)

    # prepare data
    tokenizer = AutoTokenizer.from_pretrained(args.llm, trust_remote_code=True)
    pad_token_id = tokenizer.eos_token_id
    train_data = load_from_disk(args.train_data)
    train_data.set_format(type="torch", columns=["input_ids"])
    train_dataloader = DataIterator(train_data,
                                    batch_size=distill_params.batch_size,
                                    collate_fn=Collator(pad_token_id=pad_token_id))
    
    # load base model
    llm = load_llm(checkpoint=args.llm, device_map="cpu")
    print(llm)
    if args.target_weights == "all-linear" or args.target_weights is None:
        args.target_weights = list(set([name.split(".")[-1] for name, _ in linear_iterator(llm)]))
        LOGGER.info(f"Target weights: {args.target_weights}")
    base_num_params = sum(int(p.nelement()) for p in llm.parameters())
    LOGGER.info(f'Number of parameters of the Base LLM: {base_num_params:,}')
    freeze_llm(llm)
    short_llm = None

    # search ranks
    rank_searcher = RankSearcher(llm=llm,
                                 min_rank=args.min_rank,
                                 target_weights=args.target_weights,
                                 ignore_first_layers=args.ignore_first_layers,
                                 strategy=args.strategy,
                                 reduction=args.reduction)
    num_layers, layers = rank_searcher.get_num_layers()
    rank_searcher.save_config(Path(args.output_folder))

    # load smaller LLMs
    # for the student layer we only need to load the first layers 
    # and the 'prepare_for_distillation' will share the rest non distillable layers with the teacher
    lr_llm = load_llm(checkpoint=args.llm, device_map="cpu", num_hidden_layers=num_layers)
    freeze_llm(lr_llm)
    if not args.evaluate:
        # if evaluate, we crop the base model to have the same number of layers as the student model.
        short_llm = load_llm(checkpoint=args.llm, device_map="cpu", num_hidden_layers=num_layers)
        freeze_llm(short_llm)
        crop_llm(short_llm, lr_llm)
    base_llm = llm if short_llm is None else short_llm
    base_llm = base_llm.cuda()
    lr_llm = lr_llm.cuda()

    # Low-Rank Approximation of the student layers
    LOGGER.info(f"Low-Ranking distillable weigts with {args.init_method}...")
    lowrank_llm(llm=lr_llm,
                d_model=lr_llm.config.hidden_size,
                config=rank_searcher.config,
                init_method=args.init_method)
    print(lr_llm)
    lr_llm_num_params = sum(int(p.nelement()) for n, p in lr_llm.named_parameters())
                            # if "layers" in n)
    percentage = round(lr_llm_num_params / base_num_params * 100, 2)
    LOGGER.info(f'Number of distillable parameters: {lr_llm_num_params:,} ({percentage}%)')

    # launch the distillation
    prepare_for_distillation(llm=base_llm,
                             lr_llm=lr_llm,
                             distill_params=distill_params,
                             ignore_first_layers=args.ignore_first_layers,
                             strategy=args.strategy,
                             layers=layers,
                             logger=LOGGER.info)
    print(lr_llm)
    train(train_data=train_dataloader,
          tokenizer=tokenizer,
          llm=base_llm,
          lr_llm=lr_llm,
          distill_params=distill_params)

    # save the distilled LLM
    unset_distillers(llm) # remove distiller modules from the llm
    load_best_model(lr_llm) # load the best student module
    set_distilled_layers_to_llm(lr_llm, llm) # transfer distilled layer to the base llm
    LOGGER.info(f"Saving distilled llm to '{args.output_folder}/lowrank_llm'...")
    save_distilled_llm(lr_llm=llm, lr_llm_config=rank_searcher.config, output_folder=f"{args.output_folder}/lowrank_llm")

    # evaluate
    if args.evaluate:
        eval_dir = Path(args.output_folder) / "evaluation_results"
        eval_dir.mkdir(exist_ok=True, parents=True)
        LOGGER.info("Evaluation...")
        unset_distillers(lr_llm)
        results, tasks_score = evaluate(
            llm=lr_llm,
            tokenizer=tokenizer,
        )
        LOGGER.info(f"Saving results to {eval_dir}.")
        with open(f"{args.output_folder}/full_results_0_shot.json", "w") as f:
            json.dump(results, f)
        with open(f"{args.output_folder}/0_shot_task_results.json", "w") as f:
            json.dump(tasks_score, f)

if __name__ == "__main__":
    main()

# 68.38
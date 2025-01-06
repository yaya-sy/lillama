from ..data.dataiterator import DataIterator, Collator
from .distiller import Distiller
from .config import DistillationParams
from ..evaluation.eval_ppl import get_loaders, ppl_eval
import logging
from tqdm import tqdm
from datasets import Dataset
import torch
from torch import nn

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def save_losses(lr_llm, tokens):
    for _, module in lr_llm.named_modules():
        if isinstance(module, Distiller):
            module.save_losses(tokens=tokens)

def load_best_model(lr_llm):
    for _, module in lr_llm.named_modules():
        if isinstance(module, Distiller):
            module.load_best_student()

def get_initial_ppl(lr_llm, llm, test_dataloader):
    llm_ppl = ppl_eval(model=llm.eval(), test_loader=test_dataloader, device=torch.device("cuda:0"))
    llm_ppl = round(llm_ppl, 2)
    lr_llm_ppl = ppl_eval(model=lr_llm.eval(), test_loader=test_dataloader, device=torch.device("cuda:0"))
    lr_llm_ppl = round(lr_llm_ppl, 2)
    return llm_ppl, lr_llm_ppl

def train(train_data: Dataset,
          tokenizer,
          llm: nn.Module,
          lr_llm: nn.Module,
          distill_params: DistillationParams,
          subset=None) -> None:
    """Forwards all the dataset through the LLM and computes the statistics."""
    # base perplexities
    pad_token_id=tokenizer.eos_token_id
    if distill_params.evaluate:
        test_dataloader = get_loaders(tokenizer=tokenizer)
        llm_ppl, lr_llm_ppl = get_initial_ppl(lr_llm=lr_llm, llm=llm, test_dataloader=test_dataloader)
    else:
        llm_ppl, lr_llm_ppl = None, None

    llm = llm.train()
    lr_llm = lr_llm.train()
    p_bar = tqdm(total=len(train_data))
    total_tokens = 0
    tokens = []
    tokens_ppls = [(0, lr_llm_ppl)]
    idx = 0
    LOGGER.info("Start distillation...")
    for batch in train_data:
        inputs = batch["input_ids"]
        # update statistics
        consumed_tokens = (inputs != pad_token_id).sum().item()
        total_tokens += consumed_tokens
        # forward
        inputs = inputs.to(llm.device)
        llm(inputs, use_cache=False, output_attentions=False)
        if "student" in distill_params.distillation_loss:
            lr_llm(inputs, use_cache=False, output_attentions=False)
        
        # validate
        if distill_params.evaluate and idx == distill_params.log_interval:
            ppl = ppl_eval(model=lr_llm.eval(), test_loader=test_dataloader, device=lr_llm.device)
            ppl = round(ppl, 2)
            tokens_ppls.append((total_tokens, ppl))
            LOGGER.info(f"CONSUMED TOKENS={total_tokens:,}, CURRENT PPL={ppl}, START PPL={lr_llm_ppl}, BASE MODEL PPL={llm_ppl}")
            lr_llm.train()
            idx = 0
             
        tokens.append(total_tokens)
        p_bar.update(consumed_tokens)
        idx += 1
        
    LOGGER.info("Distillation is finished. Saving the losses.")
    save_losses(lr_llm=lr_llm, tokens=tokens)
    # with open(f"{distill_params.output_folder}/tokens.txt", "w") as tokens_file:
    #     tokens_file.write("\n".join(str(n) for n in tokens))

    with open(f"{distill_params.output_folder}/perplexities.txt", "w") as ppls_file:
        ppls_file.write("\n".join(f"{t}\t{p}" for t, p in tokens_ppls))
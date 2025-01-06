import time
start_time = time.time()
from ..evaluation.load_lr_llm import load_lr_llm
from ..evaluation.custom import evaluate
from pathlib import Path
from torch.optim import AdamW
import torch
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm
from datasets import load_from_disk
from transformers import DataCollatorWithFlattening, AutoTokenizer, pipeline, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader

Path(
    '/lustre/fsn1/projects/rech/knb/urc37ho/iclr/svd/60p/13M_SlimOrca/uniform/llama2-7b/teacher+student/uniform_layers/finetuned'
).mkdir(parents=True, exist_ok=True)
# Load model
_, model, config = load_lr_llm(checkpoint="/gpfsdswork/dataset/HuggingFace_Models/meta-llama/Llama-2-7b-hf/",
                               distill_path="/lustre/fsn1/projects/rech/knb/urc37ho/iclr/svd/60p/13M_SlimOrca/uniform/llama2-7b/teacher+student/uniform_layers/checkpoints/"
                               )
tokenizer = AutoTokenizer.from_pretrained("/gpfsdswork/dataset/HuggingFace_Models/meta-llama/Llama-2-7b-hf/")
model = model.cuda()
pipe = pipeline("text-generation", model=model, do_sample=False, tokenizer=tokenizer, device=torch.device("cuda"))

# Load data
train_data = load_from_disk("/lustre/fsn1/projects/rech/knb/urc37ho/SlimOrca-tokenized-llama2/")
train_dataloader = DataLoader(train_data,
                              batch_size=4,
                              shuffle=False,
                              collate_fn=DataCollatorWithFlattening(return_tensors="pt", return_position_ids=False))
example_prompt = "Here a python function that sum up three numbers:"
with torch.no_grad():
    print(pipe(example_prompt, max_new_tokens=64, min_new_tokens=32)[0]["generated_text"])


def save_ckpt(model: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
              sum_loss: list,
              batch_idx: int):
    path = Path("/lustre/fsn1/projects/rech/knb/urc37ho/iclr/svd/60p/13M_SlimOrca/uniform/llama2-7b/teacher+student/uniform_layers/finetuned")
    lr_modules = [path.stem for path in Path(path.parent / "checkpoints").glob("*.pt")] 

    # saving layers first
    layers_path: Path = path / f"layers-{batch_idx}"
    layers_path.mkdir(exist_ok=True, parents=True)
    for n, l in model.named_modules():
        if n in lr_modules:
            layer_filename = f"{lr_modules[lr_modules.index(n)]}.pt"
            state_dict = {k: v.cpu() for k, v in l.state_dict().items()}
            torch.save(state_dict, f=(layers_path / layer_filename))
    
    # saving training check
    checkpoint_path: Path = path / f"checkpoint-{batch_idx}.pt"
    state = {
        "batch_idx": batch_idx,
        "sum_loss": sum_loss,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict()
    }
    torch.save(obj=state, f=checkpoint_path)
    pass

def load_ckpt(model: torch.nn.Module, optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler.LambdaLR):
    batch_idx = 0
    sum_loss = 0
    path = Path("/lustre/fsn1/projects/rech/knb/urc37ho/iclr/svd/60p/13M_SlimOrca/uniform/llama2-7b/teacher+student/uniform_layers/finetuned")
    if any(path.iterdir()):
        last_ckpt = max(int(str(ckpt.stem).split("-")[-1]) for ckpt in path.glob("checkpoint-*"))
        checkpoint = torch.load(path / f'checkpoint-{last_ckpt}.pt')
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        batch_idx = checkpoint["batch_idx"]
        sum_loss = checkpoint["sum_loss"]
        print(f"Loaded checkpoint checkpoint-{last_ckpt}. Resume training after {batch_idx} steps.")
    return model, optimizer, lr_scheduler, batch_idx, sum_loss

# Training Hyperparameters
LR = 5e-5
LOG_EVERY_STEPS = 2048
SAVE_EVEY_MIN = 115
        
def train(model, tokenizer):
    no_decay = ["bias", "norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.1,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=LR)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=64,
                                                   num_training_steps=len(train_dataloader))
    model, optimizer, lr_scheduler, current_batch_idx, sum_loss = load_ckpt(model, optimizer, lr_scheduler)
    current_step = 0
    if current_batch_idx > 0:
        print(f"Skiping the {current_batch_idx} first batches...")
    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training..."), start=1):
        if current_batch_idx > batch_idx:
            continue
        batch = {k: v.cuda() for k, v in batch.items()}
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        current_step += 1
        sum_loss += loss.item()

        if ((time.time() - start_time) / 60) > SAVE_EVEY_MIN:
            print("Saving checkpoints...")
            save_ckpt(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, sum_loss=sum_loss, batch_idx=batch_idx)
            print("Checkpoints saved. Exiting.")
            return
        if current_step >= LOG_EVERY_STEPS:
            mean_loss = sum_loss / batch_idx
            print(f"STEP={batch_idx}, MEAN LOSS={mean_loss}, LAST LOSS={loss.item()}, LR={optimizer.param_groups[0]['lr']}")
            print(f"Generated text at step {batch_idx}:")
            with torch.no_grad():
                pipe = pipeline("text-generation", model=model, do_sample=False, tokenizer=tokenizer, device=torch.device("cuda"))
                print(pipe(example_prompt, max_new_tokens=64)[0]["generated_text"])
            current_step = 0
    print("Saving checkpoints...")
    save_ckpt(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, sum_loss=sum_loss, batch_idx=batch_idx)
    evaluate(llm=model, tokenizer=tokenizer)
train(model, tokenizer)

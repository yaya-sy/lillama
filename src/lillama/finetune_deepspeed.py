# from ..evaluation.lm_eval_harness import eval_main
import logging
from ..evaluation.load_lr_llm import load_lr_llm
from ..evaluation.custom import evaluate
from pathlib import Path
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorWithFlattening
from datasets import load_from_disk
import torch

def main():
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/gpfsdswork/dataset/HuggingFace_Models/mistralai/Mistral-7B-v0.1/")
    _, llm, _ = load_lr_llm(checkpoint="/gpfsdswork/dataset/HuggingFace_Models/mistralai/Mistral-7B-v0.1/",
                            distill_path="/lustre/fsn1/projects/rech/knb/urc37ho/iclr/svd/38p/13M_SlimOrca/r1536/mistral-7b/teacher+student/bottom_layers/checkpoints/")

    tokenizer.padding_side = 'right' # to prevent warnings
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk("/lustre/fsn1/projects/rech/knb/urc37ho/SlimOrca-tokenized-mistral/")
    print(dataset)
    # dataset = dataset.select(range(1_000))

    data_collator = DataCollatorWithFlattening()

    # Define the training arguments
    args = TrainingArguments(
        output_dir="/lustre/fsn1/projects/rech/knb/urc37ho/mistral-4b-checkpoints-ift/", # directory to save and repository id
        num_train_epochs=1,                     # number of training epochs
        # max_steps=8_000,
        per_device_train_batch_size=4,          # batch size per device during training
        # gradient_accumulation_steps=4,          # number of steps before performing a backward/update pass
        save_strategy="steps",
        save_steps=2_000,
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        optim="adamw_torch",              # use fused adamw optimizer
        logging_steps=2000,                       # log every 1000 steps
        learning_rate=5e-5,                     # learning rate, based on QLoRA paper
        bf16=True,                              # use bfloat16 precision
        tf32=True,                              # use tf32 precision
        warmup_ratio=0.03,                       # warmup ratio based on QLoRA paper
        weight_decay=0.01,
        lr_scheduler_type="cosine",             # use constant learning rate scheduler
        push_to_hub=False,                      # push model to hub
        report_to="none",                # report metrics to tensorboard
        deepspeed={
                    "train_batch_size" : "auto",
                    "train_micro_batch_size_per_gpu": "auto",
                    "steps_per_print": 1,
                    "zero_optimization": {
                        "stage": 1
                    },
                    "bf16": {
                        "enabled": True
                    }
                    }
                )


    # Initialize the Trainer
    trainer = Trainer(
        model=llm,
        args=args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    # Start the training
    ckpt_path = "/lustre/fsn1/projects/rech/knb/urc37ho/mistral-4b-checkpoints-ift/"
    Path(ckpt_path).mkdir(exist_ok=True, parents=True)
    resume_from_checkpoint = None
    if any(Path(ckpt_path).iterdir()):
        last_chackpoint = max(int(ckpt.stem.split("-")[-1]) for ckpt in Path(ckpt_path).glob("*"))
        resume_from_checkpoint = f"{ckpt_path}/checkpoint-{last_chackpoint}"
        logging.info(f"Resume from {resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model("/lustre/fsn1/projects/rech/knb/urc37ho/mistral-4b-ift/")
    tokenizer.save_pretrained("/lustre/fsn1/projects/rech/knb/urc37ho/mistral-4b-ift/")

    evaluate(llm=trainer.model, tokenizer=tokenizer)
    
#   --learning_rate 5e-5 \
#   --weight_decay 0.0 \
#   --warmup_ratio 0.03 \
#   --lr_scheduler_type "cosine" \
  
if __name__ == "__main__":
    main()

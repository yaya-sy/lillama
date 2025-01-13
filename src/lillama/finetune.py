import os
from ..evaluation.load_lr_llm import load_lr_llm
from ..evaluation.lm_eval import evaluate
from pathlib import Path
from datasets import load_dataset
from trl import setup_chat_format, SFTTrainer
from transformers import TrainingArguments, AutoTokenizer, DataCollatorWithFlattening

def prepare_dataset(instructions: bool=True):
    def create_conversation(sample):
        sample = sample["conversations"]
        return {
            "messages" : [
                {
                    "role": "user" if data["from"] == "human" else "assistant",
                    "content": data["value"],

                } for data in sample if data["from"] != "system"
            ]
        }
    
    def create_instructions(sample):
        sample = sample["conversations"]
        prompt = [data["value"] for data in sample if data["from"] == "human"][0]
        completion = [data["value"] for data in sample if data["from"] in {"gpt", "assistant"}][0]
        return {
                "prompt": prompt,
                "completion": completion,
                }
    
    dataset = load_dataset("slimorca", split="train")
    # dataset = dataset.shuffle()
    # dataset = dataset.select(range(400_000))
    print(dataset)
    # Convert dataset to OAI messages
    map_fn = create_instructions if instructions else create_conversation
    dataset = dataset.map(map_fn, remove_columns=dataset.features, num_proc=os.cpu_count(), batched=False)
    # split dataset into 10,000 training samples and 2,500 test samples
    print(dataset[0])
    # dataset = dataset.train_test_split(test_size=0.10)
    
    return dataset

def prepare_model(hf_model, distill_path):
    _, llm, _ = load_lr_llm(distill_path=distill_path, checkpoint=hf_model)
    # Hugging Face model id
    
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokenizer.padding_side = 'right' # to prevent warnings
    tokenizer.pad_token = tokenizer.eos_token

    return llm, tokenizer

dataset = prepare_dataset()
model, tokenizer = prepare_model(hf_model="/gpfsdswork/dataset/HuggingFace_Models/mistralai/Mistral-7B-v0.1/",
                                 distill_path="/lustre/fsn1/projects/rech/knb/urc37ho/iclr/svd/40p/13M_SlimOrca/r1536/mistral-7b/teacher+student/bottom_layers/checkpoints/")
model = model.cuda()
print(f"Model size: {model.num_parameters():,}")
args = TrainingArguments(
    output_dir="/lustre/fsn1/projects/rech/knb/urc37ho/finetuned-mistral-4b-slimorca-2", # directory to save and repository id
    num_train_epochs=1,                     # number of training epochs
    per_device_train_batch_size=4,          # batch size per device during training
    # gradient_accumulation_steps=4,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=100,                       # log every 10 steps
    save_strategy="no",                     # save checkpoint every epoch
    learning_rate=2e-5,                     # learning rate, based on QLoRA paper
    bf16=True,                              # use bfloat16 precision
    tf32=True,                              # use tf32 precision
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.1,                       # warmup ratio based on QLoRA paper
    lr_scheduler_type="cosine",             # use constant learning rate scheduler
    push_to_hub=False,                      # push model to hub
    report_to="none",                # report metrics to tensorboard
    save_total_limit=1,
    load_best_model_at_end=True,
)
max_seq_length = 4096 # max sequence length for model and packing of the dataset
eos_token = tokenizer.eos_token
bos_token = tokenizer.bos_token
 
# peft_config = get_lora_config()

def format_fn(example):
    return f"{bos_token}{example['prompt']}{eos_token}{bos_token}{example['completion']}{eos_token}"

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    dataset_num_proc=os.cpu_count(),
    dataset_batch_size=32,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=format_fn,
    dataset_kwargs={
        "add_special_tokens": False,  # We template with special tokens. Todo: Try with True
        "append_concat_token": False, # No need to add additional separator token
    }
)
print(f"Fintuning a model of {model.num_parameters(only_trainable=True):,}")
trainer.train()
trainer.save_model("/lustre/fsn1/projects/rech/knb/urc37ho/mistral-4b-slimorca-2")
tokenizer.save_pretrained("/lustre/fsn1/projects/rech/knb/urc37ho/mistral-4b-slimorca-2")

evaluate(llm=trainer.model, tokenizer=tokenizer)
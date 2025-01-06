from ..data.dataiterator import Collator
from ..data.sharegpt import ShareGPT
from ..evaluation.lm_eval_harness import eval_main
from ..evaluation.load_lr_llm import load_lr_llm
import torch
from torch import nn
from torch.nn.functional import softmax, log_softmax, cross_entropy
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments

class KnowledgeDistillation(nn.Module):
    def __init__(self,
                 teacher: PreTrainedModel,
                 student: PreTrainedModel,
                 alpha: float=0.25,
                 t: float=1.0):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.alpha = alpha
        self.t = t
        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")

    def forward(self, **kwargs):
        # x in [b, s]
        x = kwargs["input_ids"]
        y = kwargs["labels"]
        b, *_ = x.shape
        with torch.no_grad():
            t_logits = self.teacher(x).logits # in [b, s, d]
            t_logits /= self.t
            t_logits[y == -100] = 1e-29
        s_logits = self.student(x).logits # in [b, s, d]
        s_logits /= self.t
        s_logits[y == -100] = 1e-29
        *_, d = t_logits.shape
        t_probs = softmax(t_logits, dim=-1)
        s_probs = log_softmax(s_logits, dim=-1)

        # kd loss
        # kl(p|q) = p * log(p/q) = p * (log(p) - log(q))
        kd_loss = torch.sum(t_probs * (t_probs.log() - s_probs)) / b * (self.t ** 2)

        # sft loss
        y = y[:, 1:].contiguous()
        logits = s_logits[..., :-1, :].contiguous()
        ce_loss = cross_entropy(logits.view(-1, d), y.view(-1), ignore_index=-100, reduction="mean")

        return {
            "loss": (self.alpha * kd_loss) + ((1 - self.alpha) * ce_loss),
            "kd_loss": kd_loss,
            "ce_loss": ce_loss,
            "logits": None
            }

def train(train_data, test_data, collate_fn, teacher, student, alpha=0.7):
    kdistiller = KnowledgeDistillation(teacher=teacher, student=student, alpha=alpha)

    args = TrainingArguments(
        output_dir="code-llama-7b-text-to-sql", # directory to save and repository id
        num_train_epochs=1,                     # number of training epochs
        per_device_train_batch_size=16,          # batch size per device during training
        gradient_accumulation_steps=4,          # number of steps before performing a backward/update pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        optim="adamw_torch_fused",              # use fused adamw optimizer
        logging_steps=50,                       # log every 10 steps
        save_strategy="epoch",                  # save checkpoint every epoch
        learning_rate=2e-5,                     # learning rate, based on QLoRA paper
        bf16=True,                              # use bfloat16 precision
        tf32=True,                              # use tf32 precision
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.1,                      # warmup ratio based on QLoRA paper
        lr_scheduler_type="cosine",           # use constant learning rate scheduler
        push_to_hub=False,                       # push model to hub
        report_to="none"                # report metrics to tensorboard
    )

    trainer = Trainer(
        model=kdistiller,                   # the instantiated 🤗 Transformers model to be trained
        args=args,                 # training arguments, defined above
        train_dataset=train_data,         # training dataset
        eval_dataset=test_data,           # evaluation dataset
        data_collator=collate_fn,
    )
    trainer.train()
    return trainer.model.student


def main():
    sharegpt = ShareGPT(
        dataset="openhermes/",
        tokenizer="/gpfsdswork/dataset/HuggingFace_Models/microsoft/phi-2/",
        target_column="conversations",
        max_length=4096,
        batch_size=2,
        num_proc=2,
    )
    train_data, test_data = sharegpt()

    tokenizer = AutoTokenizer.from_pretrained("/gpfsdswork/dataset/HuggingFace_Models/microsoft/phi-2/")
    teacher = AutoModelForCausalLM.from_pretrained("/gpfsdswork/dataset/HuggingFace_Models/microsoft/phi-2/",
                                                   # torch_dtype=torch.bfloat16,
                                                   attn_implementation="flash_attention_2",
                                                   )
    _, student, _ = load_lr_llm(distill_path="30p/13M_SlimOrca/r1024/phi2-3b/teacher+student/checkpoints/",
                                checkpoint="/gpfsdswork/dataset/HuggingFace_Models/microsoft/phi-2/")
    student.config.eos_token_id
    teacher.cuda()
    student.cuda()
    train_data.set_format(type="torch", columns=["input_ids", "labels"])
    test_data.set_format(type="torch", columns=["input_ids", "labels"])
    print(train_data)
    collate_fn=Collator(pad_token_id=tokenizer.eos_token_id)
    student = train(train_data=train_data, test_data=test_data, collate_fn=collate_fn, teacher=teacher, student=student)
    if True:
        eval_main(
            llm=student,
            tokenizer=tokenizer,
            save_dir="finetuned"
        )

main()
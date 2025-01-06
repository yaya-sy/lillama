from ..evaluation.load_lr_llm import load_lr_llm
from ..model.modeling_mixtral import MixtralForCausalLM
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import torch
import argparse

prompts = [
    (0.3, "Who was Steve Jobs?"),
    (0.3, "What is a modal verb in English?"),
    (0.0, "Write a PyTorch method that implement a simple Attention layer.")
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-d", "--distill_path", type=str, default=None)
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # with torch.no_grad():
    #    lr_llm = MixtralForCausalLM.from_pretrained(args.distill_path,
    #                                                attn_implementation="flash_attention_2",
    #                                                torch_dtype=torch.bfloat16,
    #                                                trust_remote_code=True)
    lr_llm = AutoModelForCausalLM.from_pretrained(args.model,
                                                  attn_implementation="flash_attention_2",
                                                  torch_dtype=torch.bfloat16,
                                                  trust_remote_code=True)
    # else:
    #     _, lr_llm, _ = load_lr_llm(checkpoint=args.model,
    #                                       distill_path=args.distill_path)
    print(lr_llm)
    # print(f"Number of parameters of the base LLM: {base_llm.num_parameters():,}")
    print(f"Number of parameters of the low-rank LLM: {lr_llm.num_parameters():,}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for idx, (temp, prompt) in enumerate(prompts):
        do_sample = True if temp > 0.0 else False
        lr_pipe = pipeline("text-generation", model=lr_llm, do_sample=do_sample, tokenizer=tokenizer, temperature=temp, device=device)

        print()
        print()
        print(f"Generation for '{prompt}':")
        print(lr_pipe(prompt, max_new_tokens=1024, min_new_tokens=32, top_p=0.9, top_k=10)[0]["generated_text"])
    print("FINISH\n" * 5)

<a href="https://colab.research.google.com/github/yaya-sy/lillama/blob/master/lillama.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Lillama: Large Language Models Compression via Low-Rank Feature Distillation
<p align="center">
  <img src="https://github.com/user-attachments/assets/31ea5289-ca53-4d5e-a853-2ba8f357c24f?raw=true:, width=500" alt="lillama" width=500 class="center">
</p>

# Installation

If you're working with GPUs of capability < 8.0, you can ignore the flash-attn installation.

```bash
pip install -U git+https://github.com/yaya-sy/lillama.git flash-attn
```

# Compression
Here is an example of how you can compress Phi-2 3B by 20%.

## Step 1: Prepare the dataset

```bash
HF_DATASETS_TRUST_REMOTE_CODE=True lillama-sharegpt \
  --tokenizer microsoft/phi-2 \
  --dataset Open-Orca/SlimOrca \
  --subset 13_000_00 \
  --output-folder distillation-data
```

This will prepare Slim-Orca (sharegpt format) for distillation

## Step 2: Compress the model

```bash
HF_DATASETS_TRUST_REMOTE_CODE=True lillama-distill \
  --llm microsoft/phi-2 \
  --train-data distillation-data/ \
  --output-folder distilled-phi2/ \
  --batch-size 8 \
  --log-interval 256
```
For big models (for example Mixtral-47B), you should use the argument `--no-evaluate` so the whole model will not be loaded on GPU.

The distilled weights will be saved in `distilled-phi2/checkpoints`.

# Evaluation

## 0-shot evaluation with `lm-eval'
You can evaluate the compressed model as:

```bash
HF_DATASETS_TRUST_REMOTE_CODE=True python -m lillama.evaluation.lm_eval \
  --llm microsoft/phi-2 \
  --distill-path distilled-phi2/checkpoints/ \
  --output-folder distilled-phi2-eval/
```

This will save two `.json` files. The file `full_results_0_shot.json` contains the detailed results while  `0_shot_task_results.json`contains the summarized evaluation results.

## Generate with the compressed model

You can also manually inspect the generations of the model using Huggingface Transformers:

```python
from lillama.utils import load_lr_llm
import torch
from transformers import pipeline
from transformers import AutoTokenizer

# load the model
_, lr_llm, _ = load_lr_llm(checkpoint="microsoft/phi-2",
                           distill_path="distilled-phi2/checkpoints")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
print(f"Number of parameters of the low-rank LLM: {lr_llm.num_parameters():,}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# use text-generation pipeline from Huggingface Transformers
lr_pipe = pipeline("text-generation", model=lr_llm, do_sample=True, tokenizer=tokenizer, temperature=0.3, device=device)
output = lr_pipe("What is the cause of the Civil War? Here is the story:",
                 max_new_tokens=256,
                 min_new_tokens=32,
                 top_p=0.9,
                 top_k=10)[0]["generated_text"]
```

# Share the compressed with the community on Hugginface

At the moment the model can only be loaded with `lillama`. To share the model and use it independently, you should modify manually the model file. 

First, save the compressed model as Huggingface Model:

```python
from lillama.utils import save_lr_llm

save_lr_llm(distill_path="distilled-phi2/checkpoints",
            checkpoint="microsoft/phi-2",
            output_path="hf_compressed_model")
```

This will save compressed model and its config.

Then you have to modify the model file by replacing the linear layers `torch.nn.Linear(input_feature, output_features)` with the low rank ones: `torch.nn.Sequential(torch.nn.Linear(input_feature, rank), torch.nn.Linear(rank, output_features))`. I haven't automatized this, but I've done it for `Mixtral', so you can it as template: https://huggingface.co/yaya-sy/minixtral/blob/main/modeling_mixtral.py

You also have to modify he config.json for Transformers `auto_map`. Please se how I achieved this here: https://huggingface.co/yaya-sy/minixtral/blob/main/config.json




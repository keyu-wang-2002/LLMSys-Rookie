# MLSys for Efficient LLMs - Rookies' Cookbook
List of Paper, Codebases & Blogs for rookies

updating...



## I. System Optimization Basics: Operators & Kernels

<br><br>

## II. Efficient Training
1. Unsloth: https://github.com/unslothai/unsloth
2. Megatron-LM: https://github.com/NVIDIA/Megatron-LM
3. Deepspeed: https://github.com/deepspeedai/DeepSpeed
4. LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory

<br><br>

## III. Efficient Inference
A Road Map: https://arxiv.org/abs/2402.16363

### Framework
1. vllm: https://github.com/vllm-project/vllm
2. SGLang: https://github.com/sgl-project/sglang
3. llama.cpp: https://github.com/ggml-org/llama.cpp

### 1. Early Exit
1. LayerSkip: https://arxiv.org/abs/2404.16710

### 2. Contextual Sparsity (Width-wise)
Dynamic sparsity depent on the input (context)

1. DejaVu: http://proceedings.mlr.press/v202/liu23am.html

DejaVu inspires **LLM in a flash** (https://arxiv.org/abs/2312.11514) by Apple and **PowerInfer** (https://arxiv.org/abs/2312.12456)

### 3. Mixture-of-Experts
Awesome blog: https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts

### 4. Attention/KV Compression
1. Attention Sinks: https://arxiv.org/abs/2309.17453
2. H2O: https://arxiv.org/abs/2306.14048

### 5. Speculative Reasoning

<br><br>

## IV. Model Compression

### 1. Pruning
#### - Unstructured Pruning
Selectively eliminate individual weights from a model: Fine-grained, but hard to accelerate inference

1. SparseGPT: https://proceedings.mlr.press/v202/frantar23a/frantar23a.pdf
2. Wanda: https://arxiv.org/pdf/2306.11695

Refer Wanda for neat codebase.

#### - Structured Pruning
Remove entire neurons or layers from a model: Coarse-grained, resulting in a cleaner, more regular structure and easier to accelerate inference.
Per-neuron / Per-channel (LLM-Pruner: https://arxiv.org/abs/2305.11627), Per-block, Per-layer (ShortGPT: https://arxiv.org/abs/2403.03853)

### 2. Quantization
#### - Post-Training Quantization (PTQ)
Quantizate model parameters post the LLM’s training phase (mainstream in quantizing LLMs).
 
1. GPTQ: https://arxiv.org/abs/2210.17323
2. SmoothQuant: https://arxiv.org/abs/2211.10438
3. AWQ: https://arxiv.org/abs/2306.00978
   
#### - Quantization-Aware Training (QAT)
Integrate quantization into the model’s training process or during the fine-tuning/re-training of a pre-trained LLM
 
Not common for LLMs

#### - Quantization for Parameter-Efficient Fine-Tuning (Q-PEFT)
1. QLoRA: https://arxiv.org/abs/2305.14314
   
wow, GPT-oss uses mxfp4 !

#### Useful library for Quantizing LLMs
1. bitsandbytes:  https://github.com/bitsandbytes-foundation/bitsandbytes

### 3. Distillation
Transfer of capabilities from a larger model ("teacher model") to a smaller model ("student model")

Easy codes from scratch: https://github.com/KylinC/Llama-3-Distill

<br> 



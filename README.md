# MLSys-Rookie
List of Paper, Codebases & Blogs for rookies

updating...

## I. Kernel


## II. Training
1. Unsloth: https://github.com/unslothai/unsloth
2. Megatron-LM: https://github.com/NVIDIA/Megatron-LM
3. Deepspeed: https://github.com/deepspeedai/DeepSpeed
4. LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory

## III. Inference
### Framework
1. vllm (or Slang): https://github.com/vllm-project/vllm
2. llama.cpp: https://github.com/ggml-org/llama.cpp
   
### (i) Attention Optimization (KV Compression / Sparse Atten)
**Basics**: Multi-head Attention (MHA), Multi-Query Attention (MQA), Grouped-Query Attention (GQA), Multi-head Latent Attention (GLA); Paged Attention, FlashAttention

**Paper**:
1. DejaVu: http://proceedings.mlr.press/v202/liu23am.html
2. Attention Sinks: https://arxiv.org/abs/2309.17453
3. H2O: https://arxiv.org/abs/2306.14048

### (ii) Speculative Reasoning

## IV. Model Compression

### (i) Pruning
#### Unstructured Pruning
1. SparseGPT: https://proceedings.mlr.press/v202/frantar23a/frantar23a.pdf
2. Wanda: https://arxiv.org/pdf/2306.11695
3. GBLM-Pruner: https://arxiv.org/pdf/2311.04902

Refer Wanda for neat codebase.

#### Structured Pruning
Per-channel, Per-block, Per-layer (Refer ShortGPT: https://arxiv.org/abs/2403.03853)

### (ii) Quantization
#### Post-Training Quantization (PTQ)
1. GPTQ: https://arxiv.org/abs/2210.17323
2. SmoothQuant: https://arxiv.org/abs/2211.10438
3. AWQ: https://arxiv.org/abs/2306.00978
#### Quantization-Aware Training (QAT)
1. QLoRA: https://arxiv.org/abs/2305.14314
wow, GPT-oss uses QAT by mxfp4 !
#### Useful library for Quantizing LLMs
1. bitsandbytes:  https://github.com/bitsandbytes-foundation/bitsandbytes

### (iii) Distillation
Easy codes from scratch: https://github.com/KylinC/Llama-3-Distill



# MLSys-Rookie
## I. Kernel


## II. Training
1. Megatron-LM: https://github.com/NVIDIA/Megatron-LM
2. Deepspeed: https://github.com/deepspeedai/DeepSpeed
3. LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory

## III. inference
### Framework
1. vllm: https://github.com/vllm-project/vllm
2. llama.cpp: https://github.com/ggml-org/llama.cpp
   
### Sparse Attn / KV Compression
1. DejaVu: http://proceedings.mlr.press/v202/liu23am.html
2. Attention Sinks: https://arxiv.org/abs/2309.17453
3. H2O: https://arxiv.org/abs/2306.14048

## Model Compression

### Pruning
#### Unstructured Pruning
1. SparseGPT: https://proceedings.mlr.press/v202/frantar23a/frantar23a.pdf
2. Wanda: https://arxiv.org/pdf/2306.11695
3. GBLM-Pruner: https://arxiv.org/pdf/2311.04902

Refer Wanda for neat codebase.

#### Structured Pruning
Per-channel, Per-block, Per-layer

### Quantization
#### Post-Training Quantization
1. GPTQ: https://arxiv.org/abs/2210.17323
2. SmoothQuant: https://arxiv.org/abs/2211.10438
3. AWQ: https://arxiv.org/abs/2306.00978
#### Quantization-Aware Training
1. QLoRA: https://arxiv.org/abs/2305.14314
#### Useful library for Quantizing LLMs
1. bitsandbytes:  https://github.com/bitsandbytes-foundation/bitsandbytes

### Distillation
Knowledge distillation is a specialized form of **transfer learning** in which the “knowledge” (i.e. the learned output distributions) of a larger, pretrained teacher model is transferred to a smaller or simpler student model that is being trained on the same task. 

The Process of Distillation (Denote teacher network as T, student network as S):
```
1. Keep weights of T frozen, weights of S trainable.
2. Input sample x to T and S respectively, to get corresponding outputs t_out(x) and s_out(x)
3. Calculate student loss between label y and student output s_out(x) by cross entropy loss: L_ce(s_out, y)=cross_entropy_loss(s_out, y)
4. Calculate distillation loss between teacher output t_out(x) and student output s_out(x) by KL Divergence: L_kd=kl_div(s_out, t_out)
5. The overall Loss: L = aL_ce + (1-a)L_kd
6. Backward and update: L.backward() \ optimizer.step()
```
Easy codes from scratch: https://github.com/KylinC/Llama-3-Distill



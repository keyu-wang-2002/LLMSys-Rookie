# MLSys-Rookie
## Kernel

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
4. QLoRA: https://arxiv.org/abs/2305.14314
#### Quantization-Aware Training
bitsandbytes -- library for Quantize LLMs:  https://github.com/bitsandbytes-foundation/bitsandbytes

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

Paper collections: https://github.com/FLHonker/Awesome-Knowledge-Distillation


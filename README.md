# MLSys-Rookie-Cookbook

## Model Compression

### Pruning

### Quantization

### Distillation
Knowledge distillation is a specialized form of **transfer learning** in which the “knowledge” (i.e. the learned output distributions) of a larger, pretrained teacher model is transferred to a smaller or simpler student model that is being trained on the same task. 

Whereas general transfer learning might reuse weights or align feature distributions across domains (e.g. via fine-tuning or domain adaptation), distillation focuses specifically on matching the soft-output behavior of one model to another. For example, in an image-classification setting you might take a high-capacity ResNet-101 as the teacher and train a more compact ResNet-18 student to mimic its output probabilities, thereby inheriting much of its performance while keeping the student efficient.

The Process of Distillation (Denote teacher network as T, student network as S):

1. Keep weights of T frozen, weights of S trainable.
2. Input sample x to T and S respectively, to get corresponding outputs t_out(x) and s_out(x)
3. Calculate student loss between label y and student output s_out(x) by cross entropy loss: L_ce(s_out, y)=cross_entropy_loss(s_out, y)
4. Calculate distillation loss between teacher output t_out(x) and student output s_out(x) by KL Divergence: L_kd=kl_div(s_out, t_out)
5. The overall Loss: L = aL_ce + (1-a)L_kd
6. Backward and update: L.backward() \ optimizer.step()

```
from transformers import Trainer, TrainingArguments

class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature

class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self._move_model_to_device(self.teacher, self.model.device)
        self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
        # compute student output
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss

        # compute teacher output
        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)

        # assert size
        assert outputs_student.logits.size() == outputs_teacher.logits.size()

        # Soften probabilities and compute distillation loss
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (
            loss_function(
                F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
                F.softmax(outputs_teacher.logits / self.args.temperature, dim=-1),
            )
            * (self.args.temperature ** 2)
        )
        # Return weighted student loss
        loss = self.args.alpha * student_loss + (1.0 - self.args.alpha) * loss_logits
        return (loss, outputs_student) if return_outputs else loss
```

Notes:

1. How does distillation work? The softmax output of a teacher model conveys rich information not only about the correct class but also about the relative likelihoods of the incorrect classes. Some negative classes receive much higher probabilities than others, reflecting nuanced inter-class relationships. In standard training with “hard” targets, all incorrect classes are treated equally (probability = 0). Knowledge distillation, by contrast, uses these “soft” targets so that each training example provides the student model with far more information than the traditional one-hot labels do.
(i) Easier optimization: By blending the hard target distribution from manual labels with the teacher model’s soft output distribution, the student’s loss surface becomes smoother—making it easier for the optimizer to locate the optimum and thus speeding up convergence.
(ii) Better generalization: Learn from both true label and teacher model.
2. Why do we use temperature? Raising the softmax temperature amplifies the logits of the negative classes, which flattens (smooths) the output probability distribution. A higher temperature thus reveals more detail about the teacher’s confidence across all classes, making it easier for the student to learn subtle distinctions.

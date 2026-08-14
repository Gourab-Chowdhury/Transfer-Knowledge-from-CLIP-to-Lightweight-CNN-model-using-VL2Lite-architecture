import torch
import torch.nn as nn
import torch.nn.functional as F

class VL2LiteLoss(nn.Module):
    """
    VL2Lite Loss Formulation:
    - Task Cross-Entropy Loss
    - Visual Feature MSE Distillation
    - Linguistic Text-Anchor KL Divergence on similarity distributions
    - Dynamic Linear Ramp Weighting Schedule (ramps to 1.0 within ramp_ratio of epochs)
    """
    def __init__(self, temperature=0.07, ramp_ratio=0.125):
        super().__init__()
        self.temperature = temperature
        self.ramp_ratio = ramp_ratio
        self.kl_criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, student_projected, teacher_features, text_anchors, labels, current_epoch, total_epochs):
        # 1. Dynamic Weighting Schedule
        ramp_factor = current_epoch / max(1.0, (total_epochs * self.ramp_ratio))
        w_task = min(max(ramp_factor, 0.0), 1.0)
        w_distill = 1.0 - w_task

        # 2. Classification Task Loss
        loss_task = F.cross_entropy(student_logits, labels)

        # 3. Visual Distillation Loss (MSE)
        loss_visual = F.mse_loss(student_projected, teacher_features)

        # 4. Linguistic Distillation Loss (KL Divergence on Similarity Distributions)
        logit_scale = 1.0 / self.temperature
        student_sim_logits = logit_scale * torch.matmul(student_projected, text_anchors.t())
        teacher_sim_logits = logit_scale * torch.matmul(teacher_features, text_anchors.t())

        log_student_probs = F.log_softmax(student_sim_logits, dim=-1)
        teacher_probs = F.softmax(teacher_sim_logits, dim=-1)

        loss_text = self.kl_criterion(log_student_probs, teacher_probs) * (self.temperature ** 2)

        # 5. Combined Multi-Modal Distillation Loss
        total_loss = (w_task * loss_task) + (w_distill * ((loss_visual + loss_text) / 2.0))
        return total_loss, loss_task, loss_visual, loss_text
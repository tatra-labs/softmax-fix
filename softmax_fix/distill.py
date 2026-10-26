"""Distillation losses whose gradient is exactly zero when the student has caught up.

The gradient of a distillation loss with respect to the student's logits is `p_student - p_teacher`.
Computed that way it is a difference of two numbers that are nearly equal, so when the student is
close to the teacher almost every digit cancels, and when the two probability vectors were produced
by different code paths — a teacher cached in bfloat16, a student under autocast — the leftover
rounding *is* the gradient. A student that has already matched its teacher keeps being pushed.

Writing the same quantity in terms of the logit difference removes the subtraction:

    d = z_student - z_teacher
    L = log1p( sum_j p_teacher[j] * expm1(d[j]) )          (= log sum_j p_teacher[j] exp(d[j]))
    p_student - p_teacher = p_teacher * expm1(d - L)

Every `expm1` is exactly 0 when `d` is, so `L` is exactly 0 and the gradient is exactly 0 — in any
precision, however the teacher's probabilities were obtained. The loss itself has the matching
closed form `L - sum_j p_teacher[j] d[j]`, which is also exactly 0 at a match.
"""

import torch

__all__ = ["kl_divergence", "logit_shift"]

# Beyond this the exp() in the log1p form can overflow, so fall back to the plain stable route.
_SAFE_SHIFT = 30.0


def logit_shift(student_logits, teacher_logits, teacher_log_probs=None):
    """`(L, d, p_teacher)` — the pieces both the loss and its gradient are built from."""
    student_logits = student_logits.float()
    teacher_logits = teacher_logits.float()
    d = student_logits - teacher_logits
    if teacher_log_probs is None:
        from .functional import log_softmax

        teacher_log_probs = log_softmax(teacher_logits)
    p_teacher = teacher_log_probs.exp()

    if float(d.abs().max()) <= _SAFE_SHIFT:
        total = (p_teacher * torch.expm1(d)).sum(-1, keepdim=True)
        shift = torch.log1p(total.clamp_min(-1.0 + 1e-30))
    else:
        shift = torch.logsumexp(teacher_log_probs + d, dim=-1, keepdim=True)
    return shift, d, p_teacher


class _KL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, student_logits, teacher_logits, teacher_log_probs):
        shift, d, p_teacher = logit_shift(student_logits, teacher_logits, teacher_log_probs)
        loss = (shift - (p_teacher * d).sum(-1, keepdim=True)).squeeze(-1)
        ctx.save_for_backward(shift, d, p_teacher)
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        shift, d, p_teacher = ctx.saved_tensors
        grad = p_teacher * torch.expm1(d - shift)
        return grad * grad_out.unsqueeze(-1), None, None


def kl_divergence(student_logits, teacher_logits, teacher_log_probs=None, reduction="mean"):
    """KL(teacher || student) over the last axis, exactly zero when the logits agree.

    Pass `teacher_log_probs` if you already have them; otherwise the teacher's *logits* are what
    this needs, and keeping them rather than cached probabilities is the point — the gradient then
    depends only on how far the student has drifted, not on how the teacher was rounded.
    """
    loss = _KL.apply(student_logits, teacher_logits, teacher_log_probs)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError("unknown reduction: %r" % reduction)

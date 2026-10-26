"""Cross-entropy and softmax that do not round the target's gradient to zero.

The standard route to a cross-entropy gradient is to build the full probability vector and then
subtract the one-hot target:

    p = softmax(z)
    grad = p - y

When one logit dominates, `p[target]` rounds to exactly 1.0 and `grad[target]` becomes exactly 0.
The true value is `-(1 - p[target])`, which stays representable in float32 long after `1 - p` has
stopped being: over 1000 classes a 60-nat lead leaves -8.7e-24, and float32 reaches 1e-45. So it is not
merely imprecise, it is *absent*, and the gradient vector stops summing to zero — the non-target
logits are pushed down with nothing pushing the target up.

The fix is to never form `1 - p`. Write everything relative to the target logit,

    d = z - z[target]                 so d[target] = 0 exactly
    grad[target] = -(sum of the other probabilities)

and the target's gradient is a sum of positive terms rather than a cancelling subtraction. It is
exact whenever any other probability is representable at all, and the gradient sums to zero by
construction.
"""

import torch

__all__ = ["cross_entropy", "log_softmax", "softmax", "target_log_prob"]


def _relative(logits, target):
    """Logits re-expressed relative to the target, with the shift that keeps exp() in range.

    Returns `(others, exp_target, m)` where `others` is the sum of exp(d - m) over the non-target
    entries, `exp_target` is exp(-m), and the softmax denominator is `others + exp_target`.
    """
    index = target.unsqueeze(-1)
    d = logits - logits.gather(-1, index)
    # d[target] is exactly 0, so the max is never negative and the target never overflows.
    m = d.max(-1, keepdim=True).values.clamp_min(0.0)
    shifted = (d - m).exp()
    # Zero the target entry and sum what is left. Summing positives cannot cancel; computing the
    # same quantity as `denominator - exp_target` is precisely the bug this module exists to avoid.
    others = shifted.scatter(-1, index, 0.0).sum(-1, keepdim=True)
    return shifted, others, (-m).exp(), m


def target_log_prob(logits, target):
    """log p[target], computed without ever forming 1 - p.

    For the common case where the target is the largest logit this is `-log1p(sum of the others)`,
    which has no cancellation at all.
    """
    logits = logits.float()
    _, others, exp_target, m = _relative(logits, target)
    # denominator = others + exp(-m) = 1 + (others + expm1(-m)), so log1p is exact when m is 0.
    return (-m - torch.log1p(others + torch.expm1(-m))).squeeze(-1)


class _CrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target):
        logits = logits.float()
        shifted, others, exp_target, m = _relative(logits, target)
        loss = (m + torch.log1p(others + torch.expm1(-m))).squeeze(-1)
        ctx.save_for_backward(shifted, others, exp_target, target)
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        shifted, others, exp_target, target = ctx.saved_tensors
        denominator = others + exp_target
        grad = shifted / denominator
        # The target's share is minus everything else, never 1 - p.
        grad = grad.scatter(-1, target.unsqueeze(-1), -(others / denominator))
        return grad * grad_out.unsqueeze(-1), None


def cross_entropy(logits, target, reduction="mean"):
    """Drop-in replacement for `torch.nn.functional.cross_entropy` with hard targets.

    Same value and same shape as the standard version; the difference is that the gradient on the
    target logit survives, and that the gradient sums to zero.
    """
    loss = _CrossEntropy.apply(logits, target)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError("unknown reduction: %r" % reduction)


def log_softmax(logits, dim=-1):
    """log_softmax whose largest entry is computed by log1p rather than by cancellation.

    `logsumexp(z) - z_max` loses every digit of the largest entry's log-probability once it is
    close to zero; `-log1p(sum of the rest)` keeps them.
    """
    if dim != -1:
        return log_softmax(logits.transpose(dim, -1)).transpose(dim, -1)
    logits = logits.float()
    top = logits.max(-1, keepdim=True)
    d = logits - top.values
    shifted = d.exp()
    others = shifted.scatter(-1, top.indices, 0.0).sum(-1, keepdim=True)
    return d - torch.log1p(others)


def softmax(logits, dim=-1):
    """softmax by the same route, which helps the small entries and cannot help the largest.

    Once the winner leads by more than about 17 nats its probability is not representable as
    anything but 1.0 in float32, and no implementation can change that. That is the whole reason
    the gradient here is never built by subtracting a one-hot from this vector: the information is
    gone from `p`, but it is still in `log p`, which is what `target_log_prob` returns.
    """
    return log_softmax(logits, dim=dim).exp()

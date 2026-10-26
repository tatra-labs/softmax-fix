"""Tell me whether this batch of logits is losing its target gradient.

Drop `report = collapse_report(logits, targets)` into a training loop. It compares the gradient the
standard implementation produces against the one computed relative to the target, and says how many
rows have lost the target's gradient entirely.
"""

import torch

from .functional import target_log_prob

__all__ = ["collapse_report", "standard_target_gradient"]


def standard_target_gradient(logits, target):
    """What `F.cross_entropy` actually puts on the target logit, by the usual route."""
    probabilities = torch.softmax(logits.float(), dim=-1)
    return probabilities.gather(-1, target.unsqueeze(-1)).squeeze(-1) - 1.0


def collapse_report(logits, target):
    """How much of the target gradient the standard implementation has thrown away.

    * `collapsed` — rows where the standard gradient is exactly zero but the true one is not;
    * `gradient_sum` — the standard gradient's deviation from summing to zero, per row, which is
      the net downward push the non-target logits receive with nothing pushing back;
    * `max_target_lead` — how far the target logit leads the runner-up, which is what drives it.
    """
    logits = logits.float()
    target = target.long()
    true_grad = torch.expm1(target_log_prob(logits, target))
    standard = standard_target_gradient(logits, target)

    collapsed = (standard == 0) & (true_grad != 0)
    # The standard gradient vector, summed in float64 so that the deviation is not itself rounded
    # away. For an exact softmax this is zero; what is left is the net push on the logits.
    probabilities = torch.softmax(logits, dim=-1)
    standard_grad = probabilities.double()
    index = target.unsqueeze(-1)
    standard_grad = standard_grad.scatter(
        -1, index, standard_grad.gather(-1, index) - 1.0
    )
    gradient_sum = standard_grad.sum(-1)
    # How far the target leads the runner-up: the quantity that drives the collapse.
    runner_up = logits.scatter(-1, index, float("-inf")).max(-1).values
    lead = logits.gather(-1, index).squeeze(-1) - runner_up

    relative = torch.where(
        true_grad != 0,
        (standard - true_grad).abs() / true_grad.abs(),
        torch.zeros_like(true_grad),
    )
    return {
        "rows": int(true_grad.numel()),
        "collapsed": int(collapsed.sum()),
        "collapsed_fraction": float(collapsed.float().mean()),
        "max_relative_error": float(relative.max()),
        "mean_relative_error": float(relative.mean()),
        "gradient_sum_max": float(gradient_sum.abs().max()),
        "max_target_lead": float(lead.max()),
        "max_logit": float(logits.abs().max()),
    }

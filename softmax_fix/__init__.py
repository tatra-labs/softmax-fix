"""Softmax, log-softmax and cross-entropy that keep the target's gradient."""

from .diagnose import collapse_report, standard_target_gradient
from .distill import kl_divergence, logit_shift
from .functional import cross_entropy, log_softmax, softmax, target_log_prob

__version__ = "0.1.0"
__all__ = [
    "collapse_report",
    "cross_entropy",
    "kl_divergence",
    "log_softmax",
    "logit_shift",
    "softmax",
    "standard_target_gradient",
    "target_log_prob",
]

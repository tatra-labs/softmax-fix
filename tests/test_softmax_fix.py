"""Tests for the fixed softmax, cross-entropy and distillation losses.

Two things have to hold at once: the fixed versions must agree with PyTorch wherever PyTorch is
still right, and must keep working where it is not.
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import softmax_fix as sf  # noqa: E402

FLOAT32_SMALLEST = 1.4012984643e-45


def true_target_gradient(logits, target):
    """p[target] - 1 in float64, by a route with no cancellation in it."""
    return torch.expm1(sf.target_log_prob(logits.double(), target))


# --- agreement with the standard implementation ----------------------------------------------


def test_the_loss_matches_torch():
    torch.manual_seed(0)
    logits = torch.randn(32, 1000) * 2
    target = torch.randint(0, 1000, (32,))
    mine = sf.cross_entropy(logits, target, reduction="none")
    theirs = F.cross_entropy(logits, target, reduction="none")
    assert torch.allclose(mine, theirs, atol=1e-5)


def test_the_gradient_matches_torch_where_torch_is_right():
    torch.manual_seed(0)
    logits = (torch.randn(16, 5000) * 3).requires_grad_(True)
    target = torch.randint(0, 5000, (16,))
    theirs = torch.autograd.grad(F.cross_entropy(logits, target), logits, retain_graph=True)[0]
    mine = torch.autograd.grad(sf.cross_entropy(logits, target), logits)[0]
    assert torch.allclose(mine, theirs, atol=1e-7)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_reductions_match_torch(reduction):
    torch.manual_seed(1)
    logits = torch.randn(8, 100)
    target = torch.randint(0, 100, (8,))
    assert torch.allclose(
        sf.cross_entropy(logits, target, reduction=reduction),
        F.cross_entropy(logits, target, reduction=reduction),
        atol=1e-5,
    )


def test_an_unknown_reduction_is_refused():
    with pytest.raises(ValueError):
        sf.cross_entropy(torch.randn(2, 3), torch.tensor([0, 1]), reduction="median")


def test_log_softmax_matches_torch():
    torch.manual_seed(2)
    logits = torch.randn(4, 500) * 4
    assert torch.allclose(sf.log_softmax(logits), torch.log_softmax(logits, -1), atol=1e-6)
    assert torch.allclose(sf.softmax(logits), torch.softmax(logits, -1), atol=1e-7)


def test_log_softmax_works_on_another_axis():
    torch.manual_seed(3)
    logits = torch.randn(3, 7, 11)
    assert torch.allclose(sf.log_softmax(logits, dim=1), torch.log_softmax(logits, 1), atol=1e-6)


# --- bug one: Softmax Collapse ---------------------------------------------------------------


@pytest.mark.parametrize("lead", [25, 40, 60, 80, 100])
def test_the_standard_gradient_is_gone_and_ours_is_not(lead):
    logits = torch.zeros(1, 1000)
    logits[0, 0] = float(lead)
    target = torch.zeros(1, dtype=torch.long)

    truth = float(true_target_gradient(logits, target))
    assert abs(truth) > FLOAT32_SMALLEST, "float32 can still hold this; the test is fair"

    standard = logits.clone().requires_grad_(True)
    assert float(torch.autograd.grad(F.cross_entropy(standard, target), standard)[0][0, 0]) == 0.0

    fixed = logits.clone().requires_grad_(True)
    got = float(torch.autograd.grad(sf.cross_entropy(fixed, target), fixed)[0][0, 0])
    assert got != 0.0
    assert abs(got - truth) <= 1e-3 * abs(truth)


def test_the_fixed_gradient_sums_to_zero_far_better():
    logits = torch.zeros(1, 1000)
    logits[0, 0] = 40.0
    target = torch.zeros(1, dtype=torch.long)

    standard = logits.clone().requires_grad_(True)
    theirs = torch.autograd.grad(F.cross_entropy(standard, target), standard)[0]
    fixed = logits.clone().requires_grad_(True)
    mine = torch.autograd.grad(sf.cross_entropy(fixed, target), fixed)[0]

    # The standard gradient misses by exactly the target gradient it threw away.
    truth = abs(float(true_target_gradient(logits, target)))
    assert float(theirs.double().sum()) == pytest.approx(truth, rel=1e-3)
    assert abs(float(mine.double().sum())) < truth * 1e-3


def test_target_log_prob_survives_a_dominant_logit():
    logits = torch.zeros(1, 100)
    logits[0, 3] = 50.0
    target = torch.tensor([3])
    mine = float(sf.target_log_prob(logits, target))
    theirs = float(torch.log_softmax(logits, -1)[0, 3])
    assert theirs == 0.0  # the standard route has lost every digit
    assert mine < 0.0 and mine == pytest.approx(-1.9287e-20, rel=1e-3)


def test_no_softmax_can_save_the_winning_probability():
    """A documented limitation, and the reason the gradient never goes through `p` at all."""
    logits = torch.zeros(1, 100)
    logits[0, 0] = 40.0
    assert float(torch.softmax(logits, -1)[0, 0]) == 1.0
    assert float(sf.softmax(logits)[0, 0]) == 1.0  # ours is no better, and cannot be
    # The log-probability keeps what the probability cannot, which is what the fix relies on.
    assert float(sf.target_log_prob(logits, torch.tensor([0]))) < 0.0


# --- bug two: Softmax Cancellation -----------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_a_matched_student_gets_no_gradient(dtype):
    torch.manual_seed(0)
    teacher = torch.randn(4, 4096) * 3
    cached = torch.softmax(teacher.to(dtype), -1).float()

    student = teacher.clone().requires_grad_(True)
    naive = (torch.softmax(student, -1) - cached).detach()
    assert float(naive.abs().sum()) > 1e-4, "the naive route really does invent a gradient"

    fixed = torch.autograd.grad(sf.kl_divergence(student, teacher, reduction="sum"), student)[0]
    assert float(fixed.abs().sum()) == 0.0


def test_a_matched_student_has_zero_loss():
    torch.manual_seed(0)
    teacher = torch.randn(4, 1000) * 3
    assert float(sf.kl_divergence(teacher, teacher)) == 0.0


def test_a_shifted_student_is_still_a_match():
    """Softmax ignores a constant added to every logit, and so must the gradient."""
    torch.manual_seed(0)
    teacher = torch.randn(2, 500) * 2
    student = (teacher + 3.0).requires_grad_(True)
    grad = torch.autograd.grad(sf.kl_divergence(student, teacher + 3.0, reduction="sum"), student)[0]
    assert float(grad.abs().sum()) == 0.0


def test_the_distillation_gradient_is_right_when_the_student_has_drifted():
    torch.manual_seed(0)
    teacher = torch.randn(4, 2000) * 2
    student = (teacher + torch.randn(4, 2000) * 0.3).requires_grad_(True)
    mine = torch.autograd.grad(sf.kl_divergence(student, teacher, reduction="sum"), student)[0]
    truth = torch.softmax(student.detach().double(), -1) - torch.softmax(teacher.double(), -1)
    assert torch.allclose(mine.double(), truth, atol=1e-7)


def test_distillation_falls_back_when_the_student_is_far_away():
    torch.manual_seed(0)
    teacher = torch.randn(2, 500)
    student = (teacher + torch.randn(2, 500) * 40).requires_grad_(True)
    mine = torch.autograd.grad(sf.kl_divergence(student, teacher, reduction="sum"), student)[0]
    truth = torch.softmax(student.detach().double(), -1) - torch.softmax(teacher.double(), -1)
    assert torch.isfinite(mine).all()
    assert torch.allclose(mine.double(), truth, atol=1e-6)


def test_a_matched_student_does_not_drift_over_training():
    torch.manual_seed(0)
    teacher = torch.randn(8, 1000) * 3
    student = teacher.clone().requires_grad_(True)
    optimiser = torch.optim.Adam([student], lr=1e-3)
    for _ in range(50):
        loss = sf.kl_divergence(student, teacher)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    assert float((student.detach() - teacher).abs().max()) == 0.0


# --- the diagnostic --------------------------------------------------------------------------


def test_the_report_notices_a_collapsed_batch():
    torch.manual_seed(0)
    logits = torch.randn(64, 2000)
    logits[torch.arange(64), 0] += 40.0
    report = sf.collapse_report(logits, torch.zeros(64, dtype=torch.long))
    assert report["collapsed"] == 64
    assert report["collapsed_fraction"] == 1.0
    assert report["max_relative_error"] == pytest.approx(1.0)
    assert report["max_target_lead"] > 25


def test_the_report_is_quiet_on_an_ordinary_batch():
    torch.manual_seed(0)
    logits = torch.randn(64, 2000)
    report = sf.collapse_report(logits, torch.randint(0, 2000, (64,)))
    assert report["collapsed"] == 0
    assert report["max_relative_error"] < 1e-3


# --- it has to actually train ----------------------------------------------------------------


def test_it_trains_a_small_classifier():
    torch.manual_seed(0)
    features = torch.randn(256, 20)
    weight = torch.randn(20, 5)
    labels = (features @ weight).argmax(-1)
    model = torch.nn.Linear(20, 5)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(300):
        loss = sf.cross_entropy(model(features), labels)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    assert float((model(features).argmax(-1) == labels).float().mean()) > 0.9

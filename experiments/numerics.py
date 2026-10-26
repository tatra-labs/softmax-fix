"""The two bugs, measured, with no training involved.

    python experiments/numerics.py

Bug one, Softmax Collapse: how far the target logit can lead before the standard implementation
stops producing any gradient for it at all, and what the true gradient still is at that point.

Bug two, Softmax Cancellation: a student whose logits exactly equal its teacher's, which must have
gradient zero, and does not.
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import softmax_fix as sf  # noqa: E402

FLOAT32_SMALLEST = 1.4012984643e-45  # smallest positive subnormal


def collapse_table(classes, leads, device):
    """Standard versus fixed target gradient, as the target pulls ahead."""
    rows = []
    for lead in leads:
        logits = torch.zeros(1, classes, device=device)
        logits[0, 0] = lead
        target = torch.zeros(1, dtype=torch.long, device=device)

        standard = logits.clone().requires_grad_(True)
        grad_standard = torch.autograd.grad(
            F.cross_entropy(standard, target), standard
        )[0]
        fixed = logits.clone().requires_grad_(True)
        grad_fixed = torch.autograd.grad(sf.cross_entropy(fixed, target), fixed)[0]

        truth = float(torch.expm1(sf.target_log_prob(logits.double(), target)))
        rows.append(
            {
                "lead": lead,
                "standard": float(grad_standard[0, 0]),
                "fixed": float(grad_fixed[0, 0]),
                "true": truth,
                "standard_sum": float(grad_standard.double().sum()),
                "fixed_sum": float(grad_fixed.double().sum()),
                "representable": abs(truth) > FLOAT32_SMALLEST,
            }
        )
    return rows


def cancellation_table(classes, device, seed=0):
    """A matched student, with the teacher's probabilities cached at various precisions."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    logits = (torch.randn(4, classes, generator=generator) * 3.0).to(device)
    rows = []
    for name, dtype in (("float32", torch.float32), ("bfloat16", torch.bfloat16), ("float16", torch.float16)):
        cached = torch.softmax(logits.to(dtype), dim=-1).float()
        student = logits.clone().requires_grad_(True)
        naive = (torch.softmax(student, dim=-1) - cached).detach()
        fixed = torch.autograd.grad(
            sf.kl_divergence(student, logits, reduction="sum"), student
        )[0]
        rows.append(
            {
                "teacher_cached_as": name,
                "naive_l1": float(naive.abs().sum()),
                "naive_max": float(naive.abs().max()),
                "fixed_l1": float(fixed.abs().sum()),
                "fixed_max": float(fixed.abs().max()),
            }
        )
    return rows


def drift_table(classes, device, steps=400, seed=0):
    """Let a matched student keep training against its own teacher and see whether it moves."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    teacher = (torch.randn(8, classes, generator=generator) * 3.0).to(device)
    cached = torch.softmax(teacher.bfloat16(), dim=-1).float()
    out = {}
    for name in ("naive", "fixed"):
        student = teacher.clone().requires_grad_(True)
        optimiser = torch.optim.Adam([student], lr=1e-3)
        for _ in range(steps):
            if name == "naive":
                loss = F.kl_div(
                    torch.log_softmax(student, dim=-1), cached, reduction="batchmean"
                )
            else:
                loss = sf.kl_divergence(student, teacher)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        out[name] = {
            "logit_drift": float((student.detach() - teacher).abs().max()),
            "kl_from_teacher": float(
                sf.kl_divergence(student.detach(), teacher, reduction="mean")
            ),
        }
    return out


def plot(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), dpi=160)
    leads = [row["lead"] for row in rows]
    floor = 1e-45

    axes[0].semilogy(leads, [max(abs(row["true"]), floor) for row in rows],
                     color="#333333", lw=1.6, label="true |gradient|")
    axes[0].semilogy(leads, [max(abs(row["standard"]), floor) for row in rows],
                     color="#c2492f", lw=1.6, marker="o", ms=3, label="standard")
    axes[0].semilogy(leads, [max(abs(row["fixed"]), floor) for row in rows],
                     color="#2f6fc2", lw=1.2, ls="--", marker="x", ms=4, label="fixed")
    axes[0].axhline(FLOAT32_SMALLEST, color="#888888", lw=0.8, ls=":")
    axes[0].text(leads[-1], FLOAT32_SMALLEST * 4, "smallest float32 ", fontsize=7,
                 color="#666666", ha="right")
    axes[0].set_ylim(floor / 10, 10)
    axes[0].set_xlabel("how far the target logit leads the runner-up")
    axes[0].set_ylabel("|gradient on the target logit|")
    axes[0].set_title("the standard gradient falls off a cliff", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")

    axes[1].semilogy(leads, [max(abs(row["true"]), floor) for row in rows],
                     color="#333333", lw=2.6, alpha=0.3, label="the missing gradient")
    axes[1].semilogy(leads, [max(abs(row["standard_sum"]), floor) for row in rows],
                     color="#c2492f", lw=1.6, marker="o", ms=3, label="standard")
    axes[1].semilogy(leads, [max(abs(row["fixed_sum"]), floor) for row in rows],
                     color="#2f6fc2", lw=1.6, marker="x", ms=4, label="fixed")
    axes[1].set_ylim(floor / 10, 10)
    axes[1].set_xlabel("how far the target logit leads the runner-up")
    axes[1].set_ylabel("|sum of the gradient|")
    axes[1].set_title("the standard sum misses by exactly what it dropped", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)
    for one in axes:
        one.spines[["top", "right"]].set_visible(False)
        one.grid(alpha=0.25, linewidth=0.5)
    figure.suptitle("Softmax Collapse, float32, 1000 classes", fontsize=11)
    figure.tight_layout()
    figure.savefig(path)
    print("wrote %s" % path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--classes", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="results/numerics.json")
    parser.add_argument("--figure", default="figures/numerics.png")
    args = parser.parse_args()

    leads = list(range(0, 105, 5))
    rows = collapse_table(args.classes, leads, args.device)

    print("Softmax Collapse — float32, %d classes" % args.classes)
    print("%6s %14s %14s %14s %12s" % ("lead", "standard", "fixed", "true", "repr'able"))
    for row in rows:
        if row["lead"] % 10:
            continue
        print("%6d %14.4e %14.4e %14.4e %12s" % (
            row["lead"], row["standard"], row["fixed"], row["true"],
            "yes" if row["representable"] else "no"))

    lost = [row for row in rows if row["standard"] == 0 and row["representable"]]
    if lost:
        print("\nthe standard gradient is exactly zero from a lead of %d onwards," % lost[0]["lead"])
        print("while float32 can still represent the true value up to a lead of %d." % lost[-1]["lead"])

    print("\nSoftmax Cancellation — student logits identical to the teacher's")
    print("%18s %13s %13s %13s" % ("teacher cached as", "naive L1", "fixed L1", "naive/fixed"))
    cancellation = cancellation_table(args.classes, args.device)
    for row in cancellation:
        ratio = row["naive_l1"] / row["fixed_l1"] if row["fixed_l1"] else float("inf")
        print("%18s %13.3e %13.3e %13s" % (
            row["teacher_cached_as"], row["naive_l1"], row["fixed_l1"],
            "inf" if ratio == float("inf") else "%.0fx" % ratio))

    print("\nDrift — a matched student trained 400 more steps against its own teacher")
    drift = drift_table(args.classes, args.device)
    for name, row in drift.items():
        print("%8s  max logit drift %.4e   KL from teacher %.4e" % (
            name, row["logit_drift"], row["kl_from_teacher"]))

    payload = {"collapse": rows, "cancellation": cancellation, "drift": drift}
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(payload, indent=2) + "\n")
        print("\nwrote %s" % args.out)
    if args.figure:
        os.makedirs(os.path.dirname(args.figure) or ".", exist_ok=True)
        plot(rows, args.figure)


if __name__ == "__main__":
    main()

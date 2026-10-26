"""Grokking on modular addition, with the standard cross-entropy and with the fixed one.

This is the setting where the bug has somewhere to show itself. A small transformer memorises
`(a + b) mod p` long before it generalises, and in between the logits grow until the target leads
the runner-up by tens of nats — exactly the regime where the standard implementation rounds the
target's gradient to zero and the gradient stops summing to zero.

    python experiments/grokking.py --steps 15000                 # both losses, with weight decay
    python experiments/grokking.py --weight-decay 0 --steps 15000

Everything is generated; nothing is downloaded. One run is a couple of minutes on a small GPU.
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import softmax_fix as sf  # noqa: E402


class Transformer(nn.Module):
    """One block, which is all modular addition needs."""

    def __init__(self, modulus, width=128, heads=4):
        super().__init__()
        self.embed = nn.Embedding(modulus + 1, width)
        self.position = nn.Parameter(torch.zeros(3, width))
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True, bias=False)
        self.norm_one = nn.LayerNorm(width)
        self.norm_two = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
        self.head = nn.Linear(width, modulus, bias=False)

    def forward(self, tokens):
        hidden = self.embed(tokens) + self.position
        normed = self.norm_one(hidden)
        attended, _ = self.attention(normed, normed, normed, need_weights=False)
        hidden = hidden + attended
        hidden = hidden + self.mlp(self.norm_two(hidden))
        return self.head(hidden[:, -1])


def dataset(modulus, train_fraction, seed, device):
    a, b = torch.meshgrid(torch.arange(modulus), torch.arange(modulus), indexing="ij")
    a, b = a.reshape(-1), b.reshape(-1)
    tokens = torch.stack([a, b, torch.full_like(a, modulus)], dim=-1)
    labels = (a + b) % modulus
    order = torch.randperm(tokens.shape[0], generator=torch.Generator().manual_seed(seed))
    cut = int(train_fraction * tokens.shape[0])
    train, validation = order[:cut], order[cut:]
    return (
        tokens[train].to(device),
        labels[train].to(device),
        tokens[validation].to(device),
        labels[validation].to(device),
    )


def accuracy(logits, labels):
    return float((logits.argmax(-1) == labels).float().mean())


def run(loss_name, args):
    torch.manual_seed(args.seed)
    device = args.device
    train_x, train_y, val_x, val_y = dataset(args.modulus, args.train_fraction, args.seed, device)
    model = Transformer(args.modulus, args.width).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay,
        eps=args.eps,
    )
    loss_fn = sf.cross_entropy if loss_name == "fixed" else F.cross_entropy

    history = []
    started = time.time()
    for step in range(args.steps + 1):
        model.train()
        logits = model(train_x)
        loss = loss_fn(logits, train_y)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

        if step % args.every == 0:
            model.eval()
            with torch.no_grad():
                train_logits = model(train_x)
                val_logits = model(val_x)
            report = sf.collapse_report(train_logits, train_y)
            history.append(
                {
                    "step": step,
                    "train_loss": float(loss.detach()),
                    "train_acc": accuracy(train_logits, train_y),
                    "val_acc": accuracy(val_logits, val_y),
                    "val_loss": float(sf.cross_entropy(val_logits, val_y)),
                    "max_logit": report["max_logit"],
                    "target_lead": report["max_target_lead"],
                    "collapsed": report["collapsed_fraction"],
                    "gradient_sum": report["gradient_sum_max"],
                }
            )
            if args.verbose and step % (args.every * 20) == 0:
                last = history[-1]
                print(
                    "  %-8s step %6d  train %.3f  val %.3f  |logit| %6.1f  collapsed %5.1f%%"
                    % (
                        loss_name,
                        step,
                        last["train_acc"],
                        last["val_acc"],
                        last["max_logit"],
                        100 * last["collapsed"],
                    ),
                    flush=True,
                )
    return {"loss": loss_name, "history": history, "seconds": time.time() - started}


def summarise(result, threshold):
    history = result["history"]
    grokked = next((row["step"] for row in history if row["val_acc"] >= threshold), None)
    memorised = next((row["step"] for row in history if row["train_acc"] >= 0.99), None)
    final = history[-1]
    peak = max(row["val_acc"] for row in history)
    # A slingshot is a training loss that jumps after the model had already fitted the train set.
    after = [row for row in history if memorised is not None and row["step"] > memorised]
    spikes = 0
    for earlier, later in zip(after, after[1:]):
        if later["train_loss"] > max(10 * earlier["train_loss"], 1e-3):
            spikes += 1
    return {
        "loss": result["loss"],
        "memorised_at": memorised,
        "grokked_at": grokked,
        "final_val_acc": final["val_acc"],
        "peak_val_acc": peak,
        "val_acc_drop_from_peak": peak - final["val_acc"],
        "slingshot_spikes": spikes,
        "final_max_logit": final["max_logit"],
        "final_collapsed_fraction": final["collapsed"],
        "final_gradient_sum": final["gradient_sum"],
        "seconds": result["seconds"],
    }


def aggregate(name, rows):
    """Average a summary over seeds. A quantity missing in any seed averages to None."""
    def mean(key):
        values = [row[key] for row in rows if row[key] is not None]
        return sum(values) / len(values) if len(values) == len(rows) else None

    out = {"loss": name, "seeds": len(rows)}
    for key in rows[0]:
        if key != "loss":
            out[key] = mean(key)
    out["grokked_in_all_seeds"] = all(row["grokked_at"] is not None for row in rows)
    return out


def plot(results, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"standard": "#c2492f", "fixed": "#2f6fc2"}
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), dpi=160)
    for result in results:
        history = result["history"]
        steps = [row["step"] for row in history]
        colour = colours[result["loss"]]
        axes[0].plot(steps, [row["val_acc"] for row in history], color=colour, lw=1.4, label=result["loss"])
        axes[0].plot(steps, [row["train_acc"] for row in history], color=colour, lw=0.9, alpha=0.35)
        axes[1].plot(steps, [max(row["train_loss"], 1e-12) for row in history], color=colour, lw=1.2)
        axes[2].plot(steps, [100 * row["collapsed"] for row in history], color=colour, lw=1.4)

    axes[0].set_ylabel("accuracy")
    axes[0].set_title("validation (bold) and train (faint)", fontsize=10)
    axes[0].legend(frameon=False, loc="center right")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("train loss")
    axes[1].set_title("train loss — spikes are the slingshot", fontsize=10)
    axes[2].set_ylabel("% of examples")
    axes[2].set_title("examples whose target gradient is gone", fontsize=10)
    for one in axes:
        one.set_xlabel("step")
        one.spines[["top", "right"]].set_visible(False)
        one.grid(alpha=0.25, linewidth=0.5)
    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    figure.savefig(path)
    print("wrote %s" % path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--modulus", type=int, default=97)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--train-fraction", type=float, default=0.4)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=1, help="how many seeds to average")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="results/grokking.json")
    parser.add_argument("--figure", default="figures/grokking.png")
    parser.add_argument("--quiet", dest="verbose", action="store_false")
    args = parser.parse_args()

    base_seed = args.seed
    per_seed, shown = {"standard": [], "fixed": []}, []
    for offset in range(args.seeds):
        args.seed = base_seed + offset
        for name in ("standard", "fixed"):
            result = run(name, args)
            per_seed[name].append(summarise(result, args.threshold))
            if offset == 0:
                shown.append(result)
    args.seed = base_seed

    table = [aggregate(name, rows) for name, rows in per_seed.items()]
    print("")
    print("%d seed(s), weight decay %g" % (args.seeds, args.weight_decay))
    print("")
    print("%-10s %10s %9s %9s %9s %11s %9s" % (
        "loss", "memorised", "grokked", "final val", "peak val", "slingshots", "|logit|"))
    for row in table:
        print("%-10s %10s %9s %9.3f %9.3f %11.1f %9.1f" % (
            row["loss"],
            "%.0f" % row["memorised_at"] if row["memorised_at"] is not None else "never",
            "%.0f" % row["grokked_at"] if row["grokked_at"] is not None else "never",
            row["final_val_acc"],
            row["peak_val_acc"],
            row["slingshot_spikes"],
            row["final_max_logit"],
        ))
    print("")
    print("collapsed examples at the end:  %s" % ", ".join(
        "%s %.1f%%" % (row["loss"], 100 * row["final_collapsed_fraction"]) for row in table))
    print("drop from peak validation:      %s" % ", ".join(
        "%s %.3f" % (row["loss"], row["val_acc_drop_from_peak"]) for row in table))


    payload = {"args": vars(args), "summary": table, "per_seed": per_seed, "runs": shown}
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(payload, indent=2) + "\n")
        print("wrote %s" % args.out)
    if args.figure:
        os.makedirs(os.path.dirname(args.figure) or ".", exist_ok=True)
        plot(shown, args.figure, "Grokking mod %d, weight decay %g" % (args.modulus, args.weight_decay))


if __name__ == "__main__":
    main()

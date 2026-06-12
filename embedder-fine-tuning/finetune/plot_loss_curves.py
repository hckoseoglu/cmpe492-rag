"""Plot training loss (left axis) vs eval retrieval metric (right axis) per model.

One subplot per model, ordered largest model first. Training loss is logged
every `logging_steps` and falls on the LEFT axis; the eval retrieval metric (a
different quantity, [0,1], higher=better, with the off-the-shelf baseline drawn
as a dashed reference) goes on the RIGHT axis. The classic picture: train loss
keeps falling while the eval metric peaks early then declines = overfitting.

Accepts multiple sweep logs (e.g. the original bge-small/MiniLM sweep plus the
later bge-micro run); curves are merged and one subplot is drawn per model.

    python -m finetune.plot_loss_curves \
      --sweep-log experiments/epoch_sweep/sweep.log experiments/epoch_sweep/sweep_micro.log \
      --results-dir experiments/epoch_sweep/results \
      --out experiments/epoch_sweep/plots/loss_curves.png --metric ndcg@10
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_TRAIN_MARK = re.compile(r"===TRAIN\s+([^\s(=]+)")
_TRAIN_LOSS = re.compile(r"'loss':\s*'?([\d.]+)'?.*?'epoch':\s*'?([\d.]+)'?")
_VAL_LOSS = re.compile(r"'eval_loss':\s*'?([\d.]+)'?.*?'epoch':\s*'?([\d.]+)'?")
_NAME_RE = re.compile(r"^(?P<model>.+)_epoch(?P<epoch>\d+)$")

# Approx. param counts (millions) — used to order subplots largest -> smallest.
# Unknown tags fall back to 0 (sorted last).
_MODEL_SIZE = {"bgesmall": 33.4, "minilm": 22.7, "micro": 17.4}


def _size_key(model: str) -> float:
    return _MODEL_SIZE.get(model, 0.0)


def parse_losses(sweep_logs):
    """Return train[model]=[(ep,loss)], val[model]=[(ep,loss)] merged over logs."""
    train: dict[str, list] = {}
    val: dict[str, list] = {}
    for log in sweep_logs:
        current = None
        for line in Path(log).read_text(errors="ignore").splitlines():
            tm = _TRAIN_MARK.search(line)
            if tm:
                current = tm.group(1)
                train.setdefault(current, [])
                continue
            if current is None:
                continue
            for v, e in _VAL_LOSS.findall(line):       # check eval_loss first
                val.setdefault(current, []).append((float(e), float(v)))
            if "eval_loss" not in line:
                for l, e in _TRAIN_LOSS.findall(line):
                    train[current].append((float(e), float(l)))
    return train, val


def load_eval_metric(results_dir: Path, metric: str):
    finetuned: dict[str, dict[int, float]] = {}
    baseline: dict[str, float] = {}
    for path in sorted(results_dir.glob("*.json")):
        m = _NAME_RE.match(path.stem)
        if not m:
            continue
        model, epoch = m.group("model"), int(m.group("epoch"))
        d = json.loads(path.read_text())
        finetuned.setdefault(model, {})[epoch] = d["finetuned"]["overall"][metric]
        baseline[model] = d["baseline"]["overall"][metric]
    return finetuned, baseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-log", nargs="+", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default="ndcg@10")
    args = ap.parse_args()

    train, _ = parse_losses(args.sweep_log)
    finetuned, baseline = load_eval_metric(Path(args.results_dir), args.metric)
    # subplots in descending order of model size (largest model on the left)
    models = sorted((m for m in train if train[m]), key=_size_key, reverse=True)
    if not models:
        raise SystemExit("no training-loss points parsed")

    fig, axes = plt.subplots(1, len(models), figsize=(6.5 * len(models), 5), squeeze=False)
    for ax, model in zip(axes[0], models):
        pts = sorted(train[model])
        ax.plot([e for e, _ in pts], [l for _, l in pts],
                color="tab:blue", lw=1.5, label="training loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("training loss")
        ax.set_xticks([0, 1, 2, 3])
        for b in (1, 2, 3):
            ax.axvline(b, ls=":", color="gray", alpha=0.4)
        ax.grid(alpha=0.2)

        ax2 = ax.twinx()
        if model in finetuned:
            eps = sorted(finetuned[model])
            ax2.plot(eps, [finetuned[model][e] for e in eps],
                     color="tab:orange", marker="o", lw=1.5, label=f"eval {args.metric}")
            ax2.axhline(baseline[model], ls="--", color="tab:orange", alpha=0.6,
                        label=f"baseline {args.metric}")
        ax2.set_ylabel(f"eval {args.metric}", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        ax.set_title(model)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")

    fig.suptitle("Training Loss / Eval Metric by Epoch (descending order of model size)",
                 fontsize=12)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

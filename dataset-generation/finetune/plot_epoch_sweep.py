"""Aggregate epoch-sweep comparison JSONs into a CSV + graphs.

Reads `<results-dir>/<modeltag>_epoch<N>.json` files (each produced by
`evaluate.py --out-name <modeltag>_epoch<N>`), then writes:
  - summary.csv                : one row per (model, epoch, variant)
  - overall_metrics.png        : R@1/R@5/R@10/NDCG@10 vs epoch (both models,
                                 fine-tuned lines with 95% CI error bars +
                                 baseline dashed lines)
  - delta.png                  : (fine-tuned − baseline) vs epoch per metric
  - by_style_recall5.png       : R@5 formal vs informal vs epoch

    python -m finetune.plot_epoch_sweep \
      --results-dir experiments/epoch_sweep/results \
      --out-dir experiments/epoch_sweep/plots
"""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METRICS = ["recall@1", "recall@5", "recall@10", "ndcg@10"]
_NAME_RE = re.compile(r"^(?P<model>.+)_epoch(?P<epoch>\d+)$")


def load_results(results_dir: Path):
    """Return finetuned[model][epoch] = overall-dict, baseline[model] = overall-dict."""
    finetuned: dict[str, dict[int, dict]] = {}
    baseline: dict[str, dict] = {}
    by_style: dict[str, dict[int, dict]] = {}
    for path in sorted(results_dir.glob("*.json")):
        m = _NAME_RE.match(path.stem)
        if not m:
            continue
        model, epoch = m.group("model"), int(m.group("epoch"))
        d = json.loads(path.read_text())
        finetuned.setdefault(model, {})[epoch] = d["finetuned"]["overall"]
        by_style.setdefault(model, {})[epoch] = d["finetuned"]["by_style"]
        baseline[model] = d["baseline"]["overall"]  # identical across epochs
    return finetuned, baseline, by_style


def write_csv(finetuned, baseline, out_path: Path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "epoch", "variant", *METRICS])
        for model in sorted(finetuned):
            w.writerow([model, 0, "baseline", *[round(baseline[model][m], 4) for m in METRICS]])
            for ep in sorted(finetuned[model]):
                w.writerow([model, ep, "finetuned",
                            *[round(finetuned[model][ep][m], 4) for m in METRICS]])
    print(f"wrote {out_path}")


def plot_overall(finetuned, baseline, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = plt.cm.tab10.colors
    for ax, metric in zip(axes.flat, METRICS):
        for i, model in enumerate(sorted(finetuned)):
            eps = sorted(finetuned[model])
            ys = [finetuned[model][e][metric] for e in eps]
            cis = [finetuned[model][e][f"{metric}_ci"] for e in eps]
            yerr = [[y - lo for y, (lo, hi) in zip(ys, cis)],
                    [hi - y for y, (lo, hi) in zip(ys, cis)]]
            c = colors[i % len(colors)]
            ax.errorbar(eps, ys, yerr=yerr, marker="o", capsize=4, color=c,
                        label=f"{model} (fine-tuned)")
            ax.axhline(baseline[model][metric], ls="--", color=c, alpha=0.6,
                       label=f"{model} baseline")
        ax.set_title(metric)
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.set_xticks([1, 2, 3])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Fine-tuned vs baseline retrieval by epoch (NCSA, 95% CI)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_delta(finetuned, baseline, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.tab10.colors
    styles = {"recall@1": "-", "recall@5": "--", "recall@10": ":", "ndcg@10": "-."}
    for i, model in enumerate(sorted(finetuned)):
        eps = sorted(finetuned[model])
        c = colors[i % len(colors)]
        for metric in METRICS:
            deltas = [finetuned[model][e][metric] - baseline[model][metric] for e in eps]
            ax.plot(eps, deltas, marker="o", color=c, ls=styles[metric],
                    label=f"{model} · {metric}")
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Δ (fine-tuned − baseline) by epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Δ metric")
    ax.set_xticks([1, 2, 3])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_by_style(by_style, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.tab10.colors
    for i, model in enumerate(sorted(by_style)):
        eps = sorted(by_style[model])
        c = colors[i % len(colors)]
        for style, ls in (("formal", "-"), ("informal", "--")):
            ys = [by_style[model][e][style]["recall@5"] for e in eps]
            ax.plot(eps, ys, marker="o", color=c, ls=ls, label=f"{model} · {style}")
    ax.set_title("Recall@5 by query style and epoch (fine-tuned)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("recall@5")
    ax.set_xticks([1, 2, 3])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot epoch-sweep results")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    finetuned, baseline, by_style = load_results(results_dir)
    if not finetuned:
        raise SystemExit(f"no *_epoch<N>.json files found in {results_dir}")

    write_csv(finetuned, baseline, out_dir / "summary.csv")
    plot_overall(finetuned, baseline, out_dir / "overall_metrics.png")
    plot_delta(finetuned, baseline, out_dir / "delta.png")
    plot_by_style(by_style, out_dir / "by_style_recall5.png")
    print("done")


if __name__ == "__main__":
    main()

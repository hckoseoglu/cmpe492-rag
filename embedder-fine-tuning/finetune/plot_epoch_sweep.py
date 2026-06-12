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

# Approx. param counts (millions) — order legend/lines largest -> smallest so the
# overall plot and the loss-curve plot list models in the same order.
_MODEL_SIZE = {"bgesmall": 33.4, "minilm": 22.7, "micro": 17.4}


def _size_key(model: str) -> float:
    return _MODEL_SIZE.get(model, 0.0)


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
        for i, model in enumerate(sorted(finetuned, key=_size_key, reverse=True)):
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
    fig.suptitle("Fine-tuned vs Baseline Metrics by Epoch (95% CI)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def _fname(metric: str) -> str:
    return metric.replace("@", "")


def plot_overall_separate(finetuned, baseline, out_dir: Path):
    """One file per metric: <metric> vs epoch, all models + baselines + 95% CI."""
    colors = plt.cm.tab10.colors
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(7, 5))
        for i, model in enumerate(sorted(finetuned, key=_size_key, reverse=True)):
            eps = sorted(finetuned[model])
            ys = [finetuned[model][e][metric] for e in eps]
            cis = [finetuned[model][e][f"{metric}_ci"] for e in eps]
            yerr = [[y - lo for y, (lo, hi) in zip(ys, cis)],
                    [hi - y for y, (lo, hi) in zip(ys, cis)]]
            c = colors[i % len(colors)]
            ax.errorbar(eps, ys, yerr=yerr, marker="o", markersize=9, lw=2.8,
                        elinewidth=2.0, capsize=5, capthick=2.0, color=c,
                        label=f"{model} (fine-tuned)")
            ax.axhline(baseline[model][metric], ls="--", lw=2.2, color=c, alpha=0.7,
                       label=f"{model} baseline")
        ax.set_title(f"{metric} by epoch (95% CI)", fontsize=13)
        ax.set_xlabel("epoch", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_xticks([1, 2, 3])
        ax.tick_params(labelsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        out = out_dir / f"overall_{_fname(metric)}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")


def plot_peak_bars(finetuned, baseline, out_dir: Path, metric: str):
    """Grouped vertical bars: baseline vs peak fine-tuned per model (3 pairs),
    annotated with the relative % improvement and the peak epoch."""
    models = sorted(finetuned, key=_size_key, reverse=True)
    base_vals = [baseline[m][metric] for m in models]
    peak_eps = [max(finetuned[m], key=lambda e: finetuned[m][e][metric]) for m in models]
    peak_vals = [finetuned[m][pe][metric] for m, pe in zip(models, peak_eps)]

    xs = list(range(len(models)))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.bar([x - w / 2 for x in xs], base_vals, w, label="baseline", color="0.7")
    ax.bar([x + w / 2 for x in xs], peak_vals, w, label="fine-tuned (peak)", color="tab:green")

    for x, bv, pv, pe in zip(xs, base_vals, peak_vals, peak_eps):
        ax.text(x - w / 2, bv + 0.004, f"{bv:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(x + w / 2, pv + 0.004, f"{pv:.3f}", ha="center", va="bottom", fontsize=8)
        pct = (pv - bv) / bv * 100.0
        ax.annotate(f"+{pct:.1f}%\n(epoch {pe})",
                    xy=(x + w / 2, pv), xytext=(0, 20), textcoords="offset points",
                    ha="center", fontsize=10, color="tab:green", fontweight="bold")

    labels = [f"{m}\n({_MODEL_SIZE.get(m, '?')}M)" for m in models]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel(metric)
    ax.set_title(f"Peak fine-tuned vs baseline — {metric}")
    vals = base_vals + peak_vals
    ax.set_ylim(min(vals) - 0.06, max(vals) + 0.07)  # truncated axis to show the gap
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    out = out_dir / f"peak_vs_baseline_{_fname(metric)}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


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
    plot_overall_separate(finetuned, baseline, out_dir)
    plot_peak_bars(finetuned, baseline, out_dir, "recall@5")
    plot_peak_bars(finetuned, baseline, out_dir, "ndcg@10")
    plot_delta(finetuned, baseline, out_dir / "delta.png")
    plot_by_style(by_style, out_dir / "by_style_recall5.png")
    print("done")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Even-smaller-model run: bge-micro-v2 (17.4M), 3 epochs with per-epoch snapshots
# AND a held-out validation loss (--eval-loss). Evaluates each epoch on the NCSA
# corpus, drops results into the shared sweep results dir (so the comparison
# plots show all three models), then regenerates every plot — including the new
# train/val-loss curve.
DG="/Users/hikmetcankoseoglu/dev/cmpe492-rag/dataset-generation"
cd "$DG"
source ~/.venvs/.rag/bin/activate
set -euo pipefail

RES="$DG/experiments/epoch_sweep/results"
PLOTS="$DG/experiments/epoch_sweep/plots"
mkdir -p "$RES" "$PLOTS"

TAG="micro"
BASE="TaylorAI/bge-micro-v2"
OUT="checkpoints/exp-$TAG"
EPOCHS=3
BATCH=16
SEQ=512

echo "===TRAIN $TAG ($BASE)==="
python -u -m finetune.train --device mps --loss mnrl \
  --base-model "$BASE" --batch-size "$BATCH" --epochs "$EPOCHS" \
  --max-seq-length "$SEQ" --triplets-dir triplets_filtered \
  --output-dir "$OUT" --save-each-epoch --eval-loss

for e in $(seq 1 "$EPOCHS"); do
  echo "===EVAL $TAG epoch $e==="
  python -u -m finetune.evaluate --run-dir "$OUT" \
    --finetuned-path "$OUT/epoch-$e" --baseline-model "$BASE" \
    --corpus-from-test --max-seq-length "$SEQ" --device mps \
    --out-name "${TAG}_epoch${e}"
  cp "$OUT/results/${TAG}_epoch${e}.json" "$RES/"
done

echo "===PLOT==="
python -u -m finetune.plot_epoch_sweep --results-dir "$RES" --out-dir "$PLOTS"
python -u -m finetune.plot_loss_curves \
  --sweep-log "$DG/experiments/epoch_sweep/sweep.log" "$DG/experiments/epoch_sweep/sweep_micro.log" \
  --results-dir "$RES" --out "$PLOTS/loss_curves.png" --metric ndcg@10
echo "===DONE==="

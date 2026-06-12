#!/usr/bin/env bash
# Epoch sweep: train bge-small + all-MiniLM-L6-v2 for 3 epochs each (saving a
# snapshot per epoch), evaluate each epoch on the NCSA-only corpus, then plot.
# Everything except (model, epoch) is held constant.
DG="/Users/hikmetcankoseoglu/dev/cmpe492-rag/dataset-generation"
cd "$DG"
source ~/.venvs/.rag/bin/activate
set -euo pipefail

RES="$DG/experiments/epoch_sweep/results"
PLOTS="$DG/experiments/epoch_sweep/plots"
mkdir -p "$RES" "$PLOTS"

# "tag base" pairs (bash 3.2-safe — macOS bash has no associative arrays)
MODELS=(
  "bgesmall BAAI/bge-small-en-v1.5"
  "minilm sentence-transformers/all-MiniLM-L6-v2"
)

EPOCHS=3
BATCH=16
SEQ=512

for entry in "${MODELS[@]}"; do
  set -- $entry
  tag="$1"; base="$2"
  out="checkpoints/exp-$tag"
  echo "===TRAIN $tag ($base)==="
  python -u -m finetune.train --device mps --loss mnrl \
    --base-model "$base" --batch-size "$BATCH" --epochs "$EPOCHS" \
    --max-seq-length "$SEQ" --triplets-dir triplets_filtered \
    --output-dir "$out" --save-each-epoch

  for e in $(seq 1 "$EPOCHS"); do
    echo "===EVAL $tag epoch $e==="
    python -u -m finetune.evaluate --run-dir "$out" \
      --finetuned-path "$out/epoch-$e" --baseline-model "$base" \
      --corpus-from-test --max-seq-length "$SEQ" --device mps \
      --out-name "${tag}_epoch${e}"
    cp "$out/results/${tag}_epoch${e}.json" "$RES/"
  done
done

echo "===PLOT==="
python -u -m finetune.plot_epoch_sweep --results-dir "$RES" --out-dir "$PLOTS"
echo "===DONE==="

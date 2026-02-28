from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# cross-encoder/nli-deberta-v3-small is a good balance of speed and accuracy.
# use cross-encoder/nli-deberta-v3-large if you want higher accuracy and don't mind the slower speed.
MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# deberta-v3 NLI outputs 3 logits in this order: contradiction, entailment, neutral
# (this ordering varies by model — always check model card)
LABEL_NAMES = ["contradiction", "entailment", "neutral"]

def classify_pair(premise: str, hypothesis: str) -> dict:
    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = F.softmax(logits, dim=-1).squeeze()

    return {label: round(probs[i].item(), 4) for i, label in enumerate(LABEL_NAMES)}


# 3 pairs to compare:
# - one clear contradiction
# - one subtle / hedged contradiction (real-world papers often look like this)
# - one neutral pair (same topic, no conflict) to see how scores differ
pairs = [
    {
        "label": "Clear contradiction",
        "premise":    "Creatine supplementation significantly increases muscle strength and power output in resistance-trained athletes.",
        "hypothesis": "Creatine supplementation has no effect on muscle strength or power output in athletes.",
    },
    {
        "label": "Subtle / hedged contradiction",
        "premise":    "High-intensity interval training is the most effective method for improving cardiovascular fitness.",
        "hypothesis": "Moderate continuous aerobic exercise produces superior cardiovascular adaptations compared to interval-based protocols.",
    },
    {
        "label": "No contradiction (same topic, compatible claims)",
        "premise":    "Protein intake of 1.6g per kg of bodyweight supports muscle hypertrophy in resistance training.",
        "hypothesis": "Consuming adequate dietary protein is important for muscle growth and recovery after exercise.",
    },
]

print(f"Model: {MODEL_NAME}\n")
print("=" * 70)

for pair in pairs:
    scores = classify_pair(pair["premise"], pair["hypothesis"])

    print(f"\n[{pair['label']}]")
    print(f"  Premise:    {pair['premise']}")
    print(f"  Hypothesis: {pair['hypothesis']}")
    print(f"  Scores:")
    for label, score in scores.items():
        bar = "█" * int(score * 30)
        print(f"    {label:<15} {score:.4f}  {bar}")

    top_label = max(scores, key=scores.get)
    print(f"  --> Prediction: {top_label.upper()}")

print("\n" + "=" * 70)
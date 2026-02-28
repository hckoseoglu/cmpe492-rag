import json
from langsmith import Client


DATASET_PATH = "../test_dataset/dataset.json"
DATASET_NAME = "Leakage Test Dataset"


# ── 1. Load evaluation dataset from JSON ────────────────────────────────────

print("[1/5] Loading evaluation dataset...")
with open(DATASET_PATH, "r") as f:
    eval_data = json.load(f)
print(f"✓ Loaded {len(eval_data)} questions from {DATASET_PATH}")

# Convert to LangSmith examples format
examples = []
for item in eval_data:
    examples.append(
        {
            "inputs": {"question": item["question"]},
            "outputs": {"answer": item["ground_truth"]},
            "metadata": {
                "question_type": item.get("question_type", "unknown"),
                "resource_type": item.get("resource_type", "unknown"),
                "page": item.get("page", "unknown"),
                "pages": item.get("pages", []),
                "resource": item.get("resource", "unknown"),
                "reference_text": item.get("reference_text", "unknown"),
                "reference_texts": item.get("reference_texts", []),
            },
        }
    )

print("[1/3] Setting up LangSmith dataset...")
client = Client()


if not client.has_dataset(dataset_name=DATASET_NAME):
    print(
        f"❌ Dataset '{DATASET_NAME}' not found. Please create it in LangSmith first."
    )
else:
    # Delete all existing examples
    print("[2/3] Clearing existing examples...")
    existing_examples = client.list_examples(dataset_name=DATASET_NAME)
    example_ids = [example.id for example in existing_examples]
    if example_ids:
        client.delete_examples(example_ids=example_ids)
        print(f"  Deleted {len(example_ids)} existing examples")
    else:
        print("  No existing examples found, skipping deletion")


print("[3/3] Uploading new examples...")
client.create_examples(
    dataset_name=DATASET_NAME,
    examples=examples,
)
print(f"✓ Uploaded {len(examples)} examples to dataset '{DATASET_NAME}'")

from datasets import Dataset, DatasetDict
import pandas as pd
import json
import pathlib

BASE = pathlib.Path(__file__).parent.parent


def load_json(name):
    return json.loads((BASE / "data" / name).read_text())


def build_mteb_dataset():
    queries_raw = load_json("queries.json")
    corpus_raw = load_json("corpus.json")
    qrels_raw = load_json("qrels.json")

    corpus_df = pd.DataFrame(
        [{"id": d["id"], "text": d["text"], "title": d["title"]} for d in corpus_raw]
    )
    queries_df = pd.DataFrame(
        [{"id": q["id"], "text": q["text"]} for q in queries_raw]
    )
    qrels_df = pd.DataFrame(qrels_raw)[["query-id", "corpus-id", "score"]]

    DatasetDict({"test": Dataset.from_pandas(corpus_df, preserve_index=False)}).save_to_disk(
        str(BASE / "dataset/corpus")
    )
    DatasetDict({"test": Dataset.from_pandas(queries_df, preserve_index=False)}).save_to_disk(
        str(BASE / "dataset/queries")
    )
    DatasetDict({"test": Dataset.from_pandas(qrels_df, preserve_index=False)}).save_to_disk(
        str(BASE / "dataset/qrels")
    )

    print("Dataset built:")
    print(f"  Corpus : {len(corpus_df)} chunks")
    print(f"  Queries: {len(queries_df)} queries")
    print(f"  Qrels  : {len(qrels_df)} relevance pairs")


if __name__ == "__main__":
    build_mteb_dataset()

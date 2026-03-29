import time
import json
import argparse
import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from ranx import Qrels, Run, evaluate

from turboquant.compressors import TurboQuantCompressorV2

def main():
    parser = argparse.ArgumentParser(description="Benchmark TurboQuant on SciFact")
    parser.add_argument("--mode", type=str, choices=["baseline", "2", "3", "4"], required=True,
                        help="Mode to run: 'baseline' for FP32, or '2', '3', '4' for bit precision")
    args = parser.parse_args()

    # ==========================================
    # Step 1: Download the Data
    # ==========================================
    print("Downloading/Loading SciFact...")
    corpus_ds = load_dataset("mteb/scifact", "corpus")["corpus"] 
    queries_ds = load_dataset("mteb/scifact", "queries")["queries"]
    qrels_ds = load_dataset("mteb/scifact", "default")["test"] 

    print(f"Total Corpus size: {len(corpus_ds['text'])}. Evaluating on full dataset...")
    corpus_texts = corpus_ds["text"]
    corpus_ids = corpus_ds["_id"]
    query_texts = queries_ds["text"]
    query_ids = queries_ds["_id"]

    # ==========================================
    # Step 2: Setup Model on M1 (MPS)
    # ==========================================
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    model_name = "Snowflake/snowflake-arctic-embed-s"
    model = SentenceTransformer(model_name, device=device)

    # ==========================================
    # Step 3: Embed (Both Modes Need the Embeddings First)
    # ==========================================
    print("Embedding Corpus...")
    start_time = time.time()
    # Using batch_size=4 to avoid MPS memory deadlocks
    corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True, device=device, show_progress_bar=True, batch_size=32)
    embedding_time = time.time() - start_time

    print("Embedding Queries...")
    query_embeddings = model.encode(query_texts, convert_to_tensor=True, device=device, show_progress_bar=True, batch_size=4)

    # Convert to standard normalized form for TurboQuant 
    print("Normalizing embeddings...")
    corpus_norm = F.normalize(corpus_embeddings, p=2, dim=1)
    query_norm = F.normalize(query_embeddings, p=2, dim=1)

    # Memory Size
    element_size = corpus_embeddings.element_size() 
    num_elements = corpus_embeddings.nelement()
    baseline_size_mb = (element_size * num_elements) / (1024 ** 2)

    # Prepare Qrels logic
    qrels_dict = {}
    for row in qrels_ds:
        q_id, d_id, score = str(row["query-id"]), str(row["corpus-id"]), int(row["score"])
        if q_id not in qrels_dict: qrels_dict[q_id] = {}
        qrels_dict[q_id][d_id] = score
    qrels = Qrels(qrels_dict)

    print(f"Corpus Embedding Time: {embedding_time:.2f} seconds")
    
    results = {
        "mode": args.mode,
        "corpus_embedding_time_sec": embedding_time,
        "baseline_memory_mb": baseline_size_mb
    }

    # ==========================================
    # Evaluation Loop dependent on Execution Mode
    # ==========================================
    if args.mode == "baseline":
        print("\n--- BASELINE (FP32) ---")
        print(f"Corpus Embeddings Memory Size: {baseline_size_mb:.2f} MB")

        print("\nRetrieving top 100 documents per query...")
        retrieval_start = time.time()
        # using normal dot-product on normalized vectors which is equivalent to cosine similarity
        hits = util.semantic_search(query_norm, corpus_norm, query_chunk_size=100, top_k=100)
        retrieval_time = time.time() - retrieval_start
        print(f"Baseline Retrieval Speed: {retrieval_time:.4f} seconds")

        run_dict = {}
        for i, query_hits in enumerate(hits):
            q_id = str(query_ids[i])
            run_dict[q_id] = {}
            for hit in query_hits:
                d_id = str(corpus_ids[hit['corpus_id']])
                run_dict[q_id][d_id] = float(hit['score'])

        run = Run(run_dict)
        score = evaluate(qrels, run, "ndcg@10", make_comparable=True)
        print(f"Baseline nDCG@10: {score:.4f}")

        # Record metrics for output
        results["retrieval_time_sec"] = retrieval_time
        results["ndcg_10"] = score
        results["memory_mb"] = baseline_size_mb
        results["compression_ratio"] = 1.0

    else:
        bits = int(args.mode)
        print(f"\n--- TURBOQUANT ({bits}-BIT) ---")
        embedding_dim = corpus_norm.shape[1]
        
        print(f"Applying TurboQuant Compression ({bits}-bit)...")
        # We ONLY quantize the corpus (K), the queries (Q) remain FP32/FP16.
        compress_start = time.time()
        compressor = TurboQuantCompressorV2(head_dim=embedding_dim, bits=bits, seed=42, device=device)
        
        c_norm_unsqueezed = corpus_norm.unsqueeze(0).unsqueeze(0)
        compressed_corpus = compressor.compress(c_norm_unsqueezed)
        compress_time = time.time() - compress_start
        print(f"TurboQuant Compression Time: {compress_time:.2f} seconds")

        N, D = corpus_norm.shape
        # Theoretical RAM footprints
        size_indices_mb = (N * D * bits / 8) / (1024**2)
        size_signs_mb = (N * D * 1 / 8) / (1024**2) # QJL
        size_norms_mb = (N * 2) / (1024**2) # FP16 vector magnitudes
        tq_memory_mb = size_indices_mb + size_signs_mb + size_norms_mb
        
        print(f"TurboQuant Corpus Memory Size (Theoretical {bits}-bit): {tq_memory_mb:.2f} MB")
        compression_ratio = baseline_size_mb / tq_memory_mb
        print(f"Compression Ratio: {compression_ratio:.2f}x smaller")

        print("\nRetrieving top 100 documents per query using Asymmetric Search...")
        retrieval_start = time.time()
        q_norm_unsqueezed = query_norm.unsqueeze(0).unsqueeze(0)

        # The asymmetric search estimator takes unquantized queries and compressed keys
        scores = compressor.asymmetric_attention_scores(q_norm_unsqueezed, compressed_corpus)
        scores = scores.squeeze(0).squeeze(0) 

        top_k_scores, top_k_indices = torch.topk(scores, k=100, dim=1)
        retrieval_time_tq = time.time() - retrieval_start
        print(f"TurboQuant Retrieval Speed: {retrieval_time_tq:.4f} seconds")

        tq_run_dict = {}
        for i in range(len(query_ids)):
            q_id = str(query_ids[i])
            tq_run_dict[q_id] = {}
            for j in range(100):
                corpus_idx = int(top_k_indices[i][j])
                d_id = str(corpus_ids[corpus_idx])
                tq_run_dict[q_id][d_id] = float(top_k_scores[i][j])

        tq_run = Run(tq_run_dict)
        tq_score = evaluate(qrels, tq_run, "ndcg@10", make_comparable=True)
        
        print(f"TurboQuant nDCG@10: {tq_score:.4f}")

        # Record metrics for output
        results["retrieval_time_sec"] = retrieval_time_tq
        results["ndcg_10"] = tq_score
        results["memory_mb"] = tq_memory_mb
        results["compression_time_sec"] = compress_time
        results["compression_ratio"] = compression_ratio

    out_file = f"results_{args.mode}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n================ SUMMARY ================")
    print(f"Saved JSON metrics to: {out_file}")
    
if __name__ == "__main__":
    main()

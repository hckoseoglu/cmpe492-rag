from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import torch
import time
from ranx import Qrels, Run, evaluate

# ==========================================
# Step 1: Download the Data
# ==========================================
print("Downloading TREC-COVID...")
# The dataset has subsets: 'corpus', 'queries', and the relations usually sit in a split like 'test'
corpus_ds = load_dataset("mteb/trec-covid", "corpus")["corpus"] 
queries_ds = load_dataset("mteb/trec-covid", "queries")["queries"]
qrels_ds = load_dataset("mteb/trec-covid", "default")["test"] 

# Extract text lists (limit to a subset if you want to test quickly first)
corpus_texts = corpus_ds["text"]
corpus_ids = corpus_ds["_id"]
query_texts = queries_ds["text"]
query_ids = queries_ds["_id"]

# ==========================================
# Step 2: Setup Model on M1 (MPS)
# ==========================================
# This ensures we use Apple Silicon's GPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model_name = "intfloat/multilingual-e5-large-instruct"
model = SentenceTransformer(model_name, device=device)

# ==========================================
# Step 3: Embed & Measure Speed / Size
# ==========================================
print("Embedding Corpus...")
start_time = time.time()
# encode() automatically converts text to tensors
corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True, device=device)
embedding_time = time.time() - start_time

print("Embedding Queries...")
query_embeddings = model.encode(query_texts, convert_to_tensor=True, device=device)

# Measure Size in Memory
element_size = corpus_embeddings.element_size() # Bytes per number (e.g., 4 for float32)
num_elements = corpus_embeddings.nelement()
total_size_mb = (element_size * num_elements) / (1024 ** 2)

print(f"Corpus Embedding Time: {embedding_time:.2f} seconds")
print(f"Corpus Embeddings Memory Size: {total_size_mb:.2f} MB")

# ==========================================
# Step 4: Retrieval (Semantic Search)
# ==========================================
print("Retrieving top 100 documents per query...")
retrieval_start = time.time()
# Computes cosine similarity (or dot product) and gets top k
hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k=100)
retrieval_time = time.time() - retrieval_start
print(f"Retrieval Speed: {retrieval_time:.4f} seconds")

# ==========================================
# Step 5: Manual Evaluation (nDCG@10)
# ==========================================
# Format the ground truth (qrels) for the 'ranx' library
qrels_dict = {}
for row in qrels_ds:
    q_id, d_id, score = str(row["query-id"]), str(row["corpus-id"]), int(row["score"])
    if q_id not in qrels_dict: qrels_dict[q_id] = {}
    qrels_dict[q_id][d_id] = score

# Format our model's predictions (run) for 'ranx'
run_dict = {}
for i, query_hits in enumerate(hits):
    q_id = str(query_ids[i])
    run_dict[q_id] = {}
    for hit in query_hits:
        d_id = str(corpus_ids[hit['corpus_id']])
        run_dict[q_id][d_id] = hit['score']

# Calculate exact nDCG@10
qrels = Qrels(qrels_dict)
run = Run(run_dict)
score = evaluate(qrels, run, "ndcg@10")

print(f"Baseline nDCG@10: {score:.4f}")
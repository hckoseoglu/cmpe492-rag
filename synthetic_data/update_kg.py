# ── step_2b_add_relationships.py ─────────────────────────────────────────────
# Adds only the missing transforms to the existing graph.
# Safe to run on a preloaded KG — won't touch already-populated properties.

import os
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.transforms import (
    EmbeddingExtractor,
    CosineSimilarityBuilder,
    apply_transforms,
)


client = AsyncOpenAI()

llm = llm_factory("gpt-4o-mini", client=client)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", client=client)


# ── Load existing graph ───────────────────────────────────────────────────────

kg = KnowledgeGraph.load("knowledge_graph.json")
print(f"Loaded — {len(kg.nodes)} nodes, {len(kg.relationships)} relationships")


# ── Quick sanity check before running ────────────────────────────────────────
# Inspect one node to confirm what properties already exist and what's missing.

sample = kg.nodes[0]
print(f"\nSample node properties: {list(sample.properties.keys())}")
# Expected: ['page_content', 'document_metadata', 'headlines', 'keyphrases']
# Missing:  'embedding' — which is what we need to add


# ── Run only the two missing transforms ──────────────────────────────────────
# EmbeddingExtractor reads 'page_content' and writes 'embedding' on every node.
# CosineSimilarityBuilder reads 'embedding' and creates relationship edges.
# Both are idempotent — if 'embedding' already exists on a node it gets skipped.

transforms = [
    EmbeddingExtractor(embedding_model=embeddings),
    CosineSimilarityBuilder(threshold=0.85),
]

apply_transforms(kg, transforms=transforms)

print(f"\nDone — {len(kg.nodes)} nodes, {len(kg.relationships)} relationships")

# Relationship count should now be non-zero — that confirms CosineSimilarityBuilder fired.
assert len(kg.relationships) > 0, "No relationships created — check embeddings ran correctly"


# ── Save ──────────────────────────────────────────────────────────────────────

kg.save("knowledge_graph.json")
print("Saved to knowledge_graph.json")
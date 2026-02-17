from ragas.testset.graph import KnowledgeGraph

kg = KnowledgeGraph.load("knowledge_graph.json")

print(f"Nodes        : {len(kg.nodes)}")
print(f"Relationships: {len(kg.relationships)}")

# look at a connected pair in detail
rel = kg.relationships[3]

content_a = rel.source.properties.get("page_content", "")
content_b = rel.target.properties.get("page_content", "")
score = rel.properties.get("cosine_similarity", "N/A")

print(f"\nSimilarity score: {score:.3f}")
print(f"\nNode A:\n{content_a}")
print(f"\nNode B:\n{content_b}")

for label, node in [("Node A", rel.source), ("Node B", rel.target)]:
    p = node.properties
    print(f"\n{label}")
    print("Metadata:", p.get("document_metadata", "N/A"))

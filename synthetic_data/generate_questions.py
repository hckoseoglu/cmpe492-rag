import os
import httpx
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.testset.graph import KnowledgeGraph
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers.multi_hop.abstract import (
    MultiHopAbstractQuerySynthesizer,
)
from ragas.testset.synthesizers.multi_hop.specific import (
    MultiHopSpecificQuerySynthesizer,
)

client = AsyncOpenAI(
    http_client=httpx.AsyncClient(
        timeout=httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=10.0)
    )
)

llm = llm_factory("gpt-4o-mini", client=client)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", client=client)

# 1. Load saved graph
kg = KnowledgeGraph.load("knowledge_graph.json")
print(f"Loaded graph — {len(kg.nodes)} nodes, {len(kg.relationships)} relationships")

# 2. Personas
personas = [
    Persona(
        name="Fitness Newcomer",
        role_description=(
            "An individual with little to no prior experience in structured physical training. "
            "Seeks simplified, actionable advice on foundational 'first steps,' basic exercise "
            "selection, and how to build a sustainable routine without being overwhelmed by "
            "technical jargon or complex physiological mechanisms."
        ),
    )
]

# 3. Query distribution — 75% single-hop, 25% multi-hop total
query_distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=llm, property_name="keyphrases"), 1.00)
    # (MultiHopAbstractQuerySynthesizer(llm=llm),  0.125),
    # (MultiHopSpecificQuerySynthesizer(llm=llm),  0.125),
]

# 4. Generator
generator = TestsetGenerator(
    llm=llm,
    embedding_model=embeddings,
    knowledge_graph=kg,
    persona_list=personas,
)

# 5. Generate — start small with 3 samples (one per synthesizer type) to inspect quality
print("\nGenerating 3 sample questions for inspection...")
synth = SingleHopSpecificQuerySynthesizer(llm=llm)
print(f"property_name: {synth.property_name}")

qualifying = [n for n in kg.nodes if n.properties.get(synth.property_name)]
print(f"Qualifying nodes: {len(qualifying)} / {len(kg.nodes)}")
if qualifying:
    print(f"Sample value: {qualifying[0].properties[synth.property_name]}")
sample = generator.generate(
    testset_size=3,
    query_distribution=query_distribution,
    raise_exceptions=True,
)


sample_df = sample.to_pandas()
# Inspect the number of questions generated
print(f"\nGenerated {len(sample_df)} questions:")
print("\n--- Sample questions by synthesizer type ---")
for _, row in sample_df.iterrows():
    print(f"\nSynthesizer : {row['synthesizer_name']}")
    print(f"Question    : {row['user_input']}")
    print(f"Answer      : {str(row['reference'])[:200]}")

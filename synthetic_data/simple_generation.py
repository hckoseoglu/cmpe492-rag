"""
Ragas Testset Generation - Quick Approach
Generates synthetic test questions from PDF documents for RAG evaluation
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator

# 1. Configuration
PDF_PATHS = [
    "/mnt/project/progression.pdf",  # The Scientific Principles of Hypertrophy book
    "/mnt/project/2005_11401v4.pdf"  # Research paper
]
TESTSET_SIZE = 10
OUTPUT_FILE = "testset_output.csv"


print("Starting testset generation...")
print(f"Target testset size: {TESTSET_SIZE}")
print(f"Number of documents: {len(PDF_PATHS)}")

# 2. Load documents
print("\n" + "="*50)
print("Loading PDF documents...")
print("="*50)

all_docs = []
for pdf_path in PDF_PATHS:
    print(f"\nLoading: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    all_docs.extend(docs)
    print(f"  Loaded {len(docs)} pages")

print(f"\nTotal pages loaded: {len(all_docs)}")

# 3. Setup LLM and embeddings
print("\n" + "="*50)
print("Setting up LLM and embeddings...")
print("="*50)

generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

print("LLM: gpt-4o")
print("Embeddings: OpenAI text-embedding-ada-002")

# 4. Generate testset
print("\n" + "="*50)
print("Generating testset...")
print("="*50)
print("This may take several minutes depending on document size...")

generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embeddings
)

dataset = generator.generate_with_langchain_docs(
    all_docs, 
    testset_size=TESTSET_SIZE
)

# 5. Export results
print("\n" + "="*50)
print("Exporting results...")
print("="*50)

df = dataset.to_pandas()
df.to_csv(OUTPUT_FILE, index=False)

print(f"\nTestset saved to: {OUTPUT_FILE}")
print(f"Total questions generated: {len(df)}")

# 6. Display summary
print("\n" + "="*50)
print("Testset Summary")
print("="*50)

print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst few questions:")
print("-" * 50)

for idx, row in df.head(3).iterrows():
    print(f"\nQuestion {idx + 1}:")
    print(f"  {row.get('user_input', row.get('question', 'N/A'))}")

print("\n" + "="*50)
print("Generation complete!")
print("="*50)
from openai import AsyncOpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.testset.transforms import (
    KeyphrasesExtractor,
    EmbeddingExtractor,
    CosineSimilarityBuilder,
    apply_transforms,
    Parallel,
    default_transforms
)
from ragas.run_config import RunConfig

BOOK_PDF_PATH = "./resources/pt_book.pdf"
PAPER_PDF_PATH = "./resources/progression.pdf"

# 1. Load both PDFs
book_pages = PyPDFLoader(BOOK_PDF_PATH).load()
paper_pages = PyPDFLoader(PAPER_PDF_PATH).load()

# 2. Stamp metadata so sources stay clearly separated
for doc in book_pages:
    doc.metadata["filename"] = "book_excerpt"
    doc.metadata["source_type"] = "book"

for doc in paper_pages:
    doc.metadata["filename"] = "paper"
    doc.metadata["source_type"] = "paper"

# 3. Chunk — we do this manually so HeadlineSplitter is not needed
book_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    length_function=len,
)

paper_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
)

book_chunks = book_splitter.split_documents(book_pages)
paper_chunks = paper_splitter.split_documents(paper_pages)
all_chunks = book_chunks + paper_chunks

print(f"Book chunks  : {len(book_chunks)}")
print(f"Paper chunks : {len(paper_chunks)}")
print(f"Total chunks : {len(all_chunks)}")

# 4. Build the KnowledgeGraph nodes
kg = KnowledgeGraph()

for chunk in all_chunks:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={
                "page_content": chunk.page_content,
                "document_metadata": chunk.metadata,
            },
        )
    )

print(f"\nKnowledgeGraph ready — {len(kg.nodes)} nodes")
print(f"Sample metadata: {kg.nodes[0].properties['document_metadata']}")

# 5. LLM and embeddings
client = AsyncOpenAI()

llm = llm_factory("gpt-4o-mini", client=client)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    client=client,
)

# 6. Transforms — no HeadlineSplitter, which was causing node duplication
transforms = [
    KeyphrasesExtractor(llm=llm, max_num=5),
    EmbeddingExtractor(embedding_model=embeddings),
    CosineSimilarityBuilder(threshold=0.85),
]

run_config = RunConfig(
    max_workers=32,
    timeout=180,
    max_retries=10,
    max_wait=60,
)

print("\nApplying transforms...")
apply_transforms(kg, transforms=transforms, run_config=run_config)

print(f"Nodes        : {len(kg.nodes)}")
print(f"Relationships: {len(kg.relationships)}")

kg.save("knowledge_graph.json")
print("Saved to knowledge_graph.json")
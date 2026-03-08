from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore
from langchain_core.vectorstores import InMemoryVectorStore


CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

EMBEDDING_MODEL = "text-embedding-3-large"

# Parent-child chunking defaults
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 50


def base_chunking(docs_list):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    doc_splits = text_splitter.split_documents(docs_list)
    print(f"  Split into {len(doc_splits)} chunks")
    return doc_splits


def semantic_chunking(doc_list):

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    semantic_chunkser = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        buffer_size=1,
    )
    doc_splits = semantic_chunkser.split_documents(doc_list)
    print(f"  Split into {len(doc_splits)} chunks")
    print(
        "Biggest chunk size (characters):",
        max(len(doc.page_content) for doc in doc_splits),
    )
    print(
        "Smallest chunk size (characters):",
        min(len(doc.page_content) for doc in doc_splits),
    )
    print(
        "Average chunk size (characters):",
        sum(len(doc.page_content) for doc in doc_splits) / len(doc_splits),
    )
    return doc_splits


def parent_child_chunking(
    docs_list,
    embedding_model=EMBEDDING_MODEL,
    child_chunk_size=CHILD_CHUNK_SIZE,
    child_chunk_overlap=CHILD_CHUNK_OVERLAP,
    search_k=16,
):
    """
    Parent-child chunking strategy.

    Parent splitter: SemanticChunker (creates semantically coherent large chunks)
    Child splitter: RecursiveCharacterTextSplitter (creates small chunks for search)

    At query time, child chunks are searched in the vector store, but the
    corresponding parent chunks are returned — giving the LLM richer context.

    Returns a ParentDocumentRetriever ready for .invoke() calls.
    """
    embeddings = OpenAIEmbeddings(model=embedding_model)

    # Step 1: Pre-split into semantic parent chunks
    

    parent_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        buffer_size=1,
    )
    
    
    print("  Splitting into semantic parent chunks...")
    parent_docs = parent_splitter.split_documents(docs_list)
    print(f"  Parent chunks (semantic): {len(parent_docs)}")
    print(
        f"    Biggest parent (chars): {max(len(d.page_content) for d in parent_docs)}"
    )
    print(
        f"    Smallest parent (chars): {min(len(d.page_content) for d in parent_docs)}"
    )
    print(
        f"    Average parent (chars): {sum(len(d.page_content) for d in parent_docs) / len(parent_docs):.0f}"
    )

    # Step 2: Child splitter — small chunks from each parent for precise retrieval
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
    )

    # Vector store for child chunks (this is what gets searched)
    vectorstore = InMemoryVectorStore(embedding=embeddings)

    # Document store for parent chunks (this is what gets returned)
    docstore = InMemoryStore()

    # No parent_splitter passed — each doc in add_documents() is treated
    # as a parent. The child_splitter creates searchable sub-chunks.
    
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        search_kwargs={"k": search_k},
    )

    # Index the pre-split parent docs
    print("  Indexing parent-child documents...")
    retriever.add_documents(parent_docs)

    child_docs = child_splitter.split_documents(parent_docs)
    print(f"  Child chunks (recursive): {len(child_docs)}")
    print(f"    Child chunk size: {child_chunk_size}, overlap: {child_chunk_overlap}")

    return retriever

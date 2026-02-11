import os
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PDF_PATH = "./progression.pdf"
OPENAI_MODEL = "gpt-4.1"
EMBEDDING_MODEL = "text-embedding-3-large"
CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 200  # overlap between consecutive chunks
RETRIEVAL_K = 2  # number of chunks to retrieve per query

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "rag"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

# ---------------------------------------------------------------------------
# 1. Initialise model & embeddings
# ---------------------------------------------------------------------------
model = init_chat_model(OPENAI_MODEL)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vector_store = InMemoryVectorStore(embeddings)

# ---------------------------------------------------------------------------
# 2. Load the PDF
# ---------------------------------------------------------------------------
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()
print(f"Loaded {len(docs)} page(s) from '{PDF_PATH}'")
print(f"Total characters: {sum(len(d.page_content) for d in docs)}")

# ---------------------------------------------------------------------------
# 3. Split into chunks
# ---------------------------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(docs)
print(f"Split into {len(all_splits)} chunks")

# ---------------------------------------------------------------------------
# 4. Index chunks in the vector store
# ---------------------------------------------------------------------------
document_ids = vector_store.add_documents(documents=all_splits)
print(f"Indexed {len(document_ids)} chunks (first 3 IDs: {document_ids[:3]})")

# ---------------------------------------------------------------------------
# 5. Build the agent with dynamic context injection
# ---------------------------------------------------------------------------
SYSTEM_TEMPLATE = """\
You are an expert AI personal trainer. You help users plan workouts,
choose exercises, and follow proper training rules.

IMPORTANT: Always follow the rules and exercise ordering from the
training document. Use the following retrieved context to inform
your answers. If the context does not cover the user's question,
say so honestly rather than guessing.

--- Retrieved Context ---
{context}
"""


@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Retrieve the most relevant chunks and inject them as system context."""
    # Grab the latest user message
    last_query = request.state["messages"][-1].text

    # Semantic search against the indexed PDF
    retrieved_docs = vector_store.similarity_search(last_query, k=RETRIEVAL_K)

    docs_content = "\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}] {doc.page_content}"
        for doc in retrieved_docs
    )

    # Print retrieved context for debugging
    print("\n--- Retrieved Context for Current Query ---")
    for i, doc in enumerate(retrieved_docs, 1):
        print(
            f"[Result {i}] Page {doc.metadata.get('page', '?')}: {doc.page_content[:200]}..."
        )
    print("--- End of Retrieved Context ---\n")

    return SYSTEM_TEMPLATE.format(context=docs_content)


agent = create_agent(model, tools=[], middleware=[prompt_with_context])


# ---------------------------------------------------------------------------
# 6. Interactive loop
# ---------------------------------------------------------------------------
def main():
    print("\n🏋️  AI Personal Trainer (RAG)  — type 'quit' to exit\n")
    while True:
        query = input("You: ").strip()
        if not query or query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        for step in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
        ):
            step["messages"][-1].pretty_print()
        print()  # blank line between turns


if __name__ == "__main__":
    main()

"""
AI Personal Trainer — RAG Agent
================================
Combines:
  1. RAG context injection (dynamic_prompt) — PDF rules are always available
  2. get_exercises tool — agent decides when to fetch exercises
  3. Exercise ordering — agent applies PDF rules (e.g. multi-joint first)

Dependencies:
  pip install langchain langchain-openai langchain-community pypdf pandas openpyxl

Environment:
  OPENAI_API_KEY=sk-...
  TRAINER_PDF_PATH=trainer_rules.pdf      (default)
  EXERCISE_DB_PATH=exercise_db.xlsx       (default)
"""

# Test case scenario: Hamstring, advanced

import os

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal

from exercise_filter import filter_exercises

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PDF_PATH = "./progression.pdf"
EXERCISE_DB_PATH = "./exercise_db.xlsx"
OPENAI_MODEL = "gpt-4.1"
EMBEDDING_MODEL = "text-embedding-3-large"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 2

# Set langsmith tracing
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "rag"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"


# ---------------------------------------------------------------------------
# 1. Initialise models & embeddings
# ---------------------------------------------------------------------------
model = init_chat_model(OPENAI_MODEL)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vector_store = InMemoryVectorStore(embeddings)

# Separate LLM instance for structured extraction (used inside the tool)
extraction_llm = ChatOpenAI(model="gpt-4o")

# ---------------------------------------------------------------------------
# 2. Load & index the PDF
# ---------------------------------------------------------------------------
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()
print(f"Loaded {len(docs)} page(s) from '{PDF_PATH}'")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(docs)
document_ids = vector_store.add_documents(documents=all_splits)
print(f"Indexed {len(all_splits)} chunks into vector store")


# ---------------------------------------------------------------------------
# 3. Intent extraction schema (same as your extract_user_intent)
# ---------------------------------------------------------------------------
class WorkoutIntent(BaseModel):
    """Structured extraction of user's workout preferences."""

    target_muscle: Literal[
        "chest",
        "back",
        "quadriceps",
        "hamstrings",
        "legs",
        "shoulders",
        "biceps",
        "triceps",
        "core",
        "full_body",
    ] = Field(description="Primary muscle group to target")

    experience_level: Literal["beginner", "intermediate", "advanced"] = Field(
        description="User's fitness experience level"
    )

    equipment: Literal["none", "dumbbells", "barbell", "resistance_bands", "gym"] = (
        Field(description="Available equipment for the workout")
    )

    num_exercises: int = Field(
        default=3, description="Number of exercises requested (default: 3)"
    )


# ---------------------------------------------------------------------------
# 4. Define the get_exercises tool
# ---------------------------------------------------------------------------
@tool(response_format="content_and_artifact")
def get_exercises(user_request: str) -> tuple[str, list[dict]]:
    """Fetch exercises based on the user's workout request.

    Use this tool when the user asks for a workout, exercise suggestions,
    or a training plan. The tool parses the user's request to extract
    target muscle, experience level, equipment, and number of exercises,
    then returns matching exercises from the database.

    Args:
        user_request: The user's natural language workout request,
                      e.g. "I want a chest workout with 3 exercises, I am a beginner"
    """
    # Step A: Extract structured intent from free text
    structured_llm = extraction_llm.with_structured_output(WorkoutIntent)
    intent = structured_llm.invoke(user_request)

    print(f"\n  🔎 Parsed intent:")
    print(f"     Muscle: {intent.target_muscle}")
    print(f"     Level:  {intent.experience_level}")
    print(f"     Equip:  {intent.equipment}")
    print(f"     Count:  {intent.num_exercises}")

    # Step B: Filter exercises from the Excel database
    exercises = filter_exercises(
        excel_path=EXERCISE_DB_PATH,
        target_muscle=intent.target_muscle,
        difficulty=intent.experience_level,
        equipment=intent.equipment,
        num_exercises=intent.num_exercises,
    )

    if not exercises:
        return (
            "No exercises found matching the criteria. "
            "Try adjusting the muscle group, difficulty, or equipment.",
            [],
        )

    # Step C: Format the exercises for the agent
    # Include mechanics info so the agent can determine multi-joint vs single-joint
    serialized = "Retrieved exercises (UNORDERED — you must apply ordering rules):\n\n"
    for i, ex in enumerate(exercises, 1):
        serialized += (
            f"{i}. {ex['name']}\n"
            f"   Difficulty: {ex['difficulty']}\n"
            f"   Target Muscle: {ex['target_muscle']}\n"
            f"   Equipment: {ex['equipment']}\n"
            f"   Prime Mover: {ex['prime_mover']}\n"
            f"   Secondary Muscle: {ex['secondary_muscle']}\n"
            f"   Mechanics: {ex['mechanics']}\n"
            f"   Force Type: {ex['force']}\n\n"
        )

    serialized += (
        "IMPORTANT: These exercises are returned in no particular order. "
        "You MUST reorder them according to the training rules from your context "
        "(e.g. multi-joint / compound exercises before single-joint / isolation exercises). "
        "Explain your reasoning for the ordering you choose."
    )

    return serialized, exercises


# ---------------------------------------------------------------------------
# 5. RAG context injection via dynamic_prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert AI personal trainer. You make  workout programs for users,
answer program related or general fitness/training questions.

YOUR CORE RESPONSIBILITIES:

1. ANSWER RULE QUESTIONS: When users ask about training/fitness questions,
ALWAYS use the retrieved context

2. BUILD WORKOUTS: When users request a workout, use the get_exercises tool
   to fetch exercises. After receiving the exercises, you are responsible for creating
   a training program. For the training program you MUST be careful about the following matters:
    1- Exercise ordering

3. ALWAYS REASON TRANSPARENTLY: Before presenting the final workout,
   include a "Reasoning" section where you walk through your ordering
   decisions step by step. For example:
   
   **My Reasoning:**
   - This follows the rule: "[quote the relevant rule from context]"

--- Retrieved Context (from Training Rules PDF) ---
{context}
"""


@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Retrieve the most relevant PDF chunks and inject them as system context."""
    last_query = request.state["messages"][-1].text

    retrieved_docs = vector_store.similarity_search(last_query, k=RETRIEVAL_K)

    docs_content = "\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}] {doc.page_content}"
        for doc in retrieved_docs
    )

    # Print retrieved context for debugging
    print("\n--- Retrieved Context for Current Query ---")
    for i, doc in enumerate(retrieved_docs, 1):
        print(
            f"[Result {i}] Page {doc.metadata.get('page', '?')}: {doc.page_content}..."
        )
    print("--- End of Retrieved Context ---\n")

    return SYSTEM_PROMPT.format(context=docs_content)


# ---------------------------------------------------------------------------
# 6. Create the agent
# ---------------------------------------------------------------------------
tools = [get_exercises]

agent = create_agent(model, tools, middleware=[prompt_with_context])


# ---------------------------------------------------------------------------
# 7. Interactive loop
# ---------------------------------------------------------------------------
def main():
    print("\n🏋️  AI Personal Trainer (RAG + Tools)")
    print("=" * 50)
    print("Ask me about training rules, or request a workout!")
    print("Type 'quit' to exit.\n")

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
        print()


if __name__ == "__main__":
    main()

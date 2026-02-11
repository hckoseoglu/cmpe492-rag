import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from exercise_filter import filter_exercises

DB_FOLDER = "exercise_db_index"

os.environ["LANGSMITH_TRACING"] = "true"


# ============================================
# STEP 1: Define Response Format (Schema)
# ============================================
class WorkoutIntent(BaseModel):
    """Structured extraction of user's workout preferences."""

    target_muscle: Literal[
        "chest",
        "back",
        "hamstrings",
        "quadriceps",
        "shoulders",
        "arms",
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


# ============================================
# STEP 2: Extract User Intent with Structured Output
# ============================================
llm = ChatOpenAI(model="gpt-4o")

# Use with_structured_output to get parsed Pydantic object
extraction_llm = llm.with_structured_output(WorkoutIntent)

# User query
query = "I want to work out chest at home without equipments. Please note that I am a beginner. Can you suggest some exercises for me?"

print(f"🔎 Extracting user intent from: '{query}'...\n")

# Extract structured intent
user_intent = extraction_llm.invoke(query)

print("📊 Extracted Intent:")
print(f"  - Target Muscle: {user_intent.target_muscle}")
print(f"  - Experience Level: {user_intent.experience_level}")
print(f"  - Equipment: {user_intent.equipment}")
print(f"  - Number of Exercises: {user_intent.num_exercises}\n")

# Now using the extracted intent do the filtering
exercises = filter_exercises(
    excel_path="./exercise_db.xlsx",
    target_muscle=user_intent.target_muscle,
    difficulty=user_intent.experience_level,
    equipment=user_intent.equipment,
    num_exercises=user_intent.num_exercises,
)

for e in exercises:
    print(
        f"Exercise: {e['name']}, Difficulty: {e['difficulty']}, Equipment: {e['equipment']}"
    )

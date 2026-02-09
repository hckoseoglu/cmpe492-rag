import pandas as pd
import os
import getpass
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


file_path = "exercise_db.xlsx"

df = pd.read_excel(file_path, sheet_name="Exercises", engine="openpyxl")
df = df.dropna(subset=["Exercise"])

# Convert Rows to LangChain Documents
documents = []


def clean(val):
    return str(val) if pd.notna(val) else "Unknown"


for index, row in df.iterrows():
    # Content is the part that AI reads and understand
    content = (
        f"Exercise Name: {clean(row['Exercise'])}\n"
        f"Target Muscle: {clean(row.get('Target Muscle Group ', 'Unknown'))}\n"
        f"Difficulty: {clean(row.get('Difficulty Level', 'Unknown'))}\n"
        f"Equipment: {clean(row.get('Primary Equipment ', 'None'))}\n"
    )

    # Metadata is the part that helps filtering
    metadata = {
        "source": file_path,
        "row": index,
        "difficulty": clean(row.get("Difficulty Level", "Unknown")),
        "target_muscle": clean(row.get("Target Muscle Group ", "Unknown")),
    }

    doc = Document(page_content=content, metadata=metadata)
    documents.append(doc)

print(f"Processed {len(documents)} exercises.")

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter OpenAI API Key: ")

# Embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

print("Embedding documents... please wait...")
vector_store = FAISS.from_documents(documents, embeddings)

# Save the vectors locally
vector_store.save_local("exercise_db_index")

print("Success! Vector database saved locally.")

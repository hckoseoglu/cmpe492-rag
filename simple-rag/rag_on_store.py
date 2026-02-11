import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

DB_FOLDER = "exercise_db_index"

os.environ["LANGSMITH_TRACING"] = "true"

# Load vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.load_local(
    DB_FOLDER, embeddings, allow_dangerous_deserialization=True
)

query = "I want to work out chest at home without equipments. Please note thaty I am a beginner. Can you suggest some exercises for me?"

print(f"🔎 Searching database for: '{query}'...")

# We ask the vector store for the 3 most similar documents
retrieved_docs = vector_store.similarity_search(query, k=3)

print("Printing retrieved documents:")
for i, doc in enumerate(retrieved_docs):
    print(f"[Result {i+1}]: {doc.page_content.splitlines()[0]}")


print("🤖 Sending data to GPT for an answer...")

# We combine the text of the found documents into one big string
context_text = "\n\n".join([d.page_content for d in retrieved_docs])

# We define the prompt
template = """
You are an AI personal trainer. Help the user by suggesting exercises based on ONLY the following context:

{context}

User: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model="gpt-4o")

# We run the LLM manually
chain = prompt | llm | StrOutputParser()
answer = chain.invoke({"context": context_text, "question": query})

print(answer)

from __future__ import annotations
import os
import sys
import glob
import subprocess
import json
import asyncio
import concurrent.futures
import fitz  # PyMuPDF
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from datasets import load_dataset

from agents import Agent, Runner, function_tool, set_tracing_disabled


load_dotenv()
set_tracing_disabled(True)

# Configuration
PDF_DIR = "pdfs"
RESULTS_DIR = "results"
OUTPUT_FILENAME = "financebench_pageindex_agentic_results.csv"

def get_doc_structure(doc_name):
    # Path is hardcoded ./results from run_pageindex.py
    file_path = os.path.join(RESULTS_DIR, f"{doc_name}_structure.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def extract_pages_text(pdf_path, page_numbers):
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return f"Error opening PDF: {e}"
        
    text = ""
    for page_num in page_numbers:
        # PyMuPDF is 0-indexed
        if 1 <= page_num <= len(doc):
            page_idx = page_num - 1
            page = doc[page_idx]
            text += f"\n--- Page {page_num} ---\n"
            text += page.get_text()
    
    doc.close()
    return text

def main():
    print("Loading dataset...")
    dataset = load_dataset("PatronusAI/financebench", split="train")
    df = dataset.to_pandas()
    df = df[df['dataset_subset_label'] == 'OPEN_SOURCE']
    
    # 1. Filter local trees only
    available_trees = [os.path.basename(f) for f in glob.glob(os.path.join(RESULTS_DIR, "*_structure.json"))]
    valid_doc_names = []
    doc_descriptions = {}
    
    for tree_file in available_trees:
        doc_name = tree_file.replace("_structure.json", "")
        valid_doc_names.append(doc_name)
        
        # Load description
        file_path = os.path.join(RESULTS_DIR, tree_file)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                doc_descriptions[doc_name] = data.get("doc_description", "No description available.")
        except Exception:
            doc_descriptions[doc_name] = "No description available."
    
    print(f"Total available completely generated trees in {RESULTS_DIR}/: {len(valid_doc_names)}")
    
    df['safe_doc_name'] = df['doc_name'].apply(lambda n: "".join([c for c in n if c.isalpha() or c.isdigit() or c in ' -_']).rstrip())
    
    initial_len = len(df)
    df = df[df['safe_doc_name'].isin(valid_doc_names)]
    print(f"Filtered questions to available JSON Trees: Dropped {initial_len - len(df)} questions. Kept {len(df)} questions.")
    
    if len(df) == 0:
        print("No matching JSON trees found.")
        return

    # 3. Agents / Logic setup
    print("\nInitializing Agent Pipeline...")
    
    retrieved_pages_for_current_query = []
    
    @function_tool
    def list_available_documents() -> str:
        """Returns a list of all available document names and their descriptions that you can query."""
        res = "Valid documents available:\n"
        for d in valid_doc_names:
            desc = doc_descriptions.get(d, "No description available.")
            res += f"- Document Name: {d} | Description: {desc}\n"
        return res

    @function_tool
    def get_document_structure(document_name: str) -> str:
        """Get the document's full tree structure (without text) to find relevant sections given a document name."""
        root_data = get_doc_structure(document_name)
        if not root_data:
            return f"Error: Document structure for '{document_name}' not found."
        
        # Extract the array of nodes if it's wrapped in a dict
        nodes_list = root_data.get("structure", root_data) if isinstance(root_data, dict) else root_data
        
        def simplify(nodes):
            safe_nodes = []
            if isinstance(nodes, list):
                for node in nodes:
                    new_node = node.copy()
                    new_node.pop("text", None)
                    if "nodes" in new_node:
                        new_node["nodes"] = simplify(new_node["nodes"])
                    safe_nodes.append(new_node)
            return safe_nodes
        
        safe_structure = simplify(nodes_list)
        return json.dumps(safe_structure)

    @function_tool
    def get_page_content(document_name: str, pages: list[int]) -> str:
        """
        Get the text content of specific pages. 
        Provide a list of integer page numbers, e.g. [5, 6, 7].
        """
        int_pages = []
        for p in pages:
            try:
                int_pages.append(int(p))
            except ValueError:
                pass
        retrieved_pages_for_current_query.extend(int_pages)
        pdf_path = os.path.join(PDF_DIR, f"{document_name}.pdf")
        return extract_pages_text(pdf_path, int_pages)

    AGENT_SYSTEM_PROMPT = """You are an automated document navigator evaluating your ability to find the correct document and correct pages.
TOOL USE:
- Call list_available_documents() if you don't know the exact document name. The user's query might implicitly mention what document it requires.
- Call get_document_structure(document_name) to identify relevant page ranges.
- Call get_page_content(document_name, pages) to fetch the actual text of the pages. Provide pages argument as a list of integers, e.g. [3, 4, 10].
- Answer based ONLY on tool output. Be concise.
"""

    agent = Agent(
        name="PageIndex Evaluator",
        instructions=AGENT_SYSTEM_PROMPT,
        tools=[list_available_documents, get_document_structure, get_page_content],
        model="gpt-4o"
    )

    results = []
    
    print("\nStarting evaluation loop...")
    
    async def run_agent(question):
        run = await Runner.run(agent, question)
        return run.final_output

    # ONLY RUN TOP 2 FOR QUICK TEST/DEBUG if needed, otherwise run all
    # df = df.head(1) 
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating via PageIndex Agent"):
        question = row['question']
        gold_answer = row['answer']
        financebench_id = row['financebench_id']
        expected_doc = row['safe_doc_name']
        
        retrieved_pages_for_current_query.clear()
        
        try:
            try:
                loop = asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    model_answer = pool.submit(asyncio.run, run_agent(question)).result()
            except RuntimeError:
                model_answer = asyncio.run(run_agent(question))

            model_answer = str(model_answer) if model_answer else ""
            predicted_pages = list(set(retrieved_pages_for_current_query))
        except Exception as e:
            print(f"Error processing ID {financebench_id}: {e}")
            model_answer = f"Error: {e}"
            predicted_pages = []

        # Bookkeeping
        results.append({
            "financebench_id": financebench_id,
            "question": question,
            "expected_doc_name": expected_doc,
            "gold_answer": gold_answer,
            "model_answer": model_answer,
            "retrieval_path": str(predicted_pages)
        })

    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILENAME, index=False)
    print(f"\nEvaluation complete processing {len(results)} queries. Exported to {OUTPUT_FILENAME}")

if __name__ == "__main__":
    print("Started main")
    main()

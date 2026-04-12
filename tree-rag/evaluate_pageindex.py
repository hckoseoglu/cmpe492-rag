print("here")
import os
import sys
import glob
import subprocess
import json
import fitz  # PyMuPDF
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
print("before dataset import")
from datasets import load_dataset
print("after dataset import")
from pydantic import BaseModel, Field
from typing import List
print("before langchain import")
from langchain_openai import ChatOpenAI
print(1)
from langchain_core.prompts import PromptTemplate
print(2)
from langchain_core.output_parsers import JsonOutputParser
print("after langchain import")
print("imports done")
load_dotenv()

# Configuration
PDF_DIR = "pdfs"
RESULTS_DIR = "results"
OUTPUT_FILENAME = "financebench_pageindex_results.csv"

# Make sure we use a structured output schema to grab exact lists of page numbers reliably
class ReasonerOutput(BaseModel):
    page_numbers: List[int] = Field(description="A list of integers representing the precise page numbers identified from the tree structure containing the answer.")

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
        # PyMuPDF is 0-indexed. Most tree parsers output 1-indexed representation.
        # PageIndex `page_number` in JSON: is it 1-indexed? Usually yes.
        # We assume `page_numbers` received from LLM is 1-indexed.
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
    
    # 1. Filter local trees only (skip PDF parsing)
    available_trees = [os.path.basename(f) for f in glob.glob(os.path.join(RESULTS_DIR, "*_structure.json"))]
    valid_doc_names = []
    
    # Strip _structure.json to get the doc_names
    for tree_file in available_trees:
        doc_name = tree_file.replace("_structure.json", "")
        valid_doc_names.append(doc_name)
    
    print(f"Total available completely generated trees in {RESULTS_DIR}/: {len(valid_doc_names)}")
    
    # Let's cleanly map the doc_names in df to safe names if they were saved safely
    df['safe_doc_name'] = df['doc_name'].apply(lambda n: "".join([c for c in n if c.isalpha() or c.isdigit() or c in ' -_']).rstrip())
    
    initial_len = len(df)
    df = df[df['safe_doc_name'].isin(valid_doc_names)]
    print(f"Filtered questions to available JSON Trees: Dropped {initial_len - len(df)} questions. Kept {len(df)} questions.")
    
    if len(df) == 0:
        print("No matching JSON trees found.")
        return

    unique_docs = df['safe_doc_name'].unique()
    
    # 2. Structural Tree Construction Loop (SKIPPED)
    print("\nSkipping Structural Tree Construction (User requested to only evaluate pre-built trees).")

    # 3. Agents / Logic setup
    print("\nInitializing GPT-4o reasoner pipeline...")
    
    # LLM Settings
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.01,
        max_tokens=2048,
        max_retries=3
    )
    
    # Reasoner JSON format parsing
    json_parser = JsonOutputParser(pydantic_object=ReasonerOutput)
    
    # Reasoner Prompt - asking LLM to analyze the JSON
    reasoner_prompt_template = """You are an automated document navigator. You have access to a semantic 'Tree Structure' JSON representing the chapters, sections, paragraphs, and page numbers of a document.

Tree Structure Array:
{structure_json}

The user asks: "{question}"

Analyze the tree structure carefully. Based on the "heading", "content_summary", and "page_number" fields provided in each node of the tree, identify specifically which page numbers most likely contain the exact answer to the user's question. Identify exact target pages only.
Return ONLY valid JSON matching the format instructions.

{format_instructions}"""

    reasoner_prompt = PromptTemplate(
        template=reasoner_prompt_template,
        input_variables=["structure_json", "question"],
        partial_variables={"format_instructions": json_parser.get_format_instructions()}
    )
    
    reasoner_chain = reasoner_prompt | llm | json_parser

    # Inference Prompt (Context-First)
    inference_prompt_template = """Context: 
[START OF FILING] 
{extracted_section_text} 
[END OF FILING] 

Based on the context provided above, answer the following question: {question}"""

    inference_prompt = PromptTemplate.from_template(inference_prompt_template)
    inference_chain = inference_prompt | llm
    
    results = []
    
    print("\nStarting evaluation loop...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating via PageIndex Agent"):
        question = row['question']
        gold_answer = row['answer']
        financebench_id = row['financebench_id']
        doc_name = row['safe_doc_name']
        pdf_path = os.path.join(PDF_DIR, f"{doc_name}.pdf")
        
        try:
            # 3A. Node Discovery
            structure = get_doc_structure(doc_name)
            if not structure:
                raise ValueError(f"Tree structure JSON not found for {doc_name}")
                
            # Limit the size of structure json if too large? 
            # run_pageindex.py output contains arrays of nodes. We stringify it directly.
            structure_str = json.dumps(structure, indent=2)
            
            # Request LLM reasoner
            reasoned_output = reasoner_chain.invoke({
                "structure_json": structure_str,
                "question": question
            })
            predicted_pages = reasoned_output.get("page_numbers", [])
            
            if not predicted_pages:
                predicted_pages = [1] # fallback to page 1 if none 
                print(f"[{financebench_id}] Reasoner found no pages. Fallback to 1.")
                
            # 3B. Text Extraction
            extracted_text = extract_pages_text(pdf_path, predicted_pages)
            
            # 3C. Final Generation
            final_response = inference_chain.invoke({
                "extracted_section_text": extracted_text,
                "question": question
            })
            model_answer = final_response.content
            
        except Exception as e:
            print(f"Error processing ID {financebench_id}: {e}")
            model_answer = f"Error: {e}"
            predicted_pages = []

        # 4. Bookkeeping
        results.append({
            "financebench_id": financebench_id,
            "question": question,
            "gold_answer": gold_answer,
            "model_answer": model_answer,
            "retrieval_path": str(predicted_pages)
        })

    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILENAME, index=False)
    print(f"\nEvaluation complete processing {len(results)} queries. Exported to {OUTPUT_FILENAME}")

if __name__ == "__main__":
    print("started main")
    main()

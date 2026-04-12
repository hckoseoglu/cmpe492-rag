# Repository Overview: Tree-Based RAG & Page Index

This repository implements and evaluates a novel "Tree-based RAG" (Retrieval-Augmented Generation) system—referred to as "Page Index"—and benchmarks its performance against  traditional RAG pipelines, generally utilizing the FinanceBench dataset.

## The Logic of Page Index
The core idea behind the "Page Index" is to map out a document semantically rather than just parsing it as flat text. When a document (like a long PDF report) is loaded, the system processes it to generate a **structural JSON tree**. This tree outlines the document's logical hierarchy—such as chapters, headings, sub-sections, and paragraphs—along with their exact page numbers. Instead of holding all the raw text, the nodes in this tree primarily contain higher-level context like "headings" and concise "content_summaries". 

## Tree and Tree-Based RAG
Tree-based RAG relies on this structural index to perform retrieval logically rather than mathematically:
1. **Semantic Navigation:** When a user asks a question, an LLM Reasoner/Agent examines the document's high-level JSON tree structure. 
2. **Page Identification:** By analyzing the hierarchical headings and node summaries, the Agent reasons about where the answer should logically reside and identifies specific target page numbers.
3. **Extraction & Generation:** The system pulls the full text only from those dynamically identified pages and provides it to the final LLM to generate the answer.

## Comparison with Traditional RAG
The repository scripts essentially pit this tree-based approach against standard vector-based pipelines (`evaluate_financebench.py` vs `evaluate_pageindex.py` / `evaluate_pageindex_agentic.py`).

**Traditional RAG:**
- **How it works:** Blindly divides documents into fixed-size character chunks, embeds them into high-dimensional vectors, and stores them in a vector database (like Chroma). Retrieval relies on finding chunks that are mathematically similar to the query.
- **Characteristics:** Fast and well-suited for broad queries, but risks losing the broader context of the document. Arbitrary text splits might break apart heavily contextual information (a common issue in long financial reports).

**Tree-Based (Page Index) RAG:**
- **How it works:** Maintains the document's global architecture. Retrieval is an active agentic task where an LLM reads a "table of contents on steroids" (the index tree) to decide what full pages to read.
- **Characteristics:** Strongly preserves context. Instead of retrieving isolated snippets of text, the generation model receives intact, complete pages that inherently belong to a recognized section of the document. This is particularly advantageous for complex queries where the relationship between sections matters.

# Results

--- Checking/Evaluating Standard RAG ---
Evaluating financebench_rag_results.csv from index 0 out of 118...
Evaluating financebench_rag_results.csv...
Evaluation Complete! Output saved to graded_financebench_rag_results.csv
Final Accuracy: 52/118 (44.07%)
Accuracy for Standard RAG: 44.07%

--- Checking/Evaluating PageIndex (Single Doc) ---
Evaluating financebench_pageindex_single_doc_results.csv from index 0 out of 23...
Evaluating financebench_pageindex_single_doc_results.csv...
Evaluation Complete! Output saved to graded_financebench_pageindex_single_doc_results.csv
Final Accuracy: 18/23 (78.26%)
Accuracy for PageIndex (Single Doc): 78.26%

--- Checking/Evaluating PageIndex (Agentic) ---
Evaluating financebench_pageindex_agentic_results.csv from index 0 out of 23...
Evaluating financebench_pageindex_agentic_results.csv...
Evaluation Complete! Output saved to graded_financebench_pageindex_agentic_results.csv
Final Accuracy: 15/23 (65.22%)
Accuracy for PageIndex (Agentic): 65.22%
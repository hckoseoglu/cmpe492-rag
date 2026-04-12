import os
import time
import requests
import glob
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from datasets import load_dataset
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma 
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

# Configuration
PDF_DIR = "pdfs"
VECTOR_STORE_DIR = "chroma_db"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SKIP_DOWNLOAD = True

if not os.path.exists(PDF_DIR):
    os.makedirs(PDF_DIR)

def download_pdf(doc_url, save_path, max_retries=1):
    """
    Download PDF from URL with retry logic and custom headers to avoid 403.
    """
    import urllib.parse
    import random
    
    if os.path.exists(save_path):
        return True
    
    parsed_url = urllib.parse.urlparse(doc_url)
    domain = parsed_url.netloc
    
    headers = {
        'User-Agent': 'Hikmet Can Koseoglu (student at Bogazici University; contact: hikmetcankoseoglu@bogazici.edu.tr)',
        'Accept-Encoding': 'gzip, deflate',
        'Host': domain
    }
    
    for attempt in range(max_retries):
        try:
            # Add some time between calls
            time.sleep(random.uniform(1.0, 3.0))
            
            # Use streaming with a higher timeout to resolve read timeouts
            response = requests.get(doc_url, headers=headers, stream=True, timeout=60)
            response.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
            print(f"Attempt {attempt + 1} failed for {doc_url}: {e}")
            if attempt == max_retries - 1:
                return False
            time.sleep(5)
    return False

def main():
    print("Loading dataset...")
    # 1. Dataset Access
    dataset = load_dataset("PatronusAI/financebench", split="train")
    
    # Filter for OPEN_SOURCE subset specifically
    df = dataset.to_pandas()
    df = df[df['dataset_subset_label'] == 'OPEN_SOURCE']
    print(f"Loaded {len(df)} OPEN_SOURCE cases.")

    if SKIP_DOWNLOAD:
        # 2. Local PDF Discovery & Filtering
        print("Checking available local PDFs...")
        
        available_pdfs = [os.path.basename(f) for f in glob.glob(os.path.join(PDF_DIR, "*.pdf"))]
        valid_doc_names = []
        
        for pdf_file in available_pdfs:
            doc_name = os.path.splitext(pdf_file)[0]
            valid_doc_names.append(doc_name)
        
        print(f"Total available PDFs in {PDF_DIR}/: {len(valid_doc_names)}")
        
        df['safe_doc_name'] = df['doc_name'].apply(lambda n: "".join([c for c in n if c.isalpha() or c.isdigit() or c in ' -_']).rstrip())
        
        initial_len = len(df)
        df = df[df['safe_doc_name'].isin(valid_doc_names)]
        print(f"Filtered questions to available PDFs: Dropped {initial_len - len(df)} questions. Kept {len(df)} questions.")
        
        if len(df) == 0:
            print("No matching PDFs found.")
            return

        unique_docs = df['safe_doc_name'].unique()
        downloaded_pdfs = [os.path.join(PDF_DIR, f"{n}.pdf") for n in unique_docs]
    else:
        # 2. PDF Downloader
        print("Checking and downloading PDFs...")
        unique_docs = df[['doc_name', 'doc_link']].drop_duplicates()
        print("Length of unique docs is: ", len(unique_docs))
        
        downloaded_pdfs = []
        # Using tqdm for progress tracking over the URLs
        for _, row in tqdm(unique_docs.iterrows(), total=len(unique_docs), desc="Downloading PDFs"):
            doc_name = row['doc_name']
            doc_link = row['doc_link']
            
            # Make the save path safe if doc_name contains any weird characters
            safe_doc_name = "".join([c for c in doc_name if c.isalpha() or c.isdigit() or c in ' -_']).rstrip()
            pdf_path = os.path.join(PDF_DIR, f"{safe_doc_name}.pdf")
            
            success = download_pdf(doc_link, pdf_path)
            if success:
                downloaded_pdfs.append(pdf_path)
            else:
                print(f"Failed to download: {doc_name} from {doc_link}")

        print(f"Total downloaded PDFs available locally: {len(downloaded_pdfs)}")
        
        # Apply the safe doc name to dataframe identically for downstream loop
        df['safe_doc_name'] = df['doc_name'].apply(lambda n: "".join([c for c in n if c.isalpha() or c.isdigit() or c in ' -_']).rstrip())

    # 3. Document Processing & Vector Store
    print("Parsing and indexing PDFs...")
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    # Using Chroma
    vectorstore = Chroma(
        collection_name="financebench", 
        embedding_function=embeddings, 
        persist_directory=VECTOR_STORE_DIR
    )

    # Check if we already have indexed documents to save money and time
    existing_docs = len(vectorstore.get()['ids'])
    if existing_docs == 0:
        docs = []
        for pdf_path in tqdm(downloaded_pdfs, desc="Parsing PDFs via PyMuPDFLoader"):
            try:
                loader = PyMuPDFLoader(pdf_path)
                docs.extend(loader.load())
            except Exception as e:
                print(f"Error parsing {pdf_path}: {e}")
        
        print(f"Extracted {len(docs)} pages globally. Splitting text...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=30,
        )
        splits = text_splitter.split_documents(docs)
        
        print(f"Created {len(splits)} chunks. Adding to Chroma vectorstore...")
        
        # Batch inserting into Chroma to avoid large payload errors from OpenAI
        batch_size = 200
        for i in tqdm(range(0, len(splits), batch_size), desc="Indexing Chunks"):
            vectorstore.add_documents(splits[i:i+batch_size])
    else:
        print(f"Vector store already contains {existing_docs} chunks. Skipping indexing.")

    # 4. Evaluation Loop & Inference
    # Retriever configs (k=10 chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # Model configs
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.01,
        max_tokens=2048,
        max_retries=3  # robust error handling for API calls
    )

    # Context-First Prompt Scheme
    prompt_template = """Context: 
[START OF FILING] 
{retrieved_chunks} 
[END OF FILING] 

Answer the following question: {question}"""
    
    prompt = PromptTemplate.from_template(prompt_template)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    print("Starting evaluation loop...")
    results = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating Q&A with GPT-4o"):
        question = row['question']
        gold_answer = row['answer']
        financebench_id = row['financebench_id']
        
        try:
            # Retriever fetch top 5 relevant chunks
            retrieved_docs = retriever.invoke(question)
            context = format_docs(retrieved_docs)
            
            # Record chunk metadata
            metadata_list = [doc.metadata for doc in retrieved_docs]
            
            # Combine Chain
            chain = prompt | llm | StrOutputParser()
            model_answer = chain.invoke({
                "retrieved_chunks": context,
                "question": question
            })
            
        except Exception as e:
            print(f"API/Execution Error processing id {financebench_id}: {e}")
            model_answer = f"Error: {e}"
            metadata_list = []

        # Bookkeeping
        results.append({
            "financebench_id": financebench_id,
            "question": question,
            "model_answer": model_answer,
            "gold_answer": gold_answer,
            "source_documents_metadata": str(metadata_list)  # Cast to string for CSV
        })

    # 5. Save results to CSV (Bookkeeping)
    results_df = pd.DataFrame(results)
    output_filename = "financebench_rag_results.csv"
    results_df.to_csv(output_filename, index=False)
    print(f"Evaluation complete. Results saved to {output_filename}")

if __name__ == "__main__":
    print("started main")
    main()

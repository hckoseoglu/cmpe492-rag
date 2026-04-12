import os
import time
import asyncio
import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Semaphore to control concurrency rate will be created inside the event loop

async def get_completion_async(prompt, model=None, temperature=0):
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    for i in range(3):
        try:
            messages = [{"role": "user", "content": prompt}]
            if model in ['o3-mini', 'o1-mini']:
                response = await openai_client.chat.completions.create(model=model, messages=messages)
            else:
                response = await openai_client.chat.completions.create(model=model, messages=messages, temperature=temperature)
            await openai_client.close()
            return response.choices[0].message.content
        except Exception as e:
            print(f"API Error: {e}")
            await asyncio.sleep(2 ** i)
            
    await openai_client.close()
    return "Error"

async def check_answer_equivalence(sem, answer, gold_answer, query=None, model="gpt-4o-2024-11-20"):
    query_prompt = f"- Query: {query}" if query else ""

    prompt = f"""
    You are an expert evaluator for AI-generated responses to queries. Your task is to determine whether the AI-generated answer correctly answers the query based on the golden answer provided by a human expert.

    Numerical Accuracy: 
    - Rounding differences should be **ignored** if they do not meaningfully change the conclusion.
    - You can allow some flexibility in accuracy. For example, 1.2 is considered similar to 1.23. Two numbers are considered similar if one can be rounded to the other.
    - Fractions, percentage, and numerics could be considered similar, for example: "11 of 14" is considered equivalent to "79%" and "0.79".

    Evaluation Criteria:
    - If the golden answer or any of its equivalence can be inferred or generated from the AI-generated answer, then the AI-generated answer is considered correct.
    - If any number, percentage, fraction, or figure in the golden answer is not present in the AI-generated answer, but can be inferred or generated from the AI-generated answer or implicitly exist in the AI-generated answer, then the AI-generated answer is considered correct.
    - The AI-generated answer is considered correct if it conveys the same or similar meaning, conclusion, or rationale as the golden answer.
    - If the AI-generated answer is a superset of the golden answer, it is also considered correct.
    - If the AI-generated answer provides a valid answer or reasonable interpretation compared to the golden answer, it is considered correct.
    - If the AI-generated answer contains subjective judgments or opinions, it is considered correct as long as they are reasonable and justifiable compared to the golden answer.

    - Otherwise, the AI-generated answer is incorrect.

    Inputs:
    {query_prompt}
    - AI-Generated Answer: {answer}
    - Golden Answer: {gold_answer}

    Your output should be ONLY a boolean value: `True` or `False`, nothing else.
    """

    async with sem:
        response = await get_completion_async(prompt, model=model)

    if response and "true" in response.lower()[:15]:
        return True
    elif response and "false" in response.lower()[:15]:
        return False
    
    return False

async def evaluate_row(sem, index, row, model):
    query = row['question']
    gold_answer = row['gold_answer']
    answer = row['model_answer']
    is_correct = await check_answer_equivalence(sem, answer, gold_answer, query=query, model=model)
    
    result_row = row.copy()
    result_row['is_correct'] = is_correct
    return index, result_row

async def evaluate_csv_async(input_file, output_file, model="gpt-4o-2024-11-20"):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return None

    # Load previously graded if exists to resume
    if os.path.exists(output_file):
        graded_df = pd.read_csv(output_file)
        start_index = len(graded_df)
        results = graded_df.to_dict('records')
    else:
        start_index = 0
        results = []

    df = pd.read_csv(input_file)
    total = len(df)
    
    if start_index >= total:
        print(f"{input_file} is already fully evaluated.")
        return pd.read_csv(output_file)
        
    print(f"Evaluating {input_file} from index {start_index} out of {total}...")
    
    df_to_process = df.iloc[start_index:]
    
    sem = asyncio.Semaphore(15)
    tasks = []
    for i, row in df_to_process.iterrows():
        tasks.append(evaluate_row(sem, i, row.to_dict(), model))
        
    print(f"Evaluating {os.path.basename(input_file)}...")
    evaluated_rows = await asyncio.gather(*tasks)
    
    # Needs to be sorted by index because gather might return out of order
    evaluated_rows.sort(key=lambda x: x[0])
    
    for _, result_row in evaluated_rows:
        results.append(result_row)
        
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    
    correct_count = results_df['is_correct'].sum()
    print(f"Evaluation Complete! Output saved to {output_file}")
    print(f"Final Accuracy: {correct_count}/{total} ({(correct_count/total)*100:.2f}%)")
    
    return results_df

def evaluate_csv(input_file, output_file, model="gpt-4o-2024-11-20"):
    return asyncio.run(evaluate_csv_async(input_file, output_file, model=model))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default="financebench_rag_results.csv")
    parser.add_argument("--output", "-o", type=str, default="graded_financebench_rag_results.csv")
    parser.add_argument("--model", "-m", type=str, default="gpt-4o-2024-11-20")
    args = parser.parse_args()
    
    evaluate_csv(args.input, args.output, args.model)

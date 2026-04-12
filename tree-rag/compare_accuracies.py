import os
import pandas as pd
import matplotlib.pyplot as plt
from evaluate_results import evaluate_csv

# We define the mappings of the 3 result CSVs we want to evaluate and plot
TARGETS = [
    {
        "name": "Standard RAG",
        "input_csv": "financebench_rag_results.csv",
        "graded_csv": "graded_financebench_rag_results.csv"
    },
    {
        "name": "PageIndex (Single Doc)",
        "input_csv": "financebench_pageindex_single_doc_results.csv",
        "graded_csv": "graded_financebench_pageindex_single_doc_results.csv"
    },
    {
        "name": "PageIndex (Agentic)",
        "input_csv": "financebench_pageindex_agentic_results.csv",
        "graded_csv": "graded_financebench_pageindex_agentic_results.csv"
    }
]

def main():
    names = []
    accuracies = []
    
    for target in TARGETS:
        print(f"\n--- Checking/Evaluating {target['name']} ---")
        # Evaluate or load existing evaluated csv
        graded_df = evaluate_csv(target['input_csv'], target['graded_csv'], model="gpt-4o-2024-11-20")
        
        if graded_df is not None:
            total = len(graded_df)
            if total == 0:
                print(f"Warning: {target['input_csv']} is empty.")
                accuracies.append(0.0)
            else:
                # `is_correct` boolean sum
                correct_count = graded_df['is_correct'].sum()
                acc = (correct_count / total) * 100
                accuracies.append(acc)
                print(f"Accuracy for {target['name']}: {acc:.2f}%")
        else:
            accuracies.append(0.0)
            print(f"Accuracy for {target['name']}: N/A (File missing)")
        names.append(target['name'])
        
    # Plotting
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, accuracies, color=['#4C72B0', '#DD8452', '#55A868'])
    
    plt.title('Comparison of RAG Accuracies on FinanceBench', fontsize=16)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add data labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    plt.tight_layout()
    chart_path = 'accuracies_chart.png'
    plt.savefig(chart_path)
    print(f"\nChart saved successfully to {chart_path}!")

if __name__ == "__main__":
    main()

import json
import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    modes = ["baseline", "4", "3", "2"]
    data = []
    
    for mode in modes:
        filename = f"results_{mode}.json"
        if os.path.exists(filename):
            with open(filename, "r") as f:
                content = json.load(f)
                data.append(content)
        else:
            print(f"Warning: {filename} not found.")

    if not data:
        print("No results found. Double check that results_*.json files exist in this directory.")
        return

    # Extracting standard fields
    labels = [d["mode"].upper() if d["mode"] == "baseline" else f"{d['mode']}-bit TQ" for d in data]
    memory_mb = [d.get("memory_mb", 0) for d in data]
    retrieval_time = [d.get("retrieval_time_sec", 0) for d in data]
    ndcg_10 = [d.get("ndcg_10", 0) for d in data]
    
    colors = ['#4A90E2', '#E15759', '#F28E2B', '#59A14F']
    
    # 1. Memory Profile
    plt.figure(figsize=(8, 6))
    bars1 = plt.bar(labels, memory_mb, color=colors[:len(labels)])
    plt.title('Corpus Hardware Footprint (Memory MB)', fontsize=14, fontweight='bold')
    plt.ylabel('Megabytes (MB)', fontsize=12)
    for i, v in enumerate(memory_mb):
        plt.text(i, v + (max(memory_mb)*0.02), f"{v:.2f}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('memory_results.png', dpi=300, bbox_inches='tight')
    print("Saved 'memory_results.png'")
    plt.close()

    # 2. Retrieval Speed
    plt.figure(figsize=(8, 6))
    bars2 = plt.bar(labels, retrieval_time, color=colors[:len(labels)])
    plt.title('Query Retrieval Speed', fontsize=14, fontweight='bold')
    plt.ylabel('Seconds (Lower is Better)', fontsize=12)
    for i, v in enumerate(retrieval_time):
        plt.text(i, v + (max(retrieval_time)*0.02), f"{v:.4f}s", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('speed_results.png', dpi=300, bbox_inches='tight')
    print("Saved 'speed_results.png'")
    plt.close()

    # 3. Accuracy (nDCG@10)
    plt.figure(figsize=(8, 6))
    bars3 = plt.bar(labels, ndcg_10, color=colors[:len(labels)])
    plt.title('Evaluation Accuracy (nDCG@10)', fontsize=14, fontweight='bold')
    plt.ylabel('nDCG Score (Higher is Better)', fontsize=12)
    for i, v in enumerate(ndcg_10):
        plt.text(i, v + (max(ndcg_10)*0.02), f"{v:.4f}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('ndcg_results.png', dpi=300, bbox_inches='tight')
    print("Saved 'ndcg_results.png'")
    plt.close()

if __name__ == "__main__":
    main()

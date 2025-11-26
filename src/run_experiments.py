"""
run_experiments.py
Quick experiment runner that sweeps methods and keep ratios and writes a CSV summary.
"""

import argparse
import csv
from pathlib import Path
from prune import build_token_graph, prune_graph, linearize
from evaluate import embedding_similarity
from tqdm import tqdm

METHODS = ['pagerank','degree','betweenness','hybrid']
KEEP_RATIOS = [0.9, 0.75, 0.5, 0.25]

def run_one(text, method, keep):
    doc, G = build_token_graph(text)
    keep_nodes = prune_graph(G, method=method, keep_ratio=keep)
    compressed = linearize(doc, keep_nodes, mode='orig')
    sim = embedding_similarity(text, compressed) if compressed else 0.0
    return {
        "method": method,
        "keep": keep,
        "orig_len": len(text.split()),
        "comp_len": len(compressed.split()),
        "sim": sim,
        "compressed": compressed
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--out", type=str, default="experiments.csv")
    args = parser.parse_args()

    p = Path(args.input)
    if not p.exists():
        raise FileNotFoundError(f"{args.input} not found")
    texts = [line.strip() for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]

    rows = []
    for text in tqdm(texts):
        for method in METHODS:
            for keep in KEEP_RATIOS:
                r = run_one(text, method, keep)
                rows.append({
                    "text": text,
                    "method": r["method"],
                    "keep": r["keep"],
                    "orig_len": r["orig_len"],
                    "comp_len": r["comp_len"],
                    "sim": r["sim"],
                    "compressed": r["compressed"]
                })

    # write CSV
    keys = ["text","method","keep","orig_len","comp_len","sim","compressed"]
    with open(args.out, "w", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved experiments to {args.out}")

if __name__ == "__main__":
    main()

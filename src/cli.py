"""
CLI for quick single or batch runs.
"""

import argparse
import json
from pathlib import Path
from prune import build_token_graph, prune_graph, linearize
from evaluate import embedding_similarity, bertscore_ref
from llm import infer
from tqdm import tqdm

def process_text(text, method='pagerank', keep=0.5, linearize_mode='orig', call_llm=False, llm_backend='auto'):
    doc, G = build_token_graph(text)
    keep_nodes = prune_graph(G, method=method, keep_ratio=keep)
    compressed = linearize(doc, keep_nodes, mode=linearize_mode)
    sim = embedding_similarity(text, compressed) if compressed else 0.0
    llm_out = None
    if call_llm:
        # produce a simple task instruction calling the LLM (e.g., summarization)
        prompt_for_llm = compressed
        # you might want to prepend "Summarize:" or appropriate instruction in real experiments
        llm_out = infer(prompt_for_llm, backend=llm_backend, max_tokens=128)
    return {
        "original": text,
        "compressed": compressed,
        "method": method,
        "keep": keep,
        "sim": sim,
        "llm_out": llm_out,
        "orig_len": len(text.split()),
        "comp_len": len(compressed.split())
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to newline-separated prompts file")
    parser.add_argument("--example", type=str, help="Provide single example")
    parser.add_argument("--keep", type=float, default=0.5)
    parser.add_argument("--method", type=str, default="pagerank", choices=['pagerank','degree','betweenness','hybrid'])
    parser.add_argument("--linearize", type=str, default="orig", choices=['orig','sent'])
    parser.add_argument("--out", type=str, default=None, help="Output JSON file")
    parser.add_argument("--call-llm", action="store_true", help="Call LLM on compressed prompt (requires keys or local fallback)")
    parser.add_argument("--llm-backend", type=str, default="auto", choices=['auto','openai','hf','local'])
    args = parser.parse_args()

    texts = []
    if args.input:
        p = Path(args.input)
        if not p.exists():
            raise FileNotFoundError(f"{args.input} not found")
        texts = [line.strip() for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
    elif args.example:
        texts = [args.example]
    else:
        parser.error("Provide --input or --example")

    results = []
    for t in tqdm(texts):
        res = process_text(t, method=args.method, keep=args.keep, linearize_mode=args.linearize, call_llm=args.call_llm, llm_backend=args.llm_backend)
        results.append(res)

    print(json.dumps(results, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding='utf-8')

if __name__ == "__main__":
    main()

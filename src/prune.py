"""
prune.py
Core pipeline: parsing -> dependency graph -> pruning -> linearization
"""

import spacy
import networkx as nx
import math
from typing import Tuple, Set

# load spaCy once
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    raise RuntimeError("SpaCy model not found. Run: python -m spacy download en_core_web_sm") from e

def build_token_graph(text: str) -> Tuple["spacy.tokens.Doc", nx.DiGraph]:
    doc = nlp(text)
    G = nx.DiGraph()
    for sent_id, sent in enumerate(doc.sents):
        for token in sent:
            node_id = token.i
            G.add_node(node_id, text=token.text, idx=token.i, pos=token.pos_, sent_id=sent_id)
            if token.head.i != token.i:
                G.add_edge(token.head.i, token.i, dep=token.dep_)
    return doc, G

def prune_graph(G: nx.DiGraph, method: str = "pagerank", keep_ratio: float = 0.5) -> Set[int]:
    if keep_ratio <= 0:
        return set()
    if keep_ratio >= 1.0:
        return set(G.nodes())

    num_nodes = len(G)
    n_keep = max(1, math.ceil(num_nodes * keep_ratio))

    if method == "pagerank":
        pr = nx.pagerank(G.to_undirected())
        sorted_nodes = sorted(pr.items(), key=lambda x: x[1], reverse=True)
        return {nid for nid, _ in sorted_nodes[:n_keep]}

    if method == "degree":
        deg = dict(G.degree())
        sorted_nodes = sorted(deg.items(), key=lambda x: x[1], reverse=True)
        return {nid for nid, _ in sorted_nodes[:n_keep]}

    if method == "betweenness":
        bt = nx.betweenness_centrality(G.to_undirected())
        sorted_nodes = sorted(bt.items(), key=lambda x: x[1], reverse=True)
        return {nid for nid, _ in sorted_nodes[:n_keep]}

    if method == "hybrid":
        pr = nx.pagerank(G.to_undirected())
        scores = {}
        max_idx = max((G.nodes[n]["idx"] for n in G.nodes()), default=1)
        for n in G.nodes():
            idx = G.nodes[n]["idx"]
            pos_weight = 1.0 - (idx / (max_idx + 1.0)) * 0.25
            scores[n] = pr.get(n, 0.0) * pos_weight
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return {nid for nid, _ in sorted_nodes[:n_keep]}

    raise ValueError(f"Unknown method: {method}")

def linearize(doc: "spacy.tokens.Doc", keep_nodes: Set[int], mode: str = "orig") -> str:
    if not keep_nodes:
        return ""
    if mode == "orig":
        tokens = [t.text for t in doc if t.i in keep_nodes]
        return " ".join(tokens)
    if mode == "sent":
        sents = {}
        for t in doc:
            if t.i in keep_nodes:
                sents.setdefault(t.sent.start, []).append((t.i, t.text))
        parts = []
        for start in sorted(sents.keys()):
            parts.extend([tok for _, tok in sorted(sents[start], key=lambda x: x[0])])
        return " ".join(parts)
    raise ValueError("Unknown linearize mode")

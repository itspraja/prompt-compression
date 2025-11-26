# src/streamlit_app.py
"""
Enhanced Streamlit UI for Prompt Compression via Graph Pruning

Features:
- Clean layout with tabs: Single-run, Batch-run, Sweep experiments, About
- Token highlighting with kept / dropped visual style
- Token importance bar chart (PageRank / degree / hybrid score)
- Interactive graph visualization (networkx + matplotlib)
- Embedding similarity plot and numbers
- Side-by-side LLM outputs (original vs compressed) using llm.infer (supports 'local' backend)
- Download compressed prompts / results as JSON or CSV
- Caching for spaCy and embedding model
"""

import streamlit as st
from prune import build_token_graph, prune_graph, linearize
from evaluate import embedding_similarity, get_embed_model
from llm import infer
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict
from io import BytesIO
import json
import pandas as pd
import os

st.set_page_config(page_title="Prompt Compression", layout="wide")
st.title("Prompt Compression (via Graph Pruning)")

with st.sidebar:
    st.header("Global settings")
    method = st.selectbox("Pruning method", ['pagerank','degree','betweenness','hybrid'])
    keep_ratio = st.slider("Keep ratio (fraction of tokens to keep)", min_value=0.05, max_value=1.0, value=0.5, step=0.05)
    linearize_mode = st.selectbox("Linearize mode", ['orig','sent'])
    llm_backend = st.selectbox("LLM backend (for inference)", ['local','auto','openai','hf'])
    call_llm_default = st.checkbox("Enable LLM comparison on runs by default", value=False)
    st.write("---")
    st.markdown("**Quick actions**")
    st.button("Reset UI")
    st.write("---")
    st.markdown("Project Proposal")
    proposal_path = "./data/Project_Proposal_P24CS0201.pdf"
    if os.path.exists(proposal_path):
        st.download_button("Download proposal", data=open(proposal_path, "rb").read(), file_name="Project_Proposal_P24CS0201.pdf")


@st.cache_data
def compute_importance_scores(_G, method: str):
    """
    Compute and normalize importance scores for nodes in graph _G.
    Leading underscore tells Streamlit *not* to hash that parameter.
    The function still uses the graph `_G` internally.
    """
    G = _G  # keep the local name G for readability
    if len(G) == 0:
        return {}
    H = G.to_undirected()
    if method == "pagerank":
        scores = nx.pagerank(H)
    elif method == "degree":
        scores = dict(H.degree())
    elif method == "betweenness":
        scores = nx.betweenness_centrality(H)
    elif method == "hybrid":
        pr = nx.pagerank(H)
        max_idx = max((G.nodes[n]["idx"] for n in G.nodes()), default=1)
        scores = {}
        for n in G.nodes():
            idx = G.nodes[n]["idx"]
            pos_weight = 1.0 - (idx / (max_idx + 1.0)) * 0.25
            scores[n] = pr.get(n, 0.0) * pos_weight
    else:
        scores = {n: 0.0 for n in G.nodes()}
    # normalize for plotting
    vals = list(scores.values()) if scores else [1.0]
    minv, maxv = min(vals), max(vals)
    if maxv - minv > 0:
        for k in scores:
            scores[k] = (scores[k] - minv) / (maxv - minv)
    return scores


def highlight_tokens_html(doc, keep_nodes: set, scores: Dict[int, float]) -> str:
    """Return HTML string with tokens highlighted. Kept tokens blue, others gray. Size reflects importance."""
    parts = []
    for t in doc:
        text = t.text.replace(" ", "&nbsp;")
        kept = t.i in keep_nodes
        score = scores.get(t.i, 0.0)
        size = 0.85 + 0.6 * score  # scale
        if kept:
            parts.append(f'<span style="background-color:#dff3ff; padding:2px; border-radius:4px; font-size:{size}rem; margin:2px;">{text}</span>')
        else:
            parts.append(f'<span style="color:#777777; padding:2px; font-size:{0.85}rem; margin:2px;">{text}</span>')
    return " ".join(parts)

def plot_graph(G, keep_nodes:set):
    H = G.to_undirected()
    pos = nx.spring_layout(H, seed=42, k=0.6)
    fig, ax = plt.subplots(figsize=(6,5))
    node_colors = ['#1f78b4' if n in keep_nodes else '#cccccc' for n in H.nodes()]
    nx.draw_networkx_nodes(H, pos, node_size=700, node_color=node_colors, ax=ax)
    labels = {n: G.nodes[n]['text'] for n in H.nodes()}
    nx.draw_networkx_labels(H, pos, labels, font_size=9, ax=ax)
    nx.draw_networkx_edges(H, pos, alpha=0.4, ax=ax)
    ax.set_axis_off()
    return fig

def tokens_table(doc, keep_nodes:set, scores:Dict[int,float]):
    rows = []
    for t in doc:
        rows.append({"idx": t.i, "token": t.text, "pos": t.pos_, "kept": t.i in keep_nodes, "score": round(float(scores.get(t.i, 0.0)), 3)})
    return pd.DataFrame(rows)

tabs = st.tabs(["Single Run", "Batch Run", "Sweeps", "About / Help"])

with tabs[0]:
    st.header("Interactive Prompting")
    st.write("Type a prompt, tweak pruning settings, visualize token importance and compare LLM outputs.")

    col_a, col_b = st.columns([2,1])
    with col_a:
        prompt = st.text_area("Prompt", height=200, value="The quick brown fox jumped over the lazy dog")
        run_button = st.button("Run pruning")
        call_llm = st.checkbox("Call LLM now for comparison (overrides sidebar default)", value=call_llm_default)

    with col_b:
        st.markdown("**Quick stats**")
        st.write(f"Pruning method: **{method}**")
        st.write(f"Keep ratio: **{keep_ratio:.2f}**")
        st.write(f"Linearize mode: **{linearize_mode}**")
        st.write("---")
        st.write("Actions")
        st.download_button("Download prompt (text)", data=prompt, file_name="prompt.txt")

    if run_button:
        doc, G = build_token_graph(prompt)
        keep_nodes = prune_graph(G, method=method, keep_ratio=keep_ratio)
        compressed = linearize(doc, keep_nodes, mode=linearize_mode)
        scores = compute_importance_scores(G, method=method)
        sim = embedding_similarity(prompt, compressed) if compressed else 0.0

        st.subheader("Compressed prompt")
        st.code(compressed or "(empty after pruning)")
        st.write(f"Embedding cosine similarity: **{sim:.4f}**")
        st.write(f"Original tokens: {len(prompt.split())}, Compressed tokens: {len((compressed or '').split())}")

        st.markdown("**Token highlighting**")
        html = highlight_tokens_html(doc, keep_nodes, scores)
        st.write(html, unsafe_allow_html=True)

        st.markdown("**Token importance**")
        df_tokens = tokens_table(doc, keep_nodes, scores)
        st.dataframe(df_tokens.style.format({"score":"{:.3f}"}), height=240)

        st.markdown("**Dependency graph**")
        fig = plot_graph(G, keep_nodes)
        st.pyplot(fig)

        # importance bar chart
        st.markdown("**Importance bar chart (top tokens)**")
        topk = df_tokens.sort_values("score", ascending=False).head(12)
        fig2, ax2 = plt.subplots(figsize=(6,2.6))
        ax2.barh(topk['token'][::-1], topk['score'][::-1])
        ax2.set_xlabel("Normalized importance")
        ax2.set_ylabel("Token")
        st.pyplot(fig2)

        # LLM compare
        if call_llm:
            st.markdown("**LLM comparison**")
            with st.spinner("Calling local LLM..."):
                try:
                    # We use original prompt and compressed prompt as two calls.
                    # For summarization tasks you may want to prepend "Summarize:" to compressed.
                    orig_out = infer(prompt, backend=llm_backend)
                except Exception as e:
                    orig_out = f"LLM call failed on original: {e}"
                try:
                    # For compressed prompt, prepend same instruction if compressed looks like a fragment.
                    comp_prompt = compressed if compressed.strip() else compressed
                    comp_out = infer(comp_prompt, backend=llm_backend)
                except Exception as e:
                    comp_out = f"LLM call failed on compressed: {e}"

            left, right = st.columns(2)
            with left:
                st.subheader("Original prompt -> LLM")
                st.text_area("Original LLM output", value=orig_out, height=240)
            with right:
                st.subheader("Compressed prompt -> LLM")
                st.text_area("Compressed LLM output", value=comp_out, height=240)

        # allow download of result JSON
        res = {
            "original": prompt,
            "compressed": compressed,
            "method": method,
            "keep_ratio": keep_ratio,
            "linearize_mode": linearize_mode,
            "similarity": sim,
            "tokens_original": len(prompt.split()),
            "tokens_compressed": len((compressed or "").split())
        }
        st.download_button("Download result (JSON)", data=json.dumps(res, indent=2), file_name="prune_result.json")

with tabs[1]:
    st.header("Batch Processing")
    st.write("Upload a text file with one prompt per line, run pruning over the file, and download results as CSV/JSON.")

    uploaded = st.file_uploader("Upload prompts (.txt, one per line)", type=['txt'])
    batch_call_llm = st.checkbox("Call LLM for each compressed prompt", value=False)
    batch_run = st.button("Run batch")

    if uploaded:
        raw = uploaded.read().decode('utf-8')
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        st.write(f"Loaded {len(lines)} prompts.")
        if batch_run:
            rows = []
            for i, text in enumerate(lines):
                doc, G = build_token_graph(text)
                keep_nodes = prune_graph(G, method=method, keep_ratio=keep_ratio)
                compressed = linearize(doc, keep_nodes, mode=linearize_mode)
                sim = embedding_similarity(text, compressed) if compressed else 0.0
                llm_out = None
                if batch_call_llm:
                    try:
                        llm_out = infer(compressed, backend=llm_backend)
                    except Exception as e:
                        llm_out = f"LLM failed: {e}"
                rows.append({
                    "id": i,
                    "original": text,
                    "compressed": compressed,
                    "sim": sim,
                    "llm_out": llm_out
                })
            df = pd.DataFrame(rows)
            st.dataframe(df[["id","original","compressed","sim"]], height=300)
            csv_bytes = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", data=csv_bytes, file_name="batch_results.csv")
            st.download_button("Download JSON", data=df.to_json(orient="records", indent=2), file_name="batch_results.json")

with tabs[2]:
    st.header("Sweep Experiments")
    st.write("Run a small sweep over methods and keep ratios on a small uploaded prompt file.")

    sweep_file = st.file_uploader("Upload prompts for sweep (.txt)", key="sweep", type=['txt'])
    sweep_run = st.button("Run sweep")
    default_keep_list = [0.9,0.75,0.5,0.25]
    keep_list = st.multiselect("Keep ratios to sweep", options=default_keep_list, default=default_keep_list)
    methods_sel = st.multiselect("Methods to sweep", options=['pagerank','degree','betweenness','hybrid'], default=['pagerank','hybrid'])

    if sweep_run and sweep_file:
        txt = sweep_file.read().decode('utf-8')
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        results = []
        for text in lines:
            for m in methods_sel:
                for k in keep_list:
                    doc, G = build_token_graph(text)
                    keep_nodes = prune_graph(G, method=m, keep_ratio=k)
                    compressed = linearize(doc, keep_nodes, mode=linearize_mode)
                    sim = embedding_similarity(text, compressed) if compressed else 0.0
                    results.append({"original": text, "method": m, "keep": k, "sim": sim, "compressed": compressed})
        df = pd.DataFrame(results)
        st.write("Sweep results (first 100 rows):")
        st.dataframe(df.head(100), height=300)
        st.download_button("Download sweep CSV", data=df.to_csv(index=False).encode('utf-8'), file_name="sweep_results.csv")

        # Simple summary plot (sim vs keep)
        fig, ax = plt.subplots(figsize=(6,3))
        for m in methods_sel:
            sub = df[df['method'] == m]
            if sub.empty:
                continue
            # group by keep and average sim
            agg = sub.groupby('keep')['sim'].mean().reset_index()
            ax.plot(agg['keep'], agg['sim'], marker='o', label=m)
        ax.set_xlabel("Keep ratio")
        ax.set_ylabel("Average embedding similarity")
        ax.set_title("Sweep: similarity vs keep ratio")
        ax.legend()
        st.pyplot(fig)

with tabs[3]:
    st.header("About / Help")
    st.markdown("""
    **What this prompt compressor does**
    - Parses a prompt with spaCy into tokens & dependency graph
    - Prunes tokens by graph centrality (PageRank / degree / betweenness) or a hybrid score
    - Linearizes the kept tokens into a compressed prompt
    - Computes embedding similarity (Sentence-BERT) between original and compressed

    **Tips**
    - Use `keep_ratio` to trade tokens for fidelity. 0.25 keeps fewer tokens (stronger compression).
    - `hybrid` is often robust: it mixes centrality with positional bias.
    - Use the "Sweeps" tab for quick small experiments across multiple configs.
    - Calling LLM locally can be slow. Prefer small sample sizes for demo.

    **Local LLM notes**
    - For this demonstation, using 'local' inference, by downaloading a small model . Ensure you have a small compatible model available if you use `infer(..., backend='local')`.
    - If inference fails, the UI shows an error and continues — no background tasks.

    **Reproducibility**
    - All key functions are in `src/prune.py` and `src/evaluate.py`.
    - For batch experiments, use `src/run_experiments.py`.

    """)
    st.write("---")
    st.write("Credits: Built for the Prompt Compression (via Graph Pruning) demonstration by Rohit Prajapati (P24CS0201).")

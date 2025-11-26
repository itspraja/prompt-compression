# Prompt Compression via Graph Pruning

### A Graph-Theoretic Framework for Efficient Prompt Reduction in LLMs


## Overview

Large Language Models (LLMs) incur significant computational and financial cost when processing long prompts. Most existing compression techniques use heuristic selection or truncation, often at the expense of essential semantic information.
This project presents a graph-based compression pipeline that constructs syntactic dependency graphs from prompts, identifies important tokens using centrality and sparsification techniques, prunes low-impact components, and reconstructs a shorter prompt with minimal loss of meaning. The objective is to reduce inference cost while maintaining high semantic fidelity.


## Key Features

* Dependency graph construction using spaCy
* Centrality-driven pruning (degree, PageRank, token role importance)
* Graph sparsification using spectral techniques
* Text reconstruction from pruned graph representations
* Automatic evaluation using ROUGE, BLEU, cosine similarity, and compression ratio
* Reproducible experimental pipeline with CSV logging
* CLI interface and Streamlit-based interactive dashboard


## Repository Structure

```
.
├── data/
│   └── Project_Proposal_P24CS0201.pdf
├── examples/
│   ├── prompts_batch.txt
│   └── prompts.txt
├── outputs/
│   ├── dependency_graph.png
│   ├── importance_chart_top_token.png
│   └── sweep.png
├── src/
│   ├── prune.py
│   ├── genMetrics.py
│   ├── llm.py
│   ├── evaluate.py
│   ├── run_experiments.py
│   ├── cli.py
│   └── streamlit_app.py
├── experiments.csv
├── requirements.txt
├── setup_env.sh
└── README.md
```

## Installation

Clone and initialize the environment:

```bash
git clone https://github.com/itspraja/prompt-compression.git
cd prompt-comression
bash setup_env.sh
```

Or set up manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## Usage

### Command-Line Interface

Run pruning and evaluation on a single prompt:

```bash
python src/cli.py --input "Your long prompt here"
```

### Experiment Runner

Execute batch experiments and log results to `experiments.csv`:

```bash
python src/run_experiments.py
```

### Streamlit Application

Launch the interactive dashboard:

```bash
streamlit run src/streamlit_app.py
```

The dashboard provides visualization of dependency graphs, token importance, compression–quality trade-offs, and summary statistics.


## Methodology

1. **Dependency Graph Construction**
   Prompts are parsed using a dependency parser (spaCy), where nodes represent tokens and edges represent syntactic relations.

2. **Importance Scoring**
   Token significance is quantified using structural metrics such as degree centrality, PageRank, and linguistic roles (root, subject, object, modifiers).

3. **Graph Pruning**
   Low-importance nodes are removed, or graph sparsification is applied to compress structural complexity while retaining essential information.

4. **Reconstruction**
   The pruned graph is linearized to create a shorter, coherent prompt.

5. **Evaluation**
   Outputs of the original and compressed prompts are compared using ROUGE, BLEU, embedding similarity (Sentence-BERT), and compression ratio.


## Datasets and Metrics

**Datasets**

* CNN/DailyMail
* XSum
* Custom research-oriented prompts

**Evaluation Metrics**

* ROUGE-1 / ROUGE-L
* BLEU
* Cosine similarity of embeddings
* Compression ratio
* Structural graph statistics

All experiments are logged in `experiments.csv` and visual outputs are stored in the `outputs/` directory.


## Expected Outcomes

* Substantial token reduction (30–60% depending on pruning strategy)
* Preservation of core semantics in compressed prompts
* Quantifiable trade-off curves between compression and output quality
* Reduction in inference cost for long-context LLM tasks
* A repeatable and lightweight pipeline suitable for research and production settings


## Roadmap

* Additional centrality and pruning metrics
* Optional evaluator model using LoRA
* Integration with multiple LLM backends
* Language-agnostic dependency parsing
* Multi-graph fusion for context-aware compression


## Citation

```
@misc{prajapati2025promptgraph,
  title={Prompt Compression via Graph Pruning},
  author={Rohit Prajapati},
  year={2025}
}
```

## Contributing

Contributions, suggestions, and extensions are welcome.
Please open an issue or submit a pull request for discussion.


## License

This project is released under the MIT License.


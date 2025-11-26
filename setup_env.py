#!/usr/bin/env bash
set -e
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
echo "Environment ready for Prompt Compression (via Graph Pruning) Project"

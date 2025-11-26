"""
evaluate.py
Embedding similarity + BERTScore wrapper
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from bert_score import score as bert_score

EMBED_MODEL_NAME = "all-mpnet-base-v2"
_EMBED = None

def get_embed_model():
    global _EMBED
    if _EMBED is None:
        _EMBED = SentenceTransformer(EMBED_MODEL_NAME)
    return _EMBED

def embedding_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    model = get_embed_model()
    a_vec = model.encode(a, convert_to_numpy=True)
    b_vec = model.encode(b, convert_to_numpy=True)
    denom = (np.linalg.norm(a_vec) * np.linalg.norm(b_vec) + 1e-12)
    return float((a_vec @ b_vec) / denom)

def bertscore_ref(refs, cands, lang='en'):
    """
    refs: list of reference strings
    cands: list of candidate strings
    returns list of F1 scores
    """
    if not refs or not cands:
        return []
    P, R, F1 = bert_score(cands=cands, refs=refs, lang=lang, rescale_with_baseline=True)
    return [float(f) for f in F1]

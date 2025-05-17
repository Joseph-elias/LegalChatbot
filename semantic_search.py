import json
import os
import torch
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from rank_bm25 import BM25Okapi
from sentence_transformers import util, models, SentenceTransformer

# ── A) Setup embedder ──────────────────────────────────────────────────────
# Choose your model: base or large
#MODEL_NAME = "aubmindlab/bert-large-arabertv2"
MODEL_NAME = "asafaya/bert-base-arabic"
# Build manually to avoid fallback warnings
word_emb = models.Transformer(MODEL_NAME, max_seq_length=512)
pooling  = models.Pooling(
    word_emb.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False
)
embedder = SentenceTransformer(modules=[word_emb, pooling], device="cpu")

# ── B) Paths & cache naming ────────────────────────────────────────────────
JSON_PATH = "penal_code_articles_ocr.json"
# Embed cache includes model name to avoid collisions
safe_name = MODEL_NAME.replace('/', '_')
EMB_PATH  = f"corpus_emb_{safe_name}.pt"
BM25_PATH   = "bm25_index.pkl"

# ── C) Load articles from JSON ────────────────────────────────────────────
def load_articles(json_path=JSON_PATH):
    with open(json_path, "r", encoding="utf-8") as f:
        articles = json.load(f)
    ids   = [a["article_number"] for a in articles]
    texts = [a["text"]           for a in articles]
    return ids, texts

# ── D) Build or load embeddings with dimension check ──────────────────────
def load_embeddings():
    ids, texts = load_articles()
    processed_texts = texts  # raw texts
    target_dim = embedder.get_sentence_embedding_dimension()

    if os.path.exists(EMB_PATH):
        corpus_emb = torch.load(EMB_PATH)
        # If dimension mismatch, rebuild cache
        if corpus_emb.shape[1] != target_dim:
            print(f"[Info] Detected embedding dimension change ({corpus_emb.shape[1]}→{target_dim}); rebuilding cache.")
            corpus_emb = embedder.encode(
                processed_texts,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
            torch.save(corpus_emb, EMB_PATH)
    else:
        corpus_emb = embedder.encode(
            processed_texts,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        torch.save(corpus_emb, EMB_PATH)

    return ids, texts, corpus_emb

# Initialize once
corpus_ids, corpus_texts, corpus_emb = load_embeddings()

# ── E) Pure semantic-search function ──────────────────────────────────────
def load_articles(json_path=JSON_PATH):
    with open(json_path, "r", encoding="utf-8") as f:
        articles = json.load(f)
    ids   = [a["article_number"] for a in articles]
    texts = [a["text"]           for a in articles]
    return ids, texts


# ── E) Build or load BM25 index ───────────────────────────────────────────
def load_bm25():
    _, texts = load_articles()
    # simple whitespace tokenizer for BM25
    tokenized = [t.split() for t in texts]
    if os.path.exists(BM25_PATH):
        bm25 = pickle.load(open(BM25_PATH, "rb"))
    else:
        bm25 = BM25Okapi(tokenized)
        pickle.dump(bm25, open(BM25_PATH, "wb"))
    return bm25, tokenized

# initialize once
corpus_ids, corpus_texts, corpus_emb = load_embeddings()
bm25_index, bm25_tokenized = load_bm25()

# ── F) Hybrid semantic + lexical search ────────────────────────────────────
def hybrid_search(query: str, top_k: int = 5, alpha: float = 0.6):
    # 1) Lexical BM25
    q_tokens    = query.split()
    bm25_scores = np.array(bm25_index.get_scores(q_tokens))
    # normalize BM25 scores to [0,1]
    if bm25_scores.max() - bm25_scores.min() > 0:
        bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
    else:
        bm25_norm = np.zeros_like(bm25_scores)

    # 2) Semantic cosine
    q_emb       = embedder.encode([query], convert_to_tensor=True, normalize_embeddings=True)
    cos_scores  = util.cos_sim(q_emb, corpus_emb)[0].cpu().numpy()
    # normalize semantic to [0,1]
    if cos_scores.max() - cos_scores.min() > 0:
        sem_norm = (cos_scores - cos_scores.min()) / (cos_scores.max() - cos_scores.min())
    else:
        sem_norm = np.zeros_like(cos_scores)

    # 3) Combined score
    combined_scores = alpha * bm25_norm + (1 - alpha) * sem_norm

    # 4) Rank and return top_k
    top_indices = np.argsort(combined_scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({
            "article_number": corpus_ids[idx],
            "combined":       float(combined_scores[idx]),
            "bm25":           float(bm25_scores[idx]),
            "semantic":       float(cos_scores[idx]),
            "text":           corpus_texts[idx]
        })
    return results

# ── F) Smoke-test when run directly ───────────────────────────────────────
if __name__ == "__main__":
    q = "طلب أجراً أعلى من السعر المحدد في الكشف التسعيري"
    print(f"Top 3 results for query: '{q}'\n")
    for r in hybrid_search(q, top_k=3):
        # You can print the combined score, plus the BM25 and semantic components:
        print(
            f"Article {r['article_number']} | "
            f"combined={r['combined']:.3f} | "
            f"bm25={r['bm25']:.3f} | "
            f"semantic={r['semantic']:.3f}"
        )
        print(r['text'], "\n")
import json
import os
import torch
from sentence_transformers import util, SentenceTransformer


# ── A) Setup embedder and preprocessor ─────────────────────────────────────
# Use the Base AraBERT model (large variant is not published)
MODEL_NAME   = "aubmindlab/bert-base-arabertv02"
from sentence_transformers import models
# Build a SentenceTransformer manually to avoid fallback warnings
word_emb     = models.Transformer(MODEL_NAME, max_seq_length=512)
pooling_model= models.Pooling(
    word_emb.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False
)
embedder     = SentenceTransformer(modules=[word_emb, pooling_model])
# ── B) Paths ───────────────────────────────────────────────────────────── ───────────────────────────────────────────────────────────── ─────────────────────────────────────────────────────────────
JSON_PATH = "penal_code_articles_ocr.json"
EMB_PATH  = "corpus_emb_only.pt"

# ── C) Load articles from JSON ────────────────────────────────────────────
def load_articles(json_path=JSON_PATH):
    with open(json_path, "r", encoding="utf-8") as f:
        articles = json.load(f)
    ids   = [a["article_number"] for a in articles]
    texts = [a["text"]           for a in articles]
    return ids, texts

# ── D) Build or load pre-computed embeddings ──────────────────────────────
def load_embeddings():
    ids, texts = load_articles()
    # Preprocess corpus texts for AraBERT consistency
    processed_texts = texts  # raw texts, no preprocessing
    if os.path.exists(EMB_PATH):
        corpus_emb = torch.load(EMB_PATH)
    else:
        corpus_emb = embedder.encode(
            processed_texts,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        torch.save(corpus_emb, EMB_PATH)
    return ids, texts, corpus_emb

# ── Initialize once ────────────────────────────────────────────────────────
corpus_ids, corpus_texts, corpus_emb = load_embeddings()

# ── E) Pure semantic-search function ──────────────────────────────────────
def semantic_search(query: str, top_k: int = 5):
    # Preprocess and encode query
    q_processed = query  # raw query, no preprocessing
    q_emb       = embedder.encode(
        [q_processed],
        convert_to_tensor=True,
        normalize_embeddings=True
    )
    # Retrieve top_k by cosine similarity
    hits = util.semantic_search(q_emb, corpus_emb, top_k=top_k)[0]
    results = []
    for h in hits:
        idx   = h["corpus_id"]
        results.append({
            "article_number": corpus_ids[idx],
            "score":           float(h["score"]),
            "text":            corpus_texts[idx]
        })
    return results

# ── F) Smoke-test when run directly ───────────────────────────────────────
if __name__ == "__main__":
    q = "قاصر دون الثامنة عشرة من عمره أشرية روحية"
    print(f"Top 3 results for query: '{q}'\n")
    for r in semantic_search(q, top_k=3):
        print(f"Article {r['article_number']} (score={r['score']:.3f}):")
        print(r['text'], "\n")

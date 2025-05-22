import json
import os
import torch
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import util, models, SentenceTransformer

# ── A) Setup embedder ──────────────────────────────────────────────────────
MODEL_NAME = "asafaya/bert-base-arabic"
word_emb = models.Transformer(MODEL_NAME, max_seq_length=512)
pooling  = models.Pooling(
    word_emb.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False
)
embedder = SentenceTransformer(modules=[word_emb, pooling], device="cpu")

# ── B) Paths & cache naming ────────────────────────────────────────────────
JSON_PATHS = [
    r"C:\Users\User\LegalChatbot\data\penal_code_articles_ocr.json",
    r"C:\Users\User\LegalChatbot\data\tijara_code_articles_ocr.json"
]
safe_name   = MODEL_NAME.replace('/', '_')
# include both JSON basenames so caches are unique
tag         = "_".join(os.path.splitext(os.path.basename(p))[0] for p in JSON_PATHS)
EMB_PATH    = f"corpus_emb_{safe_name}_{tag}.pt"
BM25_PATH   = f"bm25_index_{tag}.pkl"

# ── C) Load & merge articles with prefixed IDs ────────────────────────────
def load_articles(json_paths=JSON_PATHS):
    all_ids, all_texts, all_sources = [], [], []
    for path in json_paths:
        src = os.path.splitext(os.path.basename(path))[0]  # e.g. "penal_code_articles_ocr"
        with open(path, "r", encoding="utf-8") as f:
            articles = json.load(f)
        for a in articles:
            # prefix the article_number to keep it unique
            all_ids.append(f"{src}_{a['article_number']}")
            all_texts.append(a["text"])
            all_sources.append(src)
    return all_ids, all_texts, all_sources

# ── D) Build / load embeddings ────────────────────────────────────────────
def load_embeddings():
    ids, texts, _ = load_articles()
    target_dim = embedder.get_sentence_embedding_dimension()

    if os.path.exists(EMB_PATH):
        corpus_emb = torch.load(EMB_PATH)
        if corpus_emb.shape[1] != target_dim:
            corpus_emb = embedder.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
            torch.save(corpus_emb, EMB_PATH)
    else:
        corpus_emb = embedder.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        torch.save(corpus_emb, EMB_PATH)

    return ids, texts, corpus_emb

# ── E) Build / load BM25 index ────────────────────────────────────────────
def load_bm25():
    _, texts, _ = load_articles()
    tokenized   = [t.split() for t in texts]
    if os.path.exists(BM25_PATH):
        bm25 = pickle.load(open(BM25_PATH, "rb"))
    else:
        bm25 = BM25Okapi(tokenized)
        pickle.dump(bm25, open(BM25_PATH, "wb"))
    return bm25, tokenized

# ── F) Initialize once ────────────────────────────────────────────────────
corpus_ids, corpus_texts, corpus_emb    = load_embeddings()
bm25_index, bm25_tokenized              = load_bm25()

# ── G) Hybrid search now returns the prefixed ID ─────────────────────────
def hybrid_search(query: str, top_k: int = 5, alpha: float = 0.6):
    # BM25 part
    q_tokens    = query.split()
    bm25_scores = np.array(bm25_index.get_scores(q_tokens))
    if bm25_scores.max() - bm25_scores.min() > 0:
        bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
    else:
        bm25_norm = np.zeros_like(bm25_scores)

    # semantic part
    q_emb      = embedder.encode([query], convert_to_tensor=True, normalize_embeddings=True)
    cos_scores = util.cos_sim(q_emb, corpus_emb)[0].cpu().numpy()
    if cos_scores.max() - cos_scores.min() > 0:
        sem_norm = (cos_scores - cos_scores.min()) / (cos_scores.max() - cos_scores.min())
    else:
        sem_norm = np.zeros_like(cos_scores)

    # combine & rank
    combined_scores = alpha * bm25_norm + (1 - alpha) * sem_norm
    top_idxs        = np.argsort(combined_scores)[::-1][:top_k]

    results = []
    for i in top_idxs:
        num = corpus_ids[i].split("_")[-1]

        results.append({
            "article_number": num,            # so your front-end sees something here
            "doc_id":         corpus_ids[i],   # in case you still need it
            "combined":       float(combined_scores[i]),
            "bm25":           float(bm25_scores[i]),
            "semantic":       float(cos_scores[i]),
            "text":           corpus_texts[i]
        })
    return results

if __name__ == "__main__":
    for r in hybrid_search("سبب الجريمة للمجنى عليه مرضا", top_k=3):
        print(r["doc_id"], r["combined"], "\n", r["text"], "\n")
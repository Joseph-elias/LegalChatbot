import json
import os
import torch
import numpy as np
from sentence_transformers import util, SentenceTransformer
from sentence_transformers import SentenceTransformer



# ── A) Setup embedder ──────────────────────────────────────────────────────
MODEL_NAME = "intfloat/multilingual-e5-base"
embedder = SentenceTransformer(MODEL_NAME, device="cpu")  # Use "cuda" if you have a GPU

# ── B) Paths & cache naming ────────────────────────────────────────────────
JSON_PATHS = [
    r"C:/Users/User/LegalChatbot/data/penal_code_articles_ocr.json",
    r"C:/Users/User/LegalChatbot/data/tijara_code_articles_ocr.json",
    r"C:/Users/User/LegalChatbot/data/muhakamat-madaniya_code_articles_ocr.json"
]
safe_name = MODEL_NAME.replace('/', '_')
tag = "_".join(os.path.splitext(os.path.basename(p))[0] for p in JSON_PATHS)
EMB_PATH = f"corpus_emb_{safe_name}_{tag}.pt"

# ── C) Load & merge articles with prefixed IDs ────────────────────────────
def load_articles(json_paths=JSON_PATHS):
    all_ids, all_texts, all_sources = [], [], []
    for path in json_paths:
        src = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as f:
            articles = json.load(f)
        for a in articles:
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

# ── E) Initialize once ────────────────────────────────────────────────────
corpus_ids, corpus_texts, corpus_emb = load_embeddings()

# ── F) Dense-only semantic search ─────────────────────────────────────────
def semantic_search_only(query: str, top_k: int = 5):
    q_emb = embedder.encode([query], convert_to_tensor=True, normalize_embeddings=True)
    cos_scores = util.cos_sim(q_emb, corpus_emb)[0].cpu().numpy()
    top_idxs = np.argsort(cos_scores)[::-1][:top_k]

    results = []
    for i in top_idxs:
        num = corpus_ids[i].split("_")[-1]
        results.append({
            "article_number": num,
            "doc_id": corpus_ids[i],
            "score": float(cos_scores[i]),
            "text": corpus_texts[i]
        })
    return results

if __name__ == "__main__":
    for r in semantic_search_only("سبب الجريمة للمجنى عليه مرضا", top_k=3):
        print(r["doc_id"], r["score"], "\n", r["text"], "\n")

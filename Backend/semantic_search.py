import json
import os
import torch
from sentence_transformers import SentenceTransformer
import pyarabic.araby as araby

# ── Arabic Text Normalization Function ─────────────────────────────────────
def normalize_arabic_text(text):
    text = araby.strip_tashkeel(text)
    text = araby.strip_tatweel(text)
    text = araby.normalize_alef(text)
    text = araby.normalize_hamza(text)
    return text

# ── A) Setup embedder ──────────────────────────────────────────────────────
MODEL_NAME = "akhooli/Arabic-SBERT-100K" # <-- Changed
embedder = SentenceTransformer(MODEL_NAME, device="cpu")  # Use "cuda" if you have a GPU

# ── B) Paths & cache naming ────────────────────────────────────────────────
JSON_PATHS = [
    r"data/cleaned/combined_legal_articles_retrieval_with_moj_selected.json",
]
safe_name = MODEL_NAME.replace('/', '_')
tag = "_".join(os.path.splitext(os.path.basename(p))[0] for p in JSON_PATHS)
EMB_PATH = f"corpus_emb_{safe_name}_{tag}.pt"
DOC_METADATA: dict[str, dict] = {}

# ── C) Load & merge articles with prefixed IDs ────────────────────────────
def load_articles(json_paths=JSON_PATHS):
    global DOC_METADATA
    all_ids, all_texts, all_sources = [], [], []
    DOC_METADATA = {}
    for path in json_paths:
        src = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path, "r", encoding="utf-8") as f:
                articles = json.load(f)
            for i, a in enumerate(articles):
                text = normalize_arabic_text(str(a.get("text", "")))
                if not text:
                    continue

                article_number = str(
                    a.get("article_number_normalized")
                    or a.get("article_number")
                    or "unknown"
                )
                doc_id = a.get("doc_id") or f"{src}_{i}_{article_number}"
                law_code = a.get("law_code") or src

                all_ids.append(doc_id)
                all_texts.append(text)
                all_sources.append(law_code)
                DOC_METADATA[doc_id] = {
                    "law_code": law_code,
                    "law_name": a.get("law_name"),
                    "article_number": a.get("article_number_normalized") or a.get("article_number"),
                    "citation_tag": a.get("citation_tag"),
                    "provenance": a.get("provenance", {}),
                    "source_file": a.get("source_file"),
                }
        except FileNotFoundError:
            print(f"Warning: File not found {path}. Skipping.")
    return all_ids, all_texts, all_sources


def get_doc_metadata(doc_id: str) -> dict:
    return DOC_METADATA.get(doc_id, {})

# ── D) Build / load embeddings ────────────────────────────────────────────
def load_embeddings():
    ids, texts, _ = load_articles()
    # If texts is empty because files were not found, this should still proceed
    # but corpus_emb might be empty or cause issues later.
    # The primary goal is changing MODEL_NAME.
    if not texts:
        print("Warning: No texts loaded, embeddings will likely be empty or fail.")
        # Return empty tensor with expected structure if possible, or handle error.
        # For this task, focusing on MODEL_NAME change is key.
        # Fallback: create a dummy embedding if texts is empty to avoid crashing embedder.encode
        # This is a workaround for worker environment if it tries to run the script.
        texts = ["dummy text"] 
        ids = ["dummy_id"]


    target_dim = embedder.get_sentence_embedding_dimension()

    if os.path.exists(EMB_PATH):
        print(f"Attempting to load existing embeddings from {EMB_PATH}")
        corpus_emb = torch.load(EMB_PATH)
        if corpus_emb.shape[1] != target_dim:
            print(f"Dimension mismatch. Expected {target_dim}, got {corpus_emb.shape[1]}. Re-encoding.")
            corpus_emb = embedder.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
            torch.save(corpus_emb, EMB_PATH)
    else:
        print(f"No existing embeddings found at {EMB_PATH}. Encoding texts.")
        corpus_emb = embedder.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        torch.save(corpus_emb, EMB_PATH)

    return ids, texts, corpus_emb

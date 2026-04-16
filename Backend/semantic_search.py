from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import pyarabic.araby as araby
import torch
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_JSON_PATHS = [
    ROOT_DIR / "data" / "cleaned" / "combined_legal_articles_retrieval_with_moj_selected.json",
]

MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "akhooli/Arabic-SBERT-100K")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
embedder = SentenceTransformer(MODEL_NAME, device=EMBEDDING_DEVICE)

DOC_METADATA: dict[str, dict] = {}


def normalize_arabic_text(text: str) -> str:
    text = araby.strip_tashkeel(text)
    text = araby.strip_tatweel(text)
    text = araby.normalize_alef(text)
    text = araby.normalize_hamza(text)
    return text


def _parse_json_paths() -> list[Path]:
    raw = os.getenv("LEGAL_JSON_PATHS", "").strip()
    if not raw:
        return DEFAULT_JSON_PATHS

    paths: list[Path] = []
    for token in raw.split(","):
        candidate = Path(token.strip())
        if not candidate:
            continue
        paths.append(candidate if candidate.is_absolute() else (ROOT_DIR / candidate))
    return paths or DEFAULT_JSON_PATHS


def _safe_model_name() -> str:
    return MODEL_NAME.replace("/", "_")


def _corpus_fingerprint(json_paths: Iterable[Path]) -> str:
    payload: list[str] = [MODEL_NAME]
    for path in json_paths:
        resolved = path.resolve()
        if not resolved.exists():
            payload.append(f"{resolved}:missing")
            continue
        stat = resolved.stat()
        payload.append(f"{resolved}:{stat.st_size}:{stat.st_mtime_ns}")

    digest = hashlib.sha256("||".join(payload).encode("utf-8")).hexdigest()
    return digest[:16]


def get_embedding_cache_path() -> Path:
    explicit = os.getenv("EMBEDDINGS_PATH", "").strip()
    if explicit:
        target = Path(explicit)
        return target if target.is_absolute() else (ROOT_DIR / target)

    cache_dir = os.getenv("EMBEDDINGS_DIR", "").strip()
    base_dir = Path(cache_dir) if cache_dir else (ROOT_DIR / "cache")
    if not base_dir.is_absolute():
        base_dir = ROOT_DIR / base_dir

    fingerprint = _corpus_fingerprint(_parse_json_paths())
    return base_dir / f"corpus_emb_{_safe_model_name()}_{fingerprint}.pt"


def load_articles(json_paths: list[Path] | None = None) -> tuple[list[str], list[str], list[str]]:
    global DOC_METADATA

    paths = json_paths or _parse_json_paths()
    all_ids: list[str] = []
    all_texts: list[str] = []
    all_sources: list[str] = []
    DOC_METADATA = {}

    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing legal corpus files: {missing}")

    for path in paths:
        src = path.stem
        with path.open("r", encoding="utf-8") as f:
            articles = json.load(f)

        for i, article in enumerate(articles):
            text = normalize_arabic_text(str(article.get("text", "")))
            if not text:
                continue

            article_number = str(
                article.get("article_number_normalized")
                or article.get("article_number")
                or "unknown"
            )
            doc_id = article.get("doc_id") or f"{src}_{i}_{article_number}"
            law_code = article.get("law_code") or src

            all_ids.append(doc_id)
            all_texts.append(text)
            all_sources.append(law_code)
            DOC_METADATA[doc_id] = {
                "law_code": law_code,
                "law_name": article.get("law_name"),
                "article_number": article.get("article_number_normalized") or article.get("article_number"),
                "citation_tag": article.get("citation_tag"),
                "provenance": article.get("provenance", {}),
                "source_file": article.get("source_file"),
            }

    if not all_texts:
        raise ValueError("No legal article texts found after loading JSON corpus.")

    return all_ids, all_texts, all_sources


def get_doc_metadata(doc_id: str) -> dict:
    return DOC_METADATA.get(doc_id, {})


def _target_dimension() -> int:
    # sentence-transformers renamed this method; support both for compatibility.
    if hasattr(embedder, "get_embedding_dimension"):
        return int(embedder.get_embedding_dimension())
    return int(embedder.get_sentence_embedding_dimension())


def load_embeddings(
    *,
    rebuild: bool = False,
    require_existing: bool = False,
) -> tuple[list[str], list[str], torch.Tensor]:
    ids, texts, _ = load_articles()
    emb_path = get_embedding_cache_path()
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    target_dim = _target_dimension()

    if require_existing and (not emb_path.exists()):
        raise FileNotFoundError(
            f"Embedding cache is required but missing at: {emb_path}. "
            "Run scripts/precompute_embeddings.py before starting the API."
        )

    should_encode = rebuild or (not emb_path.exists())
    if not should_encode:
        corpus_emb = torch.load(emb_path)
        if corpus_emb.ndim != 2 or corpus_emb.shape[1] != target_dim or corpus_emb.shape[0] != len(ids):
            should_encode = True

    if should_encode:
        print(f"Building embeddings at {emb_path} ...")
        corpus_emb = embedder.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        torch.save(corpus_emb, emb_path)
        print(f"Saved embeddings: {emb_path}")

    return ids, texts, corpus_emb

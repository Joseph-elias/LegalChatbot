from __future__ import annotations

import os
from typing import Any

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel
from sentence_transformers import util

from semantic_search import embedder, get_doc_metadata, load_embeddings, normalize_arabic_text

# --- FastAPI Setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # keep current behavior for dev; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment.")

client = OpenAI(api_key=OPENAI_API_KEY)

corpus_ids: list[str] = []
corpus_texts: list[str] = []
corpus_emb = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 150


def llm_text(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def semantic_search_only(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    global corpus_ids, corpus_texts, corpus_emb

    query = normalize_arabic_text(query)
    q_emb = embedder.encode([query], convert_to_tensor=True, normalize_embeddings=True)

    if corpus_emb is None or not hasattr(corpus_emb, "nelement") or corpus_emb.nelement() == 0:
        corpus_ids, corpus_texts, corpus_emb = load_embeddings()

    cos_scores = util.cos_sim(q_emb, corpus_emb)[0].cpu().numpy()
    top_idxs = np.argsort(cos_scores)[::-1][:top_k]

    results: list[dict[str, Any]] = []
    for i in top_idxs:
        if i >= len(corpus_ids):
            continue

        doc_id = corpus_ids[i]
        metadata = get_doc_metadata(doc_id)
        article_number = metadata.get("article_number") or doc_id.split("_")[-1]

        results.append(
            {
                "article_number": article_number,
                "doc_id": doc_id,
                "score": float(cos_scores[i]),
                "text": corpus_texts[i],
                "law_code": metadata.get("law_code"),
                "law_name": metadata.get("law_name"),
                "citation_tag": metadata.get("citation_tag"),
                "provenance": metadata.get("provenance", {}),
            }
        )
    return results


def generate_paraphrased_questions(question: str, n: int = 54) -> list[str]:
    system_prompt = (
        "You are a legal-language rewriting assistant. "
        "Return only paraphrases, one per line, no numbering, no commentary."
    )
    user_prompt = (
        f"Rewrite this Arabic legal question into {n} formal legal paraphrases with identical meaning.\n"
        f"Question: {question}"
    )
    text = llm_text(system_prompt, user_prompt, temperature=0.0)
    lines = [line.strip().lstrip("-*").strip() for line in text.splitlines() if line.strip()]
    # keep unique order, cap at n to avoid runaway output
    seen = set()
    unique = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique.append(line)
        if len(unique) >= n:
            break
    return unique


def multi_query_search(queries: list[str], top_k: int) -> list[dict[str, Any]]:
    seen = set()
    combined: list[dict[str, Any]] = []
    for q in queries:
        for r in semantic_search_only(q, top_k=top_k):
            uid = r["doc_id"]
            if uid not in seen:
                combined.append(r)
                seen.add(uid)
    return combined


def rerank_with_llm(results: list[dict[str, Any]], user_query: str) -> str:
    context = "\n\n".join(
        [
            (
                f"[{r.get('citation_tag', 'NO-CITATION')}] "
                f"{r.get('law_name', r.get('law_code', 'unknown_law'))} "
                f"Article {r.get('article_number')}:\n{r.get('text', '')}"
            )
            for r in results
        ]
    )

    system_prompt = (
        "You are a Lebanese legal assistant. Use only provided articles. "
        "Pick one best matching article. If uncertain, say uncertain and ask a clarifying question."
    )

    user_prompt = f"""
User query:
{user_query}

Candidate legal articles:
{context}

Return EXACTLY this structure:
- Direct answer: <short practical answer>
- Legal basis: <law name + article number>
- Citation tag: <citation_tag from the chosen source>
- Article text: <verbatim selected article text from context>
- Why this article: <short explanation>
- Clarification (if needed): <one question or 'None'>
"""

    return llm_text(system_prompt, user_prompt, temperature=0.0)


def build_source_cards(results: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    cards = []
    for r in results[:limit]:
        provenance = r.get("provenance", {}) or {}
        law_name = r.get("law_name") or r.get("law_code") or "Lebanese Law"
        article_number = r.get("article_number", "N/A")
        citation_tag = r.get("citation_tag") or r.get("doc_id")
        access_url = provenance.get("law_landing_url") or provenance.get("access_url")
        excerpt = r.get("text", "")[:240].strip()

        cards.append(
            {
                "title": f"{law_name} - Article {article_number}",
                "excerpt": f"[{citation_tag}] {excerpt}",
                "url": access_url,
                "citation_tag": citation_tag,
                "publisher": provenance.get("publisher_short") or provenance.get("publisher"),
            }
        )
    return cards


def append_citation_summary(answer: str, results: list[dict[str, Any]], top_n: int = 3) -> str:
    lines = ["", "Citations used:"]
    for r in results[:top_n]:
        tag = r.get("citation_tag") or r.get("doc_id")
        law_name = r.get("law_name") or r.get("law_code") or "Lebanese Law"
        art = r.get("article_number", "N/A")
        lines.append(f"- [{tag}] {law_name}, Article {art}")
    return answer.rstrip() + "\n" + "\n".join(lines)


@app.on_event("startup")
async def startup_load_embeddings() -> None:
    global corpus_ids, corpus_texts, corpus_emb
    corpus_ids, corpus_texts, corpus_emb = load_embeddings()


@app.post("/search")
async def search(req: SearchRequest) -> dict[str, Any]:
    global corpus_ids, corpus_texts, corpus_emb

    if not corpus_ids or corpus_emb is None or not hasattr(corpus_emb, "nelement") or corpus_emb.nelement() == 0:
        corpus_ids, corpus_texts, corpus_emb = load_embeddings()

    normalized_query = normalize_arabic_text(req.query)

    paraphrased = generate_paraphrased_questions(normalized_query)
    paraphrased.insert(0, normalized_query)

    results = multi_query_search(paraphrased, top_k=req.top_k)
    results = sorted(results, key=lambda x: -x["score"])[: req.top_k]

    answer = rerank_with_llm(results, normalized_query)
    answer = append_citation_summary(answer, results)

    return {
        "answer": answer,
        "sources": build_source_cards(results),
        "raw_sources": results,
    }


if __name__ == "__main__":
    corpus_ids, corpus_texts, corpus_emb = load_embeddings()
    print(f"Loaded corpus rows: {len(corpus_ids)}")

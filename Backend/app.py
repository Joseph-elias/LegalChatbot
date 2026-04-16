from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import Depends, FastAPI
from fastapi import HTTPException
from fastapi import Header
from fastapi import Request
from fastapi import Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from openai import OpenAI
from pydantic import BaseModel
from sentence_transformers import util

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

from semantic_search import embedder, get_doc_metadata, load_embeddings, normalize_arabic_text

# --- FastAPI Setup ---
app = FastAPI()

load_dotenv()

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173").split(",") if o.strip()]
ALLOWED_HOSTS = [h.strip() for h in os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",") if h.strip()]
ALLOW_CREDENTIALS = "*" not in ALLOWED_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS or ["localhost", "127.0.0.1"])


@app.middleware("http")
async def set_security_headers(request: Request, call_next: Any) -> Response:
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    return response

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PRELOAD_EMBEDDINGS = os.getenv("PRELOAD_EMBEDDINGS", "1").strip() == "1"
EMBEDDINGS_REQUIRE_PREBUILT = os.getenv("EMBEDDINGS_REQUIRE_PREBUILT", "0").strip() == "1"
PARAPHRASE_COUNT = int(os.getenv("PARAPHRASE_COUNT", "16"))
MAX_EVIDENCE_CANDIDATES = int(os.getenv("MAX_EVIDENCE_CANDIDATES", "24"))
MAX_FINAL_EVIDENCE = int(os.getenv("MAX_FINAL_EVIDENCE", "6"))
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "dense").strip().lower()
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.75"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.25"))
HYBRID_PER_QUERY_K = int(os.getenv("HYBRID_PER_QUERY_K", "40"))
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "0").strip() == "1"
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3").strip()
RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", "30"))
RERANKER_WEIGHT = float(os.getenv("RERANKER_WEIGHT", "0.35"))
MIN_ANSWER_CONFIDENCE = float(os.getenv("MIN_ANSWER_CONFIDENCE", "0.35"))
SELF_CHECK_ENABLED = os.getenv("SELF_CHECK_ENABLED", "1").strip() == "1"
DATABASE_PATH = os.getenv("DATABASE_PATH", "chatbot.db").strip()
SESSION_DAYS = int(os.getenv("SESSION_DAYS", "30"))
POLICY_VERSION = os.getenv("POLICY_VERSION", "2026-04-16")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment.")

client = OpenAI(api_key=OPENAI_API_KEY)

corpus_ids: list[str] = []
corpus_texts: list[str] = []
corpus_emb = None
embedding_lock = threading.Lock()
bm25_lock = threading.Lock()
bm25_index: Optional[BM25Okapi] = None
bm25_tokens: list[list[str]] = []
reranker_lock = threading.Lock()
reranker_model: Optional["CrossEncoder"] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 150


class RegisterRequest(BaseModel):
    email: str
    password: str
    accept_disclaimer: bool = False
    accept_privacy: bool = False
    accept_cookies: bool = False
    policy_version: str | None = None


class LoginRequest(BaseModel):
    email: str
    password: str
    accept_disclaimer: bool = False
    accept_privacy: bool = False
    accept_cookies: bool = False
    policy_version: str | None = None


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class DeleteAccountRequest(BaseModel):
    password: str
    confirm_text: str


class CreateConversationRequest(BaseModel):
    title: str | None = None


class UpdateConversationRequest(BaseModel):
    title: str


class SendMessageRequest(BaseModel):
    content: str
    top_k: int = 40


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def db_path() -> str:
    return DATABASE_PATH


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    conn = get_db()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                policy_version TEXT,
                accepted_disclaimer_at TEXT,
                accepted_privacy_at TEXT,
                accepted_cookie_at TEXT,
                last_login_at TEXT
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
            """
        )
        user_columns = {r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
        migrations: list[str] = []
        if "policy_version" not in user_columns:
            migrations.append("ALTER TABLE users ADD COLUMN policy_version TEXT")
        if "accepted_disclaimer_at" not in user_columns:
            migrations.append("ALTER TABLE users ADD COLUMN accepted_disclaimer_at TEXT")
        if "accepted_privacy_at" not in user_columns:
            migrations.append("ALTER TABLE users ADD COLUMN accepted_privacy_at TEXT")
        if "accepted_cookie_at" not in user_columns:
            migrations.append("ALTER TABLE users ADD COLUMN accepted_cookie_at TEXT")
        if "last_login_at" not in user_columns:
            migrations.append("ALTER TABLE users ADD COLUMN last_login_at TEXT")
        for stmt in migrations:
            conn.execute(stmt)
        conn.commit()
    finally:
        conn.close()


def normalize_email(email: str) -> str:
    return email.strip().lower()


def validate_password_strength(password: str) -> None:
    if len(password) < 10:
        raise HTTPException(status_code=400, detail="Password must be at least 10 characters")
    if not re.search(r"[A-Z]", password):
        raise HTTPException(status_code=400, detail="Password must include at least one uppercase letter")
    if not re.search(r"[a-z]", password):
        raise HTTPException(status_code=400, detail="Password must include at least one lowercase letter")
    if not re.search(r"\d", password):
        raise HTTPException(status_code=400, detail="Password must include at least one number")
    if not re.search(r"[^A-Za-z0-9]", password):
        raise HTTPException(status_code=400, detail="Password must include at least one special character")


def ensure_policy_acceptance(accept_disclaimer: bool, accept_privacy: bool, accept_cookies: bool) -> None:
    if not (accept_disclaimer and accept_privacy and accept_cookies):
        raise HTTPException(
            status_code=400,
            detail="You must accept disclaimer, privacy terms, and essential cookies to continue",
        )


def hash_password(password: str, salt_b64: str) -> str:
    salt = base64.b64decode(salt_b64.encode("ascii"))
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return base64.b64encode(digest).decode("ascii")


def create_password_pair(password: str) -> tuple[str, str]:
    salt = secrets.token_bytes(16)
    salt_b64 = base64.b64encode(salt).decode("ascii")
    digest_b64 = hash_password(password, salt_b64)
    return salt_b64, digest_b64


def verify_password(password: str, salt_b64: str, expected_digest_b64: str) -> bool:
    actual = hash_password(password, salt_b64)
    return hmac.compare_digest(actual, expected_digest_b64)


def create_session_token(conn: sqlite3.Connection, user_id: int) -> str:
    token = secrets.token_urlsafe(48)
    created = datetime.now(timezone.utc)
    expires = created + timedelta(days=SESSION_DAYS)
    conn.execute(
        "INSERT INTO sessions(token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
        (token, user_id, created.isoformat(), expires.isoformat()),
    )
    conn.commit()
    return token


def apply_policy_acceptance(
    conn: sqlite3.Connection,
    user_id: int,
    *,
    accept_disclaimer: bool,
    accept_privacy: bool,
    accept_cookies: bool,
    policy_version: str | None,
) -> None:
    accepted_at = now_utc()
    conn.execute(
        """
        UPDATE users
        SET
            policy_version = ?,
            accepted_disclaimer_at = CASE
                WHEN ? THEN COALESCE(accepted_disclaimer_at, ?)
                ELSE accepted_disclaimer_at
            END,
            accepted_privacy_at = CASE
                WHEN ? THEN COALESCE(accepted_privacy_at, ?)
                ELSE accepted_privacy_at
            END,
            accepted_cookie_at = CASE
                WHEN ? THEN COALESCE(accepted_cookie_at, ?)
                ELSE accepted_cookie_at
            END
        WHERE id = ?
        """,
        (
            (policy_version or POLICY_VERSION).strip() or POLICY_VERSION,
            1 if accept_disclaimer else 0,
            accepted_at,
            1 if accept_privacy else 0,
            accepted_at,
            1 if accept_cookies else 0,
            accepted_at,
            user_id,
        ),
    )


def serialize_user(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "email": row["email"],
        "created_at": row["created_at"],
        "policy_version": row["policy_version"] if "policy_version" in row.keys() else None,
        "accepted_disclaimer": bool(row["accepted_disclaimer_at"]) if "accepted_disclaimer_at" in row.keys() else False,
        "accepted_privacy": bool(row["accepted_privacy_at"]) if "accepted_privacy_at" in row.keys() else False,
        "accepted_cookies": bool(row["accepted_cookie_at"]) if "accepted_cookie_at" in row.keys() else False,
        "accepted_disclaimer_at": row["accepted_disclaimer_at"] if "accepted_disclaimer_at" in row.keys() else None,
        "accepted_privacy_at": row["accepted_privacy_at"] if "accepted_privacy_at" in row.keys() else None,
        "accepted_cookie_at": row["accepted_cookie_at"] if "accepted_cookie_at" in row.keys() else None,
        "last_login_at": row["last_login_at"] if "last_login_at" in row.keys() else None,
    }


def parse_bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        raise HTTPException(status_code=401, detail="Invalid Authorization header")
    token = authorization[len(prefix) :].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Empty bearer token")
    return token


def get_current_user(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    token = parse_bearer_token(authorization)
    conn = get_db()
    try:
        row = conn.execute(
            """
            SELECT
                u.id,
                u.email,
                u.created_at,
                u.policy_version,
                u.accepted_disclaimer_at,
                u.accepted_privacy_at,
                u.accepted_cookie_at,
                u.last_login_at,
                s.expires_at
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token = ?
            """,
            (token,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=401, detail="Invalid session token")

        expires_at = datetime.fromisoformat(row["expires_at"])
        if expires_at < datetime.now(timezone.utc):
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            conn.commit()
            raise HTTPException(status_code=401, detail="Session expired")

        return {
            "id": row["id"],
            "email": row["email"],
            "created_at": row["created_at"],
            "policy_version": row["policy_version"],
            "accepted_disclaimer_at": row["accepted_disclaimer_at"],
            "accepted_privacy_at": row["accepted_privacy_at"],
            "accepted_cookie_at": row["accepted_cookie_at"],
            "last_login_at": row["last_login_at"],
            "token": token,
        }
    finally:
        conn.close()


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


def classify_query_mode(user_query: str) -> str:
    system_prompt = (
        "You classify user messages for a Lebanese legal assistant backend.\n"
        "Return only one token:\n"
        "- LEGAL: asks for Lebanese law/legal advice, articles, penalties, procedures, legal interpretation.\n"
        "- CHAT: greetings, small talk, general non-legal questions, writing help, coding help, or anything not legal."
    )
    label = llm_text(system_prompt, f"Message: {user_query}", temperature=0.0).strip().upper()
    return "LEGAL" if "LEGAL" in label else "CHAT"


def chat_reply(user_query: str) -> str:
    system_prompt = (
        "You are a helpful, natural conversational assistant. "
        "Reply fluently and briefly in the user's language. "
        "If the user asks legal questions later, say you can switch to legal mode."
    )
    return llm_text(system_prompt, user_query, temperature=0.4)


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def _tokenize_for_overlap(text: str) -> set[str]:
    parts = re.findall(r"[\u0600-\u06FFA-Za-z0-9]+", text.lower())
    return {p for p in parts if len(p) >= 3}


def _tokenize_for_bm25(text: str) -> list[str]:
    parts = re.findall(r"[\u0600-\u06FFA-Za-z0-9]+", text.lower())
    return [p for p in parts if len(p) >= 2]


def _contains_any(text: str, needles: list[str]) -> bool:
    return any(n in text for n in needles)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def rule_based_evidence_override(user_query: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    q = normalize_arabic_text(user_query).lower()
    normalized_rows = []
    for row in results:
        txt = normalize_arabic_text(str(row.get("text", ""))).lower()
        tag = row.get("citation_tag") or row.get("doc_id")
        normalized_rows.append((tag, txt))

    # Rule 1: Theft + multiple offenders => article text explicitly mentions "two or more".
    if _contains_any(q, ["سرقة"]) and _contains_any(q, ["شخصين", "متهم", "اكثر", "أكثر", "مشترك"]):
        for tag, txt in normalized_rows:
            if _contains_any(txt, ["بفعل شخصين او اكثر", "بفعل شخصين أو أكثر", "شخصين او اكثر", "شخصين أو أكثر"]):
                return {
                    "matched": True,
                    "selected_citation_tags": [tag],
                    "reason": "Direct article text explicitly addresses theft by two or more offenders.",
                }

    # Rule 2: Pickpocketing in public places => explicit "بنشل المارة".
    if _contains_any(q, ["نشل"]) and _contains_any(q, ["اماكن", "الأماكن", "الطرق", "عام", "عامة"]):
        for tag, txt in normalized_rows:
            if _contains_any(txt, ["بنشل المارة", "الاماكن العامة", "الأماكن العامة"]):
                return {
                    "matched": True,
                    "selected_citation_tags": [tag],
                    "reason": "Direct article text explicitly addresses pickpocketing/public-place theft.",
                }

    # Rule 3: Generic theft penalty => explicit "السرقة" + "يعاقب".
    if _contains_any(q, ["سرقة"]) and _contains_any(q, ["عقوبة", "يعاقب", "جزاء"]):
        for tag, txt in normalized_rows:
            if _contains_any(txt, ["السرقة", "سرقة"]) and _contains_any(txt, ["يعاقب", "بالحبس", "بالغرامة"]):
                return {
                    "matched": True,
                    "selected_citation_tags": [tag],
                    "reason": "Direct article text explicitly states theft punishment terms.",
                }

    return {"matched": False, "selected_citation_tags": [], "reason": ""}


def filter_by_basic_relevance(results: list[dict[str, Any]], user_query: str, limit: int = 20) -> list[dict[str, Any]]:
    q_tokens = _tokenize_for_overlap(user_query)
    if not q_tokens:
        return results[:limit]

    ranked: list[tuple[float, dict[str, Any]]] = []
    for row in results:
        text = str(row.get("text", ""))
        t_tokens = _tokenize_for_overlap(text)
        overlap = len(q_tokens & t_tokens)
        combined = (float(row.get("score", 0.0)) * 0.8) + (overlap * 0.2)
        ranked.append((combined, row))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in ranked[:limit]]


def ensure_reranker_loaded() -> None:
    global reranker_model

    if not RERANKER_ENABLED or CrossEncoder is None:
        return

    if reranker_model is not None:
        return

    with reranker_lock:
        if reranker_model is not None:
            return
        reranker_model = CrossEncoder(RERANKER_MODEL_NAME)


def apply_reranker(results: list[dict[str, Any]], user_query: str) -> list[dict[str, Any]]:
    if not results or not RERANKER_ENABLED:
        return results

    ensure_reranker_loaded()
    if reranker_model is None:
        return results

    top_n = min(RERANKER_TOP_N, len(results))
    head = results[:top_n]
    tail = results[top_n:]

    pairs = [(user_query, str(r.get("text", ""))) for r in head]
    raw_scores = reranker_model.predict(pairs)
    if hasattr(raw_scores, "tolist"):
        raw_scores = raw_scores.tolist()

    reranker_probs = [_sigmoid(float(s)) for s in raw_scores]
    reranker_max = max(reranker_probs) or 1.0

    for row, rr in zip(head, reranker_probs):
        retrieval_score = float(row.get("score", 0.0))
        retrieval_norm = max(0.0, min(1.0, retrieval_score))
        rr_norm = rr / reranker_max
        fused = ((1.0 - RERANKER_WEIGHT) * retrieval_norm) + (RERANKER_WEIGHT * rr_norm)
        row["score"] = float(fused)
        row["reranker_meta"] = {
            "model": RERANKER_MODEL_NAME,
            "raw_score": float(rr),
            "normalized": float(rr_norm),
            "weight": RERANKER_WEIGHT,
        }

    head.sort(key=lambda x: -float(x.get("score", 0.0)))
    return head + tail


def verify_evidence_with_llm(results: list[dict[str, Any]], user_query: str) -> dict[str, Any]:
    context_rows = []
    for row in results:
        tag = row.get("citation_tag") or row.get("doc_id")
        context_rows.append(
            {
                "citation_tag": tag,
                "law_name": row.get("law_name") or row.get("law_code"),
                "article_number": row.get("article_number"),
                "text": row.get("text", ""),
            }
        )

    system_prompt = (
        "You are a strict legal evidence verifier. "
        "Only accept evidence that directly addresses the exact legal issue asked by the user. "
        "If coverage is weak/indirect, mark sufficient_basis as false."
    )
    user_prompt = (
        "User query:\n"
        f"{user_query}\n\n"
        "Candidate articles JSON:\n"
        f"{json.dumps(context_rows, ensure_ascii=False)}\n\n"
        "Return ONLY valid JSON with this schema:\n"
        "{"
        "\"sufficient_basis\": true|false, "
        "\"selected_citation_tags\": [\"...\"], "
        "\"reason\": \"short reason\", "
        "\"clarification_question\": \"question in user's language or None\""
        "}"
    )

    payload = _extract_json_object(llm_text(system_prompt, user_prompt, temperature=0.0))
    selected_tags = payload.get("selected_citation_tags", [])
    if not isinstance(selected_tags, list):
        selected_tags = []

    available = {r.get("citation_tag") or r.get("doc_id") for r in results}
    selected_tags = [t for t in selected_tags if t in available]

    if not selected_tags:
        selected_tags = [(results[0].get("citation_tag") or results[0].get("doc_id"))] if results else []

    override = rule_based_evidence_override(user_query, results)
    if override.get("matched"):
        override_tags = [t for t in override.get("selected_citation_tags", []) if t in available]
        if override_tags:
            selected_tags = override_tags
        return {
            "sufficient_basis": True,
            "selected_citation_tags": selected_tags,
            "reason": override.get("reason", "Direct legal text match."),
            "clarification_question": str(payload.get("clarification_question", "None")).strip() or "None",
        }

    return {
        "sufficient_basis": bool(payload.get("sufficient_basis")) and bool(selected_tags),
        "selected_citation_tags": selected_tags,
        "reason": str(payload.get("reason", "")).strip() or "Insufficient direct legal evidence in retrieved sources.",
        "clarification_question": str(payload.get("clarification_question", "None")).strip() or "None",
    }


def build_insufficient_basis_answer(user_query: str, verifier: dict[str, Any], top_rows: list[dict[str, Any]]) -> str:
    top = top_rows[:2]
    lines = [
        "- Direct answer: لا يمكنني إعطاء حكم قانوني دقيق من المصادر المسترجعة الحالية فقط.",
        "- Simple explanation (for non-lawyers): السؤال مهم، لكن المواد التي ظهرت لا تعالج النقطة القانونية المطلوبة بشكل مباشر وواضح.",
        "- Basic legal penalty: Not explicit in retrieved sources",
        "- Aggravating circumstances: Not explicit in retrieved sources",
        "- Final legal effect: أحتاج مادة قانونية أكثر صلة قبل تأكيد النتيجة.",
        f"- Legal basis: {', '.join((r.get('law_name') or r.get('law_code') or 'Unknown law') + ' Article ' + str(r.get('article_number', 'N/A')) for r in top) if top else 'Not available'}",
        f"- Citation tag: {(top[0].get('citation_tag') or top[0].get('doc_id')) if top else 'None'}",
        f"- Article text: {str(top[0].get('text', 'None'))[:280] if top else 'None'}",
        f"- Why this article: {verifier.get('reason', 'Coverage was not direct enough for a reliable legal conclusion.')}",
        "- What to do next: اطرح سؤالاً أدق يتضمن نوع الجريمة/الواقعة أو القانون المعني (مثلاً: قانون السير اللبناني أو مادة محددة).",
        f"- Clarification (if needed): {verifier.get('clarification_question', 'None')}",
    ]
    return "\n".join(lines)


def estimate_answer_confidence(rows: list[dict[str, Any]], verifier: dict[str, Any]) -> float:
    if not rows:
        return 0.0
    top = rows[0]
    retrieval_score = float(top.get("score", 0.0))
    retrieval_score = max(0.0, min(1.0, retrieval_score))

    reranker_norm = None
    reranker_meta = top.get("reranker_meta")
    if isinstance(reranker_meta, dict):
        reranker_norm = float(reranker_meta.get("normalized", 0.0))

    if reranker_norm is None:
        model_score = retrieval_score
    else:
        model_score = (0.6 * retrieval_score) + (0.4 * max(0.0, min(1.0, reranker_norm)))

    basis = 1.0 if verifier.get("sufficient_basis", False) else 0.0
    confidence = (0.75 * model_score) + (0.25 * basis)
    return max(0.0, min(1.0, confidence))


def self_check_answer_grounding(answer: str, user_query: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not SELF_CHECK_ENABLED or not answer.strip() or not rows:
        return {"supported": True, "corrected_answer": answer, "reason": "Self-check disabled or no evidence rows."}

    context_rows = []
    for row in rows:
        context_rows.append(
            {
                "citation_tag": row.get("citation_tag") or row.get("doc_id"),
                "law_name": row.get("law_name") or row.get("law_code"),
                "article_number": row.get("article_number"),
                "text": row.get("text", ""),
            }
        )

    system_prompt = (
        "You are a strict legal groundedness reviewer. "
        "Check if the proposed answer is fully supported by provided article texts. "
        "If any legal conclusion is unsupported or overstated, mark supported=false and provide a corrected version "
        "that is conservative and explicitly says when basis is insufficient."
    )
    user_prompt = (
        "User query:\n"
        f"{user_query}\n\n"
        "Evidence articles JSON:\n"
        f"{json.dumps(context_rows, ensure_ascii=False)}\n\n"
        "Proposed answer:\n"
        f"{answer}\n\n"
        "Return ONLY JSON with keys:\n"
        "{"
        "\"supported\": true|false, "
        "\"reason\": \"short reason\", "
        "\"corrected_answer\": \"full answer text preserving the same output structure\""
        "}"
    )

    payload = _extract_json_object(llm_text(system_prompt, user_prompt, temperature=0.0))
    supported = bool(payload.get("supported"))
    corrected = str(payload.get("corrected_answer", "")).strip()
    reason = str(payload.get("reason", "")).strip() or "No reason provided."

    if not corrected:
        corrected = answer
    return {"supported": supported, "corrected_answer": corrected, "reason": reason}


def semantic_search_only(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    global corpus_ids, corpus_texts, corpus_emb

    query = normalize_arabic_text(query)
    q_emb = embedder.encode([query], convert_to_tensor=True, normalize_embeddings=True)

    ensure_embeddings_loaded()

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


def ensure_bm25_loaded() -> None:
    global bm25_index, bm25_tokens

    if BM25Okapi is None:
        return

    if bm25_index is not None and bm25_tokens:
        return

    with bm25_lock:
        if bm25_index is not None and bm25_tokens:
            return
        ensure_embeddings_loaded()
        bm25_tokens = [_tokenize_for_bm25(t) for t in corpus_texts]
        bm25_index = BM25Okapi(bm25_tokens)


def bm25_search_only(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    ensure_bm25_loaded()
    if bm25_index is None:
        return []

    q_tokens = _tokenize_for_bm25(normalize_arabic_text(query))
    if not q_tokens:
        return []

    scores = bm25_index.get_scores(q_tokens)
    top_idxs = np.argsort(scores)[::-1][:top_k]

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
                "score": float(scores[i]),
                "text": corpus_texts[i],
                "law_code": metadata.get("law_code"),
                "law_name": metadata.get("law_name"),
                "citation_tag": metadata.get("citation_tag"),
                "provenance": metadata.get("provenance", {}),
            }
        )
    return results


def generate_paraphrased_questions(question: str, n: int = PARAPHRASE_COUNT) -> list[str]:
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


def hybrid_multi_query_search(queries: list[str], top_k: int) -> list[dict[str, Any]]:
    if BM25Okapi is None:
        return multi_query_search(queries, top_k)

    dense_by_doc: dict[str, dict[str, Any]] = {}
    bm25_by_doc: dict[str, dict[str, Any]] = {}

    for q in queries:
        for r in semantic_search_only(q, top_k=HYBRID_PER_QUERY_K):
            uid = r["doc_id"]
            prev = dense_by_doc.get(uid)
            if prev is None or r["score"] > prev["score"]:
                dense_by_doc[uid] = r
        for r in bm25_search_only(q, top_k=HYBRID_PER_QUERY_K):
            uid = r["doc_id"]
            prev = bm25_by_doc.get(uid)
            if prev is None or r["score"] > prev["score"]:
                bm25_by_doc[uid] = r

    all_doc_ids = set(dense_by_doc.keys()) | set(bm25_by_doc.keys())
    if not all_doc_ids:
        return []

    max_dense = max((v["score"] for v in dense_by_doc.values()), default=1.0) or 1.0
    max_bm25 = max((v["score"] for v in bm25_by_doc.values()), default=1.0) or 1.0

    fused: list[dict[str, Any]] = []
    for doc_id in all_doc_ids:
        dense_row = dense_by_doc.get(doc_id)
        bm25_row = bm25_by_doc.get(doc_id)
        base_row = dense_row or bm25_row
        if base_row is None:
            continue

        dense_norm = (dense_row["score"] / max_dense) if dense_row else 0.0
        bm25_norm = (bm25_row["score"] / max_bm25) if bm25_row else 0.0
        fused_score = (DENSE_WEIGHT * dense_norm) + (BM25_WEIGHT * bm25_norm)

        row = dict(base_row)
        row["score"] = float(fused_score)
        row["retrieval_meta"] = {
            "dense_score_raw": float(dense_row["score"]) if dense_row else None,
            "bm25_score_raw": float(bm25_row["score"]) if bm25_row else None,
            "dense_norm": float(dense_norm),
            "bm25_norm": float(bm25_norm),
            "fusion": "weighted_sum",
        }
        fused.append(row)

    fused.sort(key=lambda x: -x["score"])
    return fused[:top_k]


def run_retrieval(queries: list[str], top_k: int) -> list[dict[str, Any]]:
    if RETRIEVAL_MODE == "hybrid":
        return hybrid_multi_query_search(queries, top_k)
    return multi_query_search(queries, top_k)


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
        "Pick one best matching article. If uncertain, say uncertain and ask a clarifying question. "
        "Be precise, practical, avoid invented legal details, and explain in simple language for non-lawyers. "
        "Do not state legal conclusions that are not explicitly supported by the provided article text."
    )

    user_prompt = f"""
User query:
{user_query}

Candidate legal articles:
{context}

Return EXACTLY this structure:
- Direct answer: <short practical answer>
- Simple explanation (for non-lawyers): <2-4 short lines in plain language>
- Basic legal penalty: <base punishment from the selected article, or 'Not explicit'>
- Aggravating circumstances: <conditions that increase severity, or 'None mentioned'>
- Final legal effect: <state whether aggravation applies to this user case or not, with reason>
- Legal basis: <law name + article number>
- Citation tag: <citation_tag from the chosen source>
- Article text: <verbatim selected article text from context>
- Why this article: <short explanation>
- What to do next: <one practical next step for the user, or 'None'>
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


def ensure_embeddings_loaded() -> None:
    global corpus_ids, corpus_texts, corpus_emb

    if corpus_emb is not None and hasattr(corpus_emb, "nelement") and corpus_emb.nelement() > 0 and corpus_ids:
        return

    with embedding_lock:
        if corpus_emb is not None and hasattr(corpus_emb, "nelement") and corpus_emb.nelement() > 0 and corpus_ids:
            return
        corpus_ids, corpus_texts, corpus_emb = load_embeddings(require_existing=EMBEDDINGS_REQUIRE_PREBUILT)


@app.on_event("startup")
async def startup_load_embeddings() -> None:
    init_db()
    if PRELOAD_EMBEDDINGS:
        ensure_embeddings_loaded()


def run_assistant_pipeline(query: str, top_k: int) -> dict[str, Any]:
    global corpus_ids, corpus_texts, corpus_emb

    try:
        ensure_embeddings_loaded()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    normalized_query = normalize_arabic_text(query)
    mode = classify_query_mode(normalized_query)

    if mode == "CHAT":
        return {
            "answer": chat_reply(query),
            "sources": [],
            "raw_sources": [],
        }

    paraphrased = generate_paraphrased_questions(normalized_query)
    paraphrased.insert(0, normalized_query)

    results = run_retrieval(paraphrased, top_k=top_k)
    results = sorted(results, key=lambda x: -x["score"])[:top_k]
    candidates = filter_by_basic_relevance(results, normalized_query, limit=min(MAX_EVIDENCE_CANDIDATES, len(results)))
    candidates = apply_reranker(candidates, normalized_query)
    verifier = verify_evidence_with_llm(candidates, normalized_query)

    selected_tags = set(verifier.get("selected_citation_tags", []))
    filtered = [r for r in candidates if (r.get("citation_tag") or r.get("doc_id")) in selected_tags][:MAX_FINAL_EVIDENCE]
    if not filtered:
        filtered = candidates[:MAX_FINAL_EVIDENCE]

    confidence = estimate_answer_confidence(filtered, verifier)
    if confidence < MIN_ANSWER_CONFIDENCE:
        verifier = dict(verifier)
        verifier["sufficient_basis"] = False
        verifier["reason"] = (
            f"{verifier.get('reason', 'Insufficient direct legal evidence in retrieved sources.')} "
            f"(low confidence={confidence:.2f})"
        )

    if not verifier.get("sufficient_basis", False):
        answer = build_insufficient_basis_answer(normalized_query, verifier, filtered)
        answer = append_citation_summary(answer, filtered)
        return {
            "answer": answer,
            "sources": build_source_cards(filtered),
            "raw_sources": filtered,
        }

    answer = rerank_with_llm(filtered, normalized_query)
    check = self_check_answer_grounding(answer, normalized_query, filtered)
    answer = check.get("corrected_answer", answer)
    if not check.get("supported", False):
        verifier = dict(verifier)
        verifier["sufficient_basis"] = False
        verifier["reason"] = f"{verifier.get('reason', '')} Self-check: {check.get('reason', 'unsupported legal claims.')}".strip()
        answer = build_insufficient_basis_answer(normalized_query, verifier, filtered)
    answer = append_citation_summary(answer, filtered)

    return {
        "answer": answer,
        "sources": build_source_cards(filtered),
        "raw_sources": filtered,
    }


@app.post("/search")
async def search(req: SearchRequest) -> dict[str, Any]:
    return run_assistant_pipeline(req.query, req.top_k)


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"status": "ok"}


@app.post("/auth/register")
async def auth_register(req: RegisterRequest) -> dict[str, Any]:
    email = normalize_email(req.email)
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Invalid email")
    validate_password_strength(req.password)
    ensure_policy_acceptance(req.accept_disclaimer, req.accept_privacy, req.accept_cookies)

    conn = get_db()
    try:
        exists = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if exists:
            raise HTTPException(status_code=409, detail="Email already registered")

        salt_b64, digest_b64 = create_password_pair(req.password)
        created_at = now_utc()
        cur = conn.execute(
            """
            INSERT INTO users(
                email,
                password_salt,
                password_hash,
                created_at,
                policy_version,
                accepted_disclaimer_at,
                accepted_privacy_at,
                accepted_cookie_at,
                last_login_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                email,
                salt_b64,
                digest_b64,
                created_at,
                (req.policy_version or POLICY_VERSION).strip() or POLICY_VERSION,
                created_at,
                created_at,
                created_at,
                created_at,
            ),
        )
        user_id = int(cur.lastrowid)
        token = create_session_token(conn, user_id)
        user_row = conn.execute(
            """
            SELECT
                id,
                email,
                created_at,
                policy_version,
                accepted_disclaimer_at,
                accepted_privacy_at,
                accepted_cookie_at,
                last_login_at
            FROM users
            WHERE id = ?
            """,
            (user_id,),
        ).fetchone()
        if user_row is None:
            raise HTTPException(status_code=500, detail="Failed to create user")
        return {"token": token, "user": serialize_user(user_row)}
    finally:
        conn.close()


@app.post("/auth/login")
async def auth_login(req: LoginRequest) -> dict[str, Any]:
    email = normalize_email(req.email)
    conn = get_db()
    try:
        row = conn.execute(
            """
            SELECT
                id,
                email,
                created_at,
                password_salt,
                password_hash,
                policy_version,
                accepted_disclaimer_at,
                accepted_privacy_at,
                accepted_cookie_at,
                last_login_at
            FROM users
            WHERE email = ?
            """,
            (email,),
        ).fetchone()
        if row is None or not verify_password(req.password, row["password_salt"], row["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        has_all_policies = bool(row["accepted_disclaimer_at"] and row["accepted_privacy_at"] and row["accepted_cookie_at"])
        if not has_all_policies:
            ensure_policy_acceptance(req.accept_disclaimer, req.accept_privacy, req.accept_cookies)
            apply_policy_acceptance(
                conn,
                int(row["id"]),
                accept_disclaimer=req.accept_disclaimer,
                accept_privacy=req.accept_privacy,
                accept_cookies=req.accept_cookies,
                policy_version=req.policy_version,
            )

        now = now_utc()
        conn.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (now, int(row["id"])))

        token = create_session_token(conn, int(row["id"]))
        user_row = conn.execute(
            """
            SELECT
                id,
                email,
                created_at,
                policy_version,
                accepted_disclaimer_at,
                accepted_privacy_at,
                accepted_cookie_at,
                last_login_at
            FROM users
            WHERE id = ?
            """,
            (int(row["id"]),),
        ).fetchone()
        if user_row is None:
            raise HTTPException(status_code=500, detail="Unable to load user profile")
        return {
            "token": token,
            "user": serialize_user(user_row),
        }
    finally:
        conn.close()


@app.post("/auth/logout")
async def auth_logout(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    conn = get_db()
    try:
        conn.execute("DELETE FROM sessions WHERE token = ?", (user["token"],))
        conn.commit()
    finally:
        conn.close()
    return {"ok": True}


@app.get("/auth/me")
async def auth_me(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    conn = get_db()
    try:
        row = conn.execute(
            """
            SELECT
                id,
                email,
                created_at,
                policy_version,
                accepted_disclaimer_at,
                accepted_privacy_at,
                accepted_cookie_at,
                last_login_at
            FROM users
            WHERE id = ?
            """,
            (user["id"],),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="User not found")
        return {"user": serialize_user(row)}
    finally:
        conn.close()


@app.get("/account/settings")
async def get_account_settings(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    conn = get_db()
    try:
        profile = conn.execute(
            """
            SELECT
                id,
                email,
                created_at,
                policy_version,
                accepted_disclaimer_at,
                accepted_privacy_at,
                accepted_cookie_at,
                last_login_at
            FROM users
            WHERE id = ?
            """,
            (user["id"],),
        ).fetchone()
        if profile is None:
            raise HTTPException(status_code=404, detail="User not found")

        active_sessions = conn.execute(
            "SELECT COUNT(*) AS c FROM sessions WHERE user_id = ? AND expires_at > ?",
            (user["id"], now_utc()),
        ).fetchone()

        return {
            "user": serialize_user(profile),
            "active_sessions": int(active_sessions["c"]) if active_sessions else 0,
            "current_policy_version": POLICY_VERSION,
        }
    finally:
        conn.close()


@app.patch("/account/password")
async def update_password(req: ChangePasswordRequest, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    validate_password_strength(req.new_password)
    if req.current_password == req.new_password:
        raise HTTPException(status_code=400, detail="New password must be different from current password")

    conn = get_db()
    try:
        row = conn.execute(
            "SELECT password_salt, password_hash FROM users WHERE id = ?",
            (user["id"],),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="User not found")
        if not verify_password(req.current_password, row["password_salt"], row["password_hash"]):
            raise HTTPException(status_code=401, detail="Current password is incorrect")

        salt_b64, digest_b64 = create_password_pair(req.new_password)
        conn.execute(
            "UPDATE users SET password_salt = ?, password_hash = ? WHERE id = ?",
            (salt_b64, digest_b64, user["id"]),
        )
        conn.execute("DELETE FROM sessions WHERE user_id = ?", (user["id"],))
        conn.commit()
        return {"ok": True, "message": "Password updated. Please log in again."}
    finally:
        conn.close()


@app.delete("/account")
async def delete_account(req: DeleteAccountRequest, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    if req.confirm_text.strip().upper() != "DELETE":
        raise HTTPException(status_code=400, detail='Type "DELETE" to confirm account deletion')

    conn = get_db()
    try:
        row = conn.execute(
            "SELECT password_salt, password_hash FROM users WHERE id = ?",
            (user["id"],),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="User not found")
        if not verify_password(req.password, row["password_salt"], row["password_hash"]):
            raise HTTPException(status_code=401, detail="Password is incorrect")

        conn.execute("DELETE FROM users WHERE id = ?", (user["id"],))
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()


@app.get("/conversations")
async def list_conversations(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    conn = get_db()
    try:
        rows = conn.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations
            WHERE user_id = ?
            ORDER BY updated_at DESC
            """,
            (user["id"],),
        ).fetchall()
        return {
            "conversations": [
                {
                    "id": r["id"],
                    "title": r["title"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                }
                for r in rows
            ]
        }
    finally:
        conn.close()


@app.post("/conversations")
async def create_conversation(req: CreateConversationRequest, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    title = (req.title or "New conversation").strip() or "New conversation"
    now = now_utc()
    conn = get_db()
    try:
        cur = conn.execute(
            "INSERT INTO conversations(user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (user["id"], title, now, now),
        )
        conn.commit()
        cid = int(cur.lastrowid)
        return {"conversation": {"id": cid, "title": title, "created_at": now, "updated_at": now}}
    finally:
        conn.close()


@app.patch("/conversations/{conversation_id}")
async def rename_conversation(conversation_id: int, req: UpdateConversationRequest, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    title = req.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title cannot be empty")

    conn = get_db()
    try:
        existing = conn.execute(
            "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user["id"]),
        ).fetchone()
        if existing is None:
            raise HTTPException(status_code=404, detail="Conversation not found")

        updated_at = now_utc()
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, updated_at, conversation_id),
        )
        conn.commit()
        return {"conversation": {"id": conversation_id, "title": title, "updated_at": updated_at}}
    finally:
        conn.close()


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: int, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    conn = get_db()
    try:
        existing = conn.execute(
            "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user["id"]),
        ).fetchone()
        if existing is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()


@app.get("/conversations/{conversation_id}/messages")
async def get_messages(conversation_id: int, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    conn = get_db()
    try:
        convo = conn.execute(
            "SELECT id, title FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user["id"]),
        ).fetchone()
        if convo is None:
            raise HTTPException(status_code=404, detail="Conversation not found")

        rows = conn.execute(
            """
            SELECT id, role, content, sources_json, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()
        messages = []
        for r in rows:
            sources = json.loads(r["sources_json"]) if r["sources_json"] else []
            messages.append(
                {
                    "id": r["id"],
                    "role": r["role"],
                    "content": r["content"],
                    "sources": sources,
                    "created_at": r["created_at"],
                }
            )
        return {"conversation": {"id": convo["id"], "title": convo["title"]}, "messages": messages}
    finally:
        conn.close()


@app.post("/conversations/{conversation_id}/messages")
async def send_message(conversation_id: int, req: SendMessageRequest, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    content = req.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    conn = get_db()
    try:
        convo = conn.execute(
            "SELECT id, title FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user["id"]),
        ).fetchone()
        if convo is None:
            raise HTTPException(status_code=404, detail="Conversation not found")

        now = now_utc()
        conn.execute(
            "INSERT INTO messages(conversation_id, role, content, sources_json, created_at) VALUES (?, 'user', ?, NULL, ?)",
            (conversation_id, content, now),
        )

        result = run_assistant_pipeline(content, req.top_k)
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        conn.execute(
            "INSERT INTO messages(conversation_id, role, content, sources_json, created_at) VALUES (?, 'assistant', ?, ?, ?)",
            (conversation_id, answer, json.dumps(sources, ensure_ascii=False), now_utc()),
        )

        updated_at = now_utc()
        existing_title = str(convo["title"] or "").strip()
        new_title = existing_title
        if existing_title.lower() == "new conversation":
            new_title = content[:60]
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (new_title, updated_at, conversation_id),
        )
        conn.commit()

        return {
            "conversation": {
                "id": conversation_id,
                "title": new_title,
                "updated_at": updated_at,
            },
            "assistant_message": {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "created_at": now_utc(),
            },
        }
    finally:
        conn.close()


if __name__ == "__main__":
    corpus_ids, corpus_texts, corpus_emb = load_embeddings()
    print(f"Loaded corpus rows: {len(corpus_ids)}")

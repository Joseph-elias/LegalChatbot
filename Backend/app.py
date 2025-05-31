from fastapi import FastAPI
from pydantic import BaseModel
from semantic_search import embedder ,normalize_arabic_text,load_embeddings # ,semantic_search_only # changed import!
import google.generativeai as genai
import json
import os
import torch
import numpy as np
from sentence_transformers import util, SentenceTransformer
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pyarabic.araby as araby
import os
from dotenv import load_dotenv

# --- FastAPI Setup ---
app = FastAPI()

origins = [
    "https://leblegalchatbot.onrender.com",  
]




app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TEMPORARY for testing — wide open
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)





load_dotenv()  

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")


# ── F) Dense-only semantic search ─────────────────────────────────────────
def semantic_search_only(query: str, top_k: int = 5):
    global corpus_ids, corpus_texts, corpus_emb
    query = normalize_arabic_text(query)
    # This function will use the globally defined 'embedder' and 'corpus_emb'
   
    q_emb = embedder.encode([query], convert_to_tensor=True, normalize_embeddings=True)
    # Ensure corpus_emb is loaded and not empty, handle defensively for worker execution
    if 'corpus_emb' not in globals() or not hasattr(corpus_emb, 'nelement') or corpus_emb.nelement() == 0:
        print("Error: corpus_emb not loaded or empty in semantic_search_only.")
        # Attempt to load corpus_emb if it's missing, for robustness during __main__ or direct calls
        corpus_ids, corpus_texts, corpus_emb = load_embeddings()
        if not hasattr(corpus_emb, 'nelement') or corpus_emb.nelement() == 0:
             print("Error: Failed to load corpus_emb, it remains empty.")
             return []


    cos_scores = util.cos_sim(q_emb, corpus_emb)[0].cpu().numpy()
    top_idxs = np.argsort(cos_scores)[::-1][:top_k]

    results = []
    # Ensure corpus_ids is also loaded and available
    if 'corpus_ids' not in globals():
        print("Error: corpus_ids not loaded in semantic_search_only.")
        return [] # Or handle error appropriately

    for i in top_idxs:
        # Defensive check for corpus_ids length
        if i < len(corpus_ids):
            num = corpus_ids[i].split("_")[-1]
            results.append({
                "article_number": num,
                "doc_id": corpus_ids[i],
                "score": float(cos_scores[i]),
                "text": corpus_texts[i] 
            })
        else:
            print(f"Warning: Index {i} out of bounds for corpus_ids.")
    return results



# --- Request Schema ---
class SearchRequest(BaseModel):
    query: str
    top_k: int = 150

# --- Step 1: Paraphrase the User's Question ---
def generate_paraphrased_questions(question: str, n: int = 54) -> list[str]:
    prompt = f"""
أعد صياغة السؤال التالي {n} مرة بصياغات مختلفة ولكن بنفس المعنى، مستخدمًا أسلوبًا أكاديميًا ورسميًا كما لو أنك تكتب لمقال علمي أو تقرير قانوني. لا تضف إجابة، فقط الصياغات.

السؤال: {question}

الصياغات:
"""
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.0}
    )
    return [line.strip("- ").strip() for line in response.text.strip().splitlines() if line.strip()]


# --- Step 2: Search Using All Reformulations ---
def multi_query_search(queries: list[str], top_k: int) -> list[dict]:
    seen = set()
    combined = []
    for q in queries:
        results = semantic_search_only(q, top_k=top_k)
        for r in results:
            uid = r["doc_id"]
            if uid not in seen:
                combined.append(r)
                seen.add(uid)
    return combined

# --- Step 3: Gemini Reranking ---
def rerank_with_llm(results, user_query):
    context = "".join([f"المادة {r['article_number']}:{r['text']}" for r in results])
    prompt = f"""
أنت مساعد قانوني متخصص. مهمتك هي اختيار المادة القانونية **الوحيدة** الأكثر تطابقًا للإجابة على سؤال المستخدم من ضمن قائمة المواد المعطاة لك فقط. اتبع التعليمات التالية بدقة:

1.  **تحليل دقيق للسؤال والمواد:** اقرأ سؤال المستخدم والمواد القانونية التالية بعناية فائقة.
2.  **شرط "جناية" و "جنحة":**
    *   إذا كان سؤال المستخدم يتعلق بـ "جناية" بشكل واضح، يجب أن تختار مادة تحتوي على كلمة "جناية" أو تتعلق بشكل مباشر بالجنايات. إذا لم تجد، اذكر ذلك.
    *   إذا كان سؤال المستخدم يتعلق بـ "جنحة" بشكل واضح، يجب أن تختار مادة تحتوي على كلمة "جنحة" أو تتعلق بشكل مباشر بالجنح. إذا لم تجد، اذكر ذلك.
    *   إذا لم يكن السؤال واضحًا بشأن "جناية" أو "جنحة"، أو لم ينطبق الشرط، اختر المادة الأكثر صلة بشكل عام.
3.  **الاختيار والإخراج:**
    *   اختر المادة **الوحيدة** الأكثر صلة.
    *   **يجب أن تقتصر إجابتك على المعلومات الموجودة ضمن المواد المعطاة فقط. لا تستخدم أي معلومات خارجية.**
    *   اذكر رقم المادة التي اخترتها.
    *   انسخ نص المادة التي اخترتها بالكامل كما هو.
    *   اشرح بإيجاز سبب اختيارك لهذه المادة وكيف تجيب على سؤال المستخدم، مع الإشارة إلى الأجزاء ذات الصلة من المادة.
4.  **في حالة عدم وجود مادة مناسبة:** إذا لم تجد أي مادة من المواد المعطاة تجيب بشكل مناسب ودقيق على سؤال المستخدم، حتى بعد تطبيق شروط "جناية" و "جنحة"، اذكر بوضوح: "لم أجد مادة قانونية مناسبة في النصوص المعطاة تجيب على سؤالك بدقة. يرجى إعادة صياغة السؤال أو توضيح المطلوب." لا تحاول اختيار مادة غير ملائمة.

المواد القانونية:
{context}

سؤال المستخدم:
{user_query}

الإجابة المختارة (رقم المادة، نص المادة، شرح السبب):
"""
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.0}
    )
    return response.text.strip()

# --- Search Endpoint ---
@app.post("/search")
async def search(req: SearchRequest):
    # Ensure corpus_ids, corpus_texts, and corpus_emb are loaded globally.
    # This would typically be done once at startup by FastAPI's lifespan events,
    # or as done in __main__. For simplicity here, ensure they are loaded.
    global corpus_ids, corpus_texts, corpus_emb
    if 'corpus_ids' not in globals() or \
       'corpus_texts' not in globals() or \
       'corpus_emb' not in globals() or \
       not hasattr(corpus_emb, 'nelement') or \
       corpus_emb.nelement() == 0:
        print("Corpus not loaded or empty in /search endpoint. Attempting to load.")
        # This call is critical for the FastAPI app if not loaded at startup
        corpus_ids, corpus_texts, corpus_emb = load_embeddings()
        if not hasattr(corpus_emb, 'nelement') or corpus_emb.nelement() == 0:
            print("Error: Failed to load corpus_emb for /search, it remains empty.")
            # Depending on requirements, might return an error response to client
            # For now, processing will likely fail in multi_query_search if emb is empty

    normalized_query = normalize_arabic_text(req.query)

    # Step 1: Reformulate the query
    paraphrased = generate_paraphrased_questions(normalized_query)
    paraphrased.insert(0, normalized_query)  # include original

    # Step 2: Dense search for all paraphrases, deduplicated
    results = multi_query_search(paraphrased, top_k=req.top_k) # This calls semantic_search_only
    results = sorted(results, key=lambda x: -x["score"])[:req.top_k]

    # Step 3: LLM reranking prompt (Gemini picks and explains)
    answer = rerank_with_llm(results, normalized_query)

    # Optional: clarification if ambiguous
    ambiguous_phrases = [
        "لا توجد في المواد", "لا يمكن الإجابة", "غير كافية", "لا توجد مادة"
    ]
    if any(p in answer for p in ambiguous_phrases):
        answer += "🔎 الرجاء توضيح سؤالك أكثر حتى أتمكن من مساعدتك بدقة."

    return {"answer": answer, "sources": results}

if __name__ == "__main__":
    # This main block is for local testing and might need genai configuration
    # For example: genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    # Ensure the API key is available if running this directly.
    
    # Critical: Initialize corpus here for __main__ execution
    # This was missing and would cause 'corpus_ids' not defined error later.
    corpus_ids, corpus_texts, corpus_emb = load_embeddings()
    
    print("Running local test for semantic_search_only:")
    # Check if corpus_emb is valid before searching
    if hasattr(corpus_emb, 'nelement') and corpus_emb.nelement() > 0 :
        for r in semantic_search_only("سبب الجريمة للمجنى عليه مرضا", top_k=3):
            print(r["doc_id"], r["score"], "", r["text"], "")
    else:
        print("Skipping semantic_search_only test as corpus_emb is empty or invalid.")
    
    # Example of testing paraphrasing and reranking (requires API key & model init)
    # user_q = "ما هي عقوبة السرقة بالإكراه؟"
    # print(f"
#Testing paraphrasing for: {user_q}")
    # paraphrases = generate_paraphrased_questions(user_q, n=2)
    # print(paraphrases)
    #
    # print("
#Testing reranking (mocked results):")
    # mock_results_for_rerank = [
    #     {"article_number": "101", "text": "نص المادة 101 عن السرقة."},
    #     {"article_number": "202", "text": "نص المادة 202 عن الإكراه."}
    # ]
    # reranked_answer = rerank_with_llm(mock_results_for_rerank, user_q)
    # print(reranked_answer)

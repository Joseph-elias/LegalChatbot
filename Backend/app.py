from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from semantic_search import semantic_search_only  # changed import!
import google.generativeai as genai

# 🔐 Hardcoded API key for testing — REMOVE in production!
genai.configure(api_key="AIzaSyBeQUdqY2x_kuhO-l3zYB--gH5lOOIYwAo")
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

# --- FastAPI Setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request Schema ---
class SearchRequest(BaseModel):
    query: str
    top_k: int = 150

# --- Step 1: Paraphrase the User's Question ---
def generate_paraphrased_questions(question: str, n: int = 15) -> list[str]:
    prompt = f"""
أعد صياغة السؤال التالي {n} مرات بصيغ مختلفة ولكن بنفس المعنى، لا تضف إجابة، فقط الصياغات:

السؤال: {question}

الصياغات:
"""
    response = model.generate_content(
    prompt,
    generation_config={"temperature": 0.3}
)
    return [line.strip("- ").strip() for line in response.text.strip().split("\n") if line.strip()]

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
    context = "\n\n".join([f"المادة {r['article_number']}:\n{r['text']}" for r in results])
    prompt = f"""
اختر المادة الوحيدة الأكثر صلة من بين المواد التالية التي تجيب عن سؤال المستخدم، لكن انتبه:
- إذا كان سؤال المستخدم يتحدث عن "جناية" يجب أن تختار فقط المواد التي تحتوي كلمة "جناية" وليس "جنحة".
- وإذا كان يتحدث عن "جنحة" اختر فقط المواد التي تحتوي كلمة "جنحة".
ثم:
- اذكر رقم المادة،
- انسخ نص المادة كاملة،
- واشرح باختصار سبب اختيارك.
إذا لم توجد مادة مناسبة بدقة، أخبر المستخدم بذلك واطلب منه توضيح سؤاله.


المواد:
{context}

سؤال المستخدم:
{user_query}

الإجابة:
"""
    response = model.generate_content(prompt)
    return response.text.strip()

# --- Search Endpoint ---
@app.post("/search")
async def search(req: SearchRequest):
    # Step 1: Reformulate the query
    paraphrased = generate_paraphrased_questions(req.query)
    paraphrased.insert(0, req.query)  # include original

    # Step 2: Dense search for all paraphrases, deduplicated
    results = multi_query_search(paraphrased, top_k=req.top_k)
    results = sorted(results, key=lambda x: -x["score"])[:req.top_k]

    # Step 3: LLM reranking prompt (Gemini picks and explains)
    answer = rerank_with_llm(results, req.query)

    # Optional: clarification if ambiguous
    ambiguous_phrases = [
        "لا توجد في المواد", "لا يمكن الإجابة", "غير كافية", "لا توجد مادة"
    ]
    if any(p in answer for p in ambiguous_phrases):
        answer += "\n\n🔎 الرجاء توضيح سؤالك أكثر حتى أتمكن من مساعدتك بدقة."

    return {"answer": answer, "sources": results}

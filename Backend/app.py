from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from semantic_search import hybrid_search
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
    top_k: int = 50
    alpha: float = 0.6

# --- Step 1: Paraphrase the User's Question ---
def generate_paraphrased_questions(question: str, n: int = 15) -> list[str]:
    prompt = f"""
أعد صياغة السؤال التالي {n} مرات بصيغ مختلفة ولكن بنفس المعنى، لا تضف إجابة، فقط الصياغات:

السؤال: {question}

الصياغات:
"""
    response = model.generate_content(prompt)
    return [line.strip("- ").strip() for line in response.text.strip().split("\n") if line.strip()]

# --- Step 2: Search Using All Reformulations ---
def multi_query_search(queries: list[str], top_k: int, alpha: float) -> list[dict]:
    seen = set()
    combined = []
    for q in queries:
        results = hybrid_search(q, top_k=top_k, alpha=alpha)
        for r in results:
            uid = r.get("id") or f"{r['article_number']}-{r['text'][:30]}"
            if uid not in seen:
                combined.append(r)
                seen.add(uid)
    return combined

# --- Step 3: Gemini Prompt + Clarification Fallback ---
@app.post("/search")
async def search(req: SearchRequest):
    # Reformulate the query
    paraphrased = generate_paraphrased_questions(req.query)
    paraphrased.insert(0, req.query)  # include original
    results = multi_query_search(paraphrased, top_k=req.top_k, alpha=req.alpha)

    # Create Gemini prompt
    context = "\n\n".join([f"المادة {r['article_number']}:\n{r['text']}" for r in results])
    prompt = f"""
أنت مساعد قانوني ذكي. إليك مجموعة من مواد قانون العقوبات اللبناني.

- أجب على السؤال باستخدام المادة **الأكثر صلة فقط** من بين المواد.
- إذا لم تكن أي مادة مناسبة، أخبر المستخدم أن المواد المقدمة غير كافية وأطلب منه توضيح سؤاله.

المواد:
{context}

سؤال المستخدم:
{req.query}

الإجابة:
"""
    # Get answer
    response = model.generate_content(prompt)
    answer = response.text.strip()

    # Optional: check for ambiguous replies manually
    ambiguous_phrases = [
        "لا توجد في المواد", "لا يمكن الإجابة", "غير كافية", "لا توجد مادة"
    ]
    if any(p in answer for p in ambiguous_phrases):
        answer += "\n\n🔎 الرجاء توضيح سؤالك أكثر حتى أتمكن من مساعدتك بدقة."

    return {"answer": answer, "sources": results}

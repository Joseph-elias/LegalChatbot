from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from semantic_search import hybrid_search
import google.generativeai as genai

# ✅ Hardcoded API key for testing — REMOVE BEFORE DEPLOYING!
genai.configure(api_key="AIzaSyBeQUdqY2x_kuhO-l3zYB--gH5lOOIYwAo")

# ✅ Recommended fast Gemini model
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str
    top_k: int = 1000
    alpha: float = 0.6

@app.post("/search")
async def search(req: SearchRequest):
    try:
        results = hybrid_search(req.query, top_k=req.top_k, alpha=req.alpha)

        context = "\n\n".join([
            f"المادة {r['article_number']}:\n{r['text']}" for r in results
        ])

        prompt = f"""أنت مساعد قانوني ذكي. إليك مجموعة من مواد قانون العقوبات اللبناني.
أجب على السؤال باستخدام **المادة الأكثر صلة فقط** من بين هذه المواد، واذكر رقم المادة في إجابتك.

السياق:
{context}

سؤال المستخدم:
{req.query}

الإجابة:"""

        response = model.generate_content(prompt)
        answer = response.text

        return {
            "answer": answer,
            "sources": results
        }

    except Exception as e:
        return {"error": str(e)}

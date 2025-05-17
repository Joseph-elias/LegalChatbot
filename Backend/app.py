# backend/app.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 1) Create the app
app = FastAPI()

# 2) Add CORS _before_ any routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # for dev you can allow all
    allow_methods=["*"],      # allow GET, POST, OPTIONS, etc.
    allow_headers=["*"],      # allow Content-Type, Authorization…
)

# 3) Health-check endpoint
@app.get("/ping")
async def ping():
    return {"ping": "pong"}

# 4) Import your semantic search
from semantic_search import hybrid_search

# 5) Define the request schema
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    alpha: float = 0.6

# 6) Your actual search endpoint
@app.post("/search")
async def search(req: SearchRequest):
    results = hybrid_search(req.query, top_k=req.top_k, alpha=req.alpha)
    return {"results": results}

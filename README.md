# ⚖️ Lebanese Legal Chatbot (GPT + Semantic Search)

A full-stack legal chatbot for Lebanese law, combining **FastAPI**, **React**, **semantic search with Arabic SBERT**, and **Google Gemini** to answer users' legal questions with precision and supporting articles.

---

## 🎯 What It Does

This chatbot takes a **user's legal question in Arabic**, reformulates it in several academic ways using **Gemini**, retrieves the most relevant **legal articles** from Lebanese law via **dense semantic search**, and uses Gemini again to **select and explain the best article** that answers the question.

---

## 💡 Key Features

- ✅ Arabic semantic search using `akhooli/Arabic-SBERT-100K`
- ✅ Reformulation of the legal query using Gemini
- ✅ Deduplicated multi-query search with article reranking
- ✅ Legal reasoning with constraints: جناية / جنحة
- ✅ React-based interactive chat UI
- ✅ CORS-enabled FastAPI API
- ✅ Local `.json` article loading (penal, commercial, civil)

---

## 🗂️ Folder Structure

```
LegalChatbot/
│
├── backend/                
│   ├── main.py             # FastAPI app
│   ├── semantic_search.py  # Normalization + embedding + search
│   ├── data/               # Legal article JSON files
│   └── corpus_emb_*.pt     # Saved SBERT embeddings
│
├── frontend/               
│   ├── src/                
│   │   ├── components/     
│   │   ├── context/        
│   │   └── App.tsx         
│   ├── public/             
│   └── .env                # Contains VITE_API_BASE=http://localhost:8000
```

---

## 🚀 Getting Started

### 1. Backend Setup (FastAPI)

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Or `venv\Scripts\activate` on Windows

pip install -r requirements.txt
```

**Set up your `.env` file:**

```
GEMINI_API_KEY=your_google_api_key
```

**Run the backend:**

```bash
uvicorn main:app --reload
```

✅ This will start the API at `http://127.0.0.1:8000`

---

### 2. Frontend Setup (React + Vite)

```bash
cd frontend
npm install
```

**Create a `.env` file:**

```
VITE_API_BASE=http://127.0.0.1:8000
```

**Start the development server:**

```bash
npm run dev
```

✅ App runs at `http://localhost:5173`

---

## 💬 API Endpoint

**POST /search**

```json
{
  "query": "ما هي عقوبة السرقة بالإكراه؟",
  "top_k": 150
}
```

**Returns:**

- `answer`: the selected legal article with reasoning (Gemini-generated)
- `sources`: list of top articles (from dense search)

---

## 📚 Requirements

### Backend

- `fastapi`
- `torch`
- `sentence-transformers`
- `google.generativeai`
- `python-dotenv`
- `pyarabic`

### Frontend

- `React`
- `Vite`
- `lucide-react`
- `TailwindCSS`

---

## 🧠 Example Use Case

> **User:** ما هي عقوبة الجريمة ضد موظف حكومي؟  
> **Bot:** المادة 123: "كل من يعتدي على موظف أثناء وظيفته..."  
> **Explanation:** اخترنا هذه المادة لأنها تتعلق مباشرةً بالاعتداء على موظفين عموميين كما هو موضح في السؤال.

---

## 📄 License

MIT — built with love for Lebanon 🇱🇧  
```Done with love to serve my country and empower access to justice.``` ❤️

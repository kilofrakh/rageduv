# RAGedu — AI Tutoring Pipeline for University Courses

## Architecture

```
Professor uploads PDFs/slides
         │
         ▼
   ┌─────────────┐
   │  ingest.py  │  Parse → Chunk → Embed → ChromaDB
   └─────────────┘
         │
         ▼
   ┌─────────────────┐
   │   ChromaDB      │  Local vector store (persisted to disk)
   └─────────────────┘
         │
    ┌────┴────────────────┐
    ▼                     ▼
┌──────────────┐   ┌──────────────────┐
│ rag_pipeline │   │ quiz_generator   │
│   .py        │   │      .py         │
│ Retrieve →   │   │ Retrieve →       │
│ Generate →   │   │ Generate MCQs    │
│ Cite sources │   └──────────────────┘
└──────────────┘
    │
    ▼
┌─────────────┐
│ analytics   │  SQLite: logs all Q&A, surfaces struggle patterns
│    .py      │
└─────────────┘
    │
    ▼
┌─────────┐
│  api.py │  FastAPI — exposes everything as REST endpoints
└─────────┘
```

## Setup

```bash
# 1. Clone / copy the project
cd ragedu

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Start the server
python api.py
# → http://localhost:8000
# → http://localhost:8000/docs  (auto-generated Swagger UI)
```

## Quick Start (CLI)

```bash
# Ingest a single file
python ingest.py ./materials/lecture1.pdf "Machine Learning" "Dr. Smith"

# Ingest an entire folder
python ingest.py ./materials/ "Machine Learning" "Dr. Smith"

# Ask a question
python rag_pipeline.py "What is the difference between supervised and unsupervised learning?"

# Generate a quiz
python quiz_generator.py "neural networks"
```

## API Reference

| Method | Endpoint | Who | Description |
|--------|----------|-----|-------------|
| POST | `/upload` | Professor | Upload a course document |
| POST | `/ask` | Student | Ask a question, get grounded answer |
| POST | `/quiz` | Professor | Generate MCQs on a topic |
| GET | `/dashboard` | Professor | Summary stats |
| GET | `/dashboard/topics` | Professor | Top struggle topics |
| GET | `/dashboard/unanswered` | Professor | Questions AI couldn't answer |
| GET | `/dashboard/volume` | Professor | Daily question volume |

## Example API Calls

### Upload a document
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@lecture1.pdf" \
  -F "subject=Machine Learning" \
  -F "professor=Dr. Smith"
```

### Ask a question
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is gradient descent?", "student_id": "student_42"}'
```

### Generate a quiz
```bash
curl -X POST http://localhost:8000/quiz \
  -H "Content-Type: application/json" \
  -d '{"topic": "backpropagation", "num_questions": 5, "difficulty": "medium"}'
```

## File Structure

```
ragedu/
├── config.py           # All settings in one place
├── ingest.py           # Stage 1: parse, chunk, embed, store
├── rag_pipeline.py     # Stage 2: retrieve, generate, cite
├── quiz_generator.py   # Stage 3: topic → MCQs
├── analytics.py        # Stage 4: log, aggregate, surface patterns
├── api.py              # FastAPI server
├── requirements.txt
├── .env.example
├── uploads/            # Uploaded files (auto-created)
└── chroma_db/          # Vector store (auto-created)
```

## Key Design Decisions

- **Local embeddings** (`all-MiniLM-L6-v2`) — no API cost, fast, runs on CPU
- **ChromaDB** — simple, file-based, no infrastructure needed for a demo
- **Source citations** — every answer cites the exact file it came from (critical for professor trust)
- **Unanswered question tracking** — highest value feature for professors; shows exactly what's missing from their materials
- **Deduplication** — re-uploading the same file doesn't create duplicate chunks (MD5 hash IDs)
- **Groq AI** — Powered by Groq's Llama 3 8B model for fast, intelligent Q&A and quiz generation

# api.py
"""
FastAPI Backend — RAGedu
------------------------
Endpoints:
  POST /upload          → Ingest a document (professor)
  POST /ask             → Student asks a question
  POST /quiz            → Professor generates a quiz
  GET  /dashboard       → Analytics summary
  GET  /dashboard/topics
  GET  /dashboard/unanswered
  GET  /dashboard/volume
  GET  /health
"""

import os
import shutil
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from ingest import ingest_file
from rag_pipeline import ask
from quiz_generator import generate_quiz
import analytics

app = FastAPI(
    title="RAGedu API",
    description="AI-powered tutoring assistant for university courses",
    version="1.0.0",
)

# Allow frontend (e.g. React on localhost:3000) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


# ── Document upload (professor) ───────────────────────────────────────────────
@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    subject:   str = Form("General"),
    professor: str = Form("Unknown"),
):
    """
    Upload a course material file (PDF, DOCX, PPTX, TXT).
    The file is saved locally then ingested into the vector store.
    """
    allowed = {".pdf", ".docx", ".pptx", ".txt"}
    ext     = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed}",
        )

    # Save the uploaded file
    save_path = os.path.join(config.UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Ingest
    try:
        result = ingest_file(save_path, subject=subject, professor=professor)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "message":  f"Successfully ingested '{file.filename}'",
        "details":  result,
    }


# ── Student Q&A ───────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question:       str
    subject_filter: Optional[str] = None
    student_id:     Optional[str] = None   # anonymized ID

@app.post("/ask")
def ask_question(body: AskRequest):
    """
    Student submits a question.
    Returns an answer grounded in uploaded course materials + source citations.
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = ask(
        question=body.question,
        subject_filter=body.subject_filter,
        student_id=body.student_id,
    )
    return result


# ── Quiz generation (professor) ───────────────────────────────────────────────
class QuizRequest(BaseModel):
    topic:          str
    num_questions:  int = 5
    subject_filter: Optional[str] = None
    difficulty:     str = "medium"

@app.post("/quiz")
def create_quiz(body: QuizRequest):
    """
    Professor requests a quiz on a specific topic.
    Returns a list of MCQs grounded in course materials.
    """
    if not body.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")
    if body.difficulty not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail="Difficulty must be easy, medium, or hard.")

    try:
        questions = generate_quiz(
            topic=body.topic,
            num_questions=body.num_questions,
            subject_filter=body.subject_filter,
            difficulty=body.difficulty,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"topic": body.topic, "questions": questions}


# ── Professor dashboard ───────────────────────────────────────────────────────
@app.get("/dashboard")
def dashboard_summary():
    """High-level stats: total questions, answer rate, top topics."""
    return analytics.get_summary()

@app.get("/dashboard/topics")
def dashboard_topics(limit: int = 10):
    """Most frequently asked-about topics."""
    return analytics.get_top_topics(limit)

@app.get("/dashboard/unanswered")
def dashboard_unanswered(limit: int = 20):
    """Questions the AI couldn't answer — needs prof attention."""
    return analytics.get_unanswered_questions(limit)

@app.get("/dashboard/volume")
def dashboard_volume():
    """Daily question volume over time."""
    return analytics.get_volume_over_time()


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

# analytics.py
"""
Stage 4 — Analytics & Struggle Detection
-----------------------------------------
Logs every student question and surfaces patterns for the professor.

Tables:
  - questions: every Q&A interaction
  - topic_frequency: aggregated topic counts (updated on each log)

Dashboard data (returned as dicts, rendered by the API/frontend):
  - top struggling topics
  - question volume over time
  - unanswered questions (no source found)
"""

from datetime import datetime
from typing import List, Optional
from collections import Counter
import json
import re

from sqlalchemy import (
    create_engine, Column, Integer, String, Text,
    DateTime, Boolean, Float, func
)
from sqlalchemy.orm import declarative_base, Session

import config

# ── DB setup ──────────────────────────────────────────────────────────────────
engine = create_engine(config.ANALYTICS_DB, connect_args={"check_same_thread": False})
Base   = declarative_base()


class QuestionLog(Base):
    __tablename__ = "questions"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    timestamp  = Column(DateTime, default=datetime.utcnow)
    question   = Column(Text, nullable=False)
    answer     = Column(Text, nullable=False)
    sources    = Column(Text)          # JSON list of filenames
    subject    = Column(String(128))
    student_id = Column(String(128))   # anonymized/optional
    answered   = Column(Boolean, default=True)   # False if no source found
    topic_tags = Column(Text)          # JSON list of extracted keywords


Base.metadata.create_all(engine)


# ── Keyword extractor (simple, no extra deps) ─────────────────────────────────
_STOPWORDS = {
    "what", "how", "why", "when", "where", "is", "are", "was", "were",
    "the", "a", "an", "of", "in", "to", "and", "or", "for", "with",
    "do", "does", "can", "could", "please", "explain", "define", "give",
    "me", "us", "i", "my", "we", "our", "you", "your",
}

def _extract_keywords(text: str, top_n: int = 3) -> List[str]:
    words  = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    counts = Counter(w for w in words if w not in _STOPWORDS)
    return [word for word, _ in counts.most_common(top_n)]


# ── Logging ───────────────────────────────────────────────────────────────────
def log_question(
    question: str,
    answer: str,
    sources: List[str],
    subject: str = "",
    student_id: Optional[str] = None,
) -> None:
    answered   = bool(sources)
    topic_tags = _extract_keywords(question)

    with Session(engine) as session:
        entry = QuestionLog(
            question   = question,
            answer     = answer,
            sources    = json.dumps(sources),
            subject    = subject,
            student_id = student_id,
            answered   = answered,
            topic_tags = json.dumps(topic_tags),
        )
        session.add(entry)
        session.commit()


# ── Dashboard queries ─────────────────────────────────────────────────────────
def get_top_topics(limit: int = 10) -> List[dict]:
    """Most frequently asked-about topics (from keyword tags)."""
    with Session(engine) as session:
        rows = session.query(QuestionLog.topic_tags).all()

    tag_counter: Counter = Counter()
    for (tags_json,) in rows:
        if tags_json:
            for tag in json.loads(tags_json):
                tag_counter[tag] += 1

    return [
        {"topic": topic, "count": count}
        for topic, count in tag_counter.most_common(limit)
    ]


def get_unanswered_questions(limit: int = 20) -> List[dict]:
    """Questions where the AI found no source material — highest priority for prof."""
    with Session(engine) as session:
        rows = (
            session.query(QuestionLog)
            .filter(QuestionLog.answered == False)
            .order_by(QuestionLog.timestamp.desc())
            .limit(limit)
            .all()
        )
    return [
        {
            "question":  r.question,
            "timestamp": r.timestamp.isoformat(),
            "subject":   r.subject,
        }
        for r in rows
    ]


def get_volume_over_time() -> List[dict]:
    """Daily question volume for the last 30 days."""
    with Session(engine) as session:
        rows = (
            session.query(
                func.date(QuestionLog.timestamp).label("date"),
                func.count(QuestionLog.id).label("count"),
            )
            .group_by(func.date(QuestionLog.timestamp))
            .order_by(func.date(QuestionLog.timestamp))
            .all()
        )
    return [{"date": str(r.date), "count": r.count} for r in rows]


def get_summary() -> dict:
    """High-level stats for the professor dashboard."""
    with Session(engine) as session:
        total      = session.query(func.count(QuestionLog.id)).scalar()
        unanswered = session.query(func.count(QuestionLog.id)).filter(
            QuestionLog.answered == False
        ).scalar()

    return {
        "total_questions":      total,
        "unanswered_questions": unanswered,
        "answer_rate":          round((total - unanswered) / total * 100, 1) if total else 0,
        "top_topics":           get_top_topics(5),
    }

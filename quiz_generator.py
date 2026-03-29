
import json
import re
from typing import List, Optional

from groq import Groq

import config
from ingest import get_vectorstore


# ── Prompt ────────────────────────────────────────────────────────────────────
QUIZ_SYSTEM_PROMPT = """You are an expert professor creating exam questions.
Given course material excerpts, generate multiple-choice questions.

Rules:
- Base every question STRICTLY on the provided material.
- Each question must have exactly 4 options (A, B, C, D).
- Only one option is correct.
- Include a brief explanation citing why the answer is correct.
- Output ONLY valid JSON — no preamble, no markdown, no extra text.

JSON format:
[
  {
    "question": "...",
    "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
    "answer": "A",
    "source": "<filename>",
    "explanation": "..."
  }
]
"""


def generate_quiz(
    topic: str,
    num_questions: int = 5,
    subject_filter: Optional[str] = None,
    difficulty: str = "medium",      
) -> List[dict]:
    """
    Generate MCQs on a given topic from course materials.

    Args:
        topic:          The topic/chapter to quiz on.
        num_questions:  How many questions to generate (max 10).
        subject_filter: Restrict retrieval to a specific subject.
        difficulty:     "easy", "medium", or "hard".

    Returns:
        List of question dicts (see module docstring for schema).
    """
    num_questions = min(num_questions, 10)  
    vs = get_vectorstore()

    # 1. Retrieve relevant material for this topic (fetch more for quiz variety)
    search_kwargs = {"k": min(num_questions * 2, 20)}
    if subject_filter:
        search_kwargs["filter"] = {"subject": subject_filter}

    docs = vs.similarity_search(topic, **search_kwargs)

    if not docs:
        raise ValueError(f"No course materials found for topic: '{topic}'")

    # 2. Build context
    context_parts = []
    for doc in docs:
        src = doc.metadata.get("source", "Unknown")
        context_parts.append(f"[From '{src}']\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    # 3. Build prompt
    user_message = f"""Topic: {topic}
Difficulty: {difficulty}
Number of questions: {num_questions}

Course material:
{context}

Generate exactly {num_questions} MCQ questions as JSON."""

    messages = [
        {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # 4. Generate
    client = Groq(api_key=config.GROQ_API_KEY)
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        max_tokens=2048,  # More tokens for quiz generation
        temperature=0.3,
    )
    raw = response.choices[0].message.content.strip()

    # 5. Parse JSON (strip any accidental markdown fences)
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    questions = json.loads(raw)

    # 6. Validate structure
    validated = []
    for q in questions:
        if all(k in q for k in ("question", "options", "answer", "explanation")):
            validated.append(q)

    return validated


# ── CLI helper ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Topic: ")
    questions = generate_quiz(topic, num_questions=5)
    for i, q in enumerate(questions, 1):
        print(f"\nQ{i}: {q['question']}")
        for letter, opt in q["options"].items():
            marker = "✓" if letter == q["answer"] else " "
            print(f"  {marker} {letter}) {opt}")
        print(f"   → {q['explanation']}")

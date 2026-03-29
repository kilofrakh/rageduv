"""
Stage 2 — Retrieval-Augmented Generation
-----------------------------------------
Steps:
  1. Embed the student's question
  2. Retrieve TOP_K most relevant chunks from ChromaDB
  3. Build a grounded prompt with retrieved context
  4. Generate answer via Claude
  5. Return answer + cited sources (filename + chunk)
"""

from typing import Optional
from groq import Groq
from langchain.schema import HumanMessage, SystemMessage

import config
from ingest import get_vectorstore


# ── LLM (loaded once) ─────────────────────────────────────────────────────────
_client = None

def get_llm():
    global _client
    if _client is None:
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an academic assistant for a university course.
Answer student questions using ONLY the provided course materials below.
Rules:
- Be clear, accurate, and helpful.
- If the answer is not in the materials, say: "I couldn't find this in the course materials. Please ask your professor."
- Always end your answer with a 'Sources:' section listing the filenames you used.
- Never invent information not present in the context.
"""

def _build_context(docs) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        parts.append(f"[{i}] From '{source}':\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ── Core RAG function ─────────────────────────────────────────────────────────
def ask(
    question: str,
    subject_filter: Optional[str] = None,
    student_id: Optional[str] = None,
) -> dict:
    """
    Answer a student question using RAG.

    Returns:
        {
            "answer":   str,
            "sources":  [{"filename": str, "subject": str}],
            "question": str,
        }
    """
    vs = get_vectorstore()

    # 1. Build search filter (optional: restrict to a specific subject)
    search_kwargs = {"k": config.TOP_K}
    if subject_filter:
        search_kwargs["filter"] = {"subject": subject_filter}

    # 2. Retrieve relevant chunks
    docs = vs.similarity_search(question, **search_kwargs)

    if not docs:
        return {
            "answer":   "I couldn't find any relevant course materials for this question. Please ask your professor.",
            "sources":  [],
            "question": question,
        }

    # 3. Build grounded prompt
    context = _build_context(docs)
    user_message = f"""Course materials:
{context}

Student question: {question}

Answer based strictly on the materials above."""

    # 4. Generate answer
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    response = get_llm().chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        max_tokens=1024,
        temperature=0.2,
    )
    answer = response.choices[0].message.content

    # 5. Deduplicate sources
    seen    = set()
    sources = []
    for doc in docs:
        fn = doc.metadata.get("source", "Unknown")
        if fn not in seen:
            seen.add(fn)
            sources.append({
                "filename": fn,
                "subject":  doc.metadata.get("subject", ""),
            })

    # 6. Log the interaction (non-blocking)
    try:
        from analytics import log_question
        log_question(
            question=question,
            answer=answer,
            sources=[s["filename"] for s in sources],
            subject=subject_filter or (sources[0]["subject"] if sources else ""),
            student_id=student_id,
        )
    except Exception:
        pass  # analytics failure should never break the student experience

    return {
        "answer":   answer,
        "sources":  sources,
        "question": question,
    }


# ── CLI helper ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Question: ")
    result   = ask(question)
    print("\n" + result["answer"])
    print("\nSources:", [s["filename"] for s in result["sources"]])

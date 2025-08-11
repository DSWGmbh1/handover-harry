from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import openai
import os
import re
from datetime import datetime

app = FastAPI()

# Set your OpenAI API Key securely via environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")


# ---------- Data Models ----------

class DocumentChunk(BaseModel):
    content: str
    filename: str
    chunk_index: int
    similarity: float
    page: Optional[int] = None
    heading: Optional[str] = None
    document_type: Optional[str] = None


class Metadata(BaseModel):
    chunks_found: int
    documents_searched: int
    processing_time_ms: int


class QuestionRequest(BaseModel):
    question: str
    user_language: str
    document_language: Optional[str] = None
    project_id: Optional[str] = None
    user_id: Optional[str] = None
    document_context: List[DocumentChunk]
    metadata: Metadata


class QuestionResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[dict]
    suggested_next_questions: List[str]
    query_type: str
    processing_time_ms: float
    chunks_analyzed: int
    document_types_used: List[str]


# ---------- Helper: Detect query type ----------

def detect_query_type(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ["maintain", "service", "wartung", "pflege", "maintenance", "interval", "wechsel"]):
        return "maintenance"
    if any(k in q for k in ["replace", "change", "tauschen", "austausch"]):
        return "replacement"
    if any(k in q for k in ["install", "mount", "einbauen", "installation"]):
        return "installation"
    if any(k in q for k in ["where", "wo ", "location", "standort"]):
        return "location"
    if any(k in q for k in ["filter", "filterklasse", "hepa", "iso 16890", "en 779"]):
        return "filter_specs"
    return "general"


# ---------- Main Endpoint ----------

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(req: QuestionRequest):
    started = datetime.utcnow()

    if not openai.api_key:
        return QuestionResponse(
            answer="",
            confidence=0.0,
            sources=[],
            suggested_next_questions=[],
            query_type="error",
            processing_time_ms=0.0,
            chunks_analyzed=0,
            document_types_used=[]
        )

    query_type = detect_query_type(req.question)

    # --- Step 1: sort by similarity ---
    sorted_chunks = sorted(req.document_context, key=lambda c: c.similarity, reverse=True)

    # Adjust thresholds based on query type
    if query_type == "maintenance":
        threshold = 0.50
    elif query_type == "location":
        threshold = 0.45
    elif query_type == "general":
        threshold = 0.55
    else:
        threshold = 0.50

    candidate_chunks = [c for c in sorted_chunks if c.similarity >= threshold][:50]

    if not candidate_chunks:
        duration = (datetime.utcnow() - started).total_seconds() * 1000.0
        return QuestionResponse(
            answer="",
            confidence=0.0,
            sources=[],
            suggested_next_questions=[],
            query_type=query_type,
            processing_time_ms=round(duration, 2),
            chunks_analyzed=0,
            document_types_used=[]
        )

    # --- Step 2: rerank (optional but helps) ---
    def chunk_score(c: DocumentChunk) -> float:
        score = c.similarity
        if query_type == "filter_specs" and re.search(r"(filter|klasse|iso|en\s?779|hepa)", c.content, re.I):
            score += 0.05
        if query_type == "maintenance" and re.search(r"(maint|service|wartung|interval|wechsel)", c.content, re.I):
            score += 0.05
        return score

    candidate_chunks = sorted(candidate_chunks, key=chunk_score, reverse=True)[:12]

    # --- Step 3: build context ---
    context_text = ""
    for c in candidate_chunks:
        heading_txt = f"[{c.heading}] " if c.heading else ""
        page_txt = f"(p.{c.page}) " if c.page else ""
        context_text += f"From '{c.filename}' {page_txt}chunk {c.chunk_index} (similarity {c.similarity:.2f}): {heading_txt}{c.content}\n\n"

    # --- Step 4: system prompt ---
    system_prompt = f"""
You are a highly accurate assistant for construction handover documentation.
Answer ONLY using the provided CONTEXT. Quote exact sentences when possible.

Rules:
- Always answer in {req.user_language}.
- If a clear, exact answer exists in CONTEXT, quote it first.
- If not found, say: "Not stated in the documents I can access."
- Avoid guessing.
- Be specific: include maintenance intervals, specs, model numbers, locations.
- Suggested questions should help the user explore related info.

--- CONTEXT START ---
{context_text}
--- CONTEXT END ---
"""

    # --- Step 5: Call GPT ---
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": req.question}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        full_response_text = completion.choices[0].message["content"].strip()
    except Exception as e:
        duration = (datetime.utcnow() - started).total_seconds() * 1000.0
        return QuestionResponse(
            answer="",
            confidence=0.0,
            sources=[],
            suggested_next_questions=[],
            query_type="error",
            processing_time_ms=round(duration, 2),
            chunks_analyzed=len(candidate_chunks),
            document_types_used=list({c.document_type for c in candidate_chunks if c.document_type})
        )

    # --- Step 6: extract suggested questions ---
    suggestions: List[str] = []
    try:
        marker = "Suggested next questions:"
        if marker in full_response_text:
            _, tail = full_response_text.split(marker, 1)
            suggestions = [x.strip(" -•\t").strip() for x in tail.strip().split("\n") if x.strip()]
            suggestions = [s for s in suggestions if len(s) > 5][:4]
    except Exception:
        pass

    # --- Step 7: confidence ---
    sim_avg = sum(c.similarity for c in candidate_chunks) / max(1, len(candidate_chunks))
    confidence = min(1.0, max(0.0, 0.65 * sim_avg + 0.25))

    # --- Step 8: NEW — Grounded check for fallback ---
    grounded = (
        ('Exact quote:' in full_response_text)
        and (len(re.sub(r'(?i)Exact quote:\s*', '', full_response_text).strip()) > 0)
        and ('Not stated in the documents I can access' not in full_response_text)
    )

    if not grounded and sim_avg < 0.42:
        # Force Lovable to fallback to OpenRouter by returning empty
        duration = (datetime.utcnow() - started).total_seconds() * 1000.0
        return QuestionResponse(
            answer="",
            confidence=0.0,
            sources=[],
            suggested_next_questions=[],
            query_type=query_type,
            processing_time_ms=round(duration, 2),
            chunks_analyzed=len(candidate_chunks),
            document_types_used=list({c.document_type for c in candidate_chunks if c.document_type})
        )

    # --- Step 9: return grounded answer ---
    srcs = [{
        "filename": c.filename,
        "chunk_index": c.chunk_index,
        "similarity": round(c.similarity, 3),
        "page": c.page,
        "heading": c.heading
    } for c in candidate_chunks[:8]]

    duration = (datetime.utcnow() - started).total_seconds() * 1000.0
    return QuestionResponse(
        answer=full_response_text.strip(),
        confidence=round(confidence, 3),
        sources=srcs,
        suggested_next_questions=suggestions,
        query_type=query_type,
        processing_time_ms=round(duration, 2),
        chunks_analyzed=len(candidate_chunks),
        document_types_used=list({c.document_type for c in candidate_chunks if c.document_type})
    )







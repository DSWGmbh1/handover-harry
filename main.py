from __future__ import annotations

import os
import re
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

import openai

# ---------------------------
# Setup
# ---------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("handab-backend")

app = FastAPI(title="HandAb Assistant API", version="2.0.0")

# OpenAI (or compatible OpenRouter with OpenAI SDK-compatible endpoint)
openai.api_key = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not openai.api_key:
    log.warning("OPENAI_API_KEY is not set. /ask will fail until it is configured.")

# ---------------------------
# Request/Response Models
# ---------------------------

class DocumentChunk(BaseModel):
    content: str = Field(..., description="Chunk text")
    filename: str = Field(..., description="Source filename")
    chunk_index: int = Field(..., description="Index of the chunk within the doc")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Retriever similarity")
    document_type: Optional[str] = Field(None, description="manual, plan, warranty, ...")
    page: Optional[int] = Field(None, description="Original PDF page number (1-based if possible)")
    heading: Optional[str] = Field(None, description="Nearest section/heading text for this chunk")
    language_detected: Optional[str] = Field(None, description="Language of the chunk, if known")
    section_types: Optional[List[str]] = Field(default_factory=list, description="Detected categories (hvac, electrical, etc.)")
    extracted_entities: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    visual_elements: Optional[List[Dict[str, Any]]] = Field(default_factory=list)

    @validator("content")
    def strip_content(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("Chunk content is empty")
        return v


class ConstructionMetadata(BaseModel):
    chunks_found: int = 0
    documents_searched: int = 0
    processing_time_ms: int = 0
    document_types_searched: List[str] = Field(default_factory=list)
    confidence_breakdown: Dict[str, float] = Field(default_factory=dict)


class AskRequest(BaseModel):
    question: str
    user_language: str = "English"
    project_id: str
    user_id: str
    document_context: List[DocumentChunk]
    metadata: Optional[ConstructionMetadata] = None
    query_type: Optional[str] = Field(
        None, description="location | maintenance | specifications | general"
    )


class AskResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    suggested_next_questions: List[str]
    query_type: str
    processing_time_ms: float
    chunks_analyzed: int
    document_types_used: List[str]


# ---------------------------
# Query classification (quick, robust)
# ---------------------------
_Q_LOC = re.compile(r"\b(where|located|position|find|standort|wo|ubicat|ubicación|dove)\b", re.I)
_Q_MAINT = re.compile(r"\b(maintain|maintenance|service|schedule|how often|interval|wechsel|tauschen|wartung|entretien|manutenzione)\b", re.I)
_Q_SPEC = re.compile(r"\b(model|serial|part|capacity|rating|size|dimensions|voltage|amperage|wattage|btu|spec|iso|en\s?779|16890|class|klasse)\b", re.I)

def classify_query(q: str) -> str:
    if _Q_LOC.search(q): return "location"
    if _Q_MAINT.search(q): return "maintenance"
    if _Q_SPEC.search(q): return "specifications"
    return "general"


# ---------------------------
# Re-ranking with heading/page boosts
# ---------------------------

def _normalize(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def _tokenize(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9äöüÄÖÜßéèêàáíóúç\-]+", s.lower())

def score_chunk(
    chunk: DocumentChunk,
    query: str,
    query_type: str,
    median_page: Optional[float]
) -> float:
    # base: retriever similarity
    score = 0.78 * float(chunk.similarity)

    q = _normalize(query)
    heading = _normalize(chunk.heading)

    q_terms = set(_tokenize(q))
    h_terms = set(_tokenize(heading))
    c_terms = set(_tokenize(chunk.content[:600]))  # look at the first 600 chars

    # heading boost: if heading shares content words with query
    overlap_h = len(q_terms & h_terms)
    if overlap_h > 0:
        score += 0.10 * min(1.0, overlap_h / 3.0)

    # keyword presence boost (chunk body)
    overlap_c = len(q_terms & c_terms)
    if overlap_c > 0:
        score += 0.06 * min(1.0, overlap_c / 5.0)

    # page proximity (helps when user mentions "page 12" or we cluster around median)
    page_hint = None
    m = re.search(r"\bpage\s*(\d{1,3})\b", q)
    if m:
        try:
            page_hint = int(m.group(1))
        except Exception:
            page_hint = None

    if page_hint and chunk.page:
        # exact or near-exact page gets a nudge
        diff = abs(int(chunk.page) - page_hint)
        score += 0.06 * max(0.0, 1.0 - min(diff, 5) / 5.0)
    elif median_page is not None and chunk.page is not None:
        # keep related pages together slightly
        diff = abs(int(chunk.page) - int(median_page))
        score += 0.03 * max(0.0, 1.0 - min(diff, 8) / 8.0)

    # light domain/taxonomy nudge
    if query_type == "maintenance":
        if any(w in c_terms for w in ("filter", "filterwechsel", "entretien", "manutenzione", "wartung", "schedule", "interval")):
            score += 0.03

    return float(min(score, 1.0))


def rerank_chunks(chunks: List[DocumentChunk], query: str, query_type: str, top_k: int = 12) -> List[DocumentChunk]:
    pages = [c.page for c in chunks if c.page is not None]
    median_page = None
    if pages:
        pages_sorted = sorted(pages)
        n = len(pages_sorted)
        median_page = pages_sorted[n // 2] if n % 2 else (pages_sorted[n // 2 - 1] + pages_sorted[n // 2]) / 2.0

    scored: List[Tuple[float, DocumentChunk]] = []
    for c in chunks:
        scored.append((score_chunk(c, query, query_type, median_page), c))

    # stable sort by (score DESC, similarity DESC)
    scored.sort(key=lambda x: (x[0], x[1].similarity), reverse=True)

    # de-duplicate by filename+page to avoid spammy repeats
    seen = set()
    result: List[DocumentChunk] = []
    for s, c in scored:
        key = (c.filename, c.page)
        if key not in seen:
            seen.add(key)
            result.append(c)
        if len(result) >= top_k:
            break
    return result


# ---------------------------
# Context builder with citations
# ---------------------------

def build_context(chunks: List[DocumentChunk], limit_chars: int = 32000) -> str:
    parts = []
    total = 0
    for c in chunks:
        tag = f"[{c.filename}"
        if c.page is not None:
            tag += f" · p.{c.page}"
        if c.heading:
            clean_h = c.heading.strip().replace("\n", " ")
            tag += f" · {clean_h[:100]}"
        tag += f" · chunk {c.chunk_index}]"

        section = f"{tag}\n{c.content.strip()}\n"
        if total + len(section) > limit_chars:
            break
        parts.append(section)
        total += len(section)
    return "\n\n".join(parts)


# ---------------------------
# System Prompt
# ---------------------------

def system_prompt(query_type: str) -> str:
    extras = ""
    if query_type == "maintenance":
        extras = (
            "- If the context provides maintenance **intervals**, **frequencies**, or **procedures**, list them explicitly.\n"
            "- Prefer exact statements over paraphrase. If the context shows **Filterwartung** or **Filterwechsel** steps, include them.\n"
        )
    elif query_type == "location":
        extras = (
            "- Be specific about floors/rooms/areas if provided. Use plan references.\n"
        )
    elif query_type == "specifications":
        extras = "- Prefer formal designations (e.g., ISO 16890 ePM1 50%, EN 779 F7) exactly as written.\n"

    return f"""You are a specialist assistant for construction handover documentation.
Use ONLY the information in the CONTEXT to answer. If info is missing, say:
"The provided documents do not contain enough information to answer this question."

General rules:
- Answer clearly in the user's language.
- Quote exact spec words when possible (e.g., filter classes, standards).
- Include inline citations using the [filename · p.X · heading · chunk N] labels that precede each context block.
- If multiple places conflict, say that and present both with citations.
- Finish with 3-4 "Suggested next questions".

Additional guidance for this query type ({query_type}):
{extras}
"""


# ---------------------------
# /health
# ---------------------------

@app.get("/health")
def health():
    return {"ok": True, "openai": bool(openai.api_key), "model": OPENAI_MODEL, "ts": datetime.utcnow().isoformat()}


# ---------------------------
# /ask
# ---------------------------

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not openai.api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

    start = datetime.utcnow()

    qtype = req.query_type or classify_query(req.question)
    # filter out empty/very low similarity chunks (keep some recall)
    raw_chunks = [c for c in req.document_context if c.content and c.similarity >= 0.35]

    if not raw_chunks:
        return AskResponse(
            answer="I couldn't find enough relevant passages in the documents to answer. Please try rephrasing or upload clearer documents.",
            confidence=0.0,
            sources=[],
            suggested_next_questions=[],
            query_type=qtype,
            processing_time_ms=0.0,
            chunks_analyzed=0,
            document_types_used=[],
        )

    # Re-rank with heading/page boosts
    selected = rerank_chunks(raw_chunks, req.question, qtype, top_k=12)

    # Build context with rich citations
    context = build_context(selected, limit_chars=32000)

    sys_prompt = system_prompt(qtype)
    user_prompt = (
        f"User question (respond in {req.user_language}): {req.question}\n\n"
        f"--- CONTEXT START ---\n{context}\n--- CONTEXT END ---\n\n"
        "Respond using ONLY the context above. Include the inline citation labels exactly as [filename · p.X · heading · chunk N] "
        "after the facts you cite. End with a line that starts with 'Suggested next questions:' followed by 3-4 short bullets."
    )

    # Call model
    try:
        completion = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            temperature=0.1,
            max_tokens=1000,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        full = completion.choices[0].message["content"].strip()
    except Exception as e:
        log.exception("OpenAI error")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")

    # Split suggested next questions if present
    suggestions: List[str] = []
    marker = "Suggested next questions:"
    answer = full
    if marker in full:
        try:
            answer, tail = full.split(marker, 1)
            suggestions = [
                s.strip(" -•\t").strip()
                for s in tail.strip().split("\n")
                if len(s.strip()) > 3
            ][:4]
        except Exception:
            pass

    # Confidence: blend of similarity + heading boost achieved in rerank
    avg_sim = sum(c.similarity for c in selected) / max(1, len(selected))
    confidence = max(0.0, min(1.0, 0.75 * avg_sim + 0.15))  # conservative nudge

    # Sources
    sources = []
    for c in selected[:10]:
        sources.append({
            "filename": c.filename,
            "page": c.page,
            "heading": c.heading,
            "chunk_index": c.chunk_index,
            "similarity": round(float(c.similarity), 3),
            "document_type": c.document_type,
        })

    dur_ms = (datetime.utcnow() - start).total_seconds() * 1000.0
    return AskResponse(
        answer=answer.strip(),
        confidence=round(confidence, 3),
        sources=sources,
        suggested_next_questions=suggestions,
        query_type=qtype,
        processing_time_ms=round(dur_ms, 2),
        chunks_analyzed=len(selected),
        document_types_used=sorted(list({c.document_type for c in selected if c.document_type})),
    )


# ---------------------------
# Root
# ---------------------------

@app.get("/")
def root():
    return {
        "name": "HandAb Assistant API",
        "version": "2.0.0",
        "endpoints": {"health": "/health", "ask": "/ask"},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))








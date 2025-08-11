# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import logging
from datetime import datetime

# Regex: use 'regex' if available, else stdlib 're'
try:
    import regex as re
except Exception:
    import re  # type: ignore

# -----------------
# Setup & logging
# -----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("handover-backend")

app = FastAPI(title="HandAb Construction Q&A", version="2.0.0")

# OpenAI init (works with both new and legacy libs)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is not set.")

# -----------------
# Models (I/O)
# -----------------
class DocumentChunk(BaseModel):
    content: str
    filename: str
    chunk_index: int
    similarity: float
    # Optional metadata (send if you have them — safe to omit)
    document_type: Optional[str] = "unknown"
    page: Optional[int] = None
    heading: Optional[str] = None
    # Extendable bucket fields
    extracted_entities: List[Dict[str, Any]] = Field(default_factory=list)
    visual_elements: List[Dict[str, Any]] = Field(default_factory=list)

class ConstructionMetadata(BaseModel):
    chunks_found: int
    documents_searched: int
    processing_time_ms: int
    document_types_searched: List[str] = Field(default_factory=list)

class EnhancedQuestionRequest(BaseModel):
    question: str
    user_language: str = "English"
    document_language: str = "English"
    project_id: str
    user_id: str
    document_context: List[DocumentChunk]
    metadata: ConstructionMetadata
    query_type: Optional[str] = None  # "location" | "maintenance" | "specifications" | "general"

class QuestionResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    suggested_next_questions: List[str] = Field(default_factory=list)
    query_type: str = "general"
    processing_time_ms: float = 0.0
    chunks_analyzed: int = 0
    document_types_used: List[str] = Field(default_factory=list)

# -----------------
# Query classifier (lightweight)
# -----------------
LOC_WORDS = ["where", "located", "find", "position", "wo", "wo ist", "dove", "ubicación", "ubicato"]
MAINT_WORDS = ["maintain", "maintenance", "wartung", "replace", "wechsel", "interval", "schedule", "intervall", "frequenz", "häufigkeit", "how often", "wie oft"]
SPEC_WORDS = ["spec", "specification", "data", "rating", "capacity", "voltage", "btu", "iso", "en", "filterklasse", "ePM"]

def classify_query(q: str) -> str:
    s = q.lower()
    if any(w in s for w in LOC_WORDS): return "location"
    if any(w in s for w in MAINT_WORDS): return "maintenance"
    if any(w in s for w in SPEC_WORDS): return "specifications"
    return "general"

# -----------------
# Spec/Maintenance patterns
# -----------------
FREQ_PATTERNS = [
    (r'\bviertelj[aä]hrlich|quartalsweise\b', 'every 3 months'),
    (r'\bhalbj[aä]hrlich\b', 'every 6 months'),
    (r'\bj[aä]hrlich|einmal pro jahr|annually|yearly\b', 'every 12 months'),
    (r'\bevery\s+(\d+)\s*(months|month|mo)\b', lambda m: f"every {m.group(1)} months"),
    (r'\balle\s+(\d+)\s*(monate|monat|mo)\b', lambda m: f"every {m.group(1)} months"),
    (r'\balle\s+(\d+)\s*(wochen|woche)\b', lambda m: f"every {int(m.group(1))*7} days"),
    (r'\balle\s+(\d+)\s*(jahre|jahr)\b', lambda m: f"every {int(m.group(1))*12} months"),
]

FILTER_SECTION_CUES = re.compile(
    r'(filter(wartung|wechsel|pflege|maintenance|replacement)|wartung|maintenance)',
    re.I
)

HEADING_RE = re.compile(r'^\s*\d+(\.\d+)*\s+\S+.*', re.M)

def normalize_frequency(text: str) -> Optional[tuple[str, str]]:
    """
    Look for maintenance frequency in the given text.
    Returns (normalized_value, exact_quote_line) or None.
    """
    if not text:
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    window = "\n".join(lines)
    for pat, norm in FREQ_PATTERNS:
        m = re.search(pat, window, flags=re.I)
        if m:
            normalized = norm(m) if callable(norm) else norm
            # pick a representative line that matched
            quote = next((ln for ln in lines if re.search(pat, ln, re.I)), lines[0])
            return normalized, quote
    return None

def nearest_heading(text: str) -> Optional[str]:
    if not text:
        return None
    m = HEADING_RE.search(text)
    return m.group(0).strip() if m else None

# -----------------
# Reranker
# -----------------
def rerank_chunks_for_answer(chunks: List[DocumentChunk], question: str) -> List[DocumentChunk]:
    """Boost chunks that look like they contain filter/maintenance or section headings."""
    q = (question or "").lower()
    scored: List[tuple[float, DocumentChunk]] = []
    for c in chunks:
        score = c.similarity
        t = (c.content or "").lower()
        if FILTER_SECTION_CUES.search(t):
            score += 0.25
        # simple multilingual cues
        for kw in ("vierteljährlich", "quartal", "quartalsweise", "monat", "monate", "jährlich", "annually", "how often", "wie oft"):
            if kw in t:
                score += 0.1
        # heading present?
        if nearest_heading(c.content or ""):
            score += 0.05
        # tiny boost for maintenance/spec queries
        if any(w in q for w in ("filter", "wartung", "maintenance", "wechsel", "replace")):
            score += 0.05
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]

# -----------------
# Context building
# -----------------
def build_context(chunks: List[DocumentChunk], max_chars: int = 32000) -> str:
    parts: List[str] = []
    total = 0
    for c in chunks:
        header_bits = []
        if c.document_type and c.document_type != "unknown":
            header_bits.append(c.document_type.upper())
        if c.filename:
            header_bits.append(f"'{c.filename}'")
        header = " - ".join(header_bits) if header_bits else c.filename
        meta = []
        if c.page: meta.append(f"p.{c.page}")
        meta.append(f"chunk {c.chunk_index}")
        meta.append(f"sim {c.similarity:.2f}")
        heading = c.heading or nearest_heading(c.content or "") or ""
        head_line = f"=== {header} ({', '.join(meta)}) ==="
        if heading:
            head_line += f"\n{heading}"
        section = f"{head_line}\n{(c.content or '').strip()}\n"
        if total + len(section) > max_chars:
            break
        parts.append(section)
        total += len(section)
    return "\n".join(parts)

# -----------------
# Health & root
# -----------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "openai": bool(OPENAI_API_KEY),
        "time": datetime.utcnow().isoformat()
    }

@app.get("/")
async def root():
    return {
        "name": "HandAb Construction Q&A",
        "version": "2.0.0",
        "endpoints": { "health": "/health", "ask": "/ask", "docs": "/docs" }
    }

# -----------------
# Main Q&A route
# -----------------
@app.post("/ask", response_model=QuestionResponse)
async def ask(req: EnhancedQuestionRequest):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    started = datetime.utcnow()

    # Classify query
    query_type = req.query_type or classify_query(req.question)

    # Sort by similarity then apply reranker
    sorted_chunks = sorted(req.document_context, key=lambda c: c.similarity, reverse=True)
    # soft threshold (allow a bit more recall)
    thresholds = {"location": 0.45, "maintenance": 0.50, "specifications": 0.50, "general": 0.55}
    th = thresholds.get(query_type, 0.55)
    candidate_chunks = [c for c in sorted_chunks if c.similarity >= th] or sorted_chunks[:12]
    candidate_chunks = candidate_chunks[:50]  # cap
    candidate_chunks = rerank_chunks_for_answer(candidate_chunks, req.question)[:12]

    if not candidate_chunks:
        return QuestionResponse(
            answer="I couldn’t find enough relevant information in your documents to answer this.",
            confidence=0.0,
            sources=[],
            suggested_next_questions=[],
            query_type=query_type,
        )

    # Try spec/maintenance scanner first (quote-first, deterministic)
    spec_hit = None
    spec_chunk = None
    if query_type in ("maintenance", "specifications", "general") or any(x in req.question.lower() for x in ["filter", "wartung", "maintenance", "wechsel", "replace"]):
        for c in candidate_chunks:
            hit = normalize_frequency(c.content or "")
            if hit:
                spec_hit = hit  # (normalized, quote)
                spec_chunk = c
                break

    if spec_hit and spec_chunk:
        normalized, quote = spec_hit
        src = spec_chunk.filename or "source"
        if spec_chunk.page:
            src = f"{src} p.{spec_chunk.page}"
        answer_text = f'**Exact quote:** "{quote}"\n\n**Answer:** Filter maintenance is **{normalized}**. [{src}]'
        duration = (datetime.utcnow() - started).total_seconds() * 1000
        return QuestionResponse(
            answer=answer_text.strip(),
            confidence=0.9,
            sources=[{
                "filename": spec_chunk.filename,
                "chunk_index": spec_chunk.chunk_index,
                "similarity": round(spec_chunk.similarity, 3),
                "page": spec_chunk.page,
                "heading": spec_chunk.heading
            }],
            suggested_next_questions=[],
            query_type=query_type,
            processing_time_ms=round(duration, 2),
            chunks_analyzed=len(candidate_chunks),
            document_types_used=list({c.document_type for c in candidate_chunks if c.document_type})
        )

    # Build context for the LLM
    context_text = build_context(candidate_chunks, max_chars=35000)

    # System & user prompts (quote-first, no guessing)
    system_prompt = f"""You are a construction handover assistant. Use ONLY the CONTEXT.

When answering:
- First, provide the exact line(s) from the CONTEXT that answer the user, under the heading "Exact quote:" (verbatim).
- Then provide a one-sentence paraphrase under "Answer:" in the user's language ({req.user_language}).
- Always include a source like [filename p.X] if page is available, else [filename].
- If the context does not contain the exact answer, reply: "Not stated in the documents I can access."
- Do NOT guess or generalize beyond the quoted text.

QUERY TYPE: {query_type}
"""

    user_prompt = f"""Question: {req.question}

CONTEXT:
{context_text}

RESPONSE FORMAT (strict):
Exact quote: "<verbatim lines from context or leave blank if none>"
Answer: "<one sentence answer in {req.user_language}>"
Sources: "[filename p.X]" or "[filename]"
Suggested next questions:
- <3 short related questions>
"""

    # Call OpenAI
    full_response_text = ""
    try:
        # New client first
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.1,
                max_tokens=900,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            full_response_text = (completion.choices[0].message.content or "").strip()
        except Exception:
            # Legacy fallback
            import openai
            openai.api_key = OPENAI_API_KEY
            completion = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                temperature=0.1,
                max_tokens=900,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            full_response_text = completion.choices[0].message["content"].strip()
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

    # Parse suggested questions (best-effort)
    suggestions: List[str] = []
    try:
        marker = "Suggested next questions:"
        if marker in full_response_text:
            _, tail = full_response_text.split(marker, 1)
            suggestions = [x.strip(" -•\t").strip() for x in tail.strip().split("\n") if x.strip()]
            suggestions = [s for s in suggestions if len(s) > 5][:4]
    except Exception:
        pass

    # Confidence heuristic
    sim_avg = sum(c.similarity for c in candidate_chunks) / max(1, len(candidate_chunks))
    confidence = min(1.0, max(0.0, 0.65 * sim_avg + 0.25))

    # Sources list
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

# -----------------
# Local dev entry
# -----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))





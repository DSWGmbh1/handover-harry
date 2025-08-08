from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import openai
import os

app = FastAPI()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------- Models ----------
class DocumentChunk(BaseModel):
    content: str
    filename: str
    chunk_index: int
    similarity: float
    # optional: include page if you have it
    # page: int | None = None

class Metadata(BaseModel):
    chunks_found: int
    documents_searched: int
    processing_time_ms: int

class QuestionRequest(BaseModel):
    question: str
    user_language: str
    document_language: str
    project_id: str
    user_id: str
    document_context: List[DocumentChunk]
    metadata: Metadata

# ---------- Helpers ----------
def dedupe_chunks(chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """Remove near-duplicate chunks by content prefix + same file/index."""
    seen = set()
    out = []
    for c in chunks:
        key = (c.filename, c.chunk_index, c.content[:120].strip().lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out

def mmr_select(chunks: List[DocumentChunk], k: int = 8, lambda_weight: float = 0.75) -> List[DocumentChunk]:
    """
    Simple MMR-like selection using only query similarity and content overlap proxy.
    Uses similarity for relevance and content-prefix overlap for diversity.
    """
    selected: List[DocumentChunk] = []
    candidates = chunks[:]
    while candidates and len(selected) < k:
        best = None
        best_score = -1e9
        for cand in candidates:
            # relevance
            rel = cand.similarity
            # diversity penalty: overlap with already selected
            if selected:
                penal = max(
                    1.0 if cand.content[:100].strip().lower() in s.content[:400].strip().lower() else 0.0
                    for s in selected
                )
            else:
                penal = 0.0
            score = lambda_weight * rel - (1 - lambda_weight) * penal
            if score > best_score:
                best_score = score
                best = cand
        selected.append(best)
        candidates = [c for c in candidates if c is not best]
    return selected

def neighbor_expand(selected: List[DocumentChunk], all_chunks: List[DocumentChunk], window: int = 0) -> List[DocumentChunk]:
    """Optionally include immediate neighbors from the same file to keep context flow."""
    if window <= 0:
        return selected
    index = {(c.filename, c.chunk_index): c for c in all_chunks}
    out = { (c.filename, c.chunk_index): c for c in selected }
    for c in selected:
        for off in range(1, window+1):
            for ni in (c.chunk_index - off, c.chunk_index + off):
                k = (c.filename, ni)
                if k in index:
                    out[k] = index[k]
    return list(out.values())

def clip_context_text(chunks: List[DocumentChunk], max_chars: int = 28000) -> str:
    """Concatenate with a hard character budget to avoid massive prompts."""
    parts = []
    total = 0
    for c in chunks:
        header = f"From '{c.filename}', chunk {c.chunk_index} (similarity {c.similarity:.2f}):\n"
        body = c.content.strip()
        piece = header + body + "\n\n"
        if total + len(piece) > max_chars:
            break
        parts.append(piece)
        total += len(piece)
    return "".join(parts)

# ---------- Route ----------
@app.post("/ask")
async def ask_question(req: QuestionRequest):
    if not openai.api_key:
        return {"error": "OpenAI API key is not set."}

    # 1) Sort by similarity, de-dupe, diversify, optionally add neighbors
    sorted_chunks = sorted(req.document_context, key=lambda c: c.similarity, reverse=True)
    clean = dedupe_chunks(sorted_chunks)

    # adjust threshold if your retriever is conservative
    clean = [c for c in clean if c.similarity >= 0.55]
    diversified = mmr_select(clean, k=8, lambda_weight=0.8)

    # include ±1 neighbor to preserve continuity (set window=0 to disable)
    with_neighbors = neighbor_expand(diversified, clean, window=1)

    if not with_neighbors:
        return {
            "answer": "The assistant could not find enough relevant information in the provided documents to answer this question.",
            "confidence": 0.0,
            "sources": [],
            "suggested_next_questions": []
        }

    # 2) Build context under budget
    context_text = clip_context_text(
        sorted(with_neighbors, key=lambda c: (c.filename, c.chunk_index)),
        max_chars=28000
    )

    # 3) Prompt – strict, cite, abstain
    system_prompt = (
        "You are an expert assistant for construction handover documentation. "
        "Use ONLY the information inside CONTEXT to answer. If any required detail is missing or unclear, "
        "explicitly say so and list what is missing. Always include bracketed citations like "
        "[filename:chunk] after each factual claim you take from the context."
    )

    user_prompt = (
        f"Question (answer in {req.user_language}): {req.question}\n\n"
        f"CONTEXT START\n{context_text}CONTEXT END\n\n"
        "Requirements:\n"
        "- Be precise and concise. No fluff.\n"
        "- If maintenance schedules or frequencies are present, include them.\n"
        "- If instructions are present, summarize steps safely.\n"
        "- Provide citations inline, e.g., [manual.pdf:12] (use the chunk index from the header).\n"
        "- If the context lacks info, say: 'Not enough information in the provided documents.' and list missing items.\n"
        "- End with a short bulleted list: 'Suggested next questions:' with 2–4 items."
    )

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4o",
            temperature=0.0,
            top_p=1.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        full_response = completion.choices[0].message["content"].strip()
    except Exception as e:
        return {"error": f"OpenAI API error: {str(e)}"}

    # 4) Split suggestions if present
    suggested_next_questions: List[str] = []
    answer = full_response
    marker = "Suggested next questions:"
    if marker in full_response:
        answer, tail = full_response.split(marker, 1)
        suggested_next_questions = [
            x.strip(" -•\t").strip()
            for x in tail.strip().split("\n")
            if x.strip()
        ][:6]

    # 5) Confidence: blend top-k similarity + abstain heuristic
    top_avg = sum(c.similarity for c in diversified) / max(len(diversified), 1)
    abstain_hit = "Not enough information" in full_response
    confidence = max(0.0, min(1.0, (0.6 * top_avg) + (0.4 * (0.0 if abstain_hit else 1.0))))

    # 6) Sources
    sources = [
        {"filename": c.filename, "chunk_index": c.chunk_index, "similarity": round(c.similarity, 3)}
        for c in sorted(with_neighbors, key=lambda c: (-c.similarity, c.filename, c.chunk_index))
    ][:12]

    return {
        "answer": answer.strip(),
        "confidence": round(confidence, 3),
        "sources": sources,
        "suggested_next_questions": suggested_next_questions,
    }


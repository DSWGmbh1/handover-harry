from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Dict, Any
import openai
import os

app = FastAPI()

# Get your OpenAI API key from the environment variable (much safer)
openai.api_key = os.getenv("OPENAI_API_KEY")

class DocumentChunk(BaseModel):
    content: str
    filename: str
    chunk_index: int
    similarity: float

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

@app.post("/ask")
async def ask_question(req: QuestionRequest):
    # Combine context for the LLM
    context_text = ""
    for c in req.document_context:
        context_text += f"From {c.filename} (chunk {c.chunk_index}, similarity {c.similarity:.2f}):\n{c.content}\n\n"

    # Compose the system and user prompt
    system_prompt = (
        f"You are an expert assistant for construction project documentation. "
        f"Answer the user's question using only the information provided in the context below. "
        f"Return the answer in {req.user_language}. "
        f"If there is not enough information, say so clearly.\n\n"
        f"CONTEXT:\n{context_text}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": req.question}
    ]

    # Call OpenAI to get the answer
    completion = openai.chat.completions.create(
        model="gpt-4o",  # Or "gpt-3.5-turbo" if preferred
        messages=messages
    )
    answer = completion.choices[0].message.content.strip()

    # Optionally, estimate a "confidence" (basic version)
    confidence = min(1.0, max(0.1, sum(c.similarity for c in req.document_context) / (len(req.document_context) or 1)))

    return {
        "answer": answer,
        "confidence": confidence
    }

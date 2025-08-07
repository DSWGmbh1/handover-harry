from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import openai
import os

app = FastAPI()

# Set your OpenAI API Key securely via environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Data models
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
    # Ensure OpenAI API key is set
    if not openai.api_key:
        return {"error": "OpenAI API key is not set."}

    # Step 1: Process document context
    sorted_chunks = sorted(req.document_context, key=lambda c: c.similarity, reverse=True)
    filtered_chunks = [c for c in sorted_chunks if c.similarity >= 0.7][:10]

    if not filtered_chunks:
        return {
            "answer": "The assistant could not find enough relevant information in the provided documents to answer this question.",
            "confidence": 0.0,
            "sources": [],
            "suggested_next_questions": []
        }

    context_text = "\n\n".join(
        f"From '{c.filename}', chunk {c.chunk_index} (similarity {c.similarity:.2f}):\n{c.content}"
        for c in filtered_chunks
    )

    # Step 2: Build enhanced system prompt
system_prompt = (
    f"You are a highly intelligent assistant for construction handover documentation. "
    f"Use only the information provided in the CONTEXT below. "
    f"Answer the user's question thoroughly in {req.language}. "
    f"Do not make anything up. If there is not enough information, say: "
    f'The provided documents do not contain enough information to answer this question.\n\n'

    f"When answering:\n"
    f"- Be clear, structured, and helpful.\n"
    f"- Provide a complete answer - not just partial.\n"
    f"- If relevant, include:\n"
    f"  • What the system is and where it's located\n"
    f"  • Maintenance or replacement schedules\n"
    f"  • Installation or usage guidance\n"
    f"  • Any helpful actions (e.g., setting reminders)\n"
    f"- Anticipate and answer likely follow-up questions the user may ask.\n"
    f"- Include references to the documents if helpful.\n"
    f"- End with a short list of suggested next questions the user could ask to continue exploring their handover information.\n\n"

    f"--- CONTEXT START ---\n{context_text}\n--- CONTEXT END ---"
)


    # Step 3: Call GPT-4o
    try:
        completion = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.question}
            ]
        )
        full_response = completion.choices[0].message.content.strip()
    except Exception as e:
        return {"error": f"OpenAI API error: {str(e)}"}

    # Step 4: Extract answer and suggestions (optional step)
    # You can split suggestions if you format them as a list at the end of the assistant's answer
    if "Suggested next questions:" in full_response:
        answer, suggestions_text = full_response.split("Suggested next questions:", 1)
        suggested_next_questions = [
            q.strip("- ").strip()
            for q in suggestions_text.strip().split("\n")
            if q.strip()
        ]
    else:
        answer = full_response
        suggested_next_questions = []

    # Step 5: Confidence score
    confidence = sum(c.similarity for c in filtered_chunks) / len(filtered_chunks)

    return {
        "answer": answer.strip(),
        "confidence": round(confidence, 3),
        "sources": [
            {
                "filename": c.filename,
                "chunk_index": c.chunk_index,
                "similarity": round(c.similarity, 3)
            }
            for c in filtered_chunks
        ],
        "suggested_next_questions": suggested_next_questions
    }




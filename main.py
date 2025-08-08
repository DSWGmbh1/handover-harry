from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import openai
import os
import json
import re
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Construction Document Q&A API", version="1.0.0")

# Initialize OpenAI client
try:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        logger.warning("OpenAI API key not found in environment variables")
except Exception as e:
    logger.error(f"Error initializing OpenAI: {e}")

# ---------- Models ----------
class DocumentChunk(BaseModel):
    content: str = Field(..., description="The text content of the document chunk")
    filename: str = Field(..., description="Name of the source document")
    chunk_index: int = Field(..., description="Index of this chunk within the document")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Similarity score to the query")
    document_type: str = Field(..., description="Type of document (plans, manual, warranty, etc.)")
    section_type: Optional[str] = Field(None, description="Section type (electrical, plumbing, hvac, etc.)")
    page: Optional[int] = Field(None, description="Page number in the original document")
    extracted_entities: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted construction entities")
    visual_elements: List[Dict[str, Any]] = Field(default_factory=list, description="Visual elements for plans/diagrams")
    coordinates: Optional[Dict[str, float]] = Field(None, description="Spatial coordinates for location queries")

class ConstructionMetadata(BaseModel):
    chunks_found: int = Field(..., ge=0, description="Number of relevant chunks found")
    documents_searched: int = Field(..., ge=0, description="Number of documents searched")
    processing_time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")
    document_types_searched: List[str] = Field(default_factory=list, description="Types of documents searched")
    confidence_breakdown: Dict[str, float] = Field(default_factory=dict, description="Confidence score breakdown")

class EnhancedQuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The construction-related question")
    user_language: str = Field(default="English", description="Preferred response language")
    document_language: str = Field(default="English", description="Language of source documents")
    project_id: str = Field(..., description="Unique identifier for the construction project")
    user_id: str = Field(..., description="Unique identifier for the user")
    document_context: List[DocumentChunk] = Field(..., description="Relevant document chunks")
    metadata: ConstructionMetadata = Field(..., description="Search metadata")
    query_type: Optional[str] = Field(None, description="Type of query (location, maintenance, specifications, etc.)")

class QuestionResponse(BaseModel):
    answer: str = Field(..., description="The generated answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    sources: List[Dict[str, Any]] = Field(..., description="Source information")
    suggested_next_questions: List[str] = Field(default_factory=list, description="Related questions")
    query_type: str = Field(..., description="Classified query type")
    processing_time_ms: float = Field(..., description="Processing time")
    chunks_analyzed: int = Field(..., description="Number of chunks analyzed")
    document_types_used: List[str] = Field(default_factory=list, description="Document types used")

# ---------- Construction-Specific NLP ----------
class ConstructionNLP:
    def __init__(self):
        self.location_patterns = [
            r"(?:main|primary|central)\s+(?:water|electrical|gas|hvac)\s+(?:shut.?off|valve|panel|switch|meter)",
            r"(?:water|electrical|gas)\s+(?:meter|panel|board|valve|shut.?off)",
            r"(?:fuse|circuit)\s+(?:box|panel|breaker)",
            r"(?:hot\s+water|water)\s+(?:heater|tank)",
            r"(?:hvac|air\s+conditioning|heating)\s+(?:unit|system|panel)",
        ]
        
        self.maintenance_patterns = [
            r"(?:inspect|check|maintain|service|replace)\s+(?:every|annually|monthly|quarterly)",
            r"(?:maintenance|service)\s+(?:schedule|frequency|interval)",
            r"(?:warranty|guarantee)\s+(?:period|coverage|expires?)",
        ]
        
        self.specification_patterns = [
            r"(?:model|serial|part)\s+(?:number|#|no\.)",
            r"(?:capacity|rating|size|dimensions)",
            r"(?:voltage|amperage|wattage|btu)",
        ]

    def classify_query(self, question: str) -> str:
        """Classify the type of construction query."""
        question_lower = question.lower()
        
        if any(re.search(pattern, question_lower) for pattern in self.location_patterns):
            return "location"
        elif any(re.search(pattern, question_lower) for pattern in self.maintenance_patterns):
            return "maintenance"
        elif any(re.search(pattern, question_lower) for pattern in self.specification_patterns):
            return "specifications"
        elif any(word in question_lower for word in ["where", "located", "find", "position"]):
            return "location"
        elif any(word in question_lower for word in ["how often", "when", "schedule", "maintain"]):
            return "maintenance"
        else:
            return "general"

    def extract_construction_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract construction-specific entities from text."""
        entities = []
        
        equipment_patterns = {
            "water_systems": r"(?:water\s+(?:heater|tank|pump|valve|meter|shut.?off)|plumbing)",
            "electrical": r"(?:electrical\s+(?:panel|board|meter|breaker)|circuit\s+breaker|fuse\s+box)",
            "hvac": r"(?:hvac|air\s+conditioning|heating|furnace|thermostat|ductwork)",
            "safety": r"(?:smoke\s+detector|carbon\s+monoxide|fire\s+extinguisher|sprinkler)",
            "appliances": r"(?:dishwasher|garbage\s+disposal|range|oven|refrigerator)",
        }
        
        for category, pattern in equipment_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    "type": category,
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })
        
        return entities

construction_nlp = ConstructionNLP()

# ---------- Utility Functions ----------
def smart_chunk_selection(chunks: List[DocumentChunk], query: str, query_type: str, k: int = 10) -> List[DocumentChunk]:
    """Select the most relevant chunks based on query type."""
    
    if query_type == "location":
        plan_chunks = [c for c in chunks if "plan" in c.document_type.lower() or "blueprint" in c.filename.lower()]
        spec_chunks = [c for c in chunks if any(word in c.content.lower() for word in ["location", "located", "position", "floor", "room"])]
        priority_chunks = plan_chunks + spec_chunks
        remaining = [c for c in chunks if c not in priority_chunks]
        combined = priority_chunks + remaining
    elif query_type == "maintenance":
        maintenance_chunks = [c for c in chunks if any(word in c.document_type.lower() for word in ["manual", "maintenance", "service"])]
        warranty_chunks = [c for c in chunks if "warranty" in c.document_type.lower()]
        priority_chunks = maintenance_chunks + warranty_chunks
        remaining = [c for c in chunks if c not in priority_chunks]
        combined = priority_chunks + remaining
    else:
        combined = chunks
    
    # Simple selection by similarity score
    return sorted(combined, key=lambda c: c.similarity, reverse=True)[:k]

def build_construction_context(chunks: List[DocumentChunk], max_chars: int = 32000) -> str:
    """Build context with construction-specific formatting."""
    context_parts = []
    total_chars = 0
    
    for chunk in sorted(chunks, key=lambda x: x.similarity, reverse=True):
        header = f"\n=== {chunk.document_type.upper()}: '{chunk.filename}' (Chunk {chunk.chunk_index}, Similarity: {chunk.similarity:.2f}) ===\n"
        
        if chunk.extracted_entities:
            entities_info = "Key Elements: " + ", ".join([f"{e['type']}({e['text']})" for e in chunk.extracted_entities[:3]]) + "\n"
            header += entities_info
        
        if chunk.visual_elements:
            visual_info = f"Visual Elements: {len(chunk.visual_elements)} diagrams/plans found\n"
            header += visual_info
        
        content = chunk.content.strip()
        section = header + content + "\n"
        
        if total_chars + len(section) > max_chars:
            break
            
        context_parts.append(section)
        total_chars += len(section)
    
    return "".join(context_parts)

# ---------- Health Check Endpoint ----------
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "openai": bool(openai.api_key)
        }
    }

# ---------- Main Route ----------
@app.post("/ask", response_model=QuestionResponse)
async def ask_construction_question(req: EnhancedQuestionRequest):
    """Main endpoint for construction document Q&A."""
    if not openai.api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key is not configured")

    start_time = datetime.now()
    
    try:
        # Classify the query
        query_type = req.query_type or construction_nlp.classify_query(req.question)
        logger.info(f"Processing {query_type} query: {req.question[:50]}...")
        
        # Enhanced chunk processing
        sorted_chunks = sorted(req.document_context, key=lambda c: c.similarity, reverse=True)
        
        # Extract entities from chunks if not already done
        for chunk in sorted_chunks:
            if not chunk.extracted_entities:
                chunk.extracted_entities = construction_nlp.extract_construction_entities(chunk.content)
        
        # Adaptive similarity threshold
        similarity_thresholds = {
            "location": 0.45,
            "maintenance": 0.55,
            "specifications": 0.50,
            "general": 0.55
        }
        
        threshold = similarity_thresholds.get(query_type, 0.55)
        relevant_chunks = [c for c in sorted_chunks if c.similarity >= threshold]
        
        if not relevant_chunks:
            relevant_chunks = sorted_chunks[:5]
        
        # Smart selection
        selected_chunks = smart_chunk_selection(relevant_chunks, req.question, query_type, k=12)
        
        if not selected_chunks:
            return QuestionResponse(
                answer="I couldn't find relevant information in your construction documents to answer this question. Please try rephrasing your question or check if the relevant documents have been uploaded.",
                confidence=0.0,
                sources=[],
                suggested_next_questions=[],
                query_type=query_type,
                processing_time_ms=0.0,
                chunks_analyzed=0,
                document_types_used=[]
            )

        # Build context
        context_text = build_construction_context(selected_chunks, max_chars=35000)
        
        # System prompt
        system_prompt = f"""You are a specialized construction handover documentation assistant with expertise in building systems, maintenance, and construction plans. Your role is to help homeowners understand their property based on their construction documentation.

EXPERTISE AREAS:
- Building plans and blueprints interpretation
- Electrical, plumbing, and HVAC systems
- Maintenance schedules and procedures
- Equipment specifications and locations
- Safety systems and procedures
- Warranty and inspection information

RESPONSE REQUIREMENTS:
- Use ONLY information from the provided CONTEXT
- For location queries, be specific about floor, room, or area
- For maintenance queries, include specific schedules, frequencies, and procedures
- Always cite sources using [filename:chunk_number] format
- Be precise about technical specifications
- If information is incomplete, explicitly state what's missing

QUERY TYPE: {query_type}"""

        # User prompt
        user_prompt_base = f"Question (answer in {req.user_language}): {req.question}\n\n"
        
        if query_type == "location":
            user_prompt_additions = """
LOCATION QUERY - Special Instructions:
- Look for floor plans, site plans, and technical drawings
- Identify specific rooms, floors, or areas
- Mention nearby landmarks or reference points
- Include access instructions if available
"""
        elif query_type == "maintenance":
            user_prompt_additions = """
MAINTENANCE QUERY - Special Instructions:
- Extract specific schedules and frequencies
- List required tools or materials
- Include safety precautions
- Note warranty implications
"""
        else:
            user_prompt_additions = ""
        
        user_prompt = (
            user_prompt_base +
            user_prompt_additions +
            f"\nCONTEXT:\n{context_text}\n\n" +
            """RESPONSE FORMAT:
- Be comprehensive but concise
- Include inline citations [filename:chunk]
- End with "Suggested next questions:" followed by 3-4 relevant questions"""
        )

        # OpenAI API call
        try:
            # Try new client first
            try:
                from openai import OpenAI
                client = OpenAI(api_key=openai.api_key)
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",  # Using mini for cost efficiency
                    temperature=0.1,
                    max_tokens=1000,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                full_response = completion.choices[0].message.content.strip()
            except ImportError:
                # Fallback to old client
                completion = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    temperature=0.1,
                    max_tokens=1000,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                full_response = completion.choices[0].message["content"].strip()
                
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

        # Process response
        suggested_next_questions: List[str] = []
        answer = full_response
        
        marker = "Suggested next questions:"
        if marker in full_response:
            try:
                answer, tail = full_response.split(marker, 1)
                suggested_next_questions = [
                    x.strip(" -•\t").strip()
                    for x in tail.strip().split("\n")
                    if x.strip() and len(x.strip()) > 5
                ][:4]
            except Exception as e:
                logger.warning(f"Error processing suggested questions: {e}")

        # Calculate confidence
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        similarity_avg = sum(c.similarity for c in selected_chunks) / len(selected_chunks) if selected_chunks else 0.0
        
        # Basic confidence calculation
        confidence = min(1.0, max(0.0, 0.7 * similarity_avg + 0.3))

        # Sources
        sources = []
        for chunk in sorted(selected_chunks, key=lambda c: (-c.similarity, c.filename, c.chunk_index)):
            source_info = {
                "filename": chunk.filename,
                "chunk_index": chunk.chunk_index,
                "similarity": round(chunk.similarity, 3),
                "document_type": chunk.document_type,
                "entities_count": len(chunk.extracted_entities),
                "has_visual_elements": len(chunk.visual_elements) > 0
            }
            sources.append(source_info)

        logger.info(f"Successfully processed query, confidence: {confidence:.3f}")

        return QuestionResponse(
            answer=answer.strip(),
            confidence=round(confidence, 3),
            sources=sources[:10],
            suggested_next_questions=suggested_next_questions,
            query_type=query_type,
            processing_time_ms=round(processing_time, 2),
            chunks_analyzed=len(selected_chunks),
            document_types_used=list(set(c.document_type for c in selected_chunks))
        )

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ---------- Root Endpoint ----------
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Construction Document Q&A API",
        "version": "1.0.0",
        "description": "Specialized API for construction handover documentation assistance",
        "endpoints": {
            "health": "/health",
            "ask": "/ask",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



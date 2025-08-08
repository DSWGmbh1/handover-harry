from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import openai
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import json
import re
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize specialized construction document embeddings
construction_embedder = SentenceTransformer('all-MiniLM-L6-v2')

# ChromaDB for vector storage with construction-specific collections
chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./construction_docs_db"
))

# ---------- Enhanced Models ----------
class DocumentChunk(BaseModel):
    content: str
    filename: str
    chunk_index: int
    similarity: float
    document_type: str  # plans, manual, warranty, inspection, etc.
    section_type: Optional[str] = None  # electrical, plumbing, hvac, etc.
    page: Optional[int] = None
    extracted_entities: List[Dict[str, Any]] = []
    visual_elements: List[Dict[str, Any]] = []  # For plans/diagrams
    coordinates: Optional[Dict[str, float]] = None  # For spatial queries

class ConstructionMetadata(BaseModel):
    chunks_found: int
    documents_searched: int
    processing_time_ms: int
    document_types_searched: List[str]
    confidence_breakdown: Dict[str, float]

class EnhancedQuestionRequest(BaseModel):
    question: str
    user_language: str
    document_language: str
    project_id: str
    user_id: str
    document_context: List[DocumentChunk]
    metadata: ConstructionMetadata
    query_type: Optional[str] = None  # location, maintenance, specifications, etc.

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
        question_lower = question.lower()
        
        # Location queries
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
        
        # Equipment/Systems
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
        
        # Model numbers, specifications
        spec_patterns = {
            "model_number": r"(?:model|part)\s*#?\s*:?\s*([A-Z0-9\-]+)",
            "serial_number": r"(?:serial|s/n)\s*#?\s*:?\s*([A-Z0-9\-]+)",
            "voltage": r"(\d+)\s*(?:v|volt|voltage)",
            "amperage": r"(\d+)\s*(?:a|amp|amperage)",
        }
        
        for spec_type, pattern in spec_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    "type": spec_type,
                    "text": match.group(),
                    "value": match.group(1) if match.groups() else match.group(),
                    "start": match.start(),
                    "end": match.end()
                })
        
        return entities

construction_nlp = ConstructionNLP()

# ---------- Enhanced Retrieval Functions ----------
def smart_chunk_selection(chunks: List[DocumentChunk], query: str, query_type: str, k: int = 10) -> List[DocumentChunk]:
    """Enhanced chunk selection based on query type and construction domain knowledge."""
    
    # Classify query for better retrieval
    if query_type == "location":
        # Prioritize plan documents and technical specifications
        plan_chunks = [c for c in chunks if "plan" in c.document_type.lower() or "blueprint" in c.filename.lower()]
        spec_chunks = [c for c in chunks if any(word in c.content.lower() for word in ["location", "located", "position", "floor", "room"])]
        priority_chunks = plan_chunks + spec_chunks
        
        # Add other relevant chunks
        remaining = [c for c in chunks if c not in priority_chunks]
        combined = priority_chunks + remaining
        
    elif query_type == "maintenance":
        # Prioritize maintenance manuals and warranty documents
        maintenance_chunks = [c for c in chunks if any(word in c.document_type.lower() for word in ["manual", "maintenance", "service"])]
        warranty_chunks = [c for c in chunks if "warranty" in c.document_type.lower()]
        priority_chunks = maintenance_chunks + warranty_chunks
        
        remaining = [c for c in chunks if c not in priority_chunks]
        combined = priority_chunks + remaining
        
    else:
        combined = chunks
    
    # Apply MMR with construction-specific diversity
    return mmr_select_enhanced(combined[:50], k=k, query=query, lambda_weight=0.7)

def mmr_select_enhanced(chunks: List[DocumentChunk], k: int = 10, query: str = "", lambda_weight: float = 0.7) -> List[DocumentChunk]:
    """Enhanced MMR with construction domain knowledge."""
    selected: List[DocumentChunk] = []
    candidates = chunks[:]
    
    while candidates and len(selected) < k:
        best = None
        best_score = -1e9
        
        for cand in candidates:
            # Relevance score (similarity + domain boost)
            relevance = cand.similarity
            
            # Boost for construction-specific content
            if any(entity["type"] in ["electrical", "water_systems", "hvac"] for entity in cand.extracted_entities):
                relevance *= 1.2
            
            # Boost for visual elements in location queries
            if "where" in query.lower() and cand.visual_elements:
                relevance *= 1.1
            
            # Diversity penalty
            diversity_penalty = 0.0
            if selected:
                for sel in selected:
                    # Content similarity penalty
                    content_overlap = calculate_content_overlap(cand.content, sel.content)
                    # Document type diversity
                    type_penalty = 0.3 if cand.document_type == sel.document_type else 0.0
                    diversity_penalty = max(diversity_penalty, content_overlap + type_penalty)
            
            score = lambda_weight * relevance - (1 - lambda_weight) * diversity_penalty
            
            if score > best_score:
                best_score = score
                best = cand
        
        if best:
            selected.append(best)
            candidates = [c for c in candidates if c is not best]
    
    return selected

def calculate_content_overlap(content1: str, content2: str) -> float:
    """Calculate content overlap between two chunks."""
    words1 = set(content1.lower().split())
    words2 = set(content2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0

def build_construction_context(chunks: List[DocumentChunk], max_chars: int = 32000) -> str:
    """Build context with construction-specific formatting."""
    context_parts = []
    total_chars = 0
    
    # Group by document type for better organization
    by_doc_type = {}
    for chunk in chunks:
        doc_type = chunk.document_type
        if doc_type not in by_doc_type:
            by_doc_type[doc_type] = []
        by_doc_type[doc_type].append(chunk)
    
    # Prioritize document types
    type_priority = ["plans", "blueprints", "specifications", "manual", "maintenance", "warranty", "inspection"]
    
    for doc_type in type_priority:
        if doc_type in by_doc_type:
            type_chunks = by_doc_type[doc_type]
            for chunk in sorted(type_chunks, key=lambda x: x.similarity, reverse=True):
                
                # Enhanced header with metadata
                header = f"\n=== {chunk.document_type.upper()}: '{chunk.filename}' (Chunk {chunk.chunk_index}, Similarity: {chunk.similarity:.2f}) ===\n"
                
                # Add extracted entities info
                if chunk.extracted_entities:
                    entities_info = "Key Elements: " + ", ".join([f"{e['type']}({e['text']})" for e in chunk.extracted_entities[:3]]) + "\n"
                    header += entities_info
                
                # Add visual elements info for plans
                if chunk.visual_elements:
                    visual_info = f"Visual Elements: {len(chunk.visual_elements)} diagrams/plans found\n"
                    header += visual_info
                
                content = chunk.content.strip()
                section = header + content + "\n"
                
                if total_chars + len(section) > max_chars:
                    break
                    
                context_parts.append(section)
                total_chars += len(section)
    
    # Add remaining chunks if space allows
    remaining_chunks = []
    for doc_type, type_chunks in by_doc_type.items():
        if doc_type not in type_priority:
            remaining_chunks.extend(type_chunks)
    
    for chunk in sorted(remaining_chunks, key=lambda x: x.similarity, reverse=True):
        header = f"\n=== {chunk.document_type.upper()}: '{chunk.filename}' (Chunk {chunk.chunk_index}, Similarity: {chunk.similarity:.2f}) ===\n"
        content = chunk.content.strip()
        section = header + content + "\n"
        
        if total_chars + len(section) > max_chars:
            break
            
        context_parts.append(section)
        total_chars += len(section)
    
    return "".join(context_parts)

# ---------- Enhanced Route ----------
@app.post("/ask")
async def ask_construction_question(req: EnhancedQuestionRequest):
    if not openai.api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key is not set")

    start_time = datetime.now()
    
    # Classify the query
    query_type = construction_nlp.classify_query(req.question)
    
    # Enhanced chunk processing
    sorted_chunks = sorted(req.document_context, key=lambda c: c.similarity, reverse=True)
    
    # Extract entities from chunks if not already done
    for chunk in sorted_chunks:
        if not chunk.extracted_entities:
            chunk.extracted_entities = construction_nlp.extract_construction_entities(chunk.content)
    
    # Adaptive similarity threshold based on query type
    similarity_thresholds = {
        "location": 0.45,  # Lower threshold for location queries
        "maintenance": 0.55,
        "specifications": 0.50,
        "general": 0.55
    }
    
    threshold = similarity_thresholds.get(query_type, 0.55)
    relevant_chunks = [c for c in sorted_chunks if c.similarity >= threshold]
    
    if not relevant_chunks:
        # Fallback with lower threshold
        relevant_chunks = sorted_chunks[:5]
    
    # Smart selection based on query type
    selected_chunks = smart_chunk_selection(relevant_chunks, req.question, query_type, k=12)
    
    if not selected_chunks:
        return {
            "answer": "I couldn't find relevant information in your construction documents to answer this question. Please try rephrasing your question or check if the relevant documents have been uploaded.",
            "confidence": 0.0,
            "sources": [],
            "suggested_next_questions": [],
            "query_type": query_type
        }

    # Build enhanced context
    context_text = build_construction_context(selected_chunks, max_chars=35000)
    
    # Construction-specialized system prompt
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
- For location queries (like "where is the main water shut-off valve"), be specific about floor, room, or area
- For maintenance queries, include specific schedules, frequencies, and procedures
- Always cite sources using [filename:chunk_number] format
- If reading plans or blueprints, describe locations clearly with reference points
- Be precise about technical specifications (model numbers, capacities, etc.)
- If information is incomplete, explicitly state what's missing

QUERY TYPE: {query_type}"""

    # Enhanced user prompt based on query type
    user_prompt_base = f"Question (answer in {req.user_language}): {req.question}\n\n"
    
    if query_type == "location":
        user_prompt_additions = """
LOCATION QUERY - Special Instructions:
- Look for floor plans, site plans, and technical drawings
- Identify specific rooms, floors, or areas
- Mention nearby landmarks or reference points
- Include access instructions if available
- Note any special tools or keys required
"""
    elif query_type == "maintenance":
        user_prompt_additions = """
MAINTENANCE QUERY - Special Instructions:
- Extract specific schedules and frequencies
- List required tools or materials
- Include safety precautions
- Note warranty implications
- Mention professional service requirements
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
- If working with plans, describe locations with clear reference points
- End with "Suggested next questions:" followed by 3-4 relevant questions
- If context is insufficient, list specific missing information needed"""
    )

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4o",
            temperature=0.1,  # Slightly higher for more natural responses
            max_tokens=1000,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        full_response = completion.choices[0].message["content"].strip()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

    # Process response
    suggested_next_questions: List[str] = []
    answer = full_response
    
    marker = "Suggested next questions:"
    if marker in full_response:
        answer, tail = full_response.split(marker, 1)
        suggested_next_questions = [
            x.strip(" -•\t").strip()
            for x in tail.strip().split("\n")
            if x.strip() and len(x.strip()) > 5
        ][:4]

    # Enhanced confidence calculation
    processing_time = (datetime.now() - start_time).total_seconds() * 1000
    
    # Multi-factor confidence scoring
    similarity_avg = sum(c.similarity for c in selected_chunks) / len(selected_chunks)
    
    # Query type confidence adjustment
    type_confidence_boost = {
        "location": 0.1 if any("plan" in c.document_type.lower() for c in selected_chunks) else 0.0,
        "maintenance": 0.1 if any("manual" in c.document_type.lower() for c in selected_chunks) else 0.0,
        "specifications": 0.05,
        "general": 0.0
    }
    
    # Entity extraction confidence boost
    entity_boost = 0.1 if any(c.extracted_entities for c in selected_chunks) else 0.0
    
    # Response completeness (not an abstain response)
    completeness_factor = 0.0 if "not enough information" in answer.lower() else 0.2
    
    confidence = min(1.0, max(0.0, 
        0.5 * similarity_avg + 
        type_confidence_boost.get(query_type, 0.0) + 
        entity_boost + 
        completeness_factor
    ))

    # Enhanced sources with metadata
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

    return {
        "answer": answer.strip(),
        "confidence": round(confidence, 3),
        "sources": sources[:10],
        "suggested_next_questions": suggested_next_questions,
        "query_type": query_type,
        "processing_time_ms": round(processing_time, 2),
        "chunks_analyzed": len(selected_chunks),
        "document_types_used": list(set(c.document_type for c in selected_chunks))
    }


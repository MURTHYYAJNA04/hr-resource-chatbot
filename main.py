import sys
import os
import json
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import logging
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HR Resource Assistant API",
    description="AI-powered HR assistant for employee resource allocation",
    version="3.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDINGS_FILE = "employee_embeddings.npy"

# Try to import OpenAI (optional)
try:
    import openai
    OPENAI_AVAILABLE = True
    logger.info("OpenAI library available")
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not installed. Using simple responses.")

# Load employee data
employees = []
try:
    with open("employees.json", "r") as file:
        data = json.load(file)
        employees = data["employees"]
    logger.info(f"Loaded {len(employees)} employees")
except Exception as e:
    logger.error(f"Error loading employees: {str(e)}")

# Initialize embedding model
model = None
try:
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Embedding model loaded")
except Exception as e:
    logger.error(f"Error loading embedding model: {str(e)}")

# Create or load embeddings
embeddings = None
if model and employees:
    if os.path.exists(EMBEDDINGS_FILE):
        embeddings = np.load(EMBEDDINGS_FILE)
        logger.info("Loaded embeddings from file")
    else:
        try:
            texts = [
                f"Name: {emp['name']} "
                f"Skills: {', '.join(emp['skills'])} "
                f"Experience: {emp['experience_years']} years "
                f"Projects: {', '.join(emp['projects'])} "
                f"Status: {emp['availability']}"
                for emp in employees
            ]
            
            embeddings = model.encode(texts, convert_to_tensor=False)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10  # Avoid division by zero
            embeddings = embeddings / norms
            
            np.save(EMBEDDINGS_FILE, embeddings)
            logger.info(f"Created embeddings for {len(employees)} employees")
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            embeddings = None

# Pydantic models
class QueryModel(BaseModel):
    text: str

class EmployeeResponse(BaseModel):
    id: int
    name: str
    skills: List[str]
    experience_years: int
    projects: List[str]
    availability: str
    similarity: float

# Search employees endpoint
@app.get("/employees/search", response_model=List[EmployeeResponse])
async def search_employees(
    query: str, 
    k: int = Query(5, description="Number of results to return"),
    min_similarity: float = Query(0.25, description="Minimum similarity score")
):
    if not employees or embeddings is None or model is None:
        raise HTTPException(503, "Service initializing or unavailable")
    
    try:
        # Encode query
        query_embedding = model.encode([query], convert_to_tensor=False)
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # Compute cosine similarity
        similarities = np.dot(embeddings, query_norm.T).flatten()
        
        # Get top matches
        indices = np.argsort(similarities)[::-1][:k]
        results = []
        
        for idx in indices:
            if similarities[idx] < min_similarity:
                continue
                
            emp = employees[idx]
            results.append({
                **emp,
                "similarity": float(similarities[idx])
            })
        
        return results
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(500, "Internal server error")

# RAG-based chat endpoint
@app.post("/chat")
async def chat(query: QueryModel):
    if not employees or embeddings is None or model is None:
        return {"response": "⚠️ Service is initializing. Please try again shortly."}
    
    try:
        # 1. Retrieval - Find relevant employees
        search_results = await search_employees(query.text, k=5, min_similarity=0.2)
        
        # Filter available candidates
        candidates = [e for e in search_results if e["availability"].lower() == "available"]
        
        if not candidates:
            return {"response": "❌ No available candidates match your query."}
        
        # 2. Augmentation - Prepare context
        context = "\n\n".join([
            f"Name: {e['name']}\n"
            f"Skills: {', '.join(e['skills'])}\n"
            f"Experience: {e['experience_years']} years\n"
            f"Projects: {', '.join(e['projects'])}"
            for e in candidates[:3]  # Only top 3
        ])
        
        # 3. Generation - Create response
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You're an HR assistant."},
                        {"role": "user", "content": f"Query: {query.text}\n\n{context}\n\nResponse format: 1) Summary 2) Top candidates 3) Next steps"}
                    ],
                    temperature=0.3,
                    max_tokens=250
                )
                return {"response": response.choices[0].message['content'].strip()}
            except Exception as e:
                logger.error(f"OpenAI error: {str(e)}")
                return generate_simple_response(candidates, query.text)
        else:
            return generate_simple_response(candidates, query.text)
            
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return {"response": f"⚠️ Error: {str(e)}"}

def generate_simple_response(candidates: List[Dict[str, Any]], query: str) -> Dict[str, str]:
    """Generate response without LLM"""
    response_lines = [
        f"🔍 Found {len(candidates)} matches for: '{query}'",
        "\nTop recommendations:"
    ]
    
    for i, candidate in enumerate(candidates[:3], 1):
        response_lines.append(
            f"{i}. **{candidate['name']}** "
            f"(Score: {candidate['similarity']:.2f})"
            f"\n   Skills: {', '.join(candidate['skills'])}"
            f"\n   Experience: {candidate['experience_years']} years"
            f"\n   Projects: {', '.join(candidate['projects'])}"
        )
    
    response_lines.append("\n💡 All candidates are currently available.")
    response_lines.append("\nAsk for more details about any candidate.")
    
    return {"response": "\n".join(response_lines)}

# Basic endpoints
@app.get("/employees")
def get_employees():
    return {"employees": employees}

@app.get("/health")
def health_check():
    return {
        "status": "ready" if employees and embeddings is not None else "initializing",
        "employees": len(employees),
        "openai_available": OPENAI_AVAILABLE and bool(os.getenv("OPENAI_API_KEY"))
    }

# Run with debug server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List
from app.rag_pipeline import RAGPipeline
import os
from dotenv import load_dotenv
import logging
import asyncio

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Insurance Policy Q&A API")

# Remove the hardcoded PDF path and RAG Pipeline initialization
rag_pipelines = {}  # Cache for RAG Pipelines

class QuestionRequest(BaseModel):
    documents: str  # URL of the document
    questions: List[str]

class QuestionResponse(BaseModel):
    answers: List[str]

@app.post("/hackrx/run")
async def process_questions(
    request: QuestionRequest,
    authorization: str = Header(..., description="Bearer <api_key>")
):
    # Verify API key
    api_key = authorization.replace("Bearer ", "")
    if api_key != os.getenv("SECURITY_API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        # Create or get cached RAG Pipeline for this document
        if request.documents not in rag_pipelines:
            rag_pipelines[request.documents] = RAGPipeline(request.documents)
        
        rag_pipeline = rag_pipelines[request.documents]
        
        # Process questions concurrently
        tasks = [rag_pipeline.aquery(question) for question in request.questions]
        responses = await asyncio.gather(*tasks)
        answers = [response["result"] for response in responses]
        
        return QuestionResponse(answers=answers)
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
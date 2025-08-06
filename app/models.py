from pydantic import BaseModel
from typing import List, Dict

# Defines the structure for the request body of our API
# Expects a JSON like: {"question": "your question here"}
class QueryRequest(BaseModel):
    question: str

# Defines the structure of a single source document
class SourceDocument(BaseModel):
    source: str
    page_content: str

# Defines the structure for the response body of our API
# Will return a JSON like: {"answer": "...", "source_documents": [...]}
class QueryResponse(BaseModel):
    answer: str
    source_documents: List[SourceDocument]
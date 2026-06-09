from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ChunkMetadata(BaseModel):
    source_file: str
    page: int
    chunk_id: str

class RetrievedChunk(BaseModel):
    text: str
    score: float
    metadata: ChunkMetadata

class AskRequest(BaseModel):
    question: str
    learner_level: Optional[str] = "novato"
    top_k: Optional[int] = 5

class Agent1Response(BaseModel):
    user_query: str
    learner_level_estimate: str
    prerequisites: List[str] = Field(default_factory=list)
    knowledge_gaps: List[str] = Field(default_factory=list)
    pedagogical_constraints: Dict[str, Any] = Field(default_factory=dict)
    retrieved_evidence: List[Dict[str, Any]] = Field(default_factory=list)

class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    learning_path: List[str]
    pedagogical_mode: str
    agent1_payload: Dict[str, Any]

from pydantic import BaseModel
from typing import Optional, List, Dict

class SessionCreate(BaseModel):
    username: str
    age: Optional[int] = None
    gender: Optional[str] = None
    education_level: Optional[str] = None
    disease_site: Optional[str] = None

class SessionChatCreate(BaseModel):
    session_id: int
    user_question: str
    parsed_question: str
    response: str
    is_verified: bool
    retrieval_similarity : Optional[float] = None
    retrieval_relevancy : Optional[float] = None
    response_eval_scores : Optional[Dict] = None
    response_analytics : Optional[Dict] = None

class SessionResponse(BaseModel):
    id: int
    username: str
    age: Optional[int]
    gender: Optional[str]
    education_level: Optional[str] 
    disease_site: Optional[str]

    class Config:
        orm_mode = True

class SessionChatResponse(BaseModel):
    id: int
    session_id: int
    user_question: str
    parsed_question: str
    response: str
    is_verified: bool
    retrieval_similarity : Optional[float] = None
    retrieval_relevancy : Optional[float] = None
    response_eval_scores : Optional[Dict] = None
    response_analytics : Optional[Dict] = None

    class Config:
        orm_mode = True

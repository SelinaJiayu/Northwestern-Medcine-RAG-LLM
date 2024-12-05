from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from cancer_rag.models.database import get_db
from cancer_rag.models.schema import SessionCreate, SessionResponse
from cancer_rag.crud import create_session, get_sessions
from typing import List

router = APIRouter()

@router.post("/sessions/", response_model=SessionResponse)
def create_new_session(session: SessionCreate, db: Session = Depends(get_db)):
    return create_session(db=db, session_data=session)

@router.get("/sessions/", response_model=List[SessionResponse])
def read_sessions(db: Session = Depends(get_db)):
    return get_sessions(db=db)

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from cancer_rag.models.database import get_db
from cancer_rag.models.schema import SessionChatCreate, SessionChatResponse
from cancer_rag.crud import create_session_chat, get_session_chats
from typing import List

router = APIRouter()

@router.post("/sessions/chats/", response_model=SessionChatResponse)
def create_new_session_chat(chat: SessionChatCreate, db: Session = Depends(get_db)):
    return create_session_chat(db=db, chat_data=chat)

@router.get("/sessions/chats/", response_model=List[SessionChatResponse])
def read_session_chats(session_id: int, db: Session = Depends(get_db)):
    return get_session_chats(db=db, session_id=session_id)

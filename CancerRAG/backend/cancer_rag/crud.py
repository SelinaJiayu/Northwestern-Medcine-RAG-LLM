from sqlalchemy.orm import Session
from cancer_rag.models.database import Session as SessionModel, SessionChat as SessionChatModel
from cancer_rag.models.schema import SessionCreate, SessionChatCreate

def create_session(db: Session, session_data: SessionCreate):
    db_session = SessionModel(**session_data.dict())
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session

def get_sessions(db: Session):
    return db.query(SessionModel).all()

def create_session_chat(db: Session, chat_data: SessionChatCreate):
    db_chat = SessionChatModel(**chat_data.dict())
    db.add(db_chat)
    db.commit()
    db.refresh(db_chat)
    return db_chat

def get_session_chats(db: Session, session_id: int):
    return db.query(SessionChatModel).filter(SessionChatModel.session_id == session_id).all()

def get_verified_data(db: Session):
    return db.query(SessionChatModel).filter(SessionChatModel.is_verified == True).all()


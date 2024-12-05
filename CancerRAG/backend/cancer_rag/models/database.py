import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Boolean, ForeignKey, Text, JSON, Float
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship, sessionmaker


#load_dotenv(Path(__file__).parent / '../../.env')

DATABASE_URI=os.getenv('DATABASE_URI')  
print(DATABASE_URI)
engine = create_engine(DATABASE_URI, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Session(Base):
    __tablename__ = 'session'

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), nullable=False)
    age = Column(Integer, nullable=True)  
    gender = Column(String(10), nullable=True) 
    disease_site = Column(String(50), nullable=True) 
    education_level = Column(String(50), nullable=True) 

    # Relationship to SessionChat table
    chats = relationship("SessionChat", back_populates="session")

    def __repr__(self):
        return f"<Session(id={self.id}, username={self.username}, age={self.age}, gender={self.gender})>"

# SessionChat table
class SessionChat(Base):
    __tablename__ = 'session_chat'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey('session.id'), nullable=False)
    user_question = Column(Text, nullable=False)
    parsed_question = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    is_verified = Column(Boolean, nullable=False, default=False)
    retrieval_similarity = Column(Float, nullable=True)
    retrieval_relevancy = Column(Float, nullable=True)
    response_eval_scores = Column(JSON, nullable=True)   
    response_analytics = Column(JSON, nullable=True)

    # Relationship back to the Session table
    session = relationship("Session", back_populates="chats")

    def __repr__(self):
        return (
            f"<SessionChat(id={self.id}, session_id={self.session_id}, user_question={self.user_question}, "
            f"parsed_question={self.parsed_question}, response={self.response}, is_verified={self.is_verified})>"
        )

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_db():
    Base.metadata.create_all(engine)

if __name__ == "__main__":
    Base.metadata.create_all(engine)
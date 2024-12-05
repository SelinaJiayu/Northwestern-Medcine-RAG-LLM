from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
import uuid
import os
import argparse
from pathlib import Path
from tqdm import tqdm

from cancer_rag.models.database import Session, SessionChat

"""
Example: 

For using Sqlite3
python scripts/data_ingestion.py \
    --data-path data/data_files/capstone_final_data_v1.csv \
    --database-uri sqlite:///$(pwd)/backend/cancer_QA.db

For using Postgres
python scripts/data_ingestion.py \
    --data-path data/data_files/capstone_final_data_v1.csv \
    --database-uri postgresql://nu_troy:p%40ssword@localhost:5432/cancer_rag
"""

def create_session(default_username="default"):
    db_session = SessionLocal()
    existing_session = db_session.query(Session).filter_by(username=default_username).first()
    if existing_session:
        print(f"Existing session ID: {existing_session.id}")
        db_session.close()
        return existing_session.id
    else:
        # Create a new session if user does not exist
        new_session = Session(username=default_username)
        db_session.add(new_session)
        db_session.commit()
        print(f"Created new session ID: {new_session.id}")
        db_session.close()
        return new_session.id

def batch_ingest(session_id, data):
    db_session = SessionLocal()
    batch_size = 50

    num_batches = (len(data) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(data), batch_size), total=num_batches, desc="Ingesting batches"):
        batch = data[i:i + batch_size]
        chats = [
            SessionChat(
                session_id=session_id,
                user_question=row['Question'],
                parsed_question=row['Question'],
                response=row['Answer'],
                is_verified=True
            ) for index, row in batch.iterrows()
        ]
        db_session.bulk_save_objects(chats)
        db_session.commit()
    db_session.close()
    print("Batch ingestion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, help="Path to Q/A data csv file")
    parser.add_argument("--database-uri", type=str, help="Path to SQlite Database")
    args = parser.parse_args()
    
    data_path = args.data_path
    database_uri = args.database_uri

    # Database connection
    engine = create_engine(database_uri)
    SessionLocal = sessionmaker(bind=engine)

    assert os.path.exists(data_path), f"{data_path.posix()} does not exist"
    #assert os.path.exists(database_uri), f"{database_uri} does not exist"

    data = pd.read_csv(data_path)
    for req_col in ['Question', 'Answer']:
        assert req_col in data.columns, f"{req_col} not present in csv data file"

    
    session_id = create_session()
    batch_ingest(session_id, data)

    


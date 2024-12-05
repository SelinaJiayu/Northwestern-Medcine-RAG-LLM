import warnings
warnings.filterwarnings('ignore')

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
import os
import argparse
from pathlib import Path

from cancer_rag.crud import get_session_chats


def format_session_data(data_list):
    formatted_data_list = []
    for entry in data_list:
        tmp_data = {
            "id" : entry.id, 
            "session_id" : entry.session_id,
            "user_question" : entry.user_question,
            "parsed_question" : entry.parsed_question,
            "response" : entry.response,
            "retrieval_similarity" : entry.retrieval_similarity,
            "retrieval_relevancy" : entry.retrieval_relevancy,
            "is_verified" : entry.is_verified
        }

        for metric, metric_data in entry.response_eval_scores.items():
            tmp_data[f'{metric}_score'] = metric_data['score']
            tmp_data[f'{metric}_reasoning'] = metric_data['reasoning']
        formatted_data_list.append(tmp_data)
    return formatted_data_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-uri", type=str, help="Path to SQlite Database")
    parser.add_argument("--session-id", type=int, help="Session id to fetch the chat")
    parser.add_argument("--save-path", type=Path, help="Path to save the session chat")
    args = parser.parse_args()
    
    database_uri = args.database_uri
    session_id = args.session_id
    save_path = args.save_path 

    if save_path is not None:
        assert str(save_path).endswith('.csv'), f"save-path can only be .csv file"
        os.makedirs(save_path.parent, exist_ok=True)

    # Database connection
    engine = create_engine(f'sqlite:///{database_uri}')
    SessionLocal = sessionmaker(bind=engine)

    assert os.path.exists(database_uri), f"{database_uri} does not exist"

    print(f"Session ID : {session_id} -- Save Path : {save_path}")
    db = SessionLocal()
    session_chat_data = get_session_chats(db, session_id)
    formatted_chat_data = format_session_data(session_chat_data)
    
    chat_df = pd.DataFrame(formatted_chat_data)
    chat_df.to_csv(save_path, index=False)
    print(f"Results successfully saved at {str(save_path)}")

    

    


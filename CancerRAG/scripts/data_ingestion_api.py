import requests
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm

"""
Example:

python scripts/data_ingestion_api.py \
    --data-path data/data_files/capstone_final_data_v1.csv \
    --api-base-url http://localhost:8000
"""

SESSION_ENDPOINT = "/sessions/"
CHAT_ENDPOINT = "/sessions/chats"

def create_session(api_base_url, default_username="default"):
    session_data = {
        "username": default_username,
        "age": None, 
        "gender": None
    }
    response = requests.post(f"{api_base_url}{SESSION_ENDPOINT}", json=session_data)
    response.raise_for_status()

    session_id = response.json().get("id")
    print(f"Created new session ID: {session_id}")
    return session_id

def batch_ingest(api_base_url, session_id, data):
    batch_size = 50
    num_batches = (len(data) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(data), batch_size), total=num_batches, desc="Ingesting batches"):
        batch = data[i:i + batch_size]
        
        for index, row in batch.iterrows():
            chat_data = {
                "session_id": session_id,
                "user_question": row['Question'],
                "parsed_question": row['Question'],
                "response": row['Answer'],
                "is_verified": True
            }
            chat_response = requests.post(f"{api_base_url}{CHAT_ENDPOINT}", json=chat_data)
            chat_response.raise_for_status() 

    print("Batch ingestion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, help="Path to Q/A data csv file")
    parser.add_argument("--api-base-url", type=str, help="Base URL for the API (e.g., http://localhost:8000)")
    args = parser.parse_args()

    data_path = args.data_path
    api_base_url = args.api_base_url

    assert data_path.exists(), f"{data_path} does not exist"

    # Load data
    data = pd.read_csv(data_path)
    for req_col in ['Question', 'Answer']:
        assert req_col in data.columns, f"{req_col} not present in csv data file"

    # Create a session and ingest data
    session_id = create_session(api_base_url)
    batch_ingest(api_base_url, session_id, data)

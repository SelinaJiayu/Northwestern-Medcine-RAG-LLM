# CancerRAG : A LLM Powered ChatEngine for Cancer Patients

## Installation

Clone the repository on your local system 

```bash
  git clone https://github.com/ayush9818/RAG-Based-LLM-for-Radiation-Oncology-Patient-Queries
  cd RAG-Based-LLM-for-Radiation-Oncology-Patient-Queries
```

Setup virtual environment and install dependencies

```bash
python3 -m venv .venv 
source .venv/bin/activate 

# For GPU 
pip install -r requirements-gpu.txt

# For CPU
pip install -r requirements-cpu.txt
```

```bash
python scripts/data_ingestion.py --data-path data/data_files/capstone_final_data_v1.csv --database-uri backend/cancer_QA.db
```

```bash
python scripts/fetch_session_chat.py --database-uri backend/cancer_QA.db --session-id 6 --save-path data/db_results/session_chat_treatmentModality.csv
```

## Setup

### Backend Service

### Frontend Service 


## Steps to run Ollama Server on docker

```bash
docker pull ollama/ollama
```

```bash
docker run -d \
  -v ollama:/root/.ollama \
  -p 11435:11434 \
  --network ollama-network \
  --name ollama_server \
  --rm \
  ollama/ollama
```


```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "model": "llama3.1:8b", 
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "Hello!" }
  ]
}' http://localhost:11435/v1/chat/completions
```

## Build Backend Image

```bash
cd backend
docker build -f Dockerfile -t cancer_rag_backend .
```

## Build Frontend Image 

```bash
cd frontend
docker build -f Dockerfile -t cancer_rag_frontend .
```


### TODO

- Better Error Handling 
- ChatBot Memory Management 
- Follow up Queries Integration
- Tone Moderation according to Human Inputs
- Moving Prompts from .py to .toml files
- Adding AWS Bedrock Configurations
- Add Greeting Chain or General Response Chain
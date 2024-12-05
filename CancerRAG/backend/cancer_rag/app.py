from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
import yaml 
import torch 
from pathlib import Path
import os 
from dotenv import load_dotenv

from cancer_rag.ai.retriever import create_retriever_chain
from cancer_rag.ai.chains import (
    create_conversion_chain, 
    create_knowledge_chain, 
    create_grader_chain,
    create_eval_chains
)
from cancer_rag.models.database import create_db
from cancer_rag.routers import session_chat_router, session_router
# export PYTHONPATH=/path/to/cancer_rag_backend:$PYTHONPATH

from cancer_rag.envs import envs
#load_dotenv(Path(__file__).parent / '../.env')


CONFIG_PATH = Path(__file__).parent / "configs/config.yaml"
DATASOURCE = envs.get('DATABASE_URI')
INIT_MODE = envs.get('INIT_MODE')
print(f"Data Source : {DATASOURCE}")

# Load configuration and device setup
config = yaml.full_load(open(CONFIG_PATH))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = FastAPI(
    title="NU Medicine ChatBot",
    version="1.0",
    description="NU Medicine Chatbot for answering cancer related queries.",
)

@app.get('/')
def healthcheck():
    return {"message" : "Welcome to NU Medicine Chatbot APIs"}

if INIT_MODE:
    create_db()
    print("Database Created")

if not INIT_MODE:
    ret_chain = create_retriever_chain(config['embedder_config'], device, DATASOURCE, config['retriever_config'])
    knowledge_chain = create_knowledge_chain(config['llm_config'], ret_chain)
    add_routes(
        app,
        knowledge_chain,
        path="/retrieve",
    )


    conversational_chain = create_conversion_chain(config['llm_config'])
    add_routes(
        app,
        conversational_chain,
        path="/chat",
    )


    context_grader = create_grader_chain(config['llm_config'])
    add_routes(
        app,
        context_grader,
        path="/grade_context",
    )

    eval_chains = create_eval_chains(config['llm_config'])
    for chain_name, eval_chain in eval_chains:
        add_routes(
            app,
            eval_chain,
            path=f"/eval/{chain_name}",
        )

# # Register routers
app.include_router(session_router.router, prefix="", tags=["Sessions"])
app.include_router(session_chat_router.router, prefix="", tags=["Session Chats"])

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
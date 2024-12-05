import warnings
warnings.filterwarnings("ignore")

import os 
from langchain_ollama import OllamaEmbeddings
import pandas as pd
import numpy as np 

import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import torch
import re
from typing import List
from langchain_core.runnables import chain
import json
import yaml 

from cancer_rag.utils import (
    load_data, 
    create_documents_from_df, 
    preprocess_text
)

question_vector_store = None
answers_vector_store = None
similarity_top_k = 10
similarity_threshold = 0.5
corpus_data = None

def init_embedder(embedder_config, device):
    embedder_params = embedder_config['params']
    if embedder_config['backend'] == "HF":
        embedder = HuggingFaceEmbeddings(
            model_name=embedder_params['model_name'],
            model_kwargs={'device': device},
            encode_kwargs=embedder_params.get('encode_kwargs', {})
        )
        print(f"Embedder Initialized with {embedder_params['model_name']}")
    elif embedder_config['backend'] == "OLLAMA":
        embedder = OllamaEmbeddings(model=embedder_params['model_name'])
        print(f"Embedder Initialized with {embedder_params['model_name']}")
    else:
        raise NotImplementedError("Embedder backend not supported")
    return embedder


def init_vectorstore(embedder):
    # Create the FAISS index for storing embeddings
    embedding_size = len(embedder.embed_query("hello world"))  # Example to get embedding size
    question_index = faiss.IndexFlatL2(embedding_size)
    answer_index = faiss.IndexFlatL2(embedding_size)
    question_vector_store = FAISS(
        embedding_function=embedder,
        index=question_index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    answers_vector_store = FAISS(
        embedding_function=embedder,
        index=answer_index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    return question_vector_store, answers_vector_store

def calculate_max_similarity(sims):
    if len(sims) == 0:
        return -1
    return float(np.max(np.array(sims)))

# Define Custom Functions as Runnables to use in retrieval chain
@chain
def retriever(query: dict) -> List[Document]:
    """Custom Retriever Logic to filter based on SIM THRESHOLD"""
    global question_vector_store, answers_vector_store, similarity_top_k, similarity_threshold
    print("Running Retriever Chain")
    query = preprocess_text(query.get('query',''))
    print("Performing Questions Level Retrieval")
    docs, scores = zip(*question_vector_store.similarity_search_with_relevance_scores(query, k=similarity_top_k))
    result = []
    for doc, score in zip(docs, scores):
        if score > similarity_threshold:
            doc.metadata["score"] = score
            result.append(doc)

    print("Performing Answers Level Retrieval")
    if len(result) == 0:
        docs, scores = zip(*answers_vector_store.similarity_search_with_relevance_scores(query, k=similarity_top_k)) 
        for doc, score in zip(docs, scores):
            if score > similarity_threshold:
                doc.metadata["score"] = score
                result.append(doc)        
    return result

@chain
def format_retrieved_docs(documents: List[Document]) -> str:
    """Context Formatter"""
    global corpus_data
    print("Formatting Retrieved Documents")
    docs = []
    for doc in documents:
        score = doc.metadata.get('score')
        index = doc.metadata.get('id')
        answer = corpus_data.iloc[index].Answer
        question = corpus_data.iloc[index].Question
        docs.append((question, answer, score))

    context_text = "\n".join([f"Q: {ctx[0]}\nA: {ctx[1]}" for ctx in docs])
    #mean_sim = calculate_mean_similarity([ctx[2] for ctx in docs])
    max_sim = calculate_max_similarity([ctx[2] for ctx in docs])
    print(context_text, max_sim)
    return {"context" : context_text, "retrieval_score" : max_sim}

def create_retriever_chain(embedder_config,
                            device,
                            datasource,
                            retriever_config
                            ):
    print("Initializing Retrieval Chain")
    global question_vector_store, answers_vector_store, similarity_top_k, similarity_threshold, corpus_data
    # Intialize Embedder and VectorStore
    embedder = init_embedder(embedder_config=embedder_config, device=device)
    question_vector_store,  answers_vector_store = init_vectorstore(embedder)

    corpus_data = load_data(datasource)
    # Add Documents to VectorStore
    question_documents = create_documents_from_df(corpus_data, index_questions=True)
    question_vector_store.add_documents(documents=question_documents, ids=corpus_data.index.tolist())
    print(f"{len(question_documents)} Documents added to Questions Vector Store")

    answer_documents = create_documents_from_df(corpus_data, index_questions=False)
    answers_vector_store.add_documents(documents=answer_documents, ids=corpus_data.index.tolist())
    print(f"{len(answer_documents)} Documents added to Answers Vector Store")

    similarity_top_k = retriever_config.get('similarity_top_k', 10)
    similarity_threshold = retriever_config.get('similarity_threshold', 0.5)

    ret_chain = retriever | format_retrieved_docs
    return ret_chain


if __name__=="__main__":
    # Quick Test 
    config = yaml.full_load(open('configs/config.yaml'))
    print(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    datasource = "/nfs/home/scg1143/RAG-Based-LLM-for-Radiation-Oncology-Patient-Queries/data/prelim_cancer_data.csv"

    ret_chain = create_retriever_chain(config['embedder_config'], 
                                       device,
                                       datasource, 
                                       config['retriever_config']
                                       )
    print(ret_chain.invoke({"query": "What are the treatment options for head and skin cancer"}))
import yaml
from chat_app.models import KnowledgeBase
from langserve import RemoteRunnable
import streamlit as st
import os
import json
from chat_app.envs import envs
from chat_app.utils import (
    create_user_chat, 
    create_user_session, 
    invoke_chain_with_retry, 
    run_text_analytics
)
import pandas as pd
import time
from argparse import ArgumentParser
from pathlib import Path 

UNSUCCESSFUL_RETRIEVAL_RESPONSE = """
Thank you for your question. Unfortunately, I wasn't able to find any relevant information for your query at the moment. 
To assist you better, could you please provide more details or clarify your question? 
Your input will help me give you a more accurate and helpful response. 
If you need general guidance on cancer-related topics, I can also provide some initial information to get us started.
"""

WELCOME_RESPONSE = "Hello {full_name}! I'm your NU Medicine Agent! How can I help you?"


def chat_gen(message, history=[]):
    global know_base, knowledge_chain, context_grader, conversational_chain
    try:
        state = {
                "input": message,
                "summary": know_base["summary"],
                "output": "",
                "know_base": know_base,
                "age" : know_base['age'],
                "gender" : know_base['gender'],
                "education_level" : know_base['education_level'],
                "response" : "" if not history else history[-1]["content"]
            }
        state["output"] = "" if not history else history[-1]["content"]
        know_base = state["know_base"]
        state = json.loads(knowledge_chain.invoke(state))
        state['summary'] = state["know_base"]["summary"]
        retrieval_score = state.get("retrieval_score")
        relevancy_score = 1.0
        state["relevancy_score"] = relevancy_score
        ret_score_token = f"Retrieval Score: {retrieval_score:.2f}\nRelevancy Score: {relevancy_score}\n"
        yield ret_score_token
        if retrieval_score == -1:
            yield UNSUCCESSFUL_RETRIEVAL_RESPONSE
            yield state
        else:
            for token in conversational_chain.stream(state):
                yield token
            yield state
    except Exception as e:
        raise e


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=Path, help="Dataset Directory")
    parser.add_argument("--save-dir", type=Path, help="Save Directory")

    args = parser.parse_args()
    data_dir = args.data_dir 
    save_dir = args.save_dir 

    os.makedirs(save_dir, exist_ok=True)

    AGE = 35
    GENDER = "Male"
    FULL_NAME = "EvalAgent"
    EDUCATION_LEVEL = "12th grade"


    knowledge_chain = RemoteRunnable(envs.get("retrieval_api"))
    conversational_chain = RemoteRunnable(envs.get("conversation_api"))
    eval_chains = [
            (eval_metric, RemoteRunnable(chain_url))
            for eval_metric, chain_url in envs.get("eval_apis", {}).items()
        ]
    context_grader = RemoteRunnable(envs.get("context_grader_api"))
    

    for file_name in os.listdir(data_dir):
        print(f"Processing File : {file_name}")
        df = pd.read_csv(data_dir / file_name)

        if "Original Question" not in df.columns:
            print("Skipping. Original Question Column not found")
            continue 

        done = 0 
        total = df.shape[0]
        failed = 0

        response_list = []
        failed_questions = []
        print("Total Rows : {}".format(df.shape[0]))
        for idx, row in df.iterrows():
            try:
                if not isinstance(row['Original Question'], str):
                    response_list.append(None)
                else:

                    know_base = dict(KnowledgeBase())
                    know_base["name"] = FULL_NAME
                    know_base["age"] = str(AGE)
                    know_base["gender"] = GENDER
                    know_base['disease_site'] = row['Diease Site']
                    know_base['education_level'] = EDUCATION_LEVEL

                    chat_history = [{"role": "AI","content": WELCOME_RESPONSE.format(full_name=FULL_NAME),}]
                    user_query = row['Original Question']
                    chat_history.append({"role": "Human", "content": user_query})

                    response_buffer = ""
                    query = None
                    context = None
                    for i, token in enumerate(chat_gen(user_query, chat_history)):
                        if i == 0:
                            continue
                        else:
                            if isinstance(token, dict):
                                query = token.get("query", None)
                                context = token.get("context", None)
                            else:
                                response_buffer += token


                    response_list.append(response_buffer)
                    # Append final AI response to chat history
                    #chat_history.append({"role": "AI", "content": response_buffer})
                    time.sleep(2)
                    done+=1
            except Exception as e:
                print(e)
                failed+=1
                #failed_questions.append(row['Question'])
            
            if (done + failed) % 10 == 0:
                print(f"{done}/{total} Done - {failed}/{total} Failed")
        
        df['Original Answer'] = response_list
        df.to_csv(save_dir / file_name, index=False)
        print(f"Saved at {save_dir / file_name}")
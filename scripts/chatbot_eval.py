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
        relevancy_score = context_grader.invoke(state)
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
    parser.add_argument("--data-path", type=Path, help="Dataset Path")
    parser.add_argument("--save-dir", type=Path, help="Path to save artifacts")

    args = parser.parse_args()
    data_path = args.data_path 
    save_dir = args.save_dir

    file_name = os.path.basename(data_path)

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
    
    df = pd.read_csv(data_path)
    done = 0 
    total = df.shape[0]
    failed = 0

    response_list = []
    failed_questions = []
    print("Total Questions : {}".format(df.shape[0]))
    for idx, row in df.iterrows():
        try:
            know_base = dict(KnowledgeBase())
            know_base["name"] = FULL_NAME
            know_base["age"] = str(AGE)
            know_base["gender"] = GENDER
            know_base['disease_site'] = row['Diease Site']
            know_base['education_level'] = EDUCATION_LEVEL

            chat_history = [{"role": "AI","content": WELCOME_RESPONSE.format(full_name=FULL_NAME),}]
            user_query = row['Modified Question']
            chat_history.append({"role": "Human", "content": user_query})

            response_buffer = ""
            query = None
            context = None
            response_dict = {}
            for i, token in enumerate(chat_gen(user_query, chat_history)):
                if i == 0:
                    continue
                else:
                    if isinstance(token, dict):
                        query = token.get("query", None)
                        context = token.get("context", None)
                        response_dict["parsed_question"] = query
                        response_dict["retrieval_similarity"] = token.get(
                            "retrieval_score", None
                        )
                        response_dict["retrieval_relevancy"] = token.get(
                            "relevancy_score", None
                        )
                    else:
                        response_buffer += token

            eval_scores = {}
            if query is not None and context is not None:
                print("Running Eval Chains....")
                eval_token = ""
                for metric, chain in eval_chains:
                    data = {"input": query, "prediction": response_buffer, "reference": context}
                    result = invoke_chain_with_retry(chain=chain, data=data, _chain_name=metric, num_retries=3)
                    if result is None:
                        result = {}
                    else:
                        result = result.get("results", {})
                    score = result.get("score", 0) / 10
                    reasoning = result.get('reasoning', '')

                    eval_token += f"**{metric}** : {score:.2f} \n\n {reasoning} \n\n"

                    eval_scores[metric] = {
                        "score": score,
                        "reasoning": reasoning,
                    }

            response_dict["Modified Question"] = user_query
            response_dict["response"] = response_buffer
            for metric in eval_scores.keys():
                response_dict[f'{metric}_score'] = eval_scores[metric]['score']
                response_dict[f'{metric}_reasoning'] = eval_scores[metric]['reasoning']

            response_analytics = run_text_analytics(response_buffer)
            for key, value in response_analytics.items():
                response_dict[key] = value

            response_list.append(response_dict)
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

    response_df = pd.DataFrame(response_list)
    final_df = df.merge(response_df, how='left', on='Modified Question')
    final_df.to_csv(save_dir / file_name, index=False)
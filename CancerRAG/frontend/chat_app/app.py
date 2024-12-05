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


TEST_MODE=envs.get('test_mode', True)

UNSUCCESSFUL_RETRIEVAL_RESPONSE = """
Thank you for your question. Unfortunately, I wasn't able to find any relevant information for your query at the moment. 
To assist you better, could you please provide more details or clarify your question? 
Your input will help me give you a more accurate and helpful response. 
If you need general guidance on cancer-related topics, I can also provide some initial information to get us started.
"""

# Streamlit app configuration
st.set_page_config(page_title="NU Medicine Agent", page_icon="🤖")
st.title("NU Medicine Agent")

# Initialize session state for chains and knowledge base if not already set
if "chains_initialized" not in st.session_state:
    st.session_state.knowledge_chain = RemoteRunnable(envs.get("retrieval_api"))
    st.session_state.conversational_chain = RemoteRunnable(envs.get("conversation_api"))
    st.session_state.eval_chains = [
        (eval_metric, RemoteRunnable(chain_url))
        for eval_metric, chain_url in envs.get("eval_apis", {}).items()
    ]
    st.session_state.context_grader = RemoteRunnable(envs.get("context_grader_api"))
    st.session_state.knowledge_base = dict(KnowledgeBase())
    st.session_state.chains_initialized = True

# Information Form to collect user's details at the start
if "user_info_collected" not in st.session_state:
    st.session_state.user_info_collected = False

if not st.session_state.user_info_collected:
    with st.form("user_info_form"):
        full_name = st.text_input("Full Name")
        age = st.number_input("Age", min_value=10, max_value=120, step=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        disease_site = st.text_input("Primary suspected location of cancer (e.g., lung, head and neck, breast, etc.)")
        education_level = st.text_input("Highest level of education completed (e.g., 1st grade, 5th grade, Bachelor's degree, etc.)")
        submitted = st.form_submit_button("Submit")

    if submitted and full_name and age:
        # Fill the KnowledgeBase with user-provided information
        st.session_state.knowledge_base["name"] = full_name
        st.session_state.knowledge_base["age"] = str(age)
        st.session_state.knowledge_base["gender"] = gender
        st.session_state.knowledge_base["disease_site"] = disease_site
        st.session_state.knowledge_base["education_level"] = education_level
        st.session_state.user_info_collected = True
        if not TEST_MODE:
            st.session_state.session_id = create_user_session(full_name, age, gender, disease_site, education_level)
            print(f"User session created. Session ID : {st.session_state.session_id}")

        st.success("Thank you for providing the information!")
        st.rerun()
else:

    def chat_gen(message, history=[]):
        try:
            knowledge_chain = st.session_state.knowledge_chain
            conversational_chain = st.session_state.conversational_chain
            context_grader = st.session_state.context_grader
            know_base = st.session_state.knowledge_base

            # Initializing the state with the necessary values
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
            st.session_state.knowledge_base = state["know_base"]
            state = json.loads(knowledge_chain.invoke(state))

            state['summary'] = state["know_base"]["summary"]

            print(state["know_base"]["summary"])
            print(state["know_base"]["query"])

            retrieval_score = state.get("retrieval_score")
            relevancy_score = context_grader.invoke(state)
            state["relevancy_score"] = relevancy_score
            ret_score_token = f"Retrieval Score: {retrieval_score:.2f}\nRelevancy Score: {relevancy_score}\n"
            yield ret_score_token

            # For unsuccessful retrieval, ask user to rephrase the quetsion
            if retrieval_score == -1:
                yield UNSUCCESSFUL_RETRIEVAL_RESPONSE
                yield state
            else:
                for token in conversational_chain.stream(state):
                    yield token
                yield state
        except Exception as e:
            print("Error Occurred:{}".format(e))
            yield "An error occurred. Please try again."

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "AI",
                "content": f"Hello {st.session_state.knowledge_base['name']}! I'm your NU Medicine Agent! How can I help you?",
            }
        ]

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    # User input
    user_query = st.chat_input("Type your message here...")
    if user_query:
        st.session_state.chat_history.append({"role": "Human", "content": user_query})

        # Display human message
        with st.chat_message("Human"):
            st.markdown(user_query)

        eval_chains = st.session_state.eval_chains

        response_container = st.empty()
        score_placeholder = st.empty()
        eval_placeholder = st.empty()
        summary_placeholder = st.empty()

        # Stream the response token by token with error handling
        response_buffer = ""
        summary_buffer = ""
        query = None
        context = None
        db_chat_mess = {}
        for i, token in enumerate(chat_gen(user_query, st.session_state.chat_history)):
            if i == 0:
                score_placeholder.markdown(
                    f'<div style="background-color: #f0f0f5; padding: 10px; border: 1px solid #ccc; border-radius: 5px;"><strong>{token}</strong></div>',
                    unsafe_allow_html=True,
                )
            else:
                if isinstance(token, dict):
                    query = token.get("query", None)
                    context = token.get("context", None)
                    db_chat_mess["parsed_question"] = query
                    db_chat_mess["retrieval_similarity"] = token.get(
                        "retrieval_score", None
                    )
                    db_chat_mess["retrieval_relevancy"] = token.get(
                        "relevancy_score", None
                    )
                    summary_buffer = token.get('summary')
                else:
                    response_buffer += token
                    response_container.markdown(response_buffer)

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

                with eval_placeholder.expander("Evaluation Results", expanded=False):
                    st.markdown(eval_token)
        
        with summary_placeholder.expander("Running Summary", expanded=False):
            st.markdown(summary_buffer)

        response_analytics = run_text_analytics(response_buffer)
        if not TEST_MODE:
            db_chat_mess["user_question"] = user_query
            db_chat_mess["response"] = response_buffer
            db_chat_mess["is_verified"] = False
            db_chat_mess["response_eval_scores"] = eval_scores
            db_chat_mess['session_id'] = st.session_state.session_id
            db_chat_mess['response_analytics'] = response_analytics
            # Ingest Chat Data into Database
            create_user_chat(db_chat_mess)
            print("Chat data ingested successfully.")
        else:
            print(response_analytics)

        # Append final AI response to chat history
        st.session_state.chat_history.append({"role": "AI", "content": response_buffer})

# References for Evaluation
# - https://docs.llamaindex.ai/en/stable/examples/evaluation/semantic_similarity_eval/
# - https://python.langchain.com/v0.1/docs/guides/productionization/evaluation/string/

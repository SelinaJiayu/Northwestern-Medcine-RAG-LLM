import os 
from dotenv import load_dotenv

load_dotenv()

env = os.environ


API_BASE_URL=env.get('BASE_URL', None)
TEST_MODE=int(env.get('TEST_MODE', 1))

def create_url(base_url, api_url, _api_name):
    if base_url is None:
        raise Exception(f"BASE_URL can not be None.")
    if api_url is None:
        raise Exception(f"API URL for {_api_name} API is None.")
    
    return base_url + api_url
    
envs = {
    "retrieval_api" : create_url(API_BASE_URL, env.get('RETRIEVAL_CHAIN_URL'), 'retrieval_api'),
    "conversation_api" : create_url(API_BASE_URL, env.get('CONVERSATIONAL_CHAIN_URL'), 'conversation_api'),
    "context_grader_api" : create_url(API_BASE_URL, env.get('CONTEXT_GRADER_URL'), 'context_grader_api'),
    "session_api" : create_url(API_BASE_URL, env.get('SESSION_API_URL'), 'session_api'),
    "session_chat_api" : create_url(API_BASE_URL, env.get('SESSION_CHAT_API_URL'), 'session_chat_api'),
    "eval_apis" : {
        "correctness" : create_url(API_BASE_URL, env.get('CORRECTNESS_EVAL_CHAIN_URL'), 'correctness'),
        "relevance" : create_url(API_BASE_URL, env.get('RELEVANCE_EVAL_CHAIN_URL'), 'relevance'),
        "harmfulness" : create_url(API_BASE_URL, env.get('HARMFULNESS_EVAL_CHAIN_URL'), 'harmfulness'),
        "conciseness" : create_url(API_BASE_URL, env.get('CONCISENESS_EVAL_CHAIN_URL'), 'conciseness'),
        "coherence" : create_url(API_BASE_URL, env.get('COHERENCE_EVAL_CHAIN_URL'), 'coherence'),
    },
    "test_mode" : True if TEST_MODE == 1 else False
}

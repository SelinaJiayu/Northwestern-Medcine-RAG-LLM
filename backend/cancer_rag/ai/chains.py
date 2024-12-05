from cancer_rag.ai.prompts import knowledge_prompt, conversation_prompt, grade_prompt
from cancer_rag.utils import RExtract
from cancer_rag.ai.models import KnowledgeBase, GradeDocuments
from cancer_rag.ai.llms import OllamaLLMProvider, AwsLLMProvider

from langchain.evaluation.scoring import ScoreStringEvalChain

from langchain.schema.runnable.passthrough import RunnableAssign
from langchain_core.runnables.base import RunnableLambda
from langchain_core.runnables import chain
import json 
from cancer_rag.envs import envs

LLM_PROVIDER_MAP = {
    "OLLAMA" : OllamaLLMProvider, 
    "AWS" : AwsLLMProvider
}

LLM_PROVIDER = LLM_PROVIDER_MAP.get(envs.get('LLM_PROVIDER', 'OLLAMA'), None)
assert LLM_PROVIDER is not None, f"Available LLM_PROVIDER are {LLM_PROVIDER_MAP.key()}"
llm_provider = LLM_PROVIDER()


@chain
def extract_query(state):
    return state.get('know_base').query

@chain
def merge_outputs(ret_results):
    result = {**ret_results[0], **ret_results[1]}
    result['know_base'] = dict(result['know_base'])
    return json.dumps(result)


@chain
def extract_know_base(x):
    if isinstance(x['know_base'], KnowledgeBase):
        return x
    know_base = KnowledgeBase(**x['know_base'])
    x['know_base'] = know_base
    return x 

def create_knowledge_chain(llm_config, ret_chain, key='knowledge_chain'):
    print("Initializing Knowledge Chain")
    llm = llm_provider.load_llm(llm_config[key])
    extractor = RExtract(KnowledgeBase, llm, knowledge_prompt)
    info_update = RunnableAssign({'know_base' : extractor})

    knowledge_chain = (
        extract_know_base 
        | info_update
        | RunnableAssign({"query" : extract_query})
        | (lambda x: (x, ret_chain.invoke(x)))
        | merge_outputs
    )
    return knowledge_chain.with_retry()

def create_conversion_chain(llm_config, key='conversation_chain'):
    print("Initializing Conversational Chain")
    llm = llm_provider.load_llm(llm_config[key])
    external_chain = conversation_prompt | llm
    return extract_know_base | external_chain.with_retry()

def create_grader_chain(llm_config, key='grader_chain'):
    print("Initializing Grader Chain")
    # LLM with function call
    grader_llm = llm_provider.load_llm(llm_config[key], chat_model=True)
    context_grader = (
                        grade_prompt 
                        | grader_llm.with_structured_output(GradeDocuments) 
                        | RunnableLambda(lambda x : x.relevancy_score)
                    )
    return context_grader.with_retry() 


def create_eval_chains(llm_config, key='eval_chain'):
    print("Initializing Evaluation Chains")
    eval_llm = llm_provider.load_llm(llm_config[key])
    chains = []
    for metric in llm_config[key]['eval_metrics']:
        eval_chain = ScoreStringEvalChain.from_llm(llm=eval_llm, criteria=metric) 
        chains.append((metric, eval_chain))
    return chains 

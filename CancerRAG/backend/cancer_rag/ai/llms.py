#from langchain_aws import BedrockLLM
#from langchain_aws import ChatBedrockConverse
from langchain_ollama import ChatOllama
from langchain_ollama.llms import OllamaLLM

from cancer_rag.envs import envs

class AwsLLMProvider:
    def __init__(self):
        pass

    def load_llm(llm_config, chat_model=False):
        pass


class OllamaLLMProvider:
    ALLOWED_LLMS = ["llama3.1:8b", "llama3.1:70b"]
    def __init__(self):
        print("OLLAMA LLM provider initialized")
    
    def load_llm(self, llm_config, chat_model=False):
        assert (
            llm_config["model_name"] in self.ALLOWED_LLMS
        ), f"{llm_config['model_name']} not supported. Supported: {self.ALLOWED_LLMS}"

        if chat_model:
            if envs.get('OLLAMA_SERVER') is None:
                print("Using default ollama server")
                llm = ChatOllama(
                    model=llm_config["model_name"],
                    temperature=llm_config.get("temperature", 0.4),
                )
            else:
                print("Using API based ollama server")
                llm = ChatOllama(
                    model=llm_config["model_name"],
                    base_url=envs.get('OLLAMA_SERVER'),
                    temperature=llm_config.get("temperature", 0.4),
                )
        else:
            if envs.get('OLLAMA_SERVER') is None:
                print("Using default ollama server")
                llm = OllamaLLM(
                    model=llm_config["model_name"],
                    temperature=llm_config.get("temperature", 0.4),
                )
            else:
                print("Using API based ollama server")
                llm = OllamaLLM(
                    model=llm_config["model_name"],
                    base_url=envs.get('OLLAMA_SERVER'),
                    temperature=llm_config.get("temperature", 0.4),
                )
        return llm

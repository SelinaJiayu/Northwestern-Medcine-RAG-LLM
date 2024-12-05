import os 
from dotenv import load_dotenv

load_dotenv()

env = os.environ


def load_boolean_vars(var_name, default=False):
    if env.get(var_name, None) == None:
        return default
    
    if env.get(var_name) == "1":
        return True 
    return False

envs = {
    "DATABASE_URI" : env.get('DATABASE_URI', None),
    "OLLAMA_SERVER" : env.get("OLLAMA_SERVER", None),
    "INIT_MODE" : load_boolean_vars('INIT_MODE', False)
}
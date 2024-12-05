import time 
from chat_app.api_handler import APIHandler
from chat_app.envs import envs
import textstat
import nltk 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download('punkt_tab')


def create_user_session(username, age, gender, education_level, disease_site):
    session_api_url = envs.get('session_api')
    payload = {
        "username" : username, 
        "age" : age, 
        "gender" : gender,
        "education_level" : education_level,
        "disease_site" : disease_site
    }   
    api_handler = APIHandler(api_url=session_api_url)
    data = api_handler.post(data=payload)
    # TODO: Handle None data better
    return data['id']


def create_user_chat(chat_data):
    """Sends chat data to the session chat API and returns the response."""
    session_api_url = envs.get('session_chat_api')
    api_handler = APIHandler(api_url=session_api_url)
    data = api_handler.post(data=chat_data)
    return data 


def invoke_chain_with_retry(chain, data, _chain_name, num_retries=3):
    """
    Invokes a chain with automatic retries on failure.

    This function attempts to invoke a specified chain with the provided data. If the chain invocation fails,
    it will retry the specified number of times, waiting 3 seconds between attempts. This function is useful for 
    handling temporary network or processing errors during chain execution.

    Parameters:
    -----------
    chain : object
        The chain object that has an `invoke` method to process the provided data.
    data : any
        The input data to be processed by the chain.
    _chain_name : str
        The name of the chain being invoked, used for logging and identification.
    num_retries : int, optional
        The maximum number of retries in case of failure (default is 3).


    Example:
    --------
    >>> invoke_chain_with_retry(my_chain, input_data, "DataProcessingChain", num_retries=5)
    Invoking Chain : DataProcessingChain
    Retry 1/5 Failed. Retrying again...
    Retry 2/5 Failed. Retrying again...
    # On successful attempt, it returns the chain's output.
    """
    print(f"Invoking Chain : {_chain_name}")
    for _retry in range(num_retries):
        try:
            output = chain.invoke(data)
            return output
        except:
            print(f"Retry {_retry+1}/{num_retries} Failed. Retrying again...")
            time.sleep(3) 
    return None

def run_text_analytics(response):
    """
    Analyzes the readability and text metrics of a given response.

    This function takes a text input (`response`) and calculates various readability and linguistic metrics
    using the `textstat` library and `nltk` for word count. It returns a dictionary containing these metrics,
    which provide insight into the text's readability, complexity, and structure.

    Parameters:
    -----------
    response : str
        The text input to be analyzed.

    Returns:
    --------
    dict
        A dictionary (`analytics_report`) containing the following readability and text metrics:
        
        - Flesch_reading_ease : A score indicating how easy the text is to read (higher is easier).
        - Flesch_kincaid_grade_level : An estimated U.S. school grade level required to comprehend the text.
        - Fog Scale : A measure of reading difficulty based on sentence length and complex words.
        - Smog_index : An estimation of the reading grade level using the SMOG formula.
        - Automated Readability index : A readability score based on sentence and word lengths.
        - Coleman_Liau_Index : A readability index based on letter counts and sentence structure.
        - Linear Write : A formula estimating the text’s readability by grade level.
        - Dale_chall_readability_score : A score reflecting difficulty for readers with grade 4+ vocabulary.
        - Readability consensus : A general readability level consensus derived from multiple metrics.
        - Syllable : The total count of syllables in the text.
        - Lexicon : The number of words in the text.
        - Sentence : The number of sentences in the text.
        - Word_count : The total number of words in the text.
        - polartity_scores : Sentiment Scores for the text 

    Example:
    --------
    >>> run_text_analytics("This is a simple sentence for testing readability metrics.")
    {
        'Flesch_reading_ease': 70.5,
        'Flesch_kincaid_grade_level': 6.2,
        'Fog Scale': 7.1,
        'Smog_index': 6.8,
        'Automated Readability index': 5.9,
        'Coleman_Liau_Index': 8.0,
        'Linear Write': 6.0,
        'Dale_chall_readability_score': 7.5,
        'Readability consensus': '6th and 7th grade',
        'Syllable': 12,
        'Lexicon': 9,
        'Sentence': 1,
        'Word_count': 9,
        'polarity_scores' : {'neg': 0.0, 'neu': 0.254, 'pos': 0.746, 'compound': 0.8316}
    }
    """
    FLESCH_EASE = 'Flesch_reading_ease'
    FLESCH_KIN ='Flesch_kincaid_grade_level'
    FOG_SCALE = 'Fog Scale'
    SMOG_INDEX = 'Smog_index'
    AUTO_READ = 'Automated Readability index'
    COLEMAN = 'Coleman_Liau_Index'
    LINEAR_WRITE = 'Linear Write'
    DALE = 'Dale_chall_readability_score'
    CONSENSUS = 'Readability consensus'
    SYLLABLE = 'Syllable'
    LEXICON = 'Lexicon'
    SENT = 'Sentence '
    WORD_COUNT = 'Word_count'
    POLARITY = 'polarity_scores'

    analyzer = SentimentIntensityAnalyzer()

    metricMap = {
        FLESCH_EASE:textstat.flesch_reading_ease,
        FLESCH_KIN:textstat.flesch_kincaid_grade,
        FOG_SCALE:textstat.gunning_fog,
        SMOG_INDEX:textstat.smog_index,
        AUTO_READ:textstat.automated_readability_index,
        COLEMAN:textstat.coleman_liau_index,
        LINEAR_WRITE:textstat.linsear_write_formula,
        DALE:textstat.dale_chall_readability_score,
        CONSENSUS:textstat.text_standard,
        SYLLABLE:textstat.syllable_count,
        LEXICON:textstat.lexicon_count,
        SENT:textstat.sentence_count,
        WORD_COUNT:lambda x:len(nltk.word_tokenize(x)),
        POLARITY:analyzer.polarity_scores
    }
    
    analytics_report = {}
    for metric_name, metric_func in metricMap.items():
        analytics_report[metric_name] = metric_func(response)

    return analytics_report






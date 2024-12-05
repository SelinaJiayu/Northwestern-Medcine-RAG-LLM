import os 
import re
import pandas as pd 
from langchain.output_parsers import PydanticOutputParser
from langchain.schema.runnable.passthrough import RunnableAssign
from langchain_core.runnables import chain

from cancer_rag.models.database import get_db
from cancer_rag.crud import get_verified_data
from langchain_core.documents import Document


def RExtract(pydantic_class, llm, prompt):
    '''
    Runnable Extraction module
    Returns a knowledge dictionary populated by slot-filling extraction
    '''
    parser = PydanticOutputParser(pydantic_object=pydantic_class)
    instruct_merge = RunnableAssign({'format_instructions' : lambda x: parser.get_format_instructions()})
    def preparse(string):
        if '{' not in string: string = '{' + string
        if '}' not in string: string = string + '}'
        string = (string
            .replace("\\_", "_")
            .replace("\n", " ")
            .replace("\]", "]")
            .replace("\[", "[")
        )
        # print(string)  ## Good for diagnostics
        return string
    return instruct_merge | prompt | llm | preparse | parser

def preprocess_text(text):
    """
    Preprocess the input text for embedding by normalizing and cleaning.
    :param text: input string to preprocess
    :return: cleaned and preprocessed string
    """
    print(text)
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

def load_data_from_db():
    db = next(get_db())
    data = get_verified_data(db)
    questions = []
    answers = []
    for row in data:
        questions.append(row.parsed_question)
        answers.append(row.response)
    
    data = pd.DataFrame({"Question" : questions, "Answer" : answers})
    print(f"Loaded {data.shape[0]} Q/A Pairs")
    return data


def load_data(datasource):
    if datasource.endswith('.csv'):
        print("Loading data from csv")
        # CSV Data source
        assert os.path.exists(datasource), f"DataSource {datasource} does not exist"
        data = pd.read_csv(datasource)
        print(f"Loaded {data.shape[0]} Q/A Pairs")
        return data 
    elif datasource.endswith('.db') or 'postgresql' in datasource:
        # SQLITE Data source
        print("Loading data from sql database")
        data = load_data_from_db()
        return data
    else:
        raise NotImplementedError("Currently can load data from .csv, .db files and postgres urls")

def create_documents_from_df(data, index_questions=True):
    data['Question'] = data['Question'].apply(lambda x : preprocess_text(x))
    data['Answer'] = data['Answer'].apply(lambda x : preprocess_text(x))
    
    if index_questions:
        print("Indexing Questions")
        documents = []
        for index, row in data.iterrows():
            question = row['Question']
            #answer = row['Answer']
            _ = Document(
                page_content=question,
                metadata={"id" : index}
            )
            documents.append(_)
        return documents
    else:
        print("Indexing Answers")
        documents = []
        for index, row in data.iterrows():
            question = row['Question']
            answer = row['Answer']
            _ = Document(
                page_content=answer,
                metadata={"id" : index, "question" : question}
            )
            documents.append(_)
        return documents
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from typing import Optional
from pydantic import BaseModel, Field
import pandas as pd 
import os 
import argparse
from tqdm import tqdm 

# Define output scheme for the LLM Response
class Question(BaseModel):
    """Directed Question based on Topic and Question Text"""
    question: str = Field(description="The precise and directed question based on topic, question text and answer")

# Define system message for chat
system_message = (
    "system",
    """
    You are a helpful assistant generates precise and directed question given a TOPIC_NAME and QUESTION_TEXT.
    The generated question should be readable by a 6th grader, short and strictly related to topic name. Don't hallucinate.
    """
)

# Define prompt for LLM
prompt = ChatPromptTemplate.from_messages(
    [   
        system_message,
        (
            "human", 
            """
            TOPIC_NAME : {topic_name}
            QUESTION_TEXT : {Question}
            """
        ),
    ]
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data-path", type=str, help="local path of input dataset")
    parser.add_argument("--output-path", type=str, help="Path to save output file")

    args = parser.parse_args()

    input_data_path = args.input_data_path
    output_path = args.output_path

    assert input_data_path is not None and os.path.exists(input_data_path), f"{input_data_path} does not exists"
    assert output_path is not None, "output path can't be None"

    # Create parent directories
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print("Input Dataset Path:", input_data_path)
    print("Output Path:",output_path)
    
    llm = ChatOllama(model="llama3.1:8b")
    structured_llm = llm.with_structured_output(Question)
    question_chain = prompt | structured_llm
    print("LLM Chain Created")

    df = pd.read_csv(input_data_path)

    generated_responses = []
    total = df.shape[0]
    done = 0 
    failed = 0 
    with tqdm(total=len(df), desc="Processing rows") as pbar:
        for index, row in df.iterrows():
            try:
                input_dict = row.to_dict()
                response = question_chain.invoke(input_dict)
                generated_responses.append(response.question)
                done += 1
            except:
                generated_responses.append(None)
                failed += 1
            
            pbar.update(1)
            pbar.set_postfix(total=total, done=done, failed=failed)

    df['Generated Question'] = generated_responses
    df=df[["topic_name", "Question", "Generated Question", "Answer"]]
    df.to_csv(output_path, index=False)

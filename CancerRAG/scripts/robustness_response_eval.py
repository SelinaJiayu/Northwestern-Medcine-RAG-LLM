import argparse
import os 
from pathlib import Path 
import pandas as pd 
from tqdm import tqdm 

tqdm.pandas()

from pydantic import BaseModel, Field
from typing import List
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

from langchain_ollama.llms import OllamaLLM

class MetricEvaluation(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    explanation: str

class FinalAssessment(BaseModel):
    overall_rating: int = Field(..., ge=1, le=5)
    explanation: str

class SimilarityEvaluation(BaseModel):
    semantic_similarity: MetricEvaluation
    key_point_alignment: MetricEvaluation
    overall_message_consistency: MetricEvaluation
    final_assessment: FinalAssessment


template = """
You are an expert judge tasked with evaluating the contextual similarity between two answers. 

Evaluation Criteria:
1. Semantic Similarity: How closely do the answers match in terms of meaning?
2. Key Point Alignment: To what extent do the main points or arguments in both answers align?
3. Overall Message Consistency: How consistent are the overall messages conveyed by both answers?

Rating Scale:
1 = Not similar at all
2 = Slightly similar
3 = Moderately similar
4 = Very similar
5 = Extremely similar or identical

Detailed Instructions:
- Rate each metric independently
- Provide a clear explanation for each rating
- Compute an overall rating that mathematically and contextually represents the aggregate similarity
- The final assessment should:
  a) Calculate an overall rating (weighted average of individual metrics)
  b) Provide a comprehensive explanation justifying the overall rating
  c) Highlight nuanced differences or similarities

Answer A:
{answer_a}

Answer B:
{answer_b}

{format_instructions}

# Weighted calculation hint in prompt
Additional Guidance for Final Assessment:
- Overall Rating Calculation:
  * Semantic Similarity: 40% weight
  * Key Point Alignment: 30% weight
  * Overall Message Consistency: 30% weight
- Round to nearest whole number
- Explanation should capture the essence of similarity comprehensively
"""

EVAL_COLUMNS = ['semantic_similarity_rating', 'semantic_similarity_explanation',
       'key_point_alignment_rating', 'key_point_alignment_explanation',
       'overall_message_consistency_rating',
       'overall_message_consistency_explanation', 'final_assessment_rating',
       'final_assessment_explanation']

def parse_results(result):
    judge_response = {}
    for metric, value in dict(result).items():
        if 'rating' in dict(value):
            rating = value.rating
        else:
            rating = value.overall_rating
        explanation = value.explanation

        judge_response[f'{metric}_rating'] = rating
        judge_response[f'{metric}_explanation'] = explanation
    return judge_response

def get_null_response():
    return {col : None  for col in EVAL_COLUMNS}


def process_one_df(df):
    original_answer = None 
    response_list = []

    total = df.shape[0]
    done = 0 
    failed = 0 

    print(f"Total Rows : {total}")

    with tqdm(total=total, desc="Processing rows") as pbar:
        for idx, row in df.iterrows():
            try:
                if isinstance(row['Original Answer'], str):
                    original_answer = row['Original Answer']
                
                answer_a = original_answer
                answer_b = row['response']

                result = chain.invoke({
                "answer_a": answer_a, 
                "answer_b": answer_b
                })

                parsed_results = parse_results(result=result)
                response_list.append(parsed_results) 
                done+=1 
            except Exception as e:
                failed+=1
                response_list.append(get_null_response())  
            finally:
                pbar.update(1)
                pbar.set_postfix({
                    "Done": done,
                    "Failed": failed,
                    "Total": total
                })

    response_df = pd.DataFrame(response_list)
    final_df = pd.concat([df, response_df], axis=1)
    return final_df  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, help="Directory Containing csv files to evaluate")
    parser.add_argument("--save-dir", type=Path, help='Directory to save the processed results')

    args = parser.parse_args()

    data_dir = args.data_dir 
    save_dir = args.save_dir 

    assert os.path.exists(data_dir), f"{data_dir} does not exist"

    os.makedirs(save_dir, exist_ok=True)
    print(f"Save Directory set to {save_dir}")

    parser = PydanticOutputParser(pydantic_object=SimilarityEvaluation)

    prompt = PromptTemplate(
        template=template,
        input_variables=["answer_a", "answer_b"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    llm = OllamaLLM(model='qwen2.5:32b')
    chain = prompt | llm | parser


    for file_name in os.listdir(data_dir):
        print(f"Processing File : {file_name}")

        df = pd.read_csv(data_dir / file_name)

        final_df = process_one_df(df)

        final_df.to_csv(save_dir / file_name, index=False)

        print(f"Saved Successfully at {save_dir / file_name}\n")




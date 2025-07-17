import pandas as pd
import numpy as np
from tqdm import tqdm
from llm_api_handler import gemini_generate, gpt_generate, claude_generate, grok_generate
import argparse


def main(llm_to_use):
    # llm_to_use = 'gpt', 'gemini', 'gemini' or 'grok' as needed
    #set random seed for reproducibility
    np.random.seed(42)

    original_df = pd.read_csv('simple_qa_test_set.csv')

    # uniformly sample 200 rows from the original DataFrame
    sampled_df = original_df.sample(n=200, random_state=42)
    sampled_df.head()


    # Instruction template
    instruction_template = """
    You are given a question and its ground-truth answer. Your task is to generate 50 answer sentences that express the same answer using different levels of confidence:

    10 with high confidence
    10 with moderate confidence
    10 with low confidence
    10 with lowest confidence
    10 with complete uncertainty, reject to reply

    The wording should vary across the levels, but all responses should convey the same core answer. Focus on natural and diverse expressions of confidence.

    Question: {}
    Answer: {}
    """


    # llm api keys
    llm_api_keys = {
        'gpt': os.getenv('GPT_API_KEY'),
        'gemini': os.getenv('GEMINI_API_KEY'),
        'claude': os.getenv('CLAUDE_API_KEY'),
        'grok': os.getenv('GROK_API_KEY')
    }
    
    llm_generate = {
        'gpt': gpt_generate, # gpt-4.1
        'gemini': gemini_generate, # gemini-2.5-pro
        'claude': claude_generate, # claude-sonnet-4-20250514
        'grok': grok_generate # grok-3
    }[llm_to_use]

    new_df = pd.DataFrame(columns=['problem', 'answer', 'raw_response', 'metadata'])

    for index, row in tqdm(sampled_df.iterrows()):
        problem = row['problem']
        answer = row['answer']
        metadata = row['metadata']
        instruction = instruction_template.format(problem, answer)
        text = llm_generate(instruction, api_key=llm_api_keys[llm_to_use])
        # add the generated text to the new df
        new_row = {
            'problem': problem,
            'answer': answer,
            'raw_response': text,
            'metadata': metadata
        }
        new_df = new_df._append(new_row, ignore_index=True)
        # save the new df to csv file
        new_df.to_csv(f'generated_text_{llm_to_use}.csv', index=False)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate dataset using LLMs')
    parser.add_argument('--llm', type=str, choices=['gpt', 'gemini', 'claude', 'grok'], required=True, help='The LLM to use for generation')
    args = parser.parse_args()
    
    main(args.llm)
    
    # python build_dataset.py --llm gemini
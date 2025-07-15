# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
from google import genai
from google.genai import types
import pandas as pd


def gemini_generate(instruction):
    client = genai.Client(
        api_key="AIzaSyC3rjmlLuj2ikuBmqxlNJr1NfUywy3jbO0"
    )

    model = "gemini-2.5-pro"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=instruction),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config = types.ThinkingConfig(
            thinking_budget=-1,
        ),
        response_mime_type="text/plain",
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    return response.text

if __name__ == "__main__":
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
                        
    df = pd.read_csv('linguistic uncertainty dataset - SimpleQA.csv')
    # create a new df to store the generated text
    new_df = pd.DataFrame(columns=['topic', 'answer_type', 'question', 'answer', 'response'])
    
    for index, row in df.iterrows():
        question = row['question']
        answer = row['answer']
        instruction = instruction_template.format(question, answer)
        text = gemini_generate(instruction)
        # add the generated text to the new df
        new_row = {
            'topic': row['topic'],
            'answer_type': row['answer_type'],
            'question': question,
            'answer': answer,
            'response': text
        }
        new_df = new_df._append(new_row, ignore_index=True)
        # save the new df to csv file
        new_df.to_csv('generated_text.csv', index=False)
        print(f"Generated {index + 1} rows")
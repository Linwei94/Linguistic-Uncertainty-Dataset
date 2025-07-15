from google import genai
from google.genai import types
from openai import OpenAI
import os
import anthropic

from xai_sdk import Client
from xai_sdk.chat import user, system

# gemini_generate, gpt_generate, claude_generate, grok_generate

# gemini handler
def gemini_generate(instruction, api_key):
    client = genai.Client(
        api_key=api_key,
    )

    model = "gemini-2.5-pro"
    contents = [
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(
                    text="You are a highly intelligent and helpful AI assistant, with expertise in linguistic pragmatics and hedging expressions."
                ),
            ],
        ),
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

# gpt handler
def gpt_generate(instruction, api_key):
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a highly intelligent and helpful AI assistant, with expertise in linguistic pragmatics and hedging expressions."
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": instruction
                }
            ]
        }
    ],
    response_format={
        "type": "text"
    },
    temperature=1,
    max_completion_tokens=2048,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    
    return response.choices[0].message.content


def claude_generate(instruction, api_key):

    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        temperature=1,
        system="You are a highly intelligent and helpful AI assistant, with expertise in linguistic pragmatics and hedging expressions.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": instruction
                    }
                ]
            }
        ]
    )
    return message.content[0].text

def grok_generate(instruction, api_key):
    client = Client(api_key=api_key)

    chat = client.chat.create(model="grok-3")
    chat.append(system("You are a highly intelligent and helpful AI assistant, with expertise in linguistic pragmatics and hedging expressions."))
    chat.append(user(instruction))

    response = chat.sample()
    return response.content

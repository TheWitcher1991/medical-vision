import os

from openai import OpenAI

GROQ_API_KEY = ""

client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

DEFAULT_MODEL = "llama-3.3-70b-versatile"


def chat(messages, model=DEFAULT_MODEL, temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content
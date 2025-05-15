import json

from dotenv import load_dotenv
from openai import OpenAI
from os import environ
from typing import Dict, List

load_dotenv()
CLIENT = OpenAI(api_key=environ.get('OPENAI_API_KEY'))

def call_openai_api(
    messages: List[Dict[str, str]],
    model: str = 'gpt-4o',
    output_as_json: bool = False,
    temperature: float = 0,
):
    # We do not check whether the indicated model support json_object.
    # Refer to https://platform.openai.com/docs/guides/json-mode for more information.
    response_format = {'type': 'json_object'} if output_as_json else {'type': 'text'}

    try:
        chat_completion = CLIENT.chat.completions.create(
            messages=messages,
            model=model,
            n=1,
            response_format=response_format,
            temperature=temperature,
        )
        response = chat_completion.choices[0].message.content
    except Exception as e:
        # Handle any unexpected error
        raise Exception(f"An unexpected error occurred: {str(e)}")

    if output_as_json:
        try:
            json_response = json.loads(response, strict=False)
            return json_response
        except json.JSONDecodeError as error:
            raise ValueError(f"JSON decoding failed: {error}")
    else:
        return response

SYSTEM_MESSAGE = (
    "You are an AI Data Scientist who creates high quality datasets that can be used for fine-tuning of Large Language Models."
    " Follow the user's instruction closely to create a dataset based on the given context."
)

USER_MESSAGE_TEMPLATE = """You are to read the following information and generate a question.
Filename: {filename}
Information: {content}

Now, generate only 1 straightforward, broad, simple and general question.

You will reply in the following JSON format:
{{
    'question': "<Question>"
}}"""

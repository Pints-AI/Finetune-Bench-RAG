SYSTEM_MESSAGE = (
    "You are an AI Data Scientist who creates high quality datasets that can be used for fine-tuning of Large Language Models."
    " Follow the user's instruction closely to create a dataset based on the given context."
)

USER_MESSAGE_TEMPLATE = """You are to read the following information and answer the question.
Filename: {filename}
Information: {content}

Now, answer the question, and you may elaborate as necessary. Do not create information that is not found in the information provided.
Question: "{question}"

You will reply in the following JSON format:
{{
    'answer': "<Answer>"
}}"""

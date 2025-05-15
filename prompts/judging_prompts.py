from typing import TypedDict
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

class OpenAIJudgeResponse(TypedDict):
    accuracy_explanation: str
    accuracy: bool
    helpfulness_explanation: str
    helpfulness: int
    relevance_explanation: str
    relevance: int
    depth_explanation: str
    depth: int

# This prompt has been deprecated. Evaluation of multiple factors with the same call
# can lead to biased results. The result for each criterion might directly 
# influence the ones after, resulting in an inaccurate or less objective assessment. 
judge_system_prompt: ChatCompletionSystemMessageParam = {
    'role': 'system',
    'content': """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below, based solely on a piece of information extracted from a file provided below. Your evaluation should consider factors such as the accuracy, helpfulness, relevance, and depth of the response. 

1. Accuracy - You will check whether the response contains extra details not found in the piece of information provided. If extra details are found, accuracy is false. Otherwise, accuracy is true. Take note that if the response partially addresses the question, but did not provide extra details not found in the piece of information provided, the response will still be considered accurate (hence accuracy = true).
2. Helpfulness - The helpfulness of the AI assistant in answering the question.
3. Relevance - Whether the response fully addresses the question.
4. Depth - The level of detail of the response in answering the question.

Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate each factor on a scale of 1 to 10 (with the exception of accuracy, where it is true or false) by strictly following this JSON format:
{
    "accuracy_explanation": <provide an explanation on accuracy, whether extra details outside the content were found.>,
    "accuracy": <true/false>,
    "helpfulness_explanation": <provide an explanation on helpfulness>,
    "helpfulness": <score>,
    "relevance_explanation": <provide an explanation on relevance>,
    "relevance": <score>,
    "depth_explanation": <provide an explanation on depth>,
    "depth": <score>
}
""",
}

judge_accuracy_system_prompt: ChatCompletionSystemMessageParam = {
    'role': 'system',
    'content': """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below, based solely on a piece of information extracted from a file provided below. Your evaluation should consider the accuracy of the response. 

You will check whether the response contains extra details not found in the piece of information provided. If extra details are found, accuracy is false. Otherwise, accuracy is true. Take note that if the response partially addresses the question, but did not provide extra details not found in the piece of information provided, the response will still be considered accurate (hence accuracy = true).

Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the accuracy with true or false by strictly following this JSON format:
{
    "accuracy_explanation": <provide an explanation on accuracy, whether extra details outside the content were found.>,
    "accuracy": <true/false>
}
""",
}

judge_helpfulness_system_prompt: ChatCompletionSystemMessageParam = {
    'role': 'system',
    'content': """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below, based solely on a piece of information extracted from a file provided below. Your evaluation should consider the helpfulness of the response. 

You will check whether the AI assistant is helpful in answering the question based on the response.

Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the helpfulness on a scale of 1 to 10 by strictly following this JSON format:
{
    "helpfulness_explanation": <provide an explanation on helpfulness>,
    "helpfulness": <score>
}
""",
}

judge_relevance_system_prompt: ChatCompletionSystemMessageParam = {
    'role': 'system',
    'content': """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below, based solely on a piece of information extracted from a file provided below. Your evaluation should consider the relevance of the response. 

You will check the relevance of the response by evaluating whether the response fully addresses the question.

Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the relevance on a scale of 1 to 10 by strictly following this JSON format:
{
    "relevance_explanation": <provide an explanation on relevance>,
    "relevance": <score>
}
""",
}

judge_depth_system_prompt: ChatCompletionSystemMessageParam = {
    'role': 'system',
    'content': """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below, based solely on a piece of information extracted from a file provided below. Your evaluation should consider the depth of the response. 

You will check the depth of the response by evaluating the level of detail of the response in answering the question.

Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the depth on a scale of 1 to 10 by strictly following this JSON format:
{
    "depth_explanation": <provide an explanation on depth>,
    "depth": <score>
}
""",
}


def get_judge_user_prompt(document) -> ChatCompletionUserMessageParam:
    return {
        'role': 'user',
        'content': f"""[The Start of Provided Information Extracted from a File]
Filename: {document['filename']}

Information: {document['content']}
[The End of Provided Information]
            
[Question]
{document['question']}

[The Start of Assistant's Response]
{document['response']}
[The End of Assistant's Response]""",
    }

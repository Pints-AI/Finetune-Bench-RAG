from logging import Logger
from pathlib import Path
from typing import List, Dict
import copy

from prompts.question_generation_prompt import SYSTEM_MESSAGE, USER_MESSAGE_TEMPLATE
from utils.dataset_utils import load_jsonl_file, write_jsonl_file
from utils.logger import setup_logger
from utils.openai import call_openai_api

def start(
    content_path: Path = "dataset/contents.jsonl",
    output_path: Path = "dataset/contents_w_questions.jsonl",
):
    assert (
        content_path.is_file()
        and content_path.suffix == ".jsonl"
    ), "Path to content data is not a jsonl file."

    logger: Logger = setup_logger(Path(__file__).stem + "_" + content_path.name)
    logger.info(f"{content_path} recognized.")

    all_content_data: List[Dict] = load_jsonl_file(content_path)

    # Ensure dataset is ready before generating questions
    for index, content_data in enumerate(all_content_data):
        missing_keys = [key for key in ["content", "filename"] if key not in content_data]
        assert not missing_keys, f"Missing key(s) in line {index + 1} of jsonl file: [{', '.join(missing_keys)}]"

    logger.info(f"Dataset checked and ready for question generation. Generating questions...")
    content_data_with_question: List[Dict] = []
    for index, content_data in enumerate(all_content_data):
        filename: str = content_data["filename"]
        content: str = content_data["content"]
        generated_question: str = generate_question(filename=filename, content=content)
        logger.info(f"Line {index + 1} of jsonl file --> Question generated for content: {content[:60].encode('unicode_escape').decode()}... Question: {generated_question}")

        cloned_content_data: Dict = copy.deepcopy(content_data)
        cloned_content_data["question"] = generated_question
        content_data_with_question.append(cloned_content_data)
    
    write_jsonl_file(content=content_data_with_question, output_path=output_path)
    logger.info(f"Generation of questions completed. Updated jsonl file saved at {output_path}")
    return

def generate_question(
    filename: str,
    content: str,
):
    user_message: str = USER_MESSAGE_TEMPLATE.format(filename=filename, content=content)
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_message}
    ]

    generated_openai_json: Dict = call_openai_api(messages=messages, output_as_json=True)

    assert "question" in generated_openai_json, "OpenAI model did not generate correct json format"
    return generated_openai_json["question"]

if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(start, as_positional=False)


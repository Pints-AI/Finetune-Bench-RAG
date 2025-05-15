from logging import Logger
from pathlib import Path
from typing import List, Dict
import copy

from prompts.answer_generation_prompt import SYSTEM_MESSAGE, USER_MESSAGE_TEMPLATE
from utils.dataset_utils import load_jsonl_file, write_jsonl_file
from utils.logger import setup_logger
from utils.openai import call_openai_api

def start(
    content_path: Path = "dataset/contents_w_questions.jsonl",
    output_path: Path = "dataset/contents_w_qa.jsonl",
):
    assert (
        content_path.is_file()
        and content_path.suffix == ".jsonl"
    ), "Path to content data is not a jsonl file."

    logger: Logger = setup_logger(Path(__file__).stem + "_" + content_path.name)
    logger.info(f"{content_path} recognized.")

    all_content_qn_data: List[Dict] = load_jsonl_file(content_path)

    # Ensure dataset is ready before generating answers
    for index, content_data in enumerate(all_content_qn_data):
        missing_keys = [key for key in ["content", "filename", "question"] if key not in content_data]
        assert not missing_keys, f"Missing key(s) in line {index + 1} of jsonl file: [{', '.join(missing_keys)}]"

    logger.info(f"Dataset checked and ready for answer generation. Generating answers...")
    content_data_with_answer: List[Dict] = []
    for index, content_data in enumerate(all_content_qn_data):
        filename: str = content_data["filename"]
        content: str = content_data["content"]
        question: str = content_data["question"]
        generated_answer: str = generate_answer(filename=filename, content=content, question=question)
        logger.info(f"Line {index + 1} of jsonl file --> Answer generated for question: {question[:60].encode('unicode_escape').decode()}... Answer: {generated_answer[:60]}...")

        cloned_content_data: Dict = copy.deepcopy(content_data)
        cloned_content_data["answer"] = generated_answer
        content_data_with_answer.append(cloned_content_data)
    
    write_jsonl_file(content=content_data_with_answer, output_path=output_path)
    logger.info(f"Generation of answers completed. Updated jsonl file saved at {output_path}")
    return

def generate_answer(
    filename: str,
    content: str,
    question: str,
):
    user_message: str = USER_MESSAGE_TEMPLATE.format(filename=filename, content=content, question=question)
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_message}
    ]

    generated_openai_json: Dict = call_openai_api(messages=messages, output_as_json=True)

    assert "answer" in generated_openai_json, "OpenAI model did not generate correct json format"
    return generated_openai_json["answer"]

if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(start, as_positional=False)
    # python -m prepare_dataset.generate_answer --content_qn_data_path dataset/documents_w_qns/sample.jsonl --output_data_path dataset/documents_w_qa/sample.jsonl

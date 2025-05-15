from logging import Logger
from pathlib import Path
from typing import List, Dict
import copy

from prompts.fictitious_content_generation_prompt import SYSTEM_MESSAGE, USER_MESSAGE_TEMPLATE
from utils.dataset_utils import load_jsonl_file, write_jsonl_file
from utils.logger import setup_logger
from utils.openai import call_openai_api

def start(
    content_path: Path = "dataset/contents_w_qa.jsonl",
    output_path: Path = "dataset/finetunerag_dataset.jsonl",
    num_fictitious_content: int = 2
):
    assert (
        content_path.is_file()
        and content_path.suffix == ".jsonl"
    ), "Path to content data is not a jsonl file."

    assert num_fictitious_content > 0, "Number of fictitious content should be more than 0"
    assert isinstance(num_fictitious_content, int), "Number of fictitious content should be an integer"

    logger: Logger = setup_logger(Path(__file__).stem + "_" + content_path.name)
    logger.info(f"{content_path} recognized.")

    all_content_data: List[Dict] = load_jsonl_file(content_path)

    # Ensure dataset is ready before generating questions
    for index, content_data in enumerate(all_content_data):
        missing_keys = [key for key in ["content", "filename"] if key not in content_data]
        assert not missing_keys, f"Missing key(s) in line {index + 1} of jsonl file: [{', '.join(missing_keys)}]"

    logger.info(f"Dataset checked and ready for fictitious content generation. Generating content...")
    fictitious_content_data: List[Dict] = []
    for index, content_data in enumerate(all_content_data):
        filename: str = content_data["filename"]
        content: str = content_data["content"]

        cloned_content_data: Dict = copy.deepcopy(content_data)
        for fictitious_content_count in range(1, num_fictitious_content + 1):
            fictitious_filename, fictitious_content = generate_fictitious_content(filename=filename, content=content)
            logger.info(f"Line {index + 1} of jsonl file --> Fictitious content {fictitious_content_count} generated. Filename: {fictitious_filename[:60]}..., Content: {fictitious_content[:60]}...")

            cloned_content_data[f"fictitious_filename{fictitious_content_count}"] = fictitious_filename
            cloned_content_data[f"fictitious_content{fictitious_content_count}"] = fictitious_content

        fictitious_content_data.append(cloned_content_data)
    
    write_jsonl_file(content=fictitious_content_data, output_path=output_path)
    logger.info(f"Generation of fictitious contents completed. Updated jsonl file saved at {output_path}")
    return

def generate_fictitious_content(
    filename: str,
    content: str,
):
    user_message: str = USER_MESSAGE_TEMPLATE.format(filename=filename, content=content)
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_message}
    ]

    # Set temperature to 0.5 for more spurious content output generation
    generated_openai_json: Dict = call_openai_api(messages=messages, output_as_json=True, temperature=0.5)
    assert (
        "fictitious_filename" in generated_openai_json
        and "fictitious_content" in generated_openai_json
    ), "OpenAI model did not generate correct JSON format"

    return generated_openai_json["fictitious_filename"], generated_openai_json["fictitious_content"]

if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(start, as_positional=False)

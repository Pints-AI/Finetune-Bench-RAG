from logging import Logger
from pathlib import Path
from typing import List, Literal, Dict

from prepare_dataset.formatting.formatter import formatter
from utils.dataset_utils import get_decider, load_jsonl_file, write_jsonl_file, get_dice
from utils.logger import setup_logger

def start(
    content_path: Path = "dataset/finetunerag_dataset.jsonl",
    output_path: Path = "dataset/adjusted_finetunerag_dataset.jsonl",
    format: Literal["baseline", "xml"] = "baseline",
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
        missing_keys = [key for key in ["content", "filename", "question", "answer", "fictitious_filename1", "fictitious_content1"] if key not in content_data]
        assert not missing_keys, f"Missing key(s) in line {index + 1} of jsonl file: [{', '.join(missing_keys)}]"

    logger.info(f"Dataset checked and ready for formatting. Preparing training data...")

    # It is assumed that every datapoint has the same number of fictitious content. Refering to the first datapoint to retrieve the number of ficitious content
    num_fictitious_content = sum(1 for key in all_content_data[0] if key.startswith("fictitious_content"))
    dice = get_dice(num_choices=num_fictitious_content)

    decider = get_decider()
    all_formatted_data = []
    for index, content_data in enumerate(all_content_data):
        filename: str = content_data["filename"]
        content: str = content_data["content"]
        question: str = content_data["question"]
        answer: str = content_data["answer"]


        selected_content_index = next(dice)
        fictitious_filename = content_data[f"fictitious_filename{selected_content_index}"]
        fictitious_content = content_data[f"fictitious_content{selected_content_index}"]

        formatted_data = formatter(
            filename=filename,
            content=content,
            fictitious_filename=fictitious_filename,
            fictitious_content=fictitious_content,
            question=question,
            answer=answer,
            decider=decider,
            template_type=format,
        )
        all_formatted_data.append(formatted_data)

    write_jsonl_file(content=all_formatted_data, output_path=output_path)
    logger.info(f"Preparation of training data completed. Training data saved at {output_path}")
    return

if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(start, as_positional=False)


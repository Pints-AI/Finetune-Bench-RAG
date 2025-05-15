import json
import numbers
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from utils.dataset_utils import (
    load_jsonl_file,
)
from finetunerag.utils import ArgumentParserPlus
from utils.openai import call_openai_api
from prompts.judging_prompts import (
    OpenAIJudgeResponse,
    get_judge_user_prompt,
    judge_accuracy_system_prompt,
    judge_depth_system_prompt,
    judge_helpfulness_system_prompt,
    judge_relevance_system_prompt,
)
from utils.logger import setup_logger


@dataclass
class InferencingArguments:
    """
    Full arguments class for inferencing from checkpoints.
    """

    answers_directory: Optional[str] = field(
        default=None,
        metadata={
            'help': 'Path to an answers directory that potentially contains multiple set of answers from a model in the form of jsonl files. This cannot be provided along with --answers_file.'
        },
    )
    answers_file: Optional[str] = field(
        default=None,
        metadata={
            'help': 'Path to a specific answers file from a model in the form of jsonl. This cannot be provided along with --answers_directory.'
        },
    )
    openai_evaluator: str = field(
        default='gpt-4o',
        metadata={
            'help': 'The evaluator to use from OpenAI. Refer to their documentation for available models.'
        },
    )
    output_directory: str = field(
        default='ragbench/judged_scores',
        metadata={'help': 'Output directory to save the judging results.'},
    )

    def __post_init__(self):
        if self.answers_directory is None and self.answers_file is None:
            raise ValueError('No answers directory or answers file provided.')

        if self.answers_directory is not None and self.answers_file is not None:
            raise ValueError(
                'Both answers directory and answers file provided. Please only provide either one.'
            )

        self.output_path = Path(self.output_directory)
        self.answers_files_path = (
            Path(self.answers_directory) if self.answers_directory else None
        )
        self.answers_file_path = Path(self.answers_file) if self.answers_file else None


JUDGE_SYSTEM_PROMPTS = [
    judge_accuracy_system_prompt,
    judge_helpfulness_system_prompt,
    judge_relevance_system_prompt,
    judge_depth_system_prompt,
]

# Global logger
logger = setup_logger(Path(__file__).name)


def start(args: InferencingArguments):
    files_to_judge: Path = []
    if args.answers_files_path:
        for file_path in args.answers_files_path.iterdir():
            if file_path.is_file() and file_path.suffix == '.jsonl':
                files_to_judge.append(file_path)
    else:
        files_to_judge.append(args.answers_file_path)

    logger.debug(
        f'These are the files identified: {list(map(lambda file_path: file_path.name, files_to_judge))}'
    )
    for file_to_judge in files_to_judge:
        aggregate_file(file_to_judge, args)


def aggregate_file(answers_file_path: Path, args: InferencingArguments):
    cumulative_stats = {'scores': 0, 'n': 0, 'failed': 0, 'trues': 0, 'falses': 0}

    documents = load_jsonl_file(answers_file_path)

    logger.debug(f'Total of {len(documents)} samples to rate.')

    args.output_path.mkdir(parents=True, exist_ok=True)
    output_file_path = args.output_path / answers_file_path.name

    files_done = set()
    # resume from checkpoint
    if output_file_path.is_file():
        output_list = load_jsonl_file(output_file_path)

        for json_row in output_list:
            files_done.add(json_row['filename'])

            # Update statistics
            if json_row['accuracy']:
                cumulative_stats['trues'] += 1
            else:
                cumulative_stats['falses'] += 1

            cumulative_stats['n'] += 1
            if json_row['average'] is not None:
                cumulative_stats['scores'] += json_row['average']

    logger.info(f"{cumulative_stats['n']} sample(s) already done.")

    # call openAI
    for index, document in enumerate(documents):
        if document['filename'] in files_done:
            continue

        try:
            logger.info(f"Attempting to judge index {index}: {document['filename']}...")
            parsed_response: OpenAIJudgeResponse = judge(
                document=document, evaluator=args.openai_evaluator
            )

        except Exception as error:
            logger.error(f"Filename: [{document['filename']}] errored.")
            logger.error(error)
            cumulative_stats['failed'] += 1
            continue

        # Compute the average score
        scores: List[numbers.Real] = [
            score
            for score in parsed_response.values()
            if isinstance(score, numbers.Real) and not isinstance(score, bool)
        ]
        if len(scores) > 0:
            parsed_response['average'] = sum(scores) / len(scores)
        else:
            parsed_response['average'] = None

        # Add the filename into the dataset
        parsed_response['filename'] = document['filename']

        parsed_response_str = json.dumps(parsed_response)
        with open(output_file_path, 'a', encoding='utf-8') as responses_file:
            responses_file.write(parsed_response_str + '\n')

        # Update statistics
        cumulative_stats['n'] += 1
        logger.info(f"{cumulative_stats['n']}/{len(documents)} inferences done")

        if parsed_response['accuracy']:
            cumulative_stats['trues'] += 1
        else:
            cumulative_stats['falses'] += 1

        if parsed_response['average'] is not None:
            cumulative_stats['scores'] += parsed_response['average']

    logger.info(cumulative_stats)
    logger.info(f"Final average: {cumulative_stats['scores'] / cumulative_stats['n']}")


def judge(
    document,
    evaluator,
) -> OpenAIJudgeResponse:
    parsed_responses = {}
    for judge_system_prompt in JUDGE_SYSTEM_PROMPTS:
        messages: List[ChatCompletionMessageParam] = []

        user_prompt = get_judge_user_prompt(document)

        messages.append(judge_system_prompt)
        messages.append(user_prompt)

        parsed_response = call_openai_api(messages=messages, model=evaluator, output_as_json=True)
        parsed_responses.update(parsed_response)

    return parsed_responses


if __name__ == '__main__':
    parser = ArgumentParserPlus((InferencingArguments))
    args = parser.parse()
    start(args)

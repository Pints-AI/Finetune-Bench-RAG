import json
import torch

from accelerate import Accelerator
from utils.dataset_utils import load_jsonl_file
from dataclasses import dataclass, field
from utils.logger import setup_logger
from prompts.prompt_styles import PromptStyle
from finetunerag.utils import ArgumentParserPlus
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

@dataclass
class InferencingArguments:
    """
    Full arguments class for inferencing from checkpoints.
    """

    checkpoints_directory: Optional[str] = field(
        default=None,
        metadata={
            'help': 'Checkpoints directory containing the checkpoints of a model. This cannot be provided along with --specific_checkpoint_directory.'
        },
    )
    specific_checkpoint_directory: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a specific checkpoint folder. This cannot be provided along with --checkpoints_directory.'},
    )
    data_directory: Optional[str] = field(
        default=None,
        metadata={'help': 'Data directory containing the content used to generate the prompts.'},
    )
    output_directory: str = field(
        default='ragbench/inferences/',
        metadata={'help': 'Output directory to save the generated answers.'},
    )
    tokenizer_directory: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to the tokenizer directory.'},
    )
    custom_chat_template: Optional[str] = field(
        default=None,
        metadata={'help': 'Name of custom chat template if the chat template from tokenizer is not desired.'},
    )

    def __post_init__(self):
        if self.checkpoints_directory == None and self.specific_checkpoint_directory == None:
            raise ValueError("No checkpoints directory or specific checkpoint directory provided.")
        
        if self.checkpoints_directory != None and self.specific_checkpoint_directory != None:
            raise ValueError("Both checkpoints directory and specific checkpoint directory provided. Please only provide either one.")
        
        if self.data_directory == None:
            raise ValueError('No data directory provided. Unable to generate from model.')
        
        if self.tokenizer_directory == None:
            raise ValueError('No tokenizer directory provided.')
        
        self.data_path = Path(self.data_directory)
        self.output_path = Path(self.output_directory)
        self.tokenizer_path = Path(self.tokenizer_directory)
        self.checkpoints_path = Path(self.checkpoints_directory) if self.checkpoints_directory else None
        self.specific_checkpoint_path = Path(self.specific_checkpoint_directory) if self.specific_checkpoint_directory else None

# Global logger
logger = setup_logger(Path(__file__).name)
        
def start(args: InferencingArguments):
    dataset = load_jsonl_file(args.data_path)
    args.output_path.mkdir(parents=True, exist_ok=True)

    if args.checkpoints_path:
        # Retrieve all checkpoints available from the path to the checkpoints
        checkpoint_paths = list(args.checkpoints_path.glob('steps_*'))
        # Sort the checkpoints by their step count
        checkpoint_paths = sorted(checkpoint_paths, key=lambda checkpoint_folder: int(checkpoint_folder.name.rsplit('-', 1)[-1]))
        logger.debug(f'These are the checkpoints identified: {checkpoint_paths}')
    else:
        checkpoint_paths = [Path(args.specific_checkpoint_directory)]

    for checkpoint_path in checkpoint_paths:
        generate_responses(checkpoint_path, args.tokenizer_path, args.output_path, dataset, args.custom_chat_template)
        logger.info(f"Finished processing {checkpoint_path}.")
    
    logger.info(f"Inference for all checkpoints complete!")


def generate_responses(
    checkpoint_path: Path,
    tokenizer_path: Path,
    output_path: Path,
    dataset: list,
    custom_chat_template: str
):
    accelerator = Accelerator()

    prompt_styler = PromptStyle.from_name(custom_chat_template)

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
        device_map={'': accelerator.process_index},
    )
    model = torch.compile(model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    model, tokenizer = accelerator.prepare(model, tokenizer)
    accelerator.wait_for_everyone() # Synchronise all processes to ensure readiness before starting generation

    generated_responses = []

    with accelerator.split_between_processes(dataset) as documents:
        for index, datarow in enumerate(documents):
            prompt_text = prompt_styler.apply(datarow['messages'], append_assistant_header=True)
            inputs = tokenizer(prompt_text, return_tensors="pt").to(accelerator.device)
            generation_output = model.generate(
                inputs.input_ids,
                max_length=6000,
                do_sample=False,
                temperature=None,
                pad_token_id=tokenizer.eos_token_id,
            )

            generated_ids = generation_output[0][inputs.input_ids.shape[1]:-1]
            response = tokenizer.decode(generated_ids)

            generated_responses.append({
                'filename': datarow['filename'],
                'content': datarow['content'],
                'question': datarow['question'],
                'response': response
            })

    gathered_responses = accelerator.gather(generated_responses)

    if accelerator.is_main_process:
        response_file_path = output_path / f'{checkpoint_path.name}.jsonl'
        with open(response_file_path, 'w') as response_file:
            for response_data in gathered_responses:
                response_file.write(json.dumps(response_data) + '\n')

    accelerator.wait_for_everyone()

if __name__ == '__main__':
    parser = ArgumentParserPlus((InferencingArguments))
    args = parser.parse()
    start(args)

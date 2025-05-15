import os
import warnings
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FinetuneArguments:
    """
    Full arguments class for fine-tuning.
    """

    exp_name: str = field(
        default=os.path.basename(__file__)[: -len('.py')],
        metadata={
            'help': (
                "The name of this experiment."
            )
        },
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            'help': (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            'help': 'Pretrained tokenizer name or path if not the same as model_name_or_path'
        },
    )
    tokenizer_revision: Optional[str] = field(
        default='main',
        metadata={
            'help': 'The specific model version to use (can be a branch name, tag name or commit id).'
        },
    )
    prompt_style: Optional[str] = field(
        default='default',
        metadata={
            'help': 'The specific prompt template to use (should be registered under one of the custom prompts).'
        },
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={'help': 'Whether to use flash attention in the model training'},
    )
    model_revision: str = field(
        default='main',
        metadata={
            'help': 'The specific model version to use (can be a branch name, tag name or commit id).'
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            'help': (
                'Whether or not to allow for custom models defined on the Hub in their own modeling files. '
                'This option should only be set to `True` for repositories you trust and in which you '
                'have read the code, as it will execute code present on the Hub on your local machine.'
            )
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            'help': (
                'It is an option to create the model as an empty shell, '
                'then only materialize its parameters when the pretrained weights are loaded. '
                'set True will benefit LLM loading time and RAM consumption.'
            )
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={'help': 'The input training data file (a json/jsonl file).'},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={'help': 'The input validation file (a json/jsonl file).'},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            'help': (
                'For debugging purposes or quicker training, truncate the number of training examples to this '
                'value if set.'
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={'help': 'The number of processes to use for the preprocessing.'},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. '
                'Sequences longer than this will be truncated,'
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={'help': 'Overwrite the cached training and evaluation sets'},
    )
    add_bos: bool = field(
        default=False,
        metadata={
            'help': 'Forcibly add bos token to the beginning of the input sequence.'
            ' Use only when tokenizer does not add bos token by default.'
        },
    )
    clip_grad_norm: float = field(
        default=-1,
        metadata={
            'help': 'Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead).'
        },
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            'help': 'Number of updates steps to accumulate before performing a backward/update pass.'
        },
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={'help': 'The initial learning rate for AdamW optimizer.'},
    )
    beta1: float = field(
        default=0.9,
        metadata={
            'help': 'The coefficient used for computing running averages of gradient and its square within the optimiser.'
        },
    )
    beta2: float = field(
        default=0.95,
        metadata={
            'help': 'The coefficient used for computing running averages of gradient and its square within the optimiser.'
        },
    )
    logging_steps: Optional[int] = field(
        default=None,
        metadata={
            'help': 'Log the training loss and learning rate every logging_steps steps.'
        },
    )
    lr_scheduler_type: str = field(
        default='linear',
        metadata={
            'help': 'The scheduler type to use for learning rate adjustment.',
            'choices': [
                'linear',
                'cosine',
                'cosine_with_restarts',
                'polynomial',
                'constant',
                'constant_with_warmup',
            ],
        },
    )
    num_train_epochs: int = field(
        default=2,
        metadata={'help': 'Total number of training epochs to perform.'},
    )
    output_dir: str = field(
        default='output/',
        metadata={
            'help': 'The output directory where the model predictions and checkpoints will be written.'
        },
    )
    per_device_train_batch_size: int = field(
        default=8,
        metadata={'help': 'Batch size per GPU/TPU core/CPU for training.'},
    )
    use_8bit_optimizer: bool = field(
        default=False,
        metadata={
            'help': 'Use 8bit optimizer from bitsandbytes. Not compatible with deepspeed.'
        },
    )
    warmup_ratio: float = field(
        default=0.03,
        metadata={'help': 'Linear warmup over warmup_ratio fraction of total steps.'},
    )
    weight_decay: float = field(
        default=0.0,
        metadata={'help': 'Weight decay for AdamW if we apply some.'},
    )
    timeout: int = field(
        default=1800,
        metadata={
            'help': 'Timeout for the training process in seconds.'
            'Useful if tokenization process is long. Default is 1800 seconds (30 minutes).'
        },
    )
    reduce_loss: str = field(
        default='mean',
        metadata={
            'help': "How to reduce loss over tokens. Options are 'mean' or 'sum'."
            "Using 'sum' can improve chat model performance."
        },
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={'help': 'If the training should continue from a checkpoint folder.'},
    )
    enable_wandb: bool = field(
        default=False,
        metadata={'help': 'Whether to enable wandb for logging.'},
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={'help': 'Entity to use for logging to wandb.'},
    )
    wandb_project: Optional[str] = field(
        default='test-runs',
        metadata={'help': 'Project name to use when logging to wandb.'},
    )
    wandb_name: Optional[str] = field(
        default='wandb',
        metadata={'help': 'Run name to use when logging to wandb.'},
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            'help': 'Turn on gradient checkpointing. Saves memory but slows training.'
        },
    )
    max_train_steps: Optional[int] = field(
        default=None,
        metadata={
            'help': 'If set, overrides the number of training steps. Otherwise, num_train_epochs is used.'
        },
    )
    seed: int = field(
        default=42,
        metadata={'help': 'Random seed for initialization and dataset shuffling.'},
    )
    checkpointing_steps: Optional[str] = field(
        default=None,
        metadata={
            'help': "Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."  # noqa
        },
    )
    validation_steps: Optional[int] = field(
        default=None,
        metadata={
            'help': 'Compute loss on validation data at the end of every n steps'
        },
    )
    keep_last_n_checkpoints: Optional[int] = field(
        default=None,
        metadata={
            'help': 'How many checkpoints to keep in the output directory. -1 for all.'
        },
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            'help': 'Overwrite the content of the output directory. Means that resumption will always start from scratch.'
        },
    )
    fused_optimizer: bool = field(
        default=True,
        metadata={
            'help': 'Whether to use fused AdamW or not.',
        },
    )

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError('The Hugging Face model name or path is not indicated.')

        if self.tokenizer_name_or_path is None:
            warnings.warn('The tokenizer name or path is not indicated. Defaulting it to model_name_or_path.')
            self.tokenizer_name_or_path = self.model_name_or_path

        if self.reduce_loss not in ['mean', 'sum']:
            raise ValueError("reduce_loss must be either 'mean' or 'sum'")

        if self.train_file is None:
            raise ValueError('Need either a dataset name, dataset mixer, or a training file.')
        else:
            extension = self.train_file.split('.')[-1]
            assert extension in [
                'json',
                'jsonl',
            ], '`train_file` should be a json or a jsonl file.'
        
        if self.validation_steps and not self.validation_file:
            raise ValueError(
                "The number of steps for every validation is indicated. However, the path to the validation dataset is not provided. Please provide it with '--validation_file'."
            )
        
        if self.validation_file and not self.validation_steps:
            warnings.warn(
                'The path to the validation dataset is provided. However, the number of steps for every validation is not indicated. Defaulted to 1.'
            )
            self.validation_steps = 1
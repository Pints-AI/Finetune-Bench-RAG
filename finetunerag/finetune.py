# Finetuning code referenced from https://github.com/allenai/open-instruct

import logging
import math
import os
import random
from datetime import timedelta
from functools import partial

import datasets
import deepspeed
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs, set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_scheduler,
)
print("Current Working Directory:", os.getcwd())

from prompts.prompt_styles import PromptStyle
from guardrag.arguments import FinetuneArguments
from guardrag.model_utils import save_with_accelerate
from guardrag.utils import (
    ArgumentParserPlus,
    clean_last_n_checkpoints,
    get_last_checkpoint_path,
    get_wandb_tags,
)

logger = get_logger(__name__)

def main(args: FinetuneArguments):

    ##########################
    # Initialise Accelerator #
    ##########################
    accelerator_log_kwargs = {}

    if args.enable_wandb:
        accelerator_log_kwargs['log_with'] = 'wandb'
        accelerator_log_kwargs['project_dir'] = args.output_dir

    # If you get timeouts (e.g. due to long tokenization) increase this.
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=args.timeout))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_seedable_sampler=True,
        **accelerator_log_kwargs,
        kwargs_handlers=[timeout_kwargs],
    )

    if args.seed is not None:
        set_seed(args.seed)
    
    if args.enable_wandb:
        experiment_config = vars(args)

        accelerator.init_trackers(
            args.wandb_project,
            experiment_config,
            init_kwargs={
                'wandb': {
                    'entity': args.wandb_entity,
                    'name': args.wandb_name,
                    'tags': [args.exp_name] + get_wandb_tags(),
                }
            },
        )

    #####################
    # Configure Logging #
    #####################
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    accelerator.wait_for_everyone()

    ##########################################
    # Load Pretrained HF Model and Tokenizer #
    ##########################################
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        revision=args.model_revision,
        token=os.getenv('HF_TOKEN', None),
    )

    tokenizer_revision = (
        args.model_revision
        if args.tokenizer_revision is None
        else args.tokenizer_revision
    )

    if tokenizer_revision != args.model_revision:
        # Warn user if tokenizer and model use different revisions; this is an unusual use case.
        warning = f"""Requested tokenizer revision `{tokenizer_revision}` is different
                   from the model revision `{args.model_revision}`."""
        logger.warn(warning)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        trust_remote_code=args.trust_remote_code,
        revision=tokenizer_revision,
        token=os.getenv('HF_TOKEN', None),
    )

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool('.ckpt' in args.model_name_or_path),
            config=config,
            trust_remote_code=args.trust_remote_code,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2'
            if args.use_flash_attn
            else 'eager',
            revision=args.model_revision,
            token=os.getenv('HF_TOKEN', None),
        )
    else:
        logger.info('Training new model from scratch')
        model = AutoModelForCausalLM.from_config(config)


    ######################
    # Embedding Resizing #
    ######################

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # gather deepspeed to get 'real' embedding size
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]
    # resize does its own gather
    if len(tokenizer) > embedding_size:
        # pad to multiple for tensor cores.
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    # update embedding size after resizing for sum loss
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]


    #######################################
    # Prepare Dataset & Set up Dataloader #
    #######################################
    data_files = {}
    dataset_args = {}

    if args.train_file:
        data_files['train'] = args.train_file
    if args.validation_file:
        data_files['validation'] = args.validation_file

    raw_datasets = load_dataset(
        'json',
        data_files=data_files,
        **dataset_args,
    )

    train_dataset = raw_datasets['train']
    validation_dataset = raw_datasets.get('validation', None)

    if 'messages' not in train_dataset.column_names:
        raise ValueError("You need to have 'messages' in your training data.")
    if validation_dataset and 'messages' not in validation_dataset.column_names:
        raise ValueError("You need to have 'messages' in your validation data.")

    # Limit training samples. Used for debugging.
    if args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), args.max_train_samples)
        logger.info(f'Limiting training samples to {max_train_samples} from {len(train_dataset)}.')
        train_dataset = train_dataset.select(range(max_train_samples))

    encode_function = partial(
        encode_messages,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        prompt_style=args.prompt_style,
        add_bos=args.add_bos,
    )

    with accelerator.main_process_first():
        train_dataset = train_dataset.map(
            encode_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            remove_columns=[
                name
                for name in train_dataset.column_names
                if name not in ['input_ids', 'labels', 'attention_mask']
            ],
            desc='Tokenizing and reformatting instruction data',
        )

        train_dataset.set_format(type='pt')
        train_dataset = train_dataset.filter(
            lambda example: (example['labels'] != -100).any()
        )

        # Log a few random samples from the training set.
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f'Sample {index} of the training set: {train_dataset[index]}.')

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=DataCollatorForSeq2Seq(
                tokenizer=tokenizer, model=model, padding='longest'
            ),
            batch_size=args.per_device_train_batch_size,
        )

        if validation_dataset:
            validation_dataset = validation_dataset.map(
                encode_function,
                batched=False,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                remove_columns=[
                    name
                    for name in validation_dataset.column_names
                    if name not in ['input_ids', 'labels', 'attention_mask']
                ],
                desc='Tokenizing and reformatting validation data',
            )
            validation_dataset.set_format(type='pt')
            validation_dataset = validation_dataset.filter(lambda example: (example['labels'] != -100).any())

            validation_dataloader = DataLoader(
                validation_dataset,
                shuffle=True,
                collate_fn=DataCollatorForSeq2Seq(
                    tokenizer=tokenizer, model=model, padding='longest'
                ),
                batch_size=args.per_device_train_batch_size,
            )


    ##############################
    # Optimizer and LR Scheduler #
    ##############################

    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ['bias', 'layer_norm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': args.weight_decay,
        },
        {
            'params': [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        fused=args.fused_optimizer,
        betas=(args.beta1, args.beta2),
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler
    # for the `num_processes` times. This is because they assume
    # the user initialize the scheduler with the entire training set.
    # In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set.
    # So each time the process needs to update the lr multiple times so that the total
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the
    # entire training set (when epochs is specified) or we need to multiply the
    # num_training_steps by num_processes so that the total number of
    # updates matches the num_training_steps.
    num_training_steps_for_scheduler = (
        args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )

    #################
    # Miscellaneous #
    #################

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    validation_steps = args.validation_steps
    if validation_steps is not None:
        validation_steps = int(validation_steps)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and str(checkpointing_steps).lower() != 'epoch':
        checkpointing_steps = int(checkpointing_steps)

    
    ##############
    # Finetuning #
    ##############

    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info('***** Running training *****')
    logger.info(f'  Num examples = {len(train_dataset)}')
    logger.info(f'  Num Epochs = {args.num_train_epochs}')
    logger.info(f'  Instantaneous batch size per device = {args.per_device_train_batch_size}')
    logger.info(f'  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}')
    logger.info(f'  Gradient Accumulation steps = {args.gradient_accumulation_steps}')
    logger.info(f'  Total optimization steps = {args.max_train_steps}')
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    last_checkpoint_path = get_last_checkpoint_path(args)
    if last_checkpoint_path:
        accelerator.print(f'Resumed from checkpoint: {last_checkpoint_path}')
        accelerator.load_state(last_checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        last_checkpoint_path = os.path.basename(last_checkpoint_path)
        training_difference = os.path.splitext(last_checkpoint_path)[0]

        if 'epoch' in training_difference:
            starting_epoch = int(training_difference.replace('epoch_', '')) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (int(training_difference.replace('step_', '')) * args.gradient_accumulation_steps)
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    print(f'Starting from epoch {starting_epoch} and step {completed_steps}.')
    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        train_dataloader.set_epoch(epoch)
        total_loss = 0
        if last_checkpoint_path and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for batch in active_dataloader:
            with accelerator.accumulate(model):
                loss = calculate_loss_and_backpropagate(
                    model,
                    batch,
                    accelerator,
                    optimizer,
                    lr_scheduler,
                    embedding_size,
                    args
                )
                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                # Perform Validation (only done by main)
                avg_val_loss = None
                if (
                    accelerator.is_local_main_process
                    and validation_steps
                    and completed_steps % validation_steps == 0
                ):
                    if completed_steps % validation_steps == 0:
                        model.eval()
                        full_val_loss = 0
                        num_val_batches = 0
                        with torch.no_grad():
                            for val_batch in validation_dataloader:
                                val_batch = {
                                    key: value.to(accelerator.device)
                                    for key, value in val_batch.items()
                                }
                                val_loss = calculate_loss(model, val_batch, embedding_size, args)

                                full_val_loss += val_loss.detach().float()
                                num_val_batches += 1

                        avg_val_loss = full_val_loss / num_val_batches
                        model.train()

                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = (
                        accelerator.gather(total_loss).mean().item()
                        / args.gradient_accumulation_steps
                        / args.logging_steps
                    )

                    val_loss_log = f', Val Loss: {avg_val_loss}' if avg_val_loss else ''
                    logger.info(f'  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Train Loss: {avg_loss}{val_loss_log}')

                    if args.enable_wandb:
                        log_data = {
                            'learning_rate': lr_scheduler.get_last_lr()[0],
                            'train_loss': avg_loss,
                        }
                        if validation_steps:
                            log_data['validation_loss'] = avg_val_loss
                        accelerator.log(log_data, step=completed_steps)

                    total_loss = 0

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f'step_{completed_steps}'
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)
                        # use this to mark the checkpoint as completely saved, to avoid restoring from garbled checkpoints
                        with open(os.path.join(get_last_checkpoint_path(args, incomplete=True), 'COMPLETED'), 'w') as f:
                            f.write('COMPLETED')

                        if (accelerator.is_local_main_process and args.keep_last_n_checkpoints):
                            clean_last_n_checkpoints(args.output_dir, args.keep_last_n_checkpoints)
                        accelerator.wait_for_everyone()

                if completed_steps >= args.max_train_steps:
                    break

        if checkpointing_steps == 'epoch':
            output_dir = f'epoch_{epoch}'
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            # use this to mark the checkpoint as completely saved, to avoid restoring from garbled checkpoints
            with open(os.path.join(get_last_checkpoint_path(args, incomplete=True), 'COMPLETED'), 'w') as f:
                f.write('COMPLETED')

            if accelerator.is_local_main_process and args.keep_last_n_checkpoints:
                clean_last_n_checkpoints(args.output_dir, args.keep_last_n_checkpoints)
            accelerator.wait_for_everyone()

    if args.output_dir is not None:
        save_with_accelerate(
            accelerator,
            model,
            tokenizer,
            args.output_dir,
        )

    accelerator.wait_for_everyone()
    if args.enable_wandb:
        accelerator.end_training()


def encode_messages(
    example,
    tokenizer,
    max_seq_length,
    prompt_style,
    add_bos=False
):
    """
    Here we assume each example has a 'messages' field.
    'messages' is a list of messages.
    Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    """
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')

    style = PromptStyle.from_name(prompt_style)

    # To support multi-turn, we need to mask all non-assistant messages.
    # To do this, we compute the length of each tokenized non-assistant messages then mask it
    segmented_prompts = []
    segmented_prompts_and_responses = []
    start_idx, end_idx = 0, 0
    while end_idx < len(messages):
        while end_idx < len(messages) and messages[end_idx]['role'] != 'assistant':
            end_idx += 1
        if start_idx <= end_idx:
            if start_idx == 0:
                # expect system prompt
                segmented_prompts.append(
                    style.apply(
                        messages[start_idx:end_idx], append_assistant_header=True
                    ) if style else tokenizer.apply_chat_template(messages[start_idx:end_idx], tokenize=False, add_generation_prompt=True)
                )
                segmented_prompts_and_responses.append(
                    style.apply(
                        messages[start_idx : end_idx + 1], append_assistant_header=False
                    ) if style else tokenizer.apply_chat_template(messages[start_idx:end_idx + 1], tokenize=False, add_generation_prompt=False)
                )
            else:
                # should not have system prompt for subsequent turns
                segmented_prompts.append(
                    style.apply(
                        messages[start_idx:end_idx],
                        no_system=True,
                        append_assistant_header=True,
                    ) if style else tokenizer.apply_chat_template(messages[start_idx:end_idx], tokenize=False, add_generation_prompt=True)
                )
                segmented_prompts_and_responses.append(
                    style.apply(
                        messages[start_idx : end_idx + 1],
                        no_system=True,
                        append_assistant_header=False,
                    ) if style else tokenizer.apply_chat_template(messages[start_idx:end_idx + 1], tokenize=False, add_generation_prompt=False)
                )
        start_idx = end_idx + 1
        end_idx += 1  # should be same as start_idx

    if add_bos:
        # add bos token to the first prompt
        segmented_prompts[0] = tokenizer.bos_token + segmented_prompts[0]
        segmented_prompts_and_responses[0] = (
            tokenizer.bos_token + segmented_prompts_and_responses[0]
        )
    encoded_segmented_prompts = list(
        map(
            lambda prompt: tokenizer(
                prompt, return_tensors='pt', max_length=max_seq_length, truncation=True
            ).input_ids.flatten(),
            segmented_prompts,
        )
    )
    encoded_segmented_prompts_and_responses = list(
        map(
            lambda prompt_and_response: tokenizer(
                prompt_and_response,
                return_tensors='pt',
                max_length=max_seq_length,
                truncation=True,
            ).input_ids.flatten(),
            segmented_prompts_and_responses,
        )
    )

    # Achieve the same effect as 'masking' by simply using ignore_index
    masked_labels = []
    num_split = len(encoded_segmented_prompts)
    for i in range(num_split):
        encoded_prompt = encoded_segmented_prompts[i]
        encoded_prompt_and_response = encoded_segmented_prompts_and_responses[i]
        label = encoded_prompt_and_response.clone()
        label[: len(encoded_prompt)] = 0
        masked_labels.append(label)

    # concatenate the segments
    encoded_prompts_and_responses = torch.cat(encoded_segmented_prompts_and_responses)
    labels = torch.cat(masked_labels)
    attention_mask = torch.ones_like(encoded_prompts_and_responses)

    return {
        'input_ids': encoded_prompts_and_responses,
        'labels': labels,
        'attention_mask': attention_mask,
    }

def calculate_loss(model, batch, embedding_size, args):
    outputs = model(**batch, use_cache=False)
    if args.reduce_loss == 'mean':
        loss = outputs.loss
    else:
        # reduce loss is sum
        # this ensures that we weight all tokens in the dataset equally,
        # rather than weighting each overall example equally when
        # using high amounts of gradient accumulation.
        # this can result in > 5 point improvements in AlpacaEval
        # see https://github.com/huggingface/transformers/issues/24725 for
        # more discussion and details.
        logits = outputs.logits
        labels = batch['labels']
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
        shift_logits = shift_logits.view(-1, embedding_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    return loss

def calculate_loss_and_backpropagate(
    model,
    batch,
    accelerator,
    optimizer,
    lr_scheduler,
    embedding_size,
    args
):
    loss = calculate_loss(model, batch, embedding_size, args)
    accelerator.backward(loss)
    # clip gradient norm. don't do this with deepspeed
    if accelerator.sync_gradients and args.clip_grad_norm > 0:
        accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
    optimizer.step()
    optimizer.zero_grad()
    lr_scheduler.step()

    return loss

if __name__ == '__main__':
    parser = ArgumentParserPlus(FinetuneArguments)
    args = parser.parse()
    main(args)

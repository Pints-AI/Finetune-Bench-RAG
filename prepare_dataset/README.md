# Dataset

The motivation here is to improve RAG (after the retrieval phase) by finetuning a model to filter out and select the intended (correct) information between intended (correct) and false positive(s). As such, the dataset is generated with the intended (correct) information and some fictitious data (simulating retrieval of incorrect data).

## Uploaded Dataset

We have uploaded the dataset we used for training to HuggingFace [here](https://huggingface.co/datasets/pints-ai/Finetune-RAG). We have curated a total of `1653` documents that is ready for train-validation-test split before fine-tuning.

## Fictitious Data Generation [OPTIONAL]

If you have your own set of document chunks and would like to curate questions, answers, and fictitious data from it for finetuning, you may do so by preparing a jsonl file that contains your document chunks per line. The structure of each line should be as follows:

```json
{
    "content": "<DOCUMENT CHUNK>",
    "filename": "<DOCUMENT FILENAME>",
}
```

> [!WARNING]  
> Via our method, your documents will be passed into GPT-4o for dataset generation. If it is according to plan, remember to include the `.env` file containing your OpenAI key at the root level. Example is given in `.env.sample`.

### Usage

Suppose your custom jsonl file is at `dataset/custom_chunks.jsonl`, run as a module at the root level:
```bash
python -m prepare_dataset.content_generation.generate_question --content_path dataset/custom_chunks.jsonl && \
python -m prepare_dataset.content_generation.generate_answer && \
python -m prepare_dataset.content_generation.generate_fictitious_content
```

## Prepare Dataset For Training

`prepare_dataset/formatting/generate_training_data.py` is the script to process the data generated for training. Essentially, it allows you to specify 2 different dialogue formats that is used to tune the model.

Using our prepared dataset from Hugging Face, or your own generated dataset, run through this preparation before fine-tuning.

### Baseline Format

```
Filename: {filename1}
Information:
{content1}

Filename: {filename2}
Information:
{content2}

Question: {question}
```

### XML Format

```
<Results>
    <Result>
        <Filename>{filename1}</Filename>
        <Information>{content1}</Information>
    </Result>
    <Result>
        <Filename>{filename2}</Filename>
        <Information>{content2}</Information>
    </Result>
</Results>

Question: {question}
```

### Usage

```bash
python -m prepare_dataset.formatting.generate_training_data --content_path dataset/finetunerag_dataset.jsonl --format baseline
```

## Split the dataset

Simply run the `prepare_dataset/formatting/split_data.py` script to get your train-validation-test splits after preparing your dataset via the instructions above.

```bash
python -m prepare_dataset.formatting.split_data --dataset_path dataset/adjusted_finetunerag_dataset.jsonl
```

import json
import random

from pathlib import Path

def start(
    dataset_path: Path = "dataset/adjusted_finetunerag_dataset.jsonl",
    output_folder_path: Path = "dataset/splits",
    seed: int = 888,
):
    with open(dataset_path, "r") as f:
        lines = f.readlines()

    random.seed(seed)
    random.shuffle(lines)

    total = len(lines)
    test_size = int(0.1 * total)
    train_size = total - (test_size * 2)

    train_data = lines[:train_size]
    val_data = lines[train_size:train_size + test_size]
    test_data = lines[train_size + test_size:]

    trimmed_test_data = []
    for line in test_data:
        obj = json.loads(line)
        if isinstance(obj.get("messages"), list) and obj["messages"]:
            obj["messages"] = obj["messages"][:-1]
            trimmed_test_data.append(json.dumps(obj) + "\n")

    output_folder_path.mkdir(parents=True, exist_ok=True)

    (output_folder_path / "train.jsonl").write_text("".join(train_data))
    (output_folder_path / "validation.jsonl").write_text("".join(val_data))
    (output_folder_path / "test.jsonl").write_text("".join(trimmed_test_data))

    print(f"Done! Files created: {(output_folder_path / 'train.jsonl')}, {(output_folder_path / 'validation.jsonl')}, {(output_folder_path / 'test.jsonl')}")

if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(start, as_positional=False)

import json
import random

from pathlib import Path
from typing import List, Dict, Generator

def load_jsonl_file(file_path: Path) -> List[Dict]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            dict_obj = json.loads(line)
            data.append(dict_obj)

    assert len(data) > 0, f'{file_path.name} is empty!'
    return data

def write_jsonl_file(content: List[dict], output_path: Path):
    # create all ancestors directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in content:
            f.write(json.dumps(entry) + '\n')

def get_dice(num_choices: int, seed: int=69) -> Generator[int, None, None]:
    if seed:
        random.seed(seed)
    
    choices = list(range(1, num_choices + 1))
    while True:
        yield random.choice(choices)

def get_decider(seed: int=69) -> Generator[int, None, None]:
    if seed:
        random.seed(seed)

    choices = [True, False]
    while True:
        yield random.choice(choices)

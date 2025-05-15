import json
import os

from pathlib import Path

def read_jsonl(file_path):
    """Reads a JSONL file and returns the data as a list of dictionaries."""
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


def compute_summary_metrics(data):
    """Computes summary metrics from a list of data dictionaries."""
    metrics = {
        'total_records': 0,
        'accuracy_true_count': 0,
        'accuracy_false_count': 0,
        'total_helpfulness': 0,
        'total_relevance': 0,
        'total_depth': 0,
        'total_average': 0.0,
    }

    # Iterate over each record to update the metrics
    for record in data:
        metrics['total_records'] += 1
        if record['accuracy']:
            metrics['accuracy_true_count'] += 1
        else:
            metrics['accuracy_false_count'] += 1
        metrics['total_helpfulness'] += record['helpfulness']
        metrics['total_relevance'] += record['relevance']
        metrics['total_depth'] += record['depth']
        metrics['total_average'] += record['average']

    # Calculate averages if there are any records
    if metrics['total_records'] > 0:
        metrics['average_helpfulness'] = (
            metrics['total_helpfulness'] / metrics['total_records']
        )
        metrics['average_relevance'] = (
            metrics['total_relevance'] / metrics['total_records']
        )
        metrics['average_depth'] = metrics['total_depth'] / metrics['total_records']
        metrics['average_average'] = metrics['total_average'] / metrics['total_records']
    else:
        metrics['average_helpfulness'] = 0
        metrics['average_relevance'] = 0
        metrics['average_depth'] = 0
        metrics['average_average'] = 0

    return metrics


def process_directory(directory_path: Path):
    """Processes all JSONL files in the specified directory."""
    filenames = sorted(os.listdir(directory_path))
    for filename in filenames:
        if filename.endswith('.jsonl'):
            file_path = os.path.join(directory_path, filename)
            data = read_jsonl(file_path)
            metrics = compute_summary_metrics(data)
            print(f'Summary Metrics for {filename}:')
            print(f"Total Records: {metrics['total_records']}")
            accuracy = metrics['accuracy_true_count'] / metrics['total_records'] * 100
            print(f'Accuracy: {accuracy:.2f}%')
            print(f"Average helpfulness: {metrics['average_helpfulness']:.2f}")
            print(f"Average Relevance: {metrics['average_relevance']:.2f}")
            print(f"Average Depth: {metrics['average_depth']:.2f}")
            print(f"Average of Averages: {metrics['average_average']:.2f}")
            print('')

if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(process_directory, as_positional=False)

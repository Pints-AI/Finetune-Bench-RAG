"""
Sample usage: python -m ragbench.plot_results --result_directory ragbench/results/Llama-3.1-8B-Instruct-Baseline ragbench/results/Llama-3.1-8B-Instruct-XML ragbench/results/Llama-3.1-8B-Instruct-Enhanced --output_directory ragbench/judge_results/
"""

import json
import re
from argparse import ArgumentParser
from pathlib import Path

from matplotlib import pyplot


def start(result_paths: list[Path], output_directory: Path):
    aggregated_results = []
    for result_path in result_paths:
        aggregated_results.extend(aggregate_scores(result_path))

    plot_scores_by_metric(aggregated_results, output_directory)


def aggregate_scores(result_path: Path):
    jsonl_file_paths = list(result_path.glob('*.jsonl'))

    results = []
    for jsonl_file_path in jsonl_file_paths:
        total_accuracy = 0
        total_helpfulness = 0
        total_relevance = 0
        total_depth = 0
        count = 0

        with open(jsonl_file_path, 'r') as jsonl_file:
            for jsonl_line in jsonl_file:
                jsonl_data = json.loads(jsonl_line.strip())

                total_accuracy += jsonl_data.get('accuracy', 0)
                total_helpfulness += jsonl_data.get('helpfulness', 0)
                total_relevance += jsonl_data.get('relevance', 0)
                total_depth += jsonl_data.get('depth', 0)
                count += 1

        avg_accuracy = total_accuracy / count
        avg_helpfulness = total_helpfulness / count
        avg_relevance = total_relevance / count
        avg_depth = total_depth / count

        results.append(
            {
                'template_type': result_path.name.rsplit('-', 1)[-1],
                'file': jsonl_file_path.name,
                'avg_accuracy': avg_accuracy,
                'avg_helpfulness': avg_helpfulness,
                'avg_relevance': avg_relevance,
                'avg_depth': avg_depth,
            }
        )

    return results


def extract_number(filename):
    match = re.search(r'steps_(\d+)', filename)
    return int(match.group(1)) if match else float('inf')


def plot_scores_by_metric(aggregated_results: list[dict], output_directory: Path):
    template_types = list(set(result['template_type'] for result in aggregated_results))

    output_directory.mkdir(parents=True, exist_ok=True)

    metrics = ['accuracy', 'helpfulness', 'relevance', 'depth']
    metric_labels = ['Accuracy', 'Helpfulness', 'Relevance', 'Depth']

    for i, metric in enumerate(metrics):
        fig, ax = pyplot.subplots(figsize=(12, 8))

        for template_type in template_types:
            files = [
                result['file']
                for result in aggregated_results
                if result['template_type'] == template_type
            ]
            values = [
                result[f'avg_{metric}']
                for result in aggregated_results
                if result['template_type'] == template_type
            ]

            sorted_pairs = sorted(
                zip(files, values), key=lambda x: extract_number(x[0])
            )
            sorted_files, sorted_values = zip(*sorted_pairs)
            sorted_files = list(map(lambda name: name[:-6], sorted_files))
            ax.plot(sorted_files, sorted_values, marker='o', label=f'{template_type}')

        ax.set_title(f'Average {metric_labels[i]} of Finetuned Llama-3.1-8B-Instruct')
        ax.set_xlabel('Checkpoint')
        ax.set_ylabel(f'Average {metric_labels[i]}')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)

        # Save the PNG file
        pyplot.tight_layout()
        pyplot.savefig(output_directory / f'{metric}_by_steps.png')
        pyplot.close()


if __name__ == '__main__':
    parser = ArgumentParser(description='Plot line graphs of ragbench results.')
    parser.add_argument(
        '--result_directory',
        type=str,
        nargs='+',
        required=True,
        help='Directories containing ragbench results in JSONL files. Multiple directories can be provided.',
    )
    parser.add_argument(
        '--output_directory',
        type=str,
        required=True,
        help='Output directory to save the plots.',
    )

    arguments = parser.parse_args()

    result_paths = [
        Path(result_directory) for result_directory in arguments.result_directory
    ]
    output_path = Path(arguments.output_directory)
    start(result_paths, output_path)

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from edu_evaluator import (
    Title,
    edu_pred_output_to_rank_tree,
    get_sample,
    markdown_text_to_rank_tree,
)
from edu_evaluator.metrics.ted import TEDMetric
from zss import Node


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch run EDU title evaluation")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="deeplang-ai/StructBench",
        help="Directory containing edu_pred_input",
    )
    parser.add_argument(
        "--inference_dir",
        type=str,
        default="edu_output",
        help="Directory to write output JSON files",
    )
    return parser.parse_args()


def print_tree(tree: Node, sentences: list[str], level: int = 1):
    if level > 1:
        print("#" * level, tree.label, sentences[tree.label])
    for node in tree.children:
        print_tree(node, sentences, level + 1)


if __name__ == "__main__":
    args = parse_args()

    dataset = load_dataset(args.data_dir, split="test")

    teds = []
    for data in dataset:
        edu_input = json.loads(data["edu_pred_input"])
        sample = get_sample(edu_input)
        true_tree = markdown_text_to_rank_tree(
            data["ground_truth"], sample, levenshtein_threshold=0.6
        )
        true_tree.label = -1

        edu_output_path = Path(args.inference_dir).joinpath(
            f"{data['instance_id']}.edu_pred_output.json"
        )
        pred_tree = edu_pred_output_to_rank_tree(
            titles=[
                Title(level + 1, sentence_index)
                for level, sentence_index, _ in json.loads(edu_output_path.read_text())
            ],
            sample=sample,
        )

        ted = TEDMetric.evaluate(pred_tree, true_tree)
        teds.append(ted)

    print("TED:", sum(teds) / len(teds))
    print("DLA(%):", sum(map(lambda ted: ted == 0, teds)) / len(teds) * 100)

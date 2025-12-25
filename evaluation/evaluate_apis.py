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
        default="/mnt/public/share/users/zhouyiqing-share/StructBench",
        help="Directory containing edu_pred_input",
    )
    parser.add_argument(
        "--inference_dir",
        type=str,
        default="outputs/gpt_4_1",
        help="Directory to write output JSON files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1",
        help="Directory containing model",
    )
    return parser.parse_args()


def rank_tree_to_markdown_text(tree: Node, sentences: list[str], level: int = 1):
    lines = []
    if level > 1:
        lines.append("#" * level + f" 句{tree.label} " + sentences[tree.label])
    for node in tree.children:
        lines.append(rank_tree_to_markdown_text(node, sentences, level+1))
    return "\n".join(lines)


def print_tree(tree: Node, sentences: list[str], level: int = 1):
    if level > 1:
        print("#" * level, tree.label, sentences[tree.label])
    for node in tree.children:
        print_tree(node, sentences, level + 1)


def get_api_output_file(model, id) -> tuple[str, str, str]:
    if model == "gpt-4.1":
        return (
            f"/mnt/public/share/users/zhangyuanwang-share/Edu-Eval_results/api_llm_results/1751350554/model_outputs/{id}/{id}.gpt41_output.md",
            f"/mnt/public/share/users/zhangyuanwang-share/Edu-Eval_results/api_llm_results/1751357623/model_outputs/{id}/{id}.gpt41_output.md",
            f"/mnt/public/share/users/zhangyuanwang-share/Edu-Eval_results/api_llm_results/1751363883/model_outputs/{id}/{id}.gpt41_output.md",
        )
    elif model == "gpt-5":
        return (
            f"/mnt/public/share/users/zhangyuanwang-share/Edu-Eval_results/gpt5_results/1754968159/model_outputs/{id}/{id}.gpt5_output.md",
            f"/mnt/public/share/users/zhangyuanwang-share/Edu-Eval_results/gpt5_results/1754968159/model_outputs/{id}/{id}.gpt5_output.md",
            f"/mnt/public/share/users/zhangyuanwang-share/Edu-Eval_results/gpt5_results/1754968159/model_outputs/{id}/{id}.gpt5_output.md",
        )
    elif model == "deepseek-r1":
        return (
            f"/mnt/public/share/users/zhangyuanwang-share/Edu-Eval_results/api_llm_results_1/model_outputs/{id}/{id}.deepseekr1_output.md",
            f"/mnt/public/share/users/zhangyuanwang-share/Edu-Eval_results/api_llm_results_2/model_outputs/{id}/{id}.deepseekr1_output.md",
            f"/mnt/public/share/users/zhangyuanwang-share/Edu-Eval_results/api_llm_results_3/model_outputs/{id}/{id}.deepseekr1_output.md",
        )
    elif model == "claude-4":
        return (
            f"/mnt/public/share/users/zhangyuanwang-share/Edu-Eval_results/api_llm_results_1/model_outputs/{id}/{id}.claude_output.md",
            f"/mnt/public/share/users/zhangyuanwang-share/Edu-Eval_results/api_llm_results_2/model_outputs/{id}/{id}.claude_output.md",
            f"/mnt/public/share/users/zhangyuanwang-share/Edu-Eval_results/api_llm_results_3/model_outputs/{id}/{id}.claude_output.md",
        )


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm

    df = pd.read_csv("token_num_statistics_models.csv")
    df = df.set_index("id")

    args = parse_args()

    dataset = load_dataset(args.data_dir, split="test")

    web_teds = []
    pdf_teds = []
    teds = []
    for data in tqdm(dataset):
        edu_input = json.loads(data["edu_pred_input"])
        sample = get_sample(edu_input)
        sample = get_sample(edu_input)
        if False:# data['doc_type'] == 'pdf':
            true_tree = markdown_text_to_rank_tree(
                Path(f"/workspace/temp/{data['instance_id']}.gt_new.md").read_text(), sample, levenshtein_threshold=0.6
            )
        else:
            true_tree = markdown_text_to_rank_tree(
                data["ground_truth"], sample, levenshtein_threshold=0.6
            )


        sub_teds = []
        for api_output_file in get_api_output_file(args.model, data['instance_id']):
            api_output = Path(api_output_file).read_text()
            pred_tree = markdown_text_to_rank_tree(
                api_output, sample, levenshtein_threshold=0.6
            )
            sub_ted = TEDMetric.evaluate(pred_tree, true_tree)
            sub_teds.append(sub_ted)

        ted = sum(sub_teds) / 3
        if data['doc_type'] == 'web':
            web_teds.append(ted)
        else:
            pdf_teds.append(ted)

        df.at[data['instance_id'], args.model] = ted
        pred_markdown = rank_tree_to_markdown_text(pred_tree, sample.sentences)
        Path(args.inference_dir).mkdir(exist_ok=True)
        Path(args.inference_dir).joinpath(
            f"{data['instance_id']}.api_output.md"
        ).write_text(f"ted={sub_ted}\n" + pred_markdown)

    teds = web_teds + pdf_teds
    print("TED:", sum(teds) / len(teds))
    print("WEB TED:", sum(web_teds) / len(web_teds))
    print("PDF TED:", sum(pdf_teds) / len(pdf_teds))
    print("DLA(%):", sum(map(lambda ted: ted == 0, teds)) / len(teds) * 100)
    df.to_csv("token_num_statistics_models.csv", index=True)

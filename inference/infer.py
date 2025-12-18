import argparse
import asyncio
import json
import os

from datasets import load_dataset
from loguru import logger

from edu_core.edu_func import TitleEduFunction
from edu_core.edu_utils import preprocess_article

semaphore = asyncio.Semaphore(8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch run EDU title parsing")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="deeplang-ai/StructBench",
        help="Directory containing input .edu_pred_input.json files",
    )
    parser.add_argument(
        "--inference_dir",
        type=str,
        default="edu_output",
        help="Directory to write output JSON files",
    )
    return parser.parse_args()


async def perform_edu_parse_local(title_func, edu_pred_input):
    sentences = preprocess_article(edu_pred_input["infos"])
    async with semaphore:
        res = await title_func.do_batch_evaluate(sentences)
    for level, sentence_index, _ in res:
        print(
            "#" * (level + 1) + f" [{sentence_index}] " + sentences[sentence_index].text
        )
    return res


async def main(dataset):
    title_func = TitleEduFunction(model="deeplang-ai/LingoEDU-4B")
    results = await asyncio.gather(
        *[
            perform_edu_parse_local(title_func, json.loads(data["edu_pred_input"]))
            for data in dataset
        ]
    )
    return results


if __name__ == "__main__":
    logger.info("start testing edu title function...")

    args = parse_args()
    dataset = load_dataset(args.data_dir, split="test")

    results = asyncio.run(main([dataset[0]]))

    os.makedirs(args.inference_dir, exist_ok=True)
    for data, res in zip(dataset, results):
        out_path = os.path.join(
            args.inference_dir,
            f"{data['instance_id']}.edu_pred_output.json",
        )
        with open(out_path, "w") as fout:
            fout.write(json.dumps(res))
    logger.info("finished testing edu title function.")

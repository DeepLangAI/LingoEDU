from __future__ import annotations

import re
import uuid

import numpy as np
from vllm.config import StructuredOutputsConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams, StructuredOutputsParams

from edu_core.edu_type import (
    Sentence,
)
from edu_core.edu_utils import (
    create_html_from_texts,
)

from loguru import logger


class TitleEduFunction:
    def __init__(self, model: str) -> None:
        super().__init__()
        self.vllm_llm_engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                model=model,
                structured_outputs_config=StructuredOutputsConfig(backend="guidance"),
            )
        )
        self.prompt = "你是一名文档内容格式化专家，请阅读文章中每一个标有序号的句子，结合文章的html标签，使用markdown形式输出正文部分的文章内容多级划分位置及其句子编号，正文部分不包括代码、表格、目录大纲、reference、致谢等。输出的每行格式应为 ##<sep_l>句xx<sep_r>句子内容-<sep_l>句xx<sep_r>句子内容 或者 ###<sep_l>句xx<sep_r>句子内容，二级标题输出切分起始和结束句，其他只输出起始句，切分时请优先按照文章标题结构进行切分，标题结构下段落较长则进行内容分割。文章如下："  # noqa: E501

    def do_begin_request(self) -> bool:
        return True

    async def do_batch_evaluate(
        self,
        sentences: list[Sentence],
    ) -> list[tuple[int, int, int | None]]:
        max_input_len = 32768
        for _ in range(10):
            try:
                origin_rsp_result = await self.get_segment_with_sequence_length_limit(
                    sentences,
                    max_input_len=max_input_len,
                )
            except ValueError:
                logger.warning(
                    f"exceed max seq len with"  # noqa: G004
                    f" max_input_len:{max_input_len}"
                )
                max_input_len -= 2048
            else:
                break

        origin_rsp_result = list(filter(lambda x: x[0] > 0, origin_rsp_result))
        last_level, last_sentence_number = 0, -1
        fixed_rsp_result = []
        for i, (level, sentence_number, ending_sentence_number) in enumerate(
            origin_rsp_result
        ):
            if level == last_level and sentence_number == last_sentence_number:
                continue
            if sentence_number < last_sentence_number:
                continue
            if level > last_level + 1:
                continue
            last_level, last_sentence_number = level, sentence_number
            fixed_rsp_result.append((level, sentence_number, ending_sentence_number))

        return fixed_rsp_result

    async def get_segment_with_sequence_length_limit(
        self,
        sentences: list[Sentence],
        max_input_len: int,
    ) -> list[tuple[int, int, int | None]]:
        """Get segment with sequence length limit."""
        tokenizer = await self.vllm_llm_engine.get_tokenizer()
        for i, sentence in enumerate(sentences):
            first = True
            atom_texts = []
            for atom in sentence.atoms:
                text = atom["txt"].replace("\n", "")
                if first:
                    text = text.lstrip()
                if not text:
                    continue
                if first:
                    text = f"<sep_l>句{i}<sep_r>" + text
                prefix, suffix = "", ""
                for tag in atom["x"].split("/"):
                    prefix += f"<{tag}>"
                    suffix += f"</{tag}>"
                atom_texts.append(prefix + text + suffix)
                first = False
            sentence.successor_seq_len = len(tokenizer.encode("".join(atom_texts)))

        sentence_seq_len_cumsum = np.cumsum(
            [sentence.successor_seq_len for sentence in sentences]
        )
        now_req_index = 0
        rsp_result = []
        while now_req_index < len(sentences):
            already_req_seq_len = sentence_seq_len_cumsum[now_req_index]
            last_req_index = now_req_index
            now_req_index = np.searchsorted(
                sentence_seq_len_cumsum,
                max_input_len + already_req_seq_len,
                side="left",
            )
            article_html_str = self.build_web_html(
                sentences[last_req_index:now_req_index]
            )
            origin_rsp_result = await self.request_model(
                article_html_str,
                sentences[last_req_index:now_req_index],
            )
            origin_rsp_result = [
                (
                    level,
                    sentence_number + last_req_index,
                    (
                        ending_sentence_number + last_req_index
                        if ending_sentence_number is not None
                        else None
                    ),
                )
                for level, sentence_number, ending_sentence_number in origin_rsp_result
            ]
            if now_req_index >= len(sentences):
                rsp_result.extend(origin_rsp_result)
                break
            level_1_rsps = list(filter(lambda x: x[0] == 1, origin_rsp_result))
            last_level1_rsp = level_1_rsps[-1] if level_1_rsps else (1, now_req_index)
            if last_level1_rsp[1] <= last_req_index:
                rsp_result.extend(origin_rsp_result)
                break
            non_last_level1_rsp_result = []
            for level, sentence_number, ending_sentence_number in origin_rsp_result:
                if (
                    level == last_level1_rsp[0]
                    and sentence_number == last_level1_rsp[1]
                ):
                    break
                non_last_level1_rsp_result.append(
                    (level, sentence_number, ending_sentence_number)
                )
            rsp_result.extend(non_last_level1_rsp_result)
            if last_req_index < last_level1_rsp[1]:
                now_req_index = last_level1_rsp[1]
        return rsp_result

    def generate_grammar(self, sentences: list[str]) -> str:
        if not sentences:
            return "start: ''"
        num_sentences = len(sentences)
        sentence_rules = []
        for i in range(num_sentences):
            escaped_sent = sentences[i].replace("\\", "\\\\").replace('"', '\\"')
            sentence_rules.append(f'sentence_{i}: "句{i}" <sep_r> "{escaped_sent}"')
        sentence_rules_str = "\n".join(sentence_rules)
        sentence_choices = " | ".join([f"sentence_{i}" for i in range(num_sentences)])
        grammar = f"""start: document
document: line+
line: heading "\\n"
heading: level_2_heading | other_level_heading
level_2_heading: "##" <sep_l> sentence_ref "-" <sep_l> sentence_ref
other_level_heading: heading_mark <sep_l> sentence_ref
heading_mark: "#" | "###" | "####" | "#####" | "######" | "#######"
sentence_ref: {sentence_choices}
{sentence_rules_str}
"""
        return grammar

    async def request_model(
        self,
        article_html_str: str,
        sentences: list[Sentence],
    ) -> list[tuple[int, int, int | None]]:
        sentence_texts = [
            sentence.text.replace("\n", "").replace("\t", " ").strip()
            for sentence in sentences
        ]
        tokenizer = await self.vllm_llm_engine.get_tokenizer()
        prompt_str = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": self.prompt,
                },
                {
                    "role": "user",
                    "content": article_html_str,
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        rsp_result: list[tuple[int, int, int | None]] = []
        grammar = self.generate_grammar(sentence_texts)

        rsp_text = ""
        async for output in self.vllm_llm_engine.generate(
            prompt=prompt_str,
            sampling_params=SamplingParams(
                temperature=0.0,
                top_k=-1,
                top_p=1.0,
                max_tokens=8 * 1024,
                skip_special_tokens=False,
                n=1,
                structured_outputs=StructuredOutputsParams(grammar=grammar),
            ),
            request_id=str(uuid.uuid4()),
        ):
            rsp_text = output.outputs[0].text

        for line in rsp_text.split("\n"):
            level = (
                len(re.match(r"^#+", line).group(0)) if re.match(r"^#+", line) else 1
            )
            match = re.findall(r"<sep_l>句(\d+)<sep_r>", line)
            if not match:
                continue
            sentence_number = int(match[0])
            ending_sentence_number = None
            if level == 2 and len(match) == 2:
                ending_sentence_number = int(match[1])
            rsp_result.append((level - 1, sentence_number, ending_sentence_number))
        logger.info(f"debug_md_str: {rsp_text}")
        return rsp_result

    def build_web_html(self, sentences: list[Sentence]) -> str:
        texts, xpaths = [], []
        for i, sentence in enumerate(sentences):
            first = True
            atom_texts = []
            for atom in sentence.atoms:
                text = atom["txt"].replace("\n", "")
                if first:
                    text = text.lstrip()
                if not text:
                    continue
                if first:
                    texts.append(f"<sep_l>句{i}<sep_r>" + text)
                else:
                    texts.append(text)
                xpaths.append(atom["x"])
                atom_texts.append(text)
                first = False
        return create_html_from_texts(texts, xpaths)

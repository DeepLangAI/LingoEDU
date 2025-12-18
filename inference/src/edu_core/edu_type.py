from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Sentence:
    """Sentence."""

    text: str
    index: int
    atoms: list[dict] = field(default_factory=list)
    successor_seq_len: int | None = None
    web_segment_id: int | None = None

    @property
    def text_len(self) -> int:
        """Text length."""
        return len(self.text)


@dataclass
class SuccessorLLMRequest:
    """LLMRequest."""

    xpath_article: str
    html_article: str
    once_answer: str
    pre_answer: str
    grammar: str
    label_id_to_sentence_index: dict[int, int]
    segment_id: int
    url: str
    guided_options_request: dict | None = None
    prompt_allowed_token_ids: list[int] | None = None
    prompt_logprobs: list | dict | None = None
    is_stage2_request: bool = False

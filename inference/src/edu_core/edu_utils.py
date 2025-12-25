from __future__ import annotations

import html
import re
from typing import Literal

from lxml import etree

from edu_core.edu_type import Sentence


def preprocess_article(
    infos: list[dict],
    doc_type: Literal["web", "pdf"],
) -> list[Sentence]:
    """Article process."""
    if infos is None:
        return None
    filtered_infos = list(
        filter(
            lambda x: x["label"] not in {"O", "figure", "table", "code", "reference"},
            infos,
        )
    )
    sentences: list[Sentence] = []
    if doc_type == "web":
        for i, info in enumerate(filtered_infos):
            txt: str = info["txt"].replace("\n", "").strip()
            sentence_number = len(sentences)
            sentences.append(
                Sentence(
                    text=txt,
                    index=sentence_number,
                    atoms=info["position"].get("atoms", [{"txt": txt, "x": ""}]),
                    web_segment_id=info["web_segment_id"],
                ),
            )
    else:
        position_ids = []
        paragraph_id = 0
        for i, info in enumerate(filtered_infos):
            pdf_positions = info["position"].get("pdf_position")
            if pdf_positions:
                position_id = pdf_positions[0].get("position_id", -1)
            else:
                position_id = -1
            if len(position_ids) == 0 or position_id != position_ids[-1]:
                paragraph_id += 1
            position_ids.append(position_id)

            txt: str = info["txt"].replace("\n", "").strip()
            sentence_number = len(sentences)
            sentences.append(
                Sentence(
                    text=txt,
                    index=sentence_number,
                    atoms=[{"txt": txt, "x": f"/html/body/p[{paragraph_id}]"}],
                    web_segment_id=info["web_segment_id"],
                ),
            )

    return sentences


def remove_duplicate_xpath_tags(
    xpath_lists: list[list[str | None]], level: int, segment_l: int, segment_r: int
) -> None:
    last_tag, last_l = None, None
    child_spans: list[tuple[int, int, str]] = []
    new_xpath_lists: list[list[str | None]] = []
    for i in range(segment_l, segment_r):
        new_xpath_lists.append([])
        xpath_list = xpath_lists[i]
        if len(xpath_list) > level:
            xpath_tag = xpath_list[level]
            if xpath_tag != last_tag:
                if last_l is not None:
                    assert last_tag is not None
                    child_spans.append((last_l, i, last_tag))
                last_tag, last_l = xpath_tag, i
        else:
            if last_l is not None and last_tag is not None:
                child_spans.append((last_l, i, last_tag))
            last_tag, last_l = None, None
    if last_l is not None and last_tag is not None:
        child_spans.append((last_l, i + 1, last_tag))
    if len(child_spans) <= 1:
        for left, right, tag in child_spans:
            remove_duplicate_xpath_tags(xpath_lists, level + 1, left, right)
            if tag.startswith(("p[", "div[", "span[")):
                for i in range(left, right):
                    xpath_lists[i][level] = None
    else:
        for left, right, _ in child_spans:
            remove_duplicate_xpath_tags(xpath_lists, level + 1, left, right)


def create_html_from_texts(
    texts: list[str],
    xpaths: list[str],
) -> str:
    xpath_lists: list[list[str | None]] = [
        list(xpath.lstrip("/").split("/")) for xpath in xpaths
    ]
    remove_duplicate_xpath_tags(xpath_lists, 0, 0, len(xpath_lists))
    xpath_lists = [
        [tag for tag in xpath_list if tag is not None] for xpath_list in xpath_lists
    ]
    root = etree.Element("html")
    body = etree.SubElement(root, "body")
    for text, parts in zip(texts, xpath_lists):
        current_element = body
        for part in parts:
            if not part:
                continue
            if "[" in part:
                tag_name = part.split("[")[0]
                index_str = part.split("[")[1].split("]")[0]
                index = int(index_str) if index_str.isdigit() else 1
            else:
                tag_name = part
                index = 1
            if not bool(re.match(r"^[a-zA-Z_][a-zA-Z0-9_\-]*$", tag_name)):
                continue
            existing_elements = current_element.findall(tag_name)
            if len(existing_elements) >= index:
                current_element = existing_elements[index - 1]
            else:
                while len(existing_elements) < index:
                    new_element = etree.SubElement(current_element, tag_name)
                    existing_elements.append(new_element)
                current_element = existing_elements[-1]
        if not current_element.text:
            current_element.text = text
        else:
            current_element.text += text
    html_str = etree.tostring(
        root, encoding="unicode", pretty_print=False, method="xml", with_tail=False
    )
    decoded_html_str = html.unescape(html_str)
    while decoded_html_str.startswith("<html><body>") and decoded_html_str.endswith(
        "</body></html>"
    ):
        decoded_html_str = decoded_html_str[
            len("<html><body>") : -len("</body></html>")
        ]
    return decoded_html_str

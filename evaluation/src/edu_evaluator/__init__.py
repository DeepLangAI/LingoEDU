__version__ = "0.3.8"

from collections import namedtuple

from edu_evaluator.definitions import DocType, NoNumMdNode, Sample, TreeNode
from edu_evaluator.utils.md_line_utils import (
    clean_md_lines_nonum,
    nonum_md_lines_to_tree,
)
from edu_evaluator.utils.md_tree_utils import (
    md_tree_to_rank_zss_tree,
    nonum_md_tree_add_num,
)
from zss import Node

Title = namedtuple(
    "Title",
    [
        "level",
        "sentence_index",
    ],
)


def get_sample(
    edu_input: dict,
) -> Sample:
    sample = Sample(
        doc_type=DocType(edu_input.get("type", "WEB").lower()),
        sentences=[],
        sentence_numbers=[],
        labels=[],
        parsing_labels=[],
    )
    filtered_infos = list(
        filter(
            lambda x: "img" not in x["tags"] and x["label"] != "O", edu_input["infos"]
        )
    )
    for i, info in enumerate(filtered_infos):
        sample.sentences.append(info["txt"])
        sample.sentence_numbers.append(i)
        sample.parsing_labels.append(info["label"])
        sample.labels.append(info["edu_l1_label"])

    return sample


def keep_titles_only(node: Node, labels: list[str]):
    if not node.children:
        return
    parsing_labels = []
    for child in node.children:
        keep_titles_only(child, labels)
        parsing_labels.append(labels[child.label])
    if not any(map(lambda pl: "title" in pl, parsing_labels)):
        node.children = []


def edu_pred_output_to_rank_tree(titles: list[Title], sample: Sample) -> Node:
    if len(titles) > 0 and titles[0].level == 1:
        root = Node(label=titles[0].sentence_index, children=[])
        titles = titles[1:]
    else:
        root = Node(label=-1, children=[])

    stack = [root]
    for title in titles:
        if title.level <= 1:
            continue
        if title.level > len(stack) + 1:
            continue
        while title.level <= len(stack):
            stack.pop()
        if title.sentence_index != stack[-1].label:
            stack[-1].children.append(Node(label=title.sentence_index, children=[]))
            stack.append(stack[-1].children[-1])
    keep_titles_only(root, sample.parsing_labels)
    return root


def markdown_text_to_rank_tree(
    markdown_text: str, sample: Sample, levenshtein_threshold: float = 0.0
) -> TreeNode:
    md_lines = markdown_text.split("</think>")[-1].splitlines()
    no_num_md_tree = nonum_md_lines_to_tree(clean_md_lines_nonum(md_lines))
    if no_num_md_tree is None:
        return TreeNode.empty_tree()
    assert isinstance(no_num_md_tree, NoNumMdNode)
    md_tree = nonum_md_tree_add_num(no_num_md_tree, sample, levenshtein_threshold)
    return md_tree_to_rank_zss_tree(md_tree)

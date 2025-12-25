import ast

from edu_evaluator.definitions import TreeNode
from zss import simple_distance


def convert_to_single_side(rank_tree: TreeNode) -> TreeNode:
    if rank_tree.label == "":
        rank_tree.label = None
    label = rank_tree.label
    if label is not None:
        if isinstance(label, str):
            label = ast.literal_eval(label)[0]
        if isinstance(label, list):
            label = label[0]
        label = str(int(label))
    return TreeNode(
        label=label,
        children=[convert_to_single_side(child) for child in rank_tree.children],
    )


def process_single_children(children: list[TreeNode]) -> list[TreeNode]:
    for child in children:
        child.children = process_single_children(child.children)
    if len(children) == 1:
        return children[0].children
    else:
        return children


class TEDMetric:
    @classmethod
    def evaluate(
        cls,
        pred_tree: TreeNode,
        true_tree: TreeNode,
        do_convert_to_single_side: bool = True,
        do_process_single_children: bool = True,
        do_ignore_article_title: bool = True,
    ) -> float:
        """
        计算TED
        """
        if do_convert_to_single_side:
            pred_tree = convert_to_single_side(pred_tree)
            true_tree = convert_to_single_side(true_tree)
        if do_process_single_children:
            process_single_children(pred_tree.children)
            process_single_children(true_tree.children)
        if do_ignore_article_title:
            true_tree.label = None
            pred_tree.label = None
        ted = simple_distance(true_tree, pred_tree)
        return ted

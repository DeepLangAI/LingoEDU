import re

import Levenshtein
import numpy as np
from edu_evaluator.definitions import MdNode, NoNumMdNode, Sample, TreeNode
from edu_evaluator.utils.md_line_utils import MD_LINE_SUFFIX_PATTERN


def calculate_match_ratio(ground_truth: str, candidate: str, match_func=0) -> bool:
    """match_func为 0 代表最长公共子串；为 1是编辑距离"""
    ground_truth_len = len(ground_truth)
    candidate_len = len(candidate)

    # 输入校验
    if ground_truth_len == 0:
        return True
    if candidate_len == 0:
        return False

    if match_func == 0:
        max_continuous_match = 0

        dp = [[0] * (candidate_len + 1) for _ in range(ground_truth_len + 1)]

        for i in range(1, ground_truth_len + 1):
            for j in range(1, candidate_len + 1):
                if ground_truth[i - 1] == candidate[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    max_continuous_match = max(max_continuous_match, dp[i][j])

        match_ratio = max_continuous_match / ground_truth_len
    elif match_func == 1:
        match_ratio = Levenshtein.ratio(ground_truth, candidate)
    return match_ratio > 0.8


def compare_trees(md_node: MdNode, gt_node: MdNode):
    if md_node.span[0] != gt_node.span[0]:
        return False, f"Span mismatch: {md_node.span} vs {gt_node.span}"
    if len(gt_node.children) != 0 and set(md_node.children.keys()) != set(
        gt_node.children.keys()
    ):
        return (
            False,
            f"Children mismatch: {md_node.span} -> {tuple(md_node.children.keys())} vs {tuple(gt_node.children.keys())}",
        )
    for key in gt_node.children:
        result, message = compare_trees(md_node.children[key], gt_node.children[key])
        if not result:
            return False, message
    return True, ""


def compare_trees_nonum(md_node: NoNumMdNode, gt_node: NoNumMdNode):
    if md_node.title != gt_node.title:
        match_ratio = calculate_match_ratio(gt_node.title, md_node.title, match_func=1)
        if not match_ratio:
            return False, f"Title mismatch: {md_node.title} vs {gt_node.title}"
    if len(gt_node.children) != 0 and set(md_node.children.keys()) != set(
        gt_node.children.keys()
    ):
        return (
            False,
            f"Children mismatch: {md_node.title} -> {tuple(md_node.children)} vs {tuple(gt_node.children)}",
        )
    for key in gt_node.children:
        result, message = compare_trees_nonum(
            md_node.children[key], gt_node.children[key]
        )
        if not result:
            return False, message
    return True, ""


def cross_validate(md_node: MdNode, *other_md_nodes: MdNode):
    for node in other_md_nodes:
        if md_node.span[0] != node.span[0]:
            return None

    for node in other_md_nodes:
        if set(md_node.children.keys()) != set(node.children.keys()):
            md_node.children.clear()
            return md_node

    for key in md_node.children:
        cross_validate(
            md_node.children[key], *[node.children[key] for node in other_md_nodes]
        )
    return md_node


def cross_validate_nonum(md_node: NoNumMdNode, *other_md_nodes: NoNumMdNode):
    for node in other_md_nodes:
        if not calculate_match_ratio(md_node.title, node.title, match_func=1):
            return None

    for node in other_md_nodes:
        if set(md_node.children.keys()) != set(node.children.keys()):
            md_node.children.clear()
            return md_node

    for key in md_node.children:
        if (
            cross_validate_nonum(
                md_node.children[key], *[node.children[key] for node in other_md_nodes]
            )
            is None
        ):
            md_node.children.clear()
            return md_node

    return md_node


def get_tree_depth(md_node: MdNode | NoNumMdNode | None) -> int:
    if md_node is None:
        return 0
    return md_node.get_tree_depth()


def build_score_matrix(
    sentences: list[str],
    titles: list[str | None],
    ignore_whitespace: bool = True,
) -> np.ndarray:
    """
    构建句子到标题的得分矩阵，并且允许标题映射到连续多个句子。
    score_matrix[i][j] = (score, k)表示：
        对于标题i从句子j开始连续匹配k句能得到得分score。
    """
    if ignore_whitespace:
        sentences = [re.sub(r"\s+", "", s) for s in sentences]
        titles = [re.sub(r"\s+", "", t) if t is not None else None for t in titles]

    num_sentences = len(sentences)
    num_titles = len(titles)
    score_matrix = np.zeros((num_sentences, num_titles, 2))
    for i in range(num_sentences):
        for j in range(num_titles):
            title = titles[j]
            if title is None:
                continue
            for k in range(i, len(sentences)):
                similarity_score = Levenshtein.ratio(
                    "".join([s for s in sentences[i : k + 1]]), title
                )
                # 为了使k循环的时间复杂度可控，如果得分单调递增则继续往更多句子匹配，否则中断。
                if similarity_score > score_matrix[i][j][0]:
                    score_matrix[i][j][0] = similarity_score
                    score_matrix[i][j][1] = k - i + 1
                else:
                    break
    return score_matrix


def find_best_mapping_with_score_matrix(
    score_matrix: np.ndarray,
    levenshtein_threshold: float = 0.0,
) -> dict[int, tuple[int, int]]:
    """
    使用动态规划找到标题到句子的最佳映射路径。
    输入score_matrix: np.ndarray, shape = (num_sentences, num_titles, 2)
    输出mapping: dict[int, tuple[int, int]]
    说明：到达(i, j)时，下一步路径为(i+k, j+1)或(i+1, j)，k为score_matrix[i][j][1]
    """
    num_sentences = score_matrix.shape[0]
    num_titles = score_matrix.shape[1]
    dp = np.zeros((num_sentences + 1, num_titles + 1), dtype=np.float32)
    backtrack: list[list[tuple[int, int] | None]] = [
        [None for _ in range(num_titles + 1)] for _ in range(num_sentences + 1)
    ]

    score_matrix[score_matrix[:, :, 0] < levenshtein_threshold] = (0, 0)

    for i in range(num_sentences - 1, -1, -1):
        for j in range(num_titles - 1, -1, -1):
            score = score_matrix[i][j][0]
            k = int(score_matrix[i][j][1])
            jump_score = dp[i + k][j + 1] + score
            move_score = dp[i + 1][j]
            if jump_score > move_score:
                dp[i][j] = jump_score
                backtrack[i][j] = (i + k, j + 1)
            else:
                dp[i][j] = move_score
                backtrack[i][j] = (i + 1, j)

    mapping: dict[int, tuple[int, int]] = {}
    # 二维矩阵dp中最大值坐标
    i, j = np.array(np.unravel_index(np.argmax(dp), dp.shape)).tolist()
    while i < num_sentences and j < num_titles:
        pos = backtrack[i][j]
        assert pos is not None
        k = int(score_matrix[i][j][1])
        if k > 0:
            mapping[j] = (i, i + k - 1)
        i, j = pos
    return mapping


def get_title_sentences(
    titles: list[str], sentences: list[str], levenshtein_threshold: float = 0.0
) -> dict[int, tuple[int, int]]:
    """
    获取标题到句子的最佳映射路径。
    """
    score_matrix = build_score_matrix(sentences, titles)
    index_to_index_span_mapping = find_best_mapping_with_score_matrix(
        score_matrix, levenshtein_threshold=levenshtein_threshold
    )
    return index_to_index_span_mapping


def nonum_gather_preorder_nodes(nonum_md_node: NoNumMdNode) -> list[NoNumMdNode]:
    nodes = [nonum_md_node]
    for key in sorted(nonum_md_node.children.keys(), key=int):
        nodes.extend(nonum_gather_preorder_nodes(nonum_md_node.children[key]))
    return nodes


def nonum_md_tree_add_num(
    nonum_md_tree: NoNumMdNode, sample: Sample, levenshtein_threshold: float = 0.0
) -> MdNode | None:
    sample = sample.filter_sentences(lambda s, n, l, pl: l != "EDU_O")
    nonum_md_nodes = nonum_gather_preorder_nodes(nonum_md_tree)
    if nonum_md_tree.line is None:
        nonum_md_nodes = nonum_md_nodes[1:]
        md_nodes: list[MdNode | None] = [
            MdNode(
                level=nonum_md_tree.level,
                span=(-1, -1),
                line=None,
                children={},
            )
        ]
    else:
        md_nodes = []
    index_to_index_span_mapping = get_title_sentences(
        titles=[n.title for n in nonum_md_nodes],
        sentences=[NoNumMdNode.process_text(s) for s in sample.sentences],
        levenshtein_threshold=levenshtein_threshold,
    )

    for i, nonum_md_node in enumerate(nonum_md_nodes):
        if i not in index_to_index_span_mapping:
            md_nodes.append(None)
            continue
        span = (
            sample.sentence_numbers[index_to_index_span_mapping[i][0]],
            sample.sentence_numbers[index_to_index_span_mapping[i][1]],
        )
        if span[0] == span[1]:
            line = f"{nonum_md_node.line}[句{span[0]}]"
        else:
            line = f"{nonum_md_node.line}[句{span[0]}-句{span[1]}]"
        md_nodes.append(
            MdNode(
                level=nonum_md_node.level,
                span=span,
                line=line,
                children={},
            )
        )

    for nonum_md_node, md_node in zip(nonum_md_nodes, md_nodes):
        if md_node is None:
            continue
        for nonum_child in nonum_md_node.children.values():
            child_index = nonum_md_nodes.index(nonum_child)
            child = md_nodes[child_index]
            if child is not None:
                md_node.children[str(child.span[0])] = child
            else:
                md_node.children.clear()
                break
    return md_nodes[0]


def md_node_del_num(md_node: MdNode, order: int = 0) -> NoNumMdNode:
    children = {}
    for i, key in enumerate(sorted(md_node.children.keys(), key=int)):
        child = md_node_del_num(md_node.children[key], i)
        children[str(child.order)] = child
    line = re.sub(MD_LINE_SUFFIX_PATTERN, "", md_node.line)
    return NoNumMdNode(
        level=md_node.level,
        order=order,
        line=line,
        children=children,
    )


def md_tree_to_zss_tree(md_node: MdNode | None) -> TreeNode | None:
    """
    将 MdNode 树结构转换为 TreeNode 树结构

    Args:
        md_node: MdNode 树结构

    Returns:
        TreeNode: 转换后的 TreeNode 树结构
    """
    if md_node is None:
        return None

    # 从 line 中提取标题文本（去除 # 号和句号标记）
    title = md_node.line.split("[")[0].strip("# ") if md_node.line else None

    # 递归转换子节点
    children = []
    for key in sorted(md_node.children.keys(), key=int):
        child_node = md_tree_to_zss_tree(md_node.children[key])
        if child_node is not None:
            children.append(child_node)

    return TreeNode(label=title, children=children)


def md_tree_to_rank_zss_tree(md_node: MdNode | None) -> TreeNode | None:
    """
    将 MdNode 树结构转换为编号树结构

    Args:
        md_node: MdNode 树结构，其中 md_node.span 包含 [起始编号, 终止编号]

    Returns:
        TreeNode: 转换后的编号树结构，节点标签格式为 (起始编号,终止编号)
    """
    if md_node is None:
        return None

    # 从 span 中获取起始和终止编号
    start_rank, end_rank = md_node.span
    label = start_rank

    # 递归转换子节点
    children = []
    for key in sorted(md_node.children.keys(), key=int):
        child_node = md_tree_to_rank_zss_tree(md_node.children[key])
        if child_node is not None:
            children.append(child_node)

    return TreeNode(label=label, children=children)

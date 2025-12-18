import re
from dataclasses import asdict, dataclass
from enum import Enum
from functools import cached_property
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger
from zss import Node


class DocType(str, Enum):
    """
    文档类型
    """

    WEB = "web"
    PDF = "pdf"


@dataclass(frozen=True)
class Sample:
    doc_type: DocType  # 文档类型
    sentences: list[str]
    sentence_numbers: list[int]
    labels: list[str]
    parsing_labels: list[str]
    level: int | None = None

    def to_dict(self):
        return asdict(self)

    def filter_sentences(
        self,
        filter_func: Callable[[str, int, str, str, bool], bool],
        level: int | None = None,
    ) -> "Sample":
        if level is None:
            level = self.level
        sentences = []
        sentence_numbers = []
        labels = []
        parsing_labels = []
        for s, n, l, pl in zip(
            self.sentences,
            self.sentence_numbers,
            self.labels,
            self.parsing_labels,
        ):
            if filter_func(s, n, l, pl):
                sentences.append(s)
                sentence_numbers.append(n)
                labels.append(l)
                parsing_labels.append(pl)
        return Sample(
            doc_type=self.doc_type,
            sentences=sentences,
            sentence_numbers=sentence_numbers,
            labels=labels,
            parsing_labels=parsing_labels,
            level=level,
        )


@dataclass(frozen=True)
class Sentence:
    text: str
    number: int


@dataclass(frozen=True)
class Title:
    level: int
    sentences: list[Sentence]

    def to_md_line(self):
        if len(self.sentences) == 1:
            prefix = "#" * self.level
            text = self.sentences[0].text
            suffix = f"[句{self.sentences[0].number}]"
        else:
            prefix = "#" * self.level
            text = "".join(s.text for s in self.sentences)
            suffix = f"[句{self.sentences[0].number}-句{self.sentences[-1].number}]"
        return f"{prefix} {text}{suffix}\n"

    def nonum_to_md_line(self):
        if len(self.sentences) == 1:
            prefix = "#" * self.level
            text = self.sentences[0].text
        else:
            prefix = "#" * self.level
            text = "".join(s.text for s in self.sentences)
        return f"{prefix} {text}\n"


@dataclass(frozen=True)
class MdNode:
    level: int
    span: tuple[int, int]
    line: str | None
    children: dict[str, "MdNode"]  # str(span[0]) -> MdNode

    def get_tree_depth(self):
        if not self.children:
            return 1
        return max(child.get_tree_depth() for child in self.children.values()) + 1

    @cached_property
    def recursive_span(self) -> tuple[int, int]:
        if not self.children:
            return self.span
        return (
            self.span[0],
            max(child.recursive_span[1] for child in self.children.values()),
        )

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            level=data["level"],
            span=data["span"],
            line=data["line"],
            children={k: cls.from_dict(v) for k, v in data["children"].items()},
        )

    def to_dict(self):
        return {
            "level": self.level,
            "span": self.span,
            "line": self.line,
            "children": {k: v.to_dict() for k, v in self.children.items()},
        }


@dataclass(frozen=True)
class NoNumMdNode:
    level: int
    order: int
    line: str | None
    children: dict[str, "NoNumMdNode"]  # str(order) -> NoNumMdNode

    @cached_property
    def title(self):
        cleaned_line = re.sub(r"^#+\s*", "", self.line)
        cleaned_line = self.process_text(cleaned_line)
        return cleaned_line

    @classmethod
    def process_text(cls, text: str) -> str:
        """
        主函数：全角转半角 -> 去除空白符 -> 过滤符号。
        """

        def convert_fullwidth_to_halfwidth(text):
            result = []
            for char in text:
                # 如果是全角字符（Unicode 范围为 65281-65374），转换为半角字符
                if 65281 <= ord(char) <= 65374:
                    result.append(chr(ord(char) - 65248))
                # 全角空格（Unicode 12288）转换为半角空格
                elif ord(char) == 12288:
                    result.append(" ")
                else:
                    result.append(char)
            return "".join(result)

        def filter_text(text):
            whitelist = ['"', "'", ":", ",", "."]
            text = text.replace(" ", "").replace("\n", "").replace("\t", "")

            whitelist_set = set(whitelist)
            result = []
            for char in text:
                # 如果字符是字母/数字，直接保留；如果是符号，检查是否在白名单中
                if char.isalnum() or char in whitelist_set:
                    result.append(char)
            return "".join(result)

        if text != "" and text is not None and text != " ":
            text = convert_fullwidth_to_halfwidth(text)
            text = filter_text(text)

        return text

    def get_tree_depth(self):
        if not self.children:
            return 1
        return max(child.get_tree_depth() for child in self.children.values()) + 1

    @classmethod
    def from_dict(cls, data: dict) -> "NoNumMdNode":
        return cls(
            level=data["level"],
            order=data["order"],
            line=data["line"],
            children={k: cls.from_dict(v) for k, v in data["children"].items()},
        )

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "order": self.order,
            "line": self.line,
            "children": {k: v.to_dict() for k, v in self.children.items()},
        }


class TreeNode(Node):
    def __init__(self, label: str | None, children: List["TreeNode"] | None = None):
        super().__init__(label, children or [])

    def __str__(self):
        return f"{self.label}({','.join(str(c) for c in self.children)})"

    def count_nodes(self) -> int:
        """计算树中节点的总数（包括当前节点）

        Returns:
            int: 树中节点的总数
        """
        if self.label is None:
            return 0
        return 1 + sum(child.count_nodes() for child in self.children)

    def get_depth(self) -> int:
        """获取树的深度

        Returns:
            int: 树的深度
        """
        if self.label is None:
            return 0
        elif len(self.children) == 0:
            return 1
        else:
            return max(child.get_depth() for child in self.children) + 1

    def cut_by_depth(self, depth: int) -> "TreeNode":
        """根据深度截断树

        Args:
            depth: 截断深度
        """
        if depth <= 0:
            return TreeNode.empty_tree()
        elif depth == 1:
            return TreeNode(self.label, [])
        else:
            return TreeNode(
                self.label, [child.cut_by_depth(depth - 1) for child in self.children]
            )

    def to_md(self, depth: int = 0) -> str:
        """将TreeNode转换为Markdown格式

        Args:
            depth: 当前节点的深度，默认为0

        Returns:
            str: 包含树结构的Markdown字符串
        """
        # 根据深度生成相应数量的井号
        prefix = "#" * (depth + 1)
        result = f"{prefix} {self.label}\n"

        # 递归处理子节点，深度加1
        for child in self.children:
            result += child.to_md(depth + 1)

        return result

    def to_dict(self) -> dict:
        """将TreeNode转换为可序列化的字典格式

        Returns:
            dict: 包含树结构的字典
        """
        return {
            "label": self.label,
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TreeNode":
        """从字典创建TreeNode对象

        Args:
            data: 包含树结构的字典

        Returns:
            TreeNode: 创建的树节点对象
        """
        if data is None:
            return None
        return cls(
            label=data["label"],
            children=[cls.from_dict(child) for child in data["children"]],
        )

    @classmethod
    def empty_tree(cls) -> "TreeNode":
        return cls(label=None, children=[])


class HierarchicalDepthMetrics:
    """层级深度相关的评估指标"""

    @staticmethod
    def get_node_depth(node: TreeNode, current_depth: int = 0) -> Dict[str, int]:
        """
        获取树中所有节点的深度信息

        Args:
            node: 树节点
            current_depth: 当前深度

        Returns:
            Dict[str, int]: 节点标签到深度的映射
        """
        depths = {node.label: current_depth}
        for child in node.children:
            depths.update(
                HierarchicalDepthMetrics.get_node_depth(child, current_depth + 1)
            )
        return depths

    @staticmethod
    def hierarchical_depth_accuracy(pred_tree: TreeNode, true_tree: TreeNode) -> float:
        """
        计算层级深度准确率

        Args:
            pred_tree: 预测的树
            true_tree: 真实的树

        Returns:
            float: 层级深度准确率 (0-1之间)
        """
        # 获取两棵树中所有节点的深度信息
        pred_depths = HierarchicalDepthMetrics.get_node_depth(pred_tree)
        true_depths = HierarchicalDepthMetrics.get_node_depth(true_tree)

        # 获取所有唯一的节点标签
        all_labels = set(pred_depths.keys()) | set(true_depths.keys())

        if not all_labels:
            return 1.0

        # 计算深度匹配的节点数
        correct_depths = 0
        total_nodes = 0

        for label in all_labels:
            if label in pred_depths and label in true_depths:
                if pred_depths[label] == true_depths[label]:
                    correct_depths += 1
                total_nodes += 1

        return correct_depths / total_nodes if total_nodes > 0 else 0.0

    @staticmethod
    def hierarchical_depth_f1(
        pred_tree: TreeNode, true_tree: TreeNode
    ) -> Tuple[float, float, float]:
        """
        计算层级深度的精确率、召回率和F1分数

        Args:
            pred_tree: 预测的树
            true_tree: 真实的树

        Returns:
            Tuple[float, float, float]: (精确率, 召回率, F1分数)
        """
        pred_depths = HierarchicalDepthMetrics.get_node_depth(pred_tree)
        true_depths = HierarchicalDepthMetrics.get_node_depth(true_tree)

        # 计算预测正确的节点数
        correct = sum(
            1
            for label in pred_depths
            if label in true_depths and pred_depths[label] == true_depths[label]
        )

        # 计算精确率
        precision = correct / len(pred_depths) if pred_depths else 0.0

        # 计算召回率
        recall = correct / len(true_depths) if true_depths else 0.0

        # 计算F1分数
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return precision, recall, f1


class StarTree:
    """从txt, rank, edu_label, is_star的DataFrame中构建文本树和编号树"""

    @classmethod
    def _find_extension_lines(cls, df: pd.DataFrame, start_idx: int, level: int) -> int:
        """查找当前节点的延伸内容行数

        Args:
            df: 包含txt, rank, edu_label, is_star的DataFrame
            start_idx: 当前节点在DataFrame中的起始索引
            level: 当前处理的层级

        Returns:
            int: 延伸内容的行数
        """
        extension_lines = 0
        i = start_idx + 1

        while i < len(df):
            row = df.iloc[i]
            edu_labels = row["edu_label"]

            # 对于根节点的特殊处理
            if level < 0:
                if len(edu_labels) > 0 and edu_labels[0] == "BOT":
                    extension_lines += 1
                    i += 1
                    continue
            # 对于其他节点的处理
            else:
                if (
                    len(edu_labels) > level + 1
                    and edu_labels[level + 1] == "BOT"
                    and (len(edu_labels) <= level or edu_labels[level] != "BOS")
                ):
                    extension_lines += 1
                    i += 1
                    continue

            # 如果不满足延伸条件，则停止查找
            break

        return extension_lines

    @classmethod
    def _find_nodes_at_level(cls, df: pd.DataFrame, level: int) -> list:
        """在指定层级查找所有符合条件的节点

        Args:
            df: 包含txt, rank, edu_label, is_star的DataFrame
            level: 要查找的层级（从0开始）

        Returns:
            list: 包含(起始行索引, 结束行索引, 节点文本, 节点rank范围)的列表
        """
        nodes = []
        i = 0
        while i < len(df):
            row = df.iloc[i]
            edu_labels = row["edu_label"]
            is_title = "title" in row["label"]
            is_stars = row["is_star"]

            # 检查当前层级是否为BOS且is_star为true
            if level < len(edu_labels) and edu_labels[level] == "BOS":

                # 找到这个节点的结束位置
                start_idx = i
                start_rank = row["rank"]

                # 获取延伸内容的行数
                extension_lines = cls._find_extension_lines(df, i, level)

                # 合并节点文本（包括延伸内容）
                node_text = row["txt"]
                for j in range(1, extension_lines + 1):
                    if i + j < len(df):
                        node_text += df.iloc[i + j]["txt"]

                # 向后查找直到遇到下一个同级节点或文件结束
                j = i + extension_lines + 1
                end_rank = df.iloc[j - 1]["rank"]

                while j < len(df):
                    next_row = df.iloc[j]
                    next_labels = next_row["edu_label"]
                    if level < len(next_labels) and next_labels[level] == "BOS":
                        break
                    j += 1

                nodes.append(
                    (
                        start_idx,
                        j,
                        node_text,
                        (start_rank, end_rank),
                        edu_labels[level + 1] == "BOT",
                        is_title,
                        is_stars[level],
                    )
                )
                i = j
            else:
                i += 1
        return nodes

    @classmethod
    def _build_subtree(
        cls, df: pd.DataFrame, start_idx: int, end_idx: int, level: int
    ) -> Tuple[list[Optional[TreeNode]], list[Optional[TreeNode]]]:
        """递归构建子树

        Args:
            df: 包含txt, rank, edu_label, is_star的DataFrame
            start_idx: 当前子树在DataFrame中的起始索引
            end_idx: 当前子树在DataFrame中的结束索引
            level: 当前处理的层级

        Returns:
            Tuple[list[TreeNode], list[TreeNode]]: 文本树和编号树节点列表
        """
        # 获取当前层级的节点
        sub_df = df.iloc[start_idx:end_idx]
        nodes = cls._find_nodes_at_level(sub_df, level)

        # 单传的非star节点，子节点上提
        if len(nodes) == 1:
            start, end, node_text, node_rank, is_bot, is_title, is_star = nodes[0]
            if not is_title:
                return cls._build_subtree(sub_df, start, end, level + 1)

        # 存储所有同级节点
        text_nodes = []
        rank_nodes = []

        # 处理所有同级节点
        for start, end, node_text, node_rank, is_bot, is_title, is_star in nodes:
            if not is_bot or not is_star:
                continue

            # 为每个节点创建新的节点
            # node_df = sub_df.iloc[start:end]
            new_text_node = TreeNode(label=node_text, children=[])
            new_rank_node = TreeNode(
                label=f"({node_rank[0]},{node_rank[1]})", children=[]
            )

            # 处理新节点的子节点
            child_text_nodes, child_rank_nodes = cls._build_subtree(
                sub_df, start, end, level + 1
            )
            if child_text_nodes:
                new_text_node.children.extend(child_text_nodes)
                new_rank_node.children.extend(child_rank_nodes)

            # 将新节点添加到同级节点列表
            text_nodes.append(new_text_node)
            rank_nodes.append(new_rank_node)

        return text_nodes, rank_nodes

    @classmethod
    def get_trees(cls, df: pd.DataFrame) -> Tuple[TreeNode, TreeNode]:
        """获取文本树和编号树

        Args:
            df: 包含txt, rank, edu_label, is_star的DataFrame

        Returns:
            Tuple[TreeNode, TreeNode]: 文本树和编号树
        """
        # 找到第一个 edu_label 为 ["BOT"] 的节点
        root_idx = None
        for i, row in df.iterrows():
            if row["edu_label"].tolist() == ["BOT"]:
                root_idx = i
                break

        if root_idx is None:
            logger.warning("No root node found")
            return TreeNode.empty_tree(), TreeNode.empty_tree()

        # 构建根节点
        root_rank_start = df.iloc[root_idx]["rank"]
        extension_lines = cls._find_extension_lines(df, root_idx, -1)
        root_rank_end = df.iloc[root_idx + extension_lines]["rank"]

        root_text = df.iloc[root_idx : root_idx + extension_lines + 1]["txt"].str.cat(
            sep=""
        )

        text_tree = TreeNode(label=root_text, children=[])
        rank_tree = TreeNode(label=f"({root_rank_start},{root_rank_end})", children=[])

        child_text_nodes, child_rank_nodes = cls._build_subtree(
            df, root_idx + extension_lines + 1, len(df), 0
        )
        if child_text_nodes:
            text_tree.children.extend(child_text_nodes)
            rank_tree.children.extend(child_rank_nodes)

        return text_tree, rank_tree

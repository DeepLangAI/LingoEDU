import re

from edu_evaluator.definitions import MdNode, NoNumMdNode

MD_LINE_PREFIX_PATTERN = r"^\#+"
MD_LINE_SUFFIX_PATTERN = r"\[句[^\[\]]*(\d+)\]$"


def clean_md_lines(md_lines: list[str]) -> list[str]:
    cleaned_md_lines = []
    start_flag = False
    for line in md_lines:
        if line.strip() == "":
            continue
        if re.match(
            f"{MD_LINE_PREFIX_PATTERN}.+{MD_LINE_SUFFIX_PATTERN}", line.strip()
        ):
            start_flag = True
            cleaned_md_lines.append(line.strip())
        else:
            if start_flag:
                break
    return cleaned_md_lines


def clean_md_lines_nonum(md_lines: list[str]) -> list[str]:
    cleaned_md_lines = []
    start_flag = False
    for line in md_lines:
        if line.strip() == "":
            continue
        if re.match("\#+.+[^\[\]]*", line.strip()):
            start_flag = True
            cleaned_md_lines.append(line.strip())
        else:
            if start_flag:
                break
    return cleaned_md_lines


def parse_level_and_sentence_span(md_line) -> tuple[int, tuple[int, int]]:
    level = -1
    for i, c in enumerate(md_line):
        if c != "#":
            level = i
            break
    md_line_suffix_match = re.search(MD_LINE_SUFFIX_PATTERN, md_line)
    assert md_line_suffix_match is not None
    md_line_suffix = md_line_suffix_match.group(0)
    matches = list(re.finditer(r"句(\d+)", md_line_suffix))
    if len(matches) == 1:
        return level, (int(matches[0].group(1)), int(matches[0].group(1)))
    elif len(matches) == 2:
        return level, (int(matches[0].group(1)), int(matches[1].group(1)))
    else:
        raise ValueError(f"Invalid md line suffix: {md_line_suffix}")


def md_lines_to_tree(lines: list[str]) -> MdNode | None:
    if len(lines) == 0:
        return None

    stack: list[MdNode] = []
    for i, line in enumerate(lines):
        level, span = parse_level_and_sentence_span(line)
        if level == 1:
            assert i == 0

        while len(stack) >= level:
            stack.pop()
        assert level == len(stack) + 1

        node = MdNode(level=level, span=span, line=line, children={})
        if level > 1:
            parent_node = stack[-1]
            parent_node.children[str(span[0])] = node
        stack.append(node)

    return stack[0]


def nonum_md_lines_to_tree(lines: list[str]) -> NoNumMdNode | None:
    if len(lines) == 0:
        return None

    has_level1 = any(line.startswith("# ") for line in lines)
    if not has_level1:
        lines = [None] + lines

    stack: list[NoNumMdNode] = []
    for idx, line in enumerate(lines):
        if line is None:
            level = 1
        else:
            level = -1
            for i, c in enumerate(line):
                if c != "#":
                    level = i
                    break

        if level == 1:
            assert idx == 0

        while len(stack) >= level:
            stack.pop()
        assert level == len(stack) + 1

        if level > 1:
            parent_node = stack[-1]
            order = len(parent_node.children)
            node = NoNumMdNode(level=level, order=order, line=line, children={})
            parent_node.children[str(order)] = node
        else:
            node = NoNumMdNode(level=level, order=0, line=line, children={})
        stack.append(node)

    return stack[0]


def tree_to_md_lines(md_node: MdNode | None) -> list[str]:
    if md_node is None:
        return []

    md_lines = []
    if md_node.line is not None:
        md_lines.append(md_node.line)
    for key in sorted(md_node.children.keys(), key=int):
        md_lines.extend(tree_to_md_lines(md_node.children[key]))
    return md_lines


def nonum_tree_to_md_lines(md_node: NoNumMdNode | None) -> list[str]:
    if md_node is None:
        return []

    md_lines = []
    if md_node.line is not None:
        md_lines.append(md_node.line)
    for key in sorted(md_node.children.keys()):
        md_lines.extend(nonum_tree_to_md_lines(md_node.children[key]))
    return md_lines


def is_legal_md_lines(md_lines: list[str], do_clean: bool = True) -> bool:
    if do_clean:
        md_lines = clean_md_lines(md_lines)

    try:
        md_lines_to_tree(md_lines)
    except AssertionError:
        return False
    return True


def is_legal_md_lines_nonum(md_lines: list[str], do_clean: bool = True) -> bool:
    if do_clean:
        md_lines = clean_md_lines_nonum(md_lines)

    try:
        nonum_md_lines_to_tree(md_lines)
    except AssertionError:
        return False
    return True

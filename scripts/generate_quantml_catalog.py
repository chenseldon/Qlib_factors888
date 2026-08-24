"""Generate the local-only QuantML adapted catalog from a Markdown table."""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import OrderedDict
from pathlib import Path


EXPECTED_SHA256 = "b5e07f40ee97a9b7df14635226f05851611bfc1927b167d28a453cb249867b29"


def read_expressions(path: Path) -> OrderedDict[str, list[int]]:
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"unexpected QuantML source SHA256: {digest}")

    expressions: OrderedDict[str, list[int]] = OrderedDict()
    for line_number, line in enumerate(source.decode("utf-8").splitlines(), start=1):
        match = re.match(r"^\|\s*(.*?)\s*\|", line)
        if match is None:
            continue
        expression = match.group(1).strip()
        if expression == "factor_name" or re.fullmatch(r":?-+:?", expression):
            continue
        expressions.setdefault(expression, []).append(line_number)
    return expressions


def render(expressions: OrderedDict[str, list[int]]) -> str:
    if sum(len(rows) for rows in expressions.values()) != 1049:
        raise ValueError("expected 1,049 QuantML source rows")
    if len(expressions) != 994:
        raise ValueError("expected 994 unique QuantML expressions")

    lines = [
        '"""Generated local-only QuantML adapted factor catalog. Do not redistribute."""',
        "",
        f'SOURCE_SHA256 = "{EXPECTED_SHA256}"',
        "SOURCE_ROW_COUNT = 1049",
        "UNIQUE_EXPRESSION_COUNT = 994",
        "DUPLICATE_ROW_COUNT = 55",
        "",
        "CATALOG_ROWS = (",
    ]
    for index, (expression, source_rows) in enumerate(expressions.items(), start=1):
        lines.append(f"    ('QMLA{index:04d}', {expression!r}, {tuple(source_rows)!r}),")
    lines.extend([")", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(render(read_expressions(args.source)), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()

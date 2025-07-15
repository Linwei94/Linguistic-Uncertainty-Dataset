#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract high / moderate / low confidence sentences (10 each) from
LLM-generated responses that use headings like:
    ### High Confidence
    **Moderate Confidence**
and list formats such as
    1. sentence...
    - sentence...
    * sentence...
"""

from __future__ import annotations
import re
import sys
from pathlib import Path
import pandas as pd

# ───────────────────────────── helper regexes ──────────────────────────────
_BOLD_HEADING_RE = re.compile(
    r'\*\*\s*(High|Moderate|Low)\s+Confidence\s*\*\*', flags=re.I
)

_SPLIT_HEADING_RE = re.compile(
    r'###\s*(High|Moderate|Low)\s+Confidence', flags=re.I
)

_LIST_SENTENCE_RE = re.compile(
    r'(?:\d+\.\s+|[-*]\s+)?'      # numbered or bullet prefix (optional)
    r'(.+?)'                      # the sentence itself (non-greedy)
    r'(?=\n(?:\d+\.\s+|[-*]\s+)|\Z)',  # stop at next list item or EOF
    flags=re.S,
)

# ───────────────────────────── core functions ──────────────────────────────
def normalize_headings(text: str) -> str:
    """Convert '**High Confidence**' → '### High Confidence' for uniform parsing."""
    return _BOLD_HEADING_RE.sub(
        lambda m: f"### {m.group(1).title()} Confidence", text
    )


def extract_levels(text: str, max_n: int = 10) -> dict[str, list[str]]:
    text = normalize_headings(text.replace("\r\n", "\n"))
    parts = _SPLIT_HEADING_RE.split(text)

    result = {"high": [], "moderate": [], "low": []}
    for i in range(1, len(parts), 2):
        label   = parts[i].strip().lower()
        section = parts[i + 1]

        # 先抓句子
        raw_sents = _LIST_SENTENCE_RE.findall(section.strip())

        # 统一去掉行首 "1.  " / "-  " / "* "
        clean_sents = [
            re.sub(r'^(?:\d+\.\s+|[-*]\s+)', '', s.strip())
            for s in raw_sents
            if s.strip()
        ][:max_n]

        result[label] = clean_sents

    return result



def parse_response_df(df: pd.DataFrame, col: str = "response") -> pd.DataFrame:
    """Expand a DataFrame column into three list columns (high/moderate/low)."""
    return pd.DataFrame(df[col].apply(extract_levels).tolist())


def flatten_levels_df(levels_df: pd.DataFrame) -> pd.DataFrame:
    """Explode the lists so each sentence is its own row."""
    records: list[dict[str, str | int]] = []
    for idx, row in levels_df.iterrows():
        for level in ("high", "moderate", "low"):
            for sent in row[level]:
                records.append({"orig_row": idx, "level": level, "sentence": sent})
    return pd.DataFrame(records)


# ────────────────────────────── CLI / main ────────────────────────────────
def main(
    input_csv: str | Path = "generated_text.csv",
    output_csv: str | Path = "extracted_sentences.csv",
    response_col: str = "response",
) -> None:
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    if not input_csv.exists():
        sys.exit(f"[Error] {input_csv} not found.")

    df_raw = pd.read_csv(input_csv)
    if response_col not in df_raw.columns:
        sys.exit(f"[Error] column '{response_col}' not found in {input_csv}")

    # 1) lists per level
    levels_df = parse_response_df(df_raw, col=response_col)

    # 2) flattened
    flat_df = flatten_levels_df(levels_df)
    flat_df.to_csv(output_csv, index=False)

    print(f"[OK] Extracted {len(flat_df)} sentences → {output_csv}")


if __name__ == "__main__":
    # 可通过命令行参数自定义路径：
    #   python extract_confidence_sentences.py input.csv output.csv
    args = sys.argv[1:]
    main(*args)  # uses defaults if no args

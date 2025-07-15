#!/usr/bin/env python3
"""
Interactive extractor for 5‑level confidence sentences.

Workflow
--------
1. Read generated_text_*.csv (each must have columns: problem, answer,
   raw_response, metadata)
2. For each raw_response, extract five uncertainty‑level buckets.
3. Auto‑pass rows that already contain exactly 5 × 10 sentences.
4. Otherwise, enter an interactive UI:
     · Enter / n : accept current distribution
     · e         : edit labels / delete sentences
     · s         : skip this row
5. Output
     · all_sentences_by_confidence.csv  (clean rows)
     · rows_needing_fix.csv             (skipped or still invalid rows)
"""

import re
import textwrap
import pandas as pd
from pathlib import Path

# ─────────────────────────────── CONFIG ─────────────────────────────── #
FILES = {
    "grok":   Path("generated_text_grok.csv"),
    "gpt":    Path("generated_text_gpt.csv"),
    "gemini": Path("generated_text_gemini.csv"),
    "claude": Path("generated_text_claude.csv"),
}

EXPECTED = 10                  # target sentences per level
USE_BULLET = True              # include - • * bullets as list items
ALL_LEVELS = ["high", "moderate", "low", "lowest", "completely uncertain"]

SHORT2LEVEL = {
    "h":  "high",
    "m":  "moderate",
    "l":  "low",
    "ls": "lowest",
    "c":  "completely uncertain",
}

# ────────────────────── 1. heading pattern helper ───────────────────── #
def pat_three_cases(phrase: str) -> re.Pattern:
    """
    Match three fixed casings:
      • Title Case          e.g. 'Complete Uncertainty'
      • ALL CAPS            e.g. 'COMPLETE UNCERTAINTY'
      • First word capital  e.g. 'Complete uncertainty'
    Any other casing is ignored.
    """
    words  = phrase.split()
    title  = ' '.join(w.capitalize() for w in words)
    caps   = phrase.upper()
    first  = words[0].capitalize() + ' ' + ' '.join(w.lower() for w in words[1:])
    pattern = '|'.join(re.escape(v) for v in (title, caps, first))
    return re.compile(rf"\b(?:{pattern})\b")

LEVEL_KEYSETS = [
    ("completely uncertain", [pat_three_cases("complete uncertainty"),
                              pat_three_cases("absolute uncertainty")]),
    ("lowest",   [pat_three_cases("lowest confidence")]),
    ("low",      [pat_three_cases("low confidence")]),
    ("moderate", [pat_three_cases("moderate confidence")]),
    ("high",     [pat_three_cases("high confidence")]),
]

def heading_level(line: str) -> str | None:
    """
    Detect heading level by regex patterns defined above.
    """
    txt = re.sub(r'^[#>*\-\s]+', '', line.strip())
    for lvl, pats in LEVEL_KEYSETS:
        if any(p.search(txt) for p in pats):
            return lvl
    # Debug print for unmatched headings that still contain key words
    if any(w in txt for w in ("Confidence", "Uncertain")):
        print("⚠️  Unmatched heading:", txt)
    return None

# ───────────────────── 2. list‑item detection helpers ───────────────── #
NUM_RE    = re.compile(r"^\s*\d+\s*[.)]\s+")
BULLET_RE = re.compile(r"^\s*[\-*•]\s+")

def is_list_item(line: str) -> bool:
    return bool(NUM_RE.match(line) or (USE_BULLET and BULLET_RE.match(line)))

def strip_item(line: str) -> str:
    if NUM_RE.match(line):
        return NUM_RE.sub("", line).strip()
    if BULLET_RE.match(line):
        return BULLET_RE.sub("", line).strip()
    return line.strip()

# ───────────────────── 3. parse a single raw_response ───────────────── #
def parse_raw(raw: str) -> dict[str, list[str]]:
    """
    Return dict: level -> list of sentences (≥2 words).
    """
    bucket = {lvl: [] for lvl in ALL_LEVELS}
    current = None
    for ln in raw.splitlines():
        l = ln.rstrip()
        if not l:
            continue
        if lvl := heading_level(l):
            current = lvl
            continue
        if current and is_list_item(l):
            sent = strip_item(l)
            if len(sent.split()) >= 2:
                bucket[current].append(sent)
    return bucket

# ─────────────────── 4. pretty‑print current bucket ─────────────────── #
def show_bucket(bucket: dict[str, list[str]], model: str, idx: int):
    print("\n" + "="*70)
    print(f"[{model}] row #{idx}")
    for lvl in ALL_LEVELS:
        print(f"\n--- {lvl.upper()} ({len(bucket[lvl])}) ---")
        for i, s in enumerate(bucket[lvl], 1):
            print(f"{i:2d}. {s}")

# ───────────────────── 5. interactive bucket editor ─────────────────── #
def edit_bucket(bucket: dict[str, list[str]]):
    """
    Per‑sentence editor.
      Enter     → keep
      d         → delete
      h/m/l/ls/c→ relabel
    """
    for lvl in ALL_LEVELS:
        i = 0
        total = len(bucket[lvl])
        while i < len(bucket[lvl]):
            s = bucket[lvl][i]
            prompt = (f"\n✏️  [{lvl.upper():>10} {i+1}/{total}] «{s}»\n"
                      f"    New level? (h/m/l/ls/c / d=delete / Enter=keep) > ")
            cmd = input(prompt).strip().lower()

            if not cmd:           # keep
                i += 1
            elif cmd == "d":      # delete
                bucket[lvl].pop(i)
                total -= 1
            elif cmd in SHORT2LEVEL:     # relabel
                new_lvl = SHORT2LEVEL[cmd]
                bucket[new_lvl].append(bucket[lvl].pop(i))
                total -= 1
            else:
                print("  ⚠️  Invalid input; keeping original label")
                i += 1
    return bucket

# ───────────────────────── 6. main interactive loop ─────────────────── #
good_rows, bad_rows = [], []

for model, path in FILES.items():
    df = pd.read_csv(path)  # columns: problem, answer, raw_response, metadata

    for idx, row in df.iterrows():
        bucket = parse_raw(str(row["raw_response"]))

        # 6.1 Auto‑pass if exactly 5 × 10 sentences
        if all(len(bucket[lvl]) == EXPECTED for lvl in ALL_LEVELS):
            print(f"✅  AUTO‑PASS | {model:<6} row={idx:<4}")
            for lvl in ALL_LEVELS:
                for sent in bucket[lvl]:
                    good_rows.append({
                        "model":      model,
                        "problem":    row["problem"],
                        "answer":     row["answer"],
                        "metadata":   row["metadata"],
                        "confidence": lvl,
                        "sentence":   sent,
                    })
            continue

        # 6.2 Otherwise interactive correction
        while True:
            show_bucket(bucket, model, idx)
            summary = " | ".join(f"{lvl}:{len(bucket[lvl])}" for lvl in ALL_LEVELS)
            print(f"\nCounts → {summary}")
            action = input("✔️  Enter/n=next | e=edit | s=skip > ").strip().lower()

            if action in ("", "n"):
                # accept current distribution
                if all(len(bucket[lvl]) >= EXPECTED for lvl in ALL_LEVELS):
                    for lvl in ALL_LEVELS:
                        for sent in bucket[lvl][:EXPECTED]:
                            good_rows.append({
                                "model":      model,
                                "problem":    row["problem"],
                                "answer":     row["answer"],
                                "metadata":   row["metadata"],
                                "confidence": lvl,
                                "sentence":   sent,
                            })
                else:
                    bad_rows.append({
                        "model":     model,
                        "row_index": idx,
                        "problem":   row["problem"],
                        "answer":    row["answer"],
                        "metadata":  row["metadata"],
                        "counts":    {lvl: len(bucket[lvl]) for lvl in ALL_LEVELS},
                        "preview":   textwrap.shorten(str(row["raw_response"]), 300),
                    })
                break

            elif action == "e":
                print("raw_response:", row["raw_response"])
                bucket = edit_bucket(bucket)
                continue

            elif action == "s":
                bad_rows.append({
                    "model":     model,
                    "row_index": idx,
                    "problem":   row["problem"],
                    "answer":    row["answer"],
                    "metadata":  row["metadata"],
                    "counts":    {lvl: len(bucket[lvl]) for lvl in ALL_LEVELS},
                    "preview":   textwrap.shorten(str(row["raw_response"]), 300),
                })
                break
            else:
                print("⚠️  Invalid command; try again")

# ───────────────────────────── 7. save results ─────────────────────── #
pd.DataFrame(good_rows).to_csv("all_sentences_by_confidence.csv", index=False)
pd.DataFrame(bad_rows).to_csv("rows_needing_fix.csv", index=False)

print("\n🎉  DONE")
print(f"Good sentences: {len(good_rows):,}  (rows={len(good_rows)//50})")
print(f"Bad rows      : {len(bad_rows)}  → rows_needing_fix.csv")

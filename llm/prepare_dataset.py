from __future__ import annotations

import argparse
import os
from typing import Optional

from datasets import load_dataset


FIELDS = ["text", "content", "article", "body"]


def normalize_text(s: str) -> str:
    s = s.replace("\r", "\n").strip()
    # collapse whitespace
    return " ".join(s.split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wikitext", help="HuggingFace dataset name")
    ap.add_argument("--config", default="wikitext-103-raw-v1", help="Dataset config name")
    ap.add_argument("--split", default="train", help="Split to stream")
    ap.add_argument("--output", default="data.txt", help="Output text file path")
    ap.add_argument("--max_bytes", type=int, default=100_000_000, help="Cap output size in bytes (approx)")
    ap.add_argument("--max_rows", type=int, default=0, help="Optional cap on number of rows (0=unbounded)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)

    written = 0
    rows = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for ex in ds:
            txt: Optional[str] = None
            for k in FIELDS:
                if k in ex and isinstance(ex[k], str) and ex[k].strip():
                    txt = ex[k]
                    break
            if not txt:
                continue
            line = normalize_text(txt) + "\n"
            f.write(line)
            written += len(line.encode("utf-8"))
            rows += 1
            if args.max_rows and rows >= args.max_rows:
                break
            if args.max_bytes and written >= args.max_bytes:
                break

    print(f"Wrote ~{written} bytes across {rows} rows to {args.output}")


if __name__ == "__main__":
    main()

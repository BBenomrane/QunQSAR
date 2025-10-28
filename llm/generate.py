from __future__ import annotations

import argparse
import json
import os

import torch

from minillm.model import GPT, GPTConfig
from minillm.tokenizer import ByteTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint .pt file")
    ap.add_argument("--prompt", default="Hello", help="Prompt text")
    ap.add_argument("--max_new", type=int, default=200, help="Max new tokens to sample")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0, help="Top-k filtering (0 disables)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    cfg_dict = ckpt.get("config")
    if cfg_dict is None:
        raise RuntimeError("Checkpoint missing 'config'")
    cfg = GPTConfig(**cfg_dict)

    model = GPT(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval().to(args.device)

    tok = ByteTokenizer()
    prompt_ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=args.device)
    out = model.generate(
        prompt_ids,
        max_new_tokens=args.max_new,
        temperature=args.temperature,
        top_k=(args.top_k if args.top_k > 0 else None),
    )
    text = tok.decode(out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()

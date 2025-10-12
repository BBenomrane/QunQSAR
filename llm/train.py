from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from minillm.data import LoaderConfig, create_dataloaders
from minillm.model import GPT, GPTConfig


def cosine_lr(step: int, max_steps: int, base_lr: float, min_lr_ratio: float = 0.1) -> float:
    if step >= max_steps:
        return base_lr * min_lr_ratio
    cosine = 0.5 * (1 + math.cos(math.pi * step / max_steps))
    return min_lr_ratio * base_lr + (base_lr - min_lr_ratio * base_lr) * cosine


def evaluate(model: GPT, val_loader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, y)
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()
    model.train()
    return total_loss / max(1, total_tokens)


@dataclass
class TrainConfig:
    data_path: str
    out_dir: str = "./out"
    block_size: int = 256
    batch_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0
    max_steps: int = 1000
    eval_interval: int = 100
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    amp: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1337


def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)

    os.makedirs(cfg.out_dir, exist_ok=True)

    # Data
    loaders_cfg = LoaderConfig(
        file_path=cfg.data_path,
        block_size=cfg.block_size,
        batch_size=cfg.batch_size,
    )
    train_loader, val_loader = create_dataloaders(loaders_cfg)

    # Model
    model_cfg = GPTConfig(
        vocab_size=256,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        block_size=cfg.block_size,
        dropout=cfg.dropout,
    )
    model = GPT(model_cfg).to(cfg.device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas)

    amp_enabled = cfg.amp and torch.cuda.is_available() and str(cfg.device).startswith("cuda")
    scaler = GradScaler(enabled=amp_enabled)

    step = 0
    best_val = float("inf")

    pbar = tqdm(total=cfg.max_steps, desc="training")

    while step < cfg.max_steps:
        for x, y in train_loader:
            if step >= cfg.max_steps:
                break
            lr = cosine_lr(step, cfg.max_steps, cfg.lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            x = x.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            step += 1
            pbar.update(1)
            if step % 10 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

            if step % cfg.eval_interval == 0 or step == cfg.max_steps:
                val = evaluate(model, val_loader, torch.device(cfg.device))
                if val < best_val:
                    best_val = val
                    ckpt_path = os.path.join(cfg.out_dir, "best.pt")
                    torch.save({
                        "model": model.state_dict(),
                        "config": model_cfg.__dict__,
                        "step": step,
                        "val_loss": val,
                    }, ckpt_path)
                last_path = os.path.join(cfg.out_dir, "last.pt")
                torch.save({
                    "model": model.state_dict(),
                    "config": model_cfg.__dict__,
                    "step": step,
                    "val_loss": val,
                }, last_path)

    pbar.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to input text file")
    ap.add_argument("--out", default="./out", help="Output directory for checkpoints")
    ap.add_argument("--block", type=int, default=256, help="Context length")
    ap.add_argument("--batch", type=int, default=64, help="Batch size")
    ap.add_argument("--layers", type=int, default=4, help="Number of transformer layers")
    ap.add_argument("--heads", type=int, default=4, help="Number of attention heads")
    ap.add_argument("--embd", type=int, default=256, help="Embedding size")
    ap.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    ap.add_argument("--steps", type=int, default=1000, help="Max training steps")
    ap.add_argument("--eval_every", type=int, default=100, help="Eval interval")
    ap.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    ap.add_argument("--wd", type=float, default=0.1, help="Weight decay")
    ap.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clip norm")
    ap.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    cfg = TrainConfig(
        data_path=args.data,
        out_dir=args.out,
        block_size=args.block,
        batch_size=args.batch,
        n_layer=args.layers,
        n_head=args.heads,
        n_embd=args.embd,
        dropout=args.dropout,
        max_steps=args.steps,
        eval_interval=args.eval_every,
        lr=args.lr,
        weight_decay=args.wd,
        grad_clip=args.grad_clip,
        amp=not args.no_amp,
        device=args.device,
        seed=args.seed,
    )

    train(cfg)


if __name__ == "__main__":
    main()

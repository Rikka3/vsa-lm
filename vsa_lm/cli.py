"""Command-line interface for VSA-LM (`python -m vsa_lm <subcommand>`)."""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser(
        prog="vsa_lm",
        description="VSA-LM: Fixed-State Linear-Attention Language Models",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # 1. `python -m vsa_lm train`
    p_train = sub.add_parser("train", help="Train or resume a VSA-LM with auto-hardware fitting")
    p_train.add_argument("--run", default="vsa", choices=["vsa", "base"], help="Model arm")
    p_train.add_argument("--max-len", type=int, default=512, help="Sequence length")
    p_train.add_argument("--work", default=None, help="Working directory for checkpoints")

    # 2. `python -m vsa_lm scan --ckpt <path>`
    p_scan = sub.add_parser("scan", help="Run static capacity & horizon diagnostics on a checkpoint")
    p_scan.add_argument("--ckpt", required=True, help="Path to .pth checkpoint")

    # 3. `python -m vsa_lm test`
    sub.add_parser("test", help="Run the Triton kernel, gradient, decoder, and erasure parity suite")

    args = p.parse_args(argv)

    if args.cmd == "train":
        from .train import main as train_main
        train_main(run=args.run, max_len=args.max_len, working=args.work)
    elif args.cmd == "scan":
        from .surgery import scan_checkpoint
        scan_checkpoint(args.ckpt)
    elif args.cmd == "test":
        from .tests import run_all
        run_all()


if __name__ == "__main__":
    main()

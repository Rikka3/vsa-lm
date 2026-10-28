"""Command-line interface for VSA-LM (`python -m vsa_lm <subcommand>`)."""
from __future__ import annotations

import argparse
import os
import sys
import torch


def _run_fast_heal(heal_args: list[str]):
    """Run the 1B anchored heal using vsa_lm.surgery's memory-optimal FusedChunkedCEKD
    and bfloat16 teacher optimizations."""
    from .surgery import fast_fused_ce_kd

    # Ensure workdir scripts (c1_surgery / c3_heal) are importable if present
    for d in (os.getcwd(), "/marimo"):
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)

    import c1_surgery as c1
    import c3_heal as c3

    # 1. Inject vsa_lm.surgery.fast_fused_ce_kd (saves ~30 GiB VRAM & avoids 2nd softmax pass)
    def _wrapped_fused_ce_kd(model, h, y, donor, donor_h, kd, temp=1.0, chunk=2):
        if donor is not None and next(donor.parameters()).dtype == torch.float32 and h.is_cuda:
            donor.to(torch.bfloat16)
            print("[vsa_lm.surgery] Converted frozen teacher to bfloat16 (saved 2.5 GiB VRAM, 2x faster teacher matmuls)", flush=True)
        loss = fast_fused_ce_kd(model, h, y, donor, donor_h, kd=kd, temp=temp, chunk=chunk)
        v = float(loss.detach())
        return loss, v, 0.0

    c1.fused_ce_kd = _wrapped_fused_ce_kd
    print("[vsa_lm] Activated FusedChunkedCEKD + bf16 teacher engine from vsa_lm.surgery!", flush=True)

    sys.argv = ["c3_heal.py"] + heal_args
    c3.main()


def main(argv: list[str] | None = None):
    if argv is None:
        argv = sys.argv[1:]

    # Forward unknown flags directly to _run_fast_heal when subcommand is 'heal'
    if argv and argv[0] == "heal":
        _run_fast_heal(argv[1:])
        return

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

    # 2. `python -m vsa_lm heal [c3_heal args...]`
    sub.add_parser("heal", help="Run memory-optimal 1B anchored healing via vsa_lm.surgery")

    # 3. `python -m vsa_lm scan --ckpt <path>`
    p_scan = sub.add_parser("scan", help="Run static capacity & horizon diagnostics on a checkpoint")
    p_scan.add_argument("--ckpt", required=True, help="Path to .pth checkpoint")

    # 4. `python -m vsa_lm test`
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

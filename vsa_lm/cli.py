"""Command-line interface for VSA-LM (`python -m vsa_lm <subcommand>`)."""
from __future__ import annotations

import argparse
import os
import sys
import torch
import torch.utils.checkpoint


def _should_use_legacy_c3(heal_args: list[str]) -> bool:
    """Use c3_heal.py only if it exists locally and the checkpoint is a C3 Llama-3.2-1B run."""
    c3_exists = any(os.path.isfile(os.path.join(d, "c3_heal.py")) for d in (os.getcwd(), "/marimo"))
    if not c3_exists:
        return False
    if "--ckpt" in heal_args:
        idx = heal_args.index("--ckpt")
        if idx + 1 < len(heal_args) and os.path.isfile(heal_args[idx + 1]):
            try:
                ck = torch.load(heal_args[idx + 1], map_location="cpu", weights_only=False)
                arch = (ck.get("config") or {}).get("donor_arch", "llama")
                if arch != "llama" or (ck.get("config") or {}).get("qkv_bias", False):
                    return False
            except Exception:
                pass
    return True


def _run_fast_heal(heal_args: list[str]):
    if not _should_use_legacy_c3(heal_args):
        p = argparse.ArgumentParser(prog="vsa_lm heal")
        p.add_argument("--ckpt", required=True, help="Path to converted VSA checkpoint")
        p.add_argument("--donor", default=None, help="Optional HF donor repo ID")
        p.add_argument("--work", default=".", help="Working directory")
        p.add_argument("--out-prefix", default="vsa_healed", help="Output checkpoint prefix")
        p.add_argument("--tokens", type=int, default=80_000_000, help="Healing token budget")
        p.add_argument("--data-tokens", type=int, default=100_000_000, help="Token cache size")
        p.add_argument("--bs", type=int, default=16)
        p.add_argument("--lr", type=float, default=1.5e-4)
        p.add_argument("--warmup", type=int, default=150)
        p.add_argument("--kd", type=float, default=1.0)
        p.add_argument("--eval-every", type=int, default=250)
        p.add_argument("--save-every", type=int, default=500)
        p.add_argument("--seed", type=int, default=7777)
        p.add_argument("--hf-repo", default=None, help="Optional HF repo for periodic checkpoint backups")
        p.add_argument("--highspeed", action="store_true", help="Use more VRAM (no block checkpointing, micro_bs=8) for faster step time")
        args, _ = p.parse_known_args(heal_args)
        from .converter import heal_universal_model
        heal_universal_model(
            args.ckpt, donor_id=args.donor, work=args.work, out_prefix=args.out_prefix,
            tokens_budget=args.tokens, data_tokens=args.data_tokens, bs=args.bs,
            lr=args.lr, warmup=args.warmup, kd=args.kd, eval_every=args.eval_every,
            save_every=args.save_every, seed=args.seed, hf_repo=args.hf_repo,
            highspeed=args.highspeed,
        )
        return

    for d in (os.getcwd(), "/marimo"):
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)

    import c1_surgery as c1
    import c3_heal as c3

    orig_fused_ce_kd = c1.fused_ce_kd

    def _wrapped_fused_ce_kd(model, h, y, donor, donor_h, kd, temp=1.0, chunk=2):
        if donor is not None and next(donor.parameters()).dtype == torch.float32 and h.is_cuda:
            donor.to(torch.bfloat16)
            print("[vsa_lm.surgery] Converted frozen teacher to bfloat16 (2x faster teacher matmuls)", flush=True)
        return orig_fused_ce_kd(model, h, y, donor, donor_h, kd, temp, chunk=chunk)

    c1.fused_ce_kd = _wrapped_fused_ce_kd
    print("[vsa_lm] Activated direct FP32 CE+KD + bf16 teacher!", flush=True)
    sys.argv = ["c3_heal.py"] + heal_args
    c3.main()


def main(argv: list[str] | None = None):
    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0] == "heal":
        _run_fast_heal(argv[1:])
        return

    p = argparse.ArgumentParser(
        prog="vsa_lm",
        description="VSA-LM: Fixed-State Linear-Attention Language Models",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_conv = sub.add_parser("convert", help="Convert any Llama/Qwen/Mistral/SmolLM model into a VSA hybrid")
    p_conv.add_argument("--donor", required=True, help="HuggingFace repo ID or local snapshot directory")
    p_conv.add_argument("--out", default="vsa_student_init.pth", help="Output .pth checkpoint path")
    p_conv.add_argument("--hybrid-every", type=int, default=4, help="Keep 1 Softmax layer every N layers (default: 4)")
    p_conv.add_argument("--max-T", type=int, default=1024, help="Training sequence length (default: 1024)")
    p_conv.add_argument("--gamma-cap", type=float, default=0.992, help="Max initial decay horizon (default: 0.992)")
    p_conv.add_argument("--hf-token", default=None, help="Optional HuggingFace token for gated models")

    sub.add_parser("heal", help="Run memory-optimal anchored healing on any converted VSA checkpoint")

    p_train = sub.add_parser("train", help="Train or resume a VSA-LM from scratch with auto-hardware fitting")
    p_train.add_argument("--run", default="vsa", choices=["vsa", "base"], help="Model arm")
    p_train.add_argument("--max-len", type=int, default=512, help="Sequence length")
    p_train.add_argument("--work", default=None, help="Working directory for checkpoints")

    p_scan = sub.add_parser("scan", help="Run static capacity & horizon diagnostics on a checkpoint")
    p_scan.add_argument("--ckpt", required=True, help="Path to .pth checkpoint")

    sub.add_parser("test", help="Run the Triton kernel, gradient, decoder, and erasure parity suite")

    args = p.parse_args(argv)

    if args.cmd == "convert":
        from .converter import convert_donor_model
        convert_donor_model(
            args.donor, args.out, hybrid_every=args.hybrid_every,
            max_T=args.max_T, gamma_cap=args.gamma_cap, hf_token=args.hf_token,
        )
    elif args.cmd == "train":
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

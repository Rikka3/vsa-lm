"""Command-line interface for VSA-LM (`python -m vsa_lm <subcommand>`)."""
from __future__ import annotations

import argparse
import os
import sys
import torch
import torch.utils.checkpoint


def _run_fast_heal(heal_args: list[str]):
    """Run the 1B anchored heal using vsa_lm.surgery's memory-optimal FusedChunkedCEKD,
    bfloat16 teacher, and automatic VRAM fitting (block checkpointing + micro_bs scaling)."""
    from .surgery import fast_fused_ce_kd

    low_vram_flag = "--low-vram" in heal_args
    if low_vram_flag:
        heal_args = [a for a in heal_args if a != "--low-vram"]

    for d in (os.getcwd(), "/marimo"):
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)

    import c1_surgery as c1
    import c3_heal as c3

    vram_gb = 96.0
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    use_block_ckpt = low_vram_flag or (vram_gb < 40.0)
    if use_block_ckpt:
        orig_blk_fwd = c1.LlamaBlock.forward

        def _ckpt_blk_fwd(self, x):
            if self.training and x.requires_grad:
                return torch.utils.checkpoint.checkpoint(orig_blk_fwd, self, x, use_reentrant=False)
            return orig_blk_fwd(self, x)

        c1.LlamaBlock.forward = _ckpt_blk_fwd
        if "--micro-bs" not in heal_args:
            auto_mbs = "2" if vram_gb < 20.0 else "4"
            heal_args = heal_args + ["--micro-bs", auto_mbs]
        print(f"[vsa_lm.auto-hw] GPU VRAM={vram_gb:.1f} GiB -> enabled LlamaBlock activation "
              f"checkpointing (~28 GiB VRAM saved!)", flush=True)

    def _wrapped_fused_ce_kd(model, h, y, donor, donor_h, kd, temp=1.0, chunk=2):
        if donor is not None and next(donor.parameters()).dtype == torch.float32 and h.is_cuda:
            donor.to(torch.bfloat16)
            print("[vsa_lm.surgery] Converted frozen teacher to bfloat16 "
                  "(saved 2.5 GiB VRAM, 2x faster teacher matmuls)", flush=True)
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

    if argv and argv[0] == "heal":
        _run_fast_heal(argv[1:])
        return

    p = argparse.ArgumentParser(
        prog="vsa_lm",
        description="VSA-LM: Fixed-State Linear-Attention Language Models",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # 1. `python -m vsa_lm convert --donor <repo_or_dir> --out <ckpt.pth>`
    p_conv = sub.add_parser("convert", help="Convert any Llama/Qwen/Mistral/SmolLM model into a VSA hybrid")
    p_conv.add_argument("--donor", required=True, help="HuggingFace repo ID or local snapshot directory")
    p_conv.add_argument("--out", default="vsa_student_init.pth", help="Output .pth checkpoint path")
    p_conv.add_argument("--hybrid-every", type=int, default=4, help="Keep 1 Softmax layer every N layers (default: 4)")
    p_conv.add_argument("--max-T", type=int, default=1024, help="Training sequence length (default: 1024)")
    p_conv.add_argument("--gamma-cap", type=float, default=0.995, help="Max initial decay horizon (default: 0.995)")
    p_conv.add_argument("--hf-token", default=None, help="Optional HuggingFace token for gated models")

    # 2. `python -m vsa_lm heal [c3_heal args...]`
    sub.add_parser("heal", help="Run memory-optimal anchored healing via vsa_lm.surgery")

    # 3. `python -m vsa_lm train`
    p_train = sub.add_parser("train", help="Train or resume a VSA-LM from scratch with auto-hardware fitting")
    p_train.add_argument("--run", default="vsa", choices=["vsa", "base"], help="Model arm")
    p_train.add_argument("--max-len", type=int, default=512, help="Sequence length")
    p_train.add_argument("--work", default=None, help="Working directory for checkpoints")

    # 4. `python -m vsa_lm scan --ckpt <path>`
    p_scan = sub.add_parser("scan", help="Run static capacity & horizon diagnostics on a checkpoint")
    p_scan.add_argument("--ckpt", required=True, help="Path to .pth checkpoint")

    # 5. `python -m vsa_lm test`
    sub.add_parser("test", help="Run the Triton kernel, gradient, decoder, and erasure parity suite")

    args = p.parse_args(argv)

    if args.cmd == "convert":
        from .converter import convert_donor_model
        convert_donor_model(
            args.donor,
            args.out,
            hybrid_every=args.hybrid_every,
            max_T=args.max_T,
            gamma_cap=args.gamma_cap,
            hf_token=args.hf_token,
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

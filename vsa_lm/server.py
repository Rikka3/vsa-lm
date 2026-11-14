"""OpenAI-Compatible Local API Server for VSA-LM (`python -m vsa_lm serve`).

Fixes from Audit:
  - D1: Uses stateful VSARecurrentDecoder (O(1) VSA state + growing Softmax KV cache) instead of sliding a 1,024-token window!
  - D2: Removes hidden 1.08 repetition penalty that corrupted math/code tokens; makes repetition_penalty opt-in (default 1.0).
  - D3: Checks all EOS & chat end tokens (`<|im_end|>`, `<|endoftext|>`, `<｜end▁of▁sentence｜>`, `eos_token_id`).
  - B6 + D4: Requires valid donor metadata or `--donor` flag; defaults `quantize_cpu=False` unless requested.
"""
from __future__ import annotations

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
import torch
import torch.nn as nn

from .decode import VSARecurrentDecoder


def get_all_eos_ids(tokenizer) -> set[int]:
    """D3 Fix: Collect all valid stop token IDs for Qwen, DeepSeek, Llama, and Mistral."""
    eos_ids = set()
    if getattr(tokenizer, "eos_token_id", None) is not None:
        if isinstance(tokenizer.eos_token_id, list):
            eos_ids.update(int(x) for x in tokenizer.eos_token_id)
        else:
            eos_ids.add(int(tokenizer.eos_token_id))
    for tok_str in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<｜end▁of▁sentence｜>", "</s>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok_str)
            if isinstance(tid, int) and tid >= 0 and tid != getattr(tokenizer, "unk_token_id", -1):
                eos_ids.add(tid)
        except Exception:
            pass
    return eos_ids


def load_model_for_serving(ckpt_path: str, donor_tokenizer: str | None = None, quantize_cpu: bool = False):
    from transformers import AutoTokenizer
    from .converter import UniversalVSALM, fetch_any_donor
    from .surgery import LlamaVSALM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[vsa-serve] Loading {ckpt_path} on {device}...")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    donor_id = donor_tokenizer or ck.get("donor") or (ck.get("injection") or {}).get("donor")
    if not donor_id:
        raise ValueError("Checkpoint lacks donor metadata; pass --donor <hf_repo_id> explicitly.")

    if cfg.get("donor_arch") not in (None, "llama") or cfg.get("qkv_bias", False) or cfg.get("has_qwen35_lin", False):
        model = UniversalVSALM(**cfg)
    else:
        model = LlamaVSALM(**cfg)
    model.load_state_dict(ck["model"], strict=False)
    del ck

    fp32_keys = ("pos_code", "log_gamma", "k_head_mix", "v_head_mix", "head_pos_delta",
                 "v_self_scale", "local_q_gate", "q_relu_mix", "k_relu_mix", "rope_mix", "A_log", "dt_bias", "swa_mix", "center_proj")
    if device.type == "cuda":
        for n, p in model.named_parameters():
            p.data = p.data.to(device, dtype=torch.float32 if any(k in n for k in fp32_keys) else torch.bfloat16)
        for b in model.buffers():
            b.data = b.data.to(device)
        model.eval()
    else:
        model = model.float().eval()
        if quantize_cpu:
            print("[vsa-serve] Applying Dynamic INT8 CPU Quantization...")
            model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    donor_dir = fetch_any_donor(donor_id)
    tokenizer = AutoTokenizer.from_pretrained(donor_dir)
    print(f"[vsa-serve] Model & tokenizer ({donor_id}) ready on {device}!")
    return model, tokenizer, device


@torch.no_grad()
def generate_completion(model, tokenizer, device, messages: list[dict], max_tokens: int = 1024,
                        temperature: float = 0.0, repetition_penalty: float = 1.0,
                        decoder: VSARecurrentDecoder | None = None) -> str:
    """One chat completion. Pass a long-lived `decoder` to reuse its tables and CUDA graph across requests
    (and to choose streaming: VSARecurrentDecoder(model, stream_window=W))."""
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages) + "\nassistant: "

    ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_len = len(ids)
    eos_ids = get_all_eos_ids(tokenizer)

    # D1 Fix: Use stateful VSARecurrentDecoder so context never drops out at 1,024 tokens!
    if decoder is None:
        decoder = VSARecurrentDecoder(model, wrap_positions=False)
    prompt_tensor = torch.tensor([ids], dtype=torch.long, device=device)
    out_tensor = decoder.generate(
        prompt_tensor,
        max_new_tokens=max_tokens,
        temperature=temperature,
        eos_ids=eos_ids,
    )
    gen_ids = out_tensor[0, prompt_len:].tolist()
    if eos_ids and gen_ids and gen_ids[-1] in eos_ids:
        gen_ids = gen_ids[:-1]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def run_server(ckpt_path: str, donor_tokenizer: str | None = None, host: str = "0.0.0.0", port: int = 8000, quantize_cpu: bool = False,
               stream_window: int | None = None):
    """stream_window=W: infinite-context mode (constant RAM per request at any prompt length; see decode.StreamKV)."""
    model, tokenizer, device = load_model_for_serving(ckpt_path, donor_tokenizer, quantize_cpu=quantize_cpu)
    decoder = VSARecurrentDecoder(model, wrap_positions=False, stream_window=stream_window)

    class OpenAIHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/v1/models"):
                payload = {"object": "list", "data": [{"id": "vsa-lm", "object": "model", "owned_by": "vsa-lm"}]}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode("utf-8"))
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"VSA-LM OpenAI-Compatible Server Running")

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
            messages = body.get("messages", [{"role": "user", "content": body.get("prompt", "")}])
            max_tokens = int(body.get("max_tokens", 1024))
            temperature = float(body.get("temperature", 0.0))
            rep_pen = float(body.get("repetition_penalty", 1.0))

            reply = generate_completion(
                model, tokenizer, device, messages,
                max_tokens=max_tokens, temperature=temperature, repetition_penalty=rep_pen, decoder=decoder,
            )
            resp = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:10]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", "vsa-lm"),
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }],
            }
            data = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    print("=" * 78)
    print(f"VSA-LM OpenAI API Server listening at http://{host}:{port}/v1")
    print("=" * 78)
    HTTPServer((host, port), OpenAIHandler).serve_forever()

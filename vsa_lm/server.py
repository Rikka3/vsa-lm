"""OpenAI-Compatible Local API Server for VSA-LM (`python -m vsa_lm serve`).

Allows coding agents like `opencode`, `aider`, `Continue.dev`, and `Cline` to
connect directly to any VSA-LM checkpoint at `http://localhost:8000/v1`.
Supports both CUDA GPUs and low-RAM CPUs (e.g. 8GB RAM laptops with no GPU)
via automatic dynamic INT8 weight quantization (`--quantize`).
"""
from __future__ import annotations

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_model_for_serving(ckpt_path: str, donor_tokenizer: str | None = None, quantize_cpu: bool = True):
    from transformers import AutoTokenizer
    from .converter import UniversalVSALM
    from .surgery import LlamaVSALM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[vsa-serve] Loading {ckpt_path} on {device}...")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    donor_id = donor_tokenizer or ck.get("donor") or (ck.get("injection") or {}).get("donor") or "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    if cfg.get("donor_arch") not in (None, "llama") or cfg.get("qkv_bias", False):
        model = UniversalVSALM(**cfg)
    else:
        model = LlamaVSALM(**cfg)
    model.load_state_dict(ck["model"], strict=True)
    del ck

    if device.type == "cuda":
        model = model.to(device, dtype=torch.bfloat16).eval()
    else:
        model = model.float().eval()
        if quantize_cpu:
            print("[vsa-serve] Applying Dynamic INT8 CPU Quantization (cuts RAM by ~50% & speeds up CPU matmuls)...")
            model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    tokenizer = AutoTokenizer.from_pretrained(donor_id)
    print(f"[vsa-serve] Model & tokenizer ({donor_id}) ready on {device}!")
    return model, tokenizer, device


@torch.no_grad()
def generate_completion(model, tokenizer, device, messages: list[dict], max_tokens: int = 512, temperature: float = 0.6) -> str:
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages) + "\nassistant: "

    ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_len = len(ids)
    max_T = model.config.get("max_T", 1024)

    for _ in range(max_tokens):
        win = ids[-max_T:]
        w_len = len(win)
        pad = (-w_len) % 32
        x = torch.tensor([win + ([0] * pad if pad else [])], dtype=torch.long, device=device)
        with torch.autocast("cuda", torch.bfloat16, enabled=(device.type == "cuda")):
            h = model(x)
            logits = model.lm_head(h[0, w_len - 1]).float()

        if len(ids) > prompt_len:
            seen = torch.tensor(list(set(ids[-64:])), device=device, dtype=torch.long)
            s_vals = logits[seen]
            logits[seen] = torch.where(s_vals > 0, s_vals / 1.08, s_vals * 1.08)

        if temperature <= 0.01:
            nxt = int(logits.argmax(dim=-1).item())
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            nxt = int(torch.multinomial(probs, num_samples=1).item())

        ids.append(nxt)
        if nxt == tokenizer.eos_token_id:
            break

    return tokenizer.decode(ids[prompt_len:], skip_special_tokens=True).strip()


def run_server(ckpt_path: str, donor_tokenizer: str | None = None, host: str = "0.0.0.0", port: int = 8000, quantize_cpu: bool = True):
    model, tokenizer, device = load_model_for_serving(ckpt_path, donor_tokenizer, quantize_cpu=quantize_cpu)

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
            max_tokens = int(body.get("max_tokens", 512))
            temperature = float(body.get("temperature", 0.6))

            reply = generate_completion(model, tokenizer, device, messages, max_tokens=max_tokens, temperature=temperature)
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
    print("  Connect opencode / aider / Continue.dev to:")
    print(f"    Base URL : http://localhost:{port}/v1")
    print("    Model ID : vsa-lm")
    print("=" * 78)
    HTTPServer((host, port), OpenAIHandler).serve_forever()

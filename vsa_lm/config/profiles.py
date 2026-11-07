"""Donor profiles: the settings that depend on what kind of model the donor is, not on VSA itself.

  reasoning  chat template with <think> (DeepSeek-R1 distills, QwQ, ...): long sampled chains
  chat       chat template without <think> (instruct models): shorter replies, the donor's own sampling
  base       no chat template: plain-text prompts, corpus-heavy healing mix

The profile is picked from the donor's tokenizer (`detect_profile`); `--profile` overrides it.
Stage-3 prompts, sampling, the healing data mix, diagnose and hubcheck all read it from here.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Profile:
    name: str
    temperature: float
    top_p: float
    max_new: int
    # Healing data mix (fractions of the token budget): CoT/math, code, chat. Must sum to 1.
    mix_cot: float
    mix_code: float
    mix_chat: float
    # Stage-3 prompt sources besides GSM8K train: "stratos" (CoT user turns) or "ultrachat" (chat user turns).
    stage3_extra: str | None
    use_chat_template: bool


PROFILES: dict[str, Profile] = {
    # The settings every R1-distill run so far used (DeepSeek's recommended sampling).
    "reasoning": Profile("reasoning", temperature=0.6, top_p=0.95, max_new=2048,
                         mix_cot=0.40, mix_code=0.35, mix_chat=0.25, stage3_extra="stratos", use_chat_template=True),
    "chat": Profile("chat", temperature=0.7, top_p=0.9, max_new=1024,
                    mix_cot=0.15, mix_code=0.30, mix_chat=0.55, stage3_extra="ultrachat", use_chat_template=True),
    "base": Profile("base", temperature=0.7, top_p=0.95, max_new=512,
                    mix_cot=0.10, mix_code=0.30, mix_chat=0.60, stage3_extra=None, use_chat_template=False),
}


def detect_profile_name(tok) -> str:
    template = getattr(tok, "chat_template", None) or ""
    if isinstance(template, dict):  # some tokenizers ship several named templates
        template = " ".join(str(v) for v in template.values())
    if not template:
        return "base"
    return "reasoning" if "<think>" in template else "chat"


def _donor_sampling(donor_dir: str | None) -> dict:
    """temperature / top_p from the donor's generation_config.json, when it sets them."""
    if not donor_dir:
        return {}
    path = os.path.join(donor_dir, "generation_config.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            gc = json.load(f)
    except (OSError, ValueError):
        return {}
    out = {}
    for key in ("temperature", "top_p"):
        val = gc.get(key)
        if isinstance(val, (int, float)) and val > 0:
            out[key] = float(val)
    return out


def resolve_profile(tok, name: str | None = None, donor_dir: str | None = None) -> Profile:
    """The named profile, or the one detected from the tokenizer. The donor's own generation_config sampling
    wins over the profile defaults (it is what the donor was tuned and evaluated with)."""
    if name is None:
        name = detect_profile_name(tok)
    if name not in PROFILES:
        raise ValueError(f"Unknown profile {name!r}; choose from {sorted(PROFILES)}")
    return replace(PROFILES[name], **_donor_sampling(donor_dir))


def format_user_prompt(tok, profile: Profile, text: str) -> str:
    """One user turn, ready for generation: the chat template when the profile uses one, plain text otherwise."""
    if profile.use_chat_template and getattr(tok, "chat_template", None):
        return tok.apply_chat_template([{"role": "user", "content": text.strip()}], tokenize=False, add_generation_prompt=True)
    return f"Question: {text.strip()}\nAnswer:"

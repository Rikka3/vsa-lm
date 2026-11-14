"""Declarative per-family model conversion specs for VSA-LM."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelFamilySpec:
    name: str
    swa_window: int = 256
    out_norm_init: float = 0.05
    pin_endpoints: bool = True
    center_unbinding: bool = True
    max_T: int = 2048
    hybrid_every: int = 4


MODEL_FAMILIES: dict[str, ModelFamilySpec] = {
    "qwen3": ModelFamilySpec(name="qwen3", swa_window=256, out_norm_init=0.05, pin_endpoints=True, center_unbinding=True, max_T=2048, hybrid_every=4),
    "qwen2": ModelFamilySpec(name="qwen2", swa_window=128, out_norm_init=0.05, pin_endpoints=True, center_unbinding=True, max_T=2048, hybrid_every=4),
    "llama": ModelFamilySpec(name="llama", swa_window=128, out_norm_init=0.05, pin_endpoints=True, center_unbinding=True, max_T=2048, hybrid_every=4),
    "mistral": ModelFamilySpec(name="mistral", swa_window=128, out_norm_init=0.05, pin_endpoints=True, center_unbinding=True, max_T=2048, hybrid_every=4),
    "qwen3_5": ModelFamilySpec(name="qwen3_5", swa_window=64, out_norm_init=0.05, pin_endpoints=False, center_unbinding=False, max_T=2048, hybrid_every=4),
    "glm": ModelFamilySpec(name="glm", swa_window=256, out_norm_init=0.05, pin_endpoints=True, center_unbinding=True, max_T=2048, hybrid_every=4),
    "deepseek": ModelFamilySpec(name="deepseek", swa_window=256, out_norm_init=0.05, pin_endpoints=True, center_unbinding=True, max_T=2048, hybrid_every=4),
    "default": ModelFamilySpec(name="default", swa_window=128, out_norm_init=0.05, pin_endpoints=True, center_unbinding=True, max_T=2048, hybrid_every=4),
}


def get_model_spec(arch_or_cfg: str | dict) -> ModelFamilySpec:
    arch = str(arch_or_cfg.get("donor_arch", "default") if isinstance(arch_or_cfg, dict) else (arch_or_cfg or "default")).lower()
    if arch in MODEL_FAMILIES:
        return MODEL_FAMILIES[arch]
    for k, spec in MODEL_FAMILIES.items():
        if k != "default" and k in arch:
            return spec
    return MODEL_FAMILIES["default"]

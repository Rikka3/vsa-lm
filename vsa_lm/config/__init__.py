"""Modular model family configurations and attention layers for VSA-LM."""
from .inspector import (
    fetch_any_donor,
    load_all_safetensors,
    inspect_donor_config,
    map_any_hf_to_student,
)
from .models import ModelFamilySpec, MODEL_FAMILIES, get_model_spec
from .qwen35 import (
    Qwen35ZeroCenteredRMSNorm,
    Qwen35RMSNormGated,
    Qwen35PartialRoPE,
    Qwen35VSAAttn,
    preserve_visual_tower,
    reattach_visual_tower,
)
from .standard import (
    UniversalSoftmaxAttn,
    UniversalVSAAttn,
    UniversalBlock,
    UniversalVSALM,
)

__all__ = [
    "fetch_any_donor",
    "load_all_safetensors",
    "inspect_donor_config",
    "map_any_hf_to_student",
    "ModelFamilySpec",
    "MODEL_FAMILIES",
    "get_model_spec",
    "Qwen35ZeroCenteredRMSNorm",
    "Qwen35RMSNormGated",
    "Qwen35PartialRoPE",
    "Qwen35VSAAttn",
    "preserve_visual_tower",
    "reattach_visual_tower",
    "UniversalSoftmaxAttn",
    "UniversalVSAAttn",
    "UniversalBlock",
    "UniversalVSALM",
]

"""VSA-LM: fixed-state linear-attention language models with exact memory editing."""
from .model import VSALanguageModel
from .decode import VSARecurrentDecoder
from .functional import VSAChunkFunction, vsa_parallel_attention
from .surgery import LlamaVSALM, scan_checkpoint
from .train import auto_fit_hardware

__all__ = [
    "VSALanguageModel",
    "LlamaVSALM",
    "VSARecurrentDecoder",
    "VSAChunkFunction",
    "vsa_parallel_attention",
    "auto_fit_hardware",
    "scan_checkpoint",
]

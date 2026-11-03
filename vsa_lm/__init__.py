"""VSA-LM: fixed-state linear-attention language models with exact memory editing."""
from .model import VSALanguageModel
from .decode import VSARecurrentDecoder
from .functional import VSAChunkFunction, vsa_parallel_attention
from .surgery import LlamaVSALM, scan_checkpoint
from .converter import UniversalVSALM, convert_donor_model
from .thinking import OpenMythosEngine, LatentThinkController
from .train import auto_fit_hardware

__all__ = [
    "VSALanguageModel",
    "LlamaVSALM",
    "UniversalVSALM",
    "VSARecurrentDecoder",
    "OpenMythosEngine",
    "LatentThinkController",
    "VSAChunkFunction",
    "vsa_parallel_attention",
    "auto_fit_hardware",
    "convert_donor_model",
    "scan_checkpoint",
]

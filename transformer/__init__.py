"""Decoder-only language model and training entry point."""

from .model import MODEL_SIZES, TransformerConfig, TransformerLM

__all__ = ["MODEL_SIZES", "TransformerConfig", "TransformerLM"]

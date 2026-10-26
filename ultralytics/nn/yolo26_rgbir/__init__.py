"""RGB/IR dynamic fusion modules for YOLO26."""

from .modules import (
    RGBIRDynamicFusion,
    RGBIRDualC2PSA,
    RGBIRDualC3k2,
    RGBIRDualConv,
    RGBIRDualSPPF,
    RGBIRSplit,
    RGBIRStem,
    collect_rgbir_modal_weights,
)

__all__ = (
    "RGBIRDynamicFusion",
    "RGBIRDualC2PSA",
    "RGBIRDualC3k2",
    "RGBIRDualConv",
    "RGBIRDualSPPF",
    "RGBIRSplit",
    "RGBIRStem",
    "collect_rgbir_modal_weights",
)

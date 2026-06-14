"""
vision_tool.py — Backward-compatibility shim.

All tools have been moved to the vision_tools/ package.
This module re-exports everything so existing imports continue to work.
"""

from vision_tools import (  # noqa: F401
    VisionTool,
    TranscriptTool,
    KeyFrameExtractionTool,
    MetadataTool,
    MetadataAnalyzerTool,
    OCRTool,
    DescriptionTool,
    NERTool,
    WeatherDetectionTool,
    GeolocationTool,
    LipSyncDetectionTool,
    AiDetectionTool,
    DeepFakeDetectionTool,
    FacialRecognitionTool,
    ReverseImageSearchTool,
    KnowledgeGraphTool,
)
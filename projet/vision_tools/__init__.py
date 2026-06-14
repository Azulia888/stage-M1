"""
vision_tools — package of individual AFC vision tool modules.

Import from here for convenience:
    from vision_tools import TranscriptTool, DescriptionTool, ...
"""

from vision_tools.base import VisionTool, _StubTool, _make_tool_json, _ollama_post, _ollama_response
from vision_tools.transcript import TranscriptTool
from vision_tools.keyframes import KeyFrameExtractionTool
from vision_tools.metadata import MetadataTool
from vision_tools.metadata_analyzer import MetadataAnalyzerTool
from vision_tools.ocr import OCRTool
from vision_tools.description import DescriptionTool
from vision_tools.ner import NERTool
from vision_tools.weather import WeatherDetectionTool
from vision_tools.geolocation import GeolocationTool
from vision_tools.lipsync import LipSyncDetectionTool
from vision_tools.stubs import AiDetectionTool, DeepFakeDetectionTool, FacialRecognitionTool
from vision_tools.ris import ReverseImageSearchTool
from vision_tools.knowledge_graph import KnowledgeGraphTool


__all__ = [
    "VisionTool",
    "TranscriptTool",
    "KeyFrameExtractionTool",
    "MetadataTool",
    "MetadataAnalyzerTool",
    "OCRTool",
    "DescriptionTool",
    "NERTool",
    "WeatherDetectionTool",
    "GeolocationTool",
    "LipSyncDetectionTool",
    "AiDetectionTool",
    "DeepFakeDetectionTool",
    "FacialRecognitionTool",
    "ReverseImageSearchTool",
    "KnowledgeGraphTool",
]
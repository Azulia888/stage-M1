"""
stubs.py — Stub tools for detectors not yet implemented.
"""

from vision_tools.base import _StubTool


class AiDetectionTool(_StubTool):
    TOOL_NAME = "AI Detection"
    INPUTS = ["Image", "Video"]


class DeepFakeDetectionTool(_StubTool):
    TOOL_NAME = "Deepfake Detection"
    INPUTS = ["Video"]


class FacialRecognitionTool(_StubTool):
    TOOL_NAME = "Facial Recognition"
    INPUTS = ["Image", "Video", "Keyframes"]
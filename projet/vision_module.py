from vision_tool import *
from data_manager import DataManager

class VisionModule:
    data: DataManager
    tools: list[VisionTool]

    def __init__(self):
        self.data = None
        self.tools = [TranscriptTool(), KeyFrameExtractionTool(), MetadataTool(), OCRTool(), DescriptionTool(),
                      DeepFakeDetectionTool(), LipSyncDetectionTool(), AiDetectionTool(), WeatherDetectionTool(),
                      GeolocationTool(), FacialRecognitionTool(), NERTool()]

    def run(self, media_path: str, metadata_path: str, isVideo: bool):
        self.data = DataManager(media_path, metadata_path, isVideo)
        for tool in self.tools:
            tool.run(self.data)
            tool.addData(self.data)
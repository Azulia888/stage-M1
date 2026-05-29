class DataManager:
    originalMedia: str
    keyframes: list[str]
    description: str
    metadata: str
    metadata_path: str
    ocr: str
    transcript: str
    isVideo: bool
    toolResult: dict

    def __init__(self, originalMedia: str, metadata_path: str, isVideo: bool):
        self.originalMedia = originalMedia
        self.metadata_path = metadata_path
        self.isVideo = isVideo
        self.keyframes = []
        self.description = None
        self.metadata = None
        self.ocr = None
        self.transcript = None
        self.toolResult = None

    def addToolResult(self, toolJson: dict):
        if ("ToolName" in toolJson):
            self.toolResult[toolJson["ToolName"]] = toolJson

            if (toolJson["ToolName"] == "Description"):
                self.description = toolJson["Output"]
            if (toolJson["ToolName"] == "Metadata"):
                self.metadata = toolJson["Output"]
            if (toolJson["ToolName"] == "OCR"):
                self.ocr = toolJson["Output"]
            if (toolJson["ToolName"] == "Keyframes"):
                self.keyframes = toolJson["Output"]
            if (toolJson["ToolName"] == "Transcript"):
                self.transcript = toolJson["Output"]

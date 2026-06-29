# AFC — Automatic Fact-Checking (proof of concept)

Explainable-by-design pipeline of vision/audio/text tools that analyse a
video or image and produce a result graph a human can inspect tool by tool.

## 1. Requirements

### Python
Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

### spaCy model (only if you use the "spacy" or "ensemble" NER backend)
```bash
python -m spacy download en_core_web_sm
```

### NLTK data (only if you use the "nltk" or "ensemble" NER backend)
Downloaded automatically on first run by `NERTool`. If automatic download
fails (no internet at runtime), run manually:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('maxent_ne_chunker'); nltk.download('maxent_ne_chunker_tab'); nltk.download('words')"
```

### System tools (not installed via pip)

- **ffmpeg / ffprobe** — required by `MetadataTool` for video metadata.
- **exiftool** — required by `MetadataTool` for EXIF/file metadata.

Install on Debian/Ubuntu:
```bash
sudo apt-get install ffmpeg libimage-exiftool-perl
```
Install on macOS (Homebrew):
```bash
brew install ffmpeg exiftool
```

### Ollama (required for vision/LLM tools)

The project calls a local Ollama server for `Description`, `OCR`,
`Geolocation`, `Weather Detection`, `NER` (llm/ensemble backend),
`Metadata Analyzer`, and `Reverse Image Search` (frame scoring).

1. Install Ollama: https://ollama.com/download
2. Pull a vision-capable model, e.g.:
   ```bash
   ollama pull qwen2.5vl:7b
   ```
3. Start the server (usually automatic after install):
   ```bash
   ollama serve
   ```

Configure via environment variables (defaults shown):
```bash
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_VISION_MODEL=qwen3.5:2b   # set to a model you actually pulled
export OLLAMA_SYNTH_MODEL=$OLLAMA_VISION_MODEL
```

### Whisper (transcription)

`openai-whisper` and `torch` are in requirements.txt. The model size is
controlled by:
```bash
export WHISPER_MODEL=base   # tiny / base / small / medium / large
```
First run downloads the chosen model automatically.

### SyncNet (lip-sync detection — optional)

`LipSyncDetectionTool` requires a separate clone of SyncNet:
```bash
git clone https://github.com/joonson/syncnet_python
cd syncnet_python
sh download_model.sh
cd ..
export SYNCNET_DIR=$(pwd)/syncnet_python
export SYNCNET_MODEL=$SYNCNET_DIR/data/syncnet_v2.model
```
If this is not set up, the tool reports "not implemented/installed" and the
pipeline continues normally.

### Reverse Image Search (optional)

`ReverseImageSearchTool` uses ImgBB (image hosting) and SerpApi (Google
reverse image search). API keys are currently hard-coded in
`vision_tools/ris.py` as a stopgap — replace them with your own keys or move
them to environment variables before any real/shared deployment:
```python
IMGBB_API_KEY = "..."
SERPAPI_KEY = "..."
```

### Knowledge Graph (optional)

`KnowledgeGraphTool` only uses `requests` (already in requirements.txt) to
query the public Wikidata API — no extra setup needed, just internet access.

## 2. Project structure

```
projet/
  data_manager.py        # DataManager: shared state passed between tools
  vision_module.py        # VisionModule: orchestrates the tool pipeline
  vision_tool.py           # backward-compat shim re-exporting vision_tools/*
  vision_tools/            # one module per tool
  pipeline_timer.py        # shared elapsed-time clock for a pipeline run
  exports/                 # auto-saved console logs (and manual .pkl exports)
  gui/                     # PySide6 desktop GUI
    main.py
    launch_window.py
    run_window.py
    graph_window.py
    graph_model.py
    detail_dialog.py
    pipeline_worker.py
```

## 3. Running the GUI

```bash
cd projet
python gui/main.py
```

1. Choose a source: a URL (downloaded via yt-dlp) or a local file.
2. Choose the media type: Video or Image.
3. Click "Run analysis". A console log streams tool output live, each line
   timestamped with the time elapsed since the pipeline launched; each tool
   reports its own run time once it finishes.
4. When the run ends, the total elapsed time is shown in the top-right
   corner, and the full console output is saved to
   `exports/console_log.txt`.
5. A graph then opens: one node per tool, colour-coded by confidence. Click
   any node for its explanation, confidence, and raw output.

## 4. Running the pipeline without the GUI

```python
from vision_module import VisionModule

module = VisionModule()

# Local file
module.run("/path/to/video.mp4", metadata_path="", isVideo=True)

# Or from a URL (downloads with yt-dlp first)
module.runURL("https://www.youtube.com/watch?v=...", isVideo=True)

for tool_name, result in module.data.toolResult.items():
    print(tool_name, result)
```

## 5. Notes

- Tools that fail or are not configured return `hasRun: 0` rather than
  raising, so the pipeline always completes and the graph shows which tools
  were skipped and why.
- `AiDetectionTool`, `DeepFakeDetectionTool`, and `FacialRecognitionTool` are
  stubs (`Not implemented.`) and are placeholders for future work.
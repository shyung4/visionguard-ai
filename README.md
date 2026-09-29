# VisionGuard AI

VisionGuard AI is an event-driven video analytics system for real-time object detection, multi-object tracking, ROI-based intrusion detection, and automated event capture.

The current version uses YOLO26, ByteTrack, and OpenCV to detect and track objects in video, monitor user-defined regions of interest, and save structured event data with contextual frames.

## Features

- Real-time object detection with YOLO26
- Multi-object tracking with ByteTrack
- Interactive ROI selection
- Restricted-zone intrusion detection
- Track-based event triggering
- Clean event keyframe capture
- Annotated event keyframe capture
- Before/after context frame capture
- Structured event metadata saved as JSON

## Pipeline

Video Input  
→ YOLO26 Detection  
→ ByteTrack Tracking  
→ ROI Event Detection  
→ Event Metadata Generation  
→ Context Frame Capture

## Project Structure

```text
visionguard-ai/
├── src/
│   ├── detector.py
│   ├── event_engine.py
│   ├── event_writer.py
│   ├── roi_selector.py
│   └── vlm_analyzer.py
│
├── configs/
│   ├── bytetrack.yaml
│   └── roi.json
│
├── assets/
│   └── screenshots/
│
├── README.md
├── requirements.txt
└── .gitignore
from pathlib import Path
import json


class VLMAnalyzer:
    def __init__(self):
        pass

    def load_event_data(self, event_dir):
        event_dir = Path(event_dir)

        event_json_path = event_dir / "event.json"

        with open(
            event_json_path,
            "r",
            encoding="utf-8",
        ) as file:
            event_data = json.load(file)

        return event_data

    def build_prompt(self, event_data):
        prompt = f"""
You are analyzing a video surveillance event.

Event metadata:
- Event type: {event_data["event_type"]}
- Object class: {event_data["class_name"]}
- Track ID: {event_data["track_id"]}
- Detection confidence: {event_data["confidence"]:.3f}
- Timestamp: {event_data["timestamp_sec"]:.2f} seconds

You are given three frames:
1. Before the event
2. Event keyframe
3. After the event

Analyze only what is visually supported.

Return:
- whether the detected event appears valid
- a concise description of what happened
- visible evidence supporting the conclusion
- relevant uncertainty
- a risk level: low, medium, or high
- a recommended review action
"""

        return prompt

    def analyze(self, event_dir):
        event_dir = Path(event_dir)

        event_data = self.load_event_data(event_dir)
        prompt = self.build_prompt(event_data)

        before_path = event_dir / "before.jpg"
        keyframe_path = event_dir / "keyframe.jpg"
        after_path = event_dir / "after.jpg"

        print("\n--- VLM INPUT ---")
        print(f"Event directory: {event_dir}")
        print(f"Before: {before_path}")
        print(f"Keyframe: {keyframe_path}")
        print(f"After: {after_path}")
        print("\nPrompt:")
        print(prompt)

        # Actual VLM inference comes next.

        return {
            "event_data": event_data,
            "prompt": prompt,
        }
    
if __name__ == "__main__":
    analyzer = VLMAnalyzer()

    analyzer.analyze(
        "outputs/events/traffic_540_32fps/event_0001"
    )
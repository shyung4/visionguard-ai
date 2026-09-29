from pathlib import Path
from typing import Literal
import base64
import json
import mimetypes

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from utils import find_video

load_dotenv()


class VLMAnalysis(BaseModel):
    event_valid: bool

    description: str

    visual_evidence: list[str]

    uncertainties: list[str]

    risk_level: Literal[
        "low",
        "medium",
        "high"
    ]

    recommended_action: str


class VLMAnalyzer:
    def __init__(
        self,
        model="gpt-5.6-luna",
    ):
        self.client = OpenAI()
        self.model = model

    def load_event_data(
        self,
        event_dir,
    ):
        event_dir = Path(event_dir)

        event_json_path = (
            event_dir
            / "event.json"
        )

        with open(
            event_json_path,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    def encode_image(
        self,
        image_path,
    ):
        image_path = Path(image_path)

        mime_type, _ = mimetypes.guess_type(
            image_path
        )

        if mime_type is None:
            mime_type = "image/jpeg"

        with open(
            image_path,
            "rb",
        ) as file:
            image_bytes = (
                base64.b64encode(
                    file.read()
                )
                .decode("utf-8")
            )

        return (
            f"data:{mime_type};"
            f"base64,{image_bytes}"
        )

    def build_prompt(
        self,
        event_data,
    ):
        return f"""
    You are analyzing a video surveillance event.

    You are given three chronological images:

    1. BEFORE frame:
    A clean frame captured approximately one second before the event.

    2. EVENT KEYFRAME:
    An annotated frame showing the restricted ROI,
    tracked object, bounding box, and event information.

    3. AFTER frame:
    A clean frame captured approximately one second after the event.

    Computer vision metadata:

    Event type:
    {event_data["event_type"]}

    Tracked object:
    {event_data["class_name"]}

    Track ID:
    {event_data["track_id"]}

    Detection confidence:
    {event_data["confidence"]:.3f}

    Event timestamp:
    {event_data["timestamp_sec"]:.2f} seconds

    Determine whether the computer vision event is visually supported.

    Important rules:

    - Use only visually observable evidence.
    - Treat the CV metadata as a hypothesis, not ground truth.
    - Use the annotated keyframe to understand the ROI boundary.
    - Compare object position across all three frames.
    - Do not infer identity, intent, or events not visible in the images.
    - Explicitly describe uncertainty when evidence is ambiguous.

    Assess:

    1. Whether the event appears valid.
    2. What visibly happened.
    3. Evidence supporting the conclusion.
    4. Any uncertainty or limitations.
    5. Risk level.
    6. Appropriate review action.
    """

    def analyze(
        self,
        event_dir,
    ):
        event_dir = Path(event_dir)

        event_data = (
            self.load_event_data(
                event_dir
            )
        )

        before_path = (
            event_dir
            / "before.jpg"
        )

        keyframe_path = (
            event_dir
            / "keyframe_annotated.jpg"
        )

        after_path = (
            event_dir
            / "after.jpg"
        )

        required_paths = [
            before_path,
            keyframe_path,
            after_path,
        ]

        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing image: {path}"
                )

        prompt = self.build_prompt(
            event_data
        )

        before_image = self.encode_image(
            before_path
        )

        keyframe_image = self.encode_image(
            keyframe_path
        )

        after_image = self.encode_image(
            after_path
        )

        response = (
            self.client.responses.parse(
                model=self.model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": prompt,
                            },
                            {
                                "type": "input_image",
                                "image_url":
                                    before_image,
                            },
                            {
                                "type": "input_image",
                                "image_url":
                                    keyframe_image,
                            },
                            {
                                "type": "input_image",
                                "image_url":
                                    after_image,
                            },
                        ],
                    }
                ],
                text_format=VLMAnalysis,
            )
        )

        analysis = (
            response.output_parsed
        )

        if analysis is None:
            raise RuntimeError(
                "Model returned no "
                "structured analysis."
            )

        output_path = (
            event_dir
            / "analysis.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                analysis.model_dump(),
                file,
                indent=4,
                ensure_ascii=False,
            )

        print(
            "\nVLM analysis complete"
        )

        print(
            json.dumps(
                analysis.model_dump(),
                indent=4,
                ensure_ascii=False,
            )
        )

        print(
            f"\nSaved to: "
            f"{output_path}"
        )

        return analysis

    def analyze_all(
        self,
        events_dir,
        skip_existing=True,
    ):
        events_dir = Path(events_dir)

        event_dirs = sorted(
            path
            for path in events_dir.iterdir()
            if path.is_dir()
            and path.name.startswith("event_")
        )

        print(
            f"\nFound {len(event_dirs)} events."
        )

        results = []

        for event_dir in event_dirs:

            analysis_path = (
                event_dir
                / "analysis.json"
            )

            if (
                skip_existing
                and analysis_path.exists()
            ):
                print(
                    f"Skipping {event_dir.name}: "
                    f"analysis already exists."
                )
                continue

            print(
                f"\nAnalyzing "
                f"{event_dir.name}..."
            )

            try:
                analysis = self.analyze(
                    event_dir
                )

                results.append(
                    {
                        "event_dir": event_dir,
                        "analysis": analysis,
                    }
                )

            except Exception as error:
                print(
                    f"Failed to analyze "
                    f"{event_dir.name}: {error}"
                )

        print(
            "\nBatch analysis complete."
        )

        return results


if __name__ == "__main__":
    analyzer = VLMAnalyzer()

    video_path = find_video()
    events_dir = (
        Path("outputs")
        / "events"
        / video_path.stem
    )

    analyzer.analyze_all(
        events_dir
    )
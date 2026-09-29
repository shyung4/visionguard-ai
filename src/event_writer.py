from pathlib import Path
import json

import cv2


class EventWriter:
    def __init__(self, output_root, video_name):
        self.output_root = Path(output_root)
        self.video_name = Path(video_name).stem

        self.video_output_dir = (
            self.output_root
            / "events"
            / self.video_name
        )

        self.video_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.event_counter = 0

    def create_event(
        self,
        event,
        before_frame,
        keyframe,
        annotated_frame=None,
    ):
        self.event_counter += 1

        event_dir = (
            self.video_output_dir
            / f"event_{self.event_counter:04d}"
        )

        event_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        event_data = event.to_dict()

        with open(
            event_dir / "event.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                event_data,
                file,
                indent=4,
            )

        if before_frame is not None:
            cv2.imwrite(
                str(event_dir / "before.jpg"),
                before_frame,
            )

        # Clean image for VLM input
        cv2.imwrite(
            str(event_dir / "keyframe.jpg"),
            keyframe,
        )

        # Annotated image for visualization / README
        if annotated_frame is not None:
            cv2.imwrite(
                str(event_dir / "keyframe_annotated.jpg"),
                annotated_frame,
            )

        print(
            f"Created event {self.event_counter:04d} "
            f"at {event_dir}"
        )

        return {
            "event_dir": event_dir,
            "event_number": self.event_counter,
        }

    def save_after_frame(
        self,
        event_dir,
        after_frame,
    ):
        cv2.imwrite(
            str(event_dir / "after.jpg"),
            after_frame,
        )

        print(
            f"Saved after frame: {event_dir}"
        )
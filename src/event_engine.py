from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass
class Event:
    event_type: str
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    frame_index: int
    timestamp_sec: float
    x: float
    y: float

    def to_dict(self):
        return asdict(self)


class EventEngine:
    def __init__(self, roi_polygon):
        """
        roi_polygon:
            [(x1, y1), (x2, y2), (x3, y3), ...]
        """

        self.roi_polygon = np.array(
            roi_polygon,
            dtype=np.int32
        ).reshape((-1, 1, 2))

        # Previous inside/outside state for each tracked object
        self.previous_state = {}

    def get_bottom_center(self, box):
        """
        box:
            [x1, y1, x2, y2]
        """

        x1, y1, x2, y2 = box

        center_x = (x1 + x2) / 2
        bottom_y = y2

        return center_x, bottom_y

    def is_inside_roi(self, point):
        """
        Returns True if point is inside or on the ROI boundary.
        """

        x, y = point

        result = cv2.pointPolygonTest(
            self.roi_polygon,
            (float(x), float(y)),
            False
        )

        return result >= 0

    def update(
        self,
        track_id,
        class_id,
        class_name,
        confidence,
        box,
        frame_index,
        fps
    ):
        """
        Update one tracked object.

        Returns:
            Event if the object enters the ROI.
            None otherwise.
        """

        point = self.get_bottom_center(box)

        inside_now = self.is_inside_roi(point)

        inside_before = self.previous_state.get(
            track_id,
            False
        )

        # Save current state
        self.previous_state[track_id] = inside_now

        # Event occurs only when:
        # outside -> inside
        if inside_now and not inside_before:

            timestamp_sec = frame_index / fps

            event = Event(
                event_type="restricted_zone_intrusion",
                track_id=int(track_id),
                class_id=int(class_id),
                class_name=class_name,
                confidence=float(confidence),
                frame_index=int(frame_index),
                timestamp_sec=float(timestamp_sec),
                x=float(point[0]),
                y=float(point[1])
            )

            return event

        return None
from pathlib import Path
import cv2, json
from collections import deque
from ultralytics import YOLO
from event_engine import EventEngine
from event_writer import EventWriter
from utils import find_video
import shutil

roi_path = Path("configs") / "roi.json"

def load_roi(path, expected_video):
    with open(path, "r") as file:
        data = json.load(file)

    roi_video = data.get("video")

    if roi_video != expected_video:
        raise ValueError(
            f"ROI was created for '{roi_video}', "
            f"but current video is '{expected_video}'."
        )

    return data["roi_polygon"]

def run_detection():
    
    video_path = find_video()
    video = video_path.name
    output_name = video_path.stem
    model = YOLO("yolo26n.pt")
    roi_polygon = load_roi(
    roi_path,
    expected_video=video,
    )
    event_engine = EventEngine(roi_polygon)

    events_dir = (
    Path("outputs")
    / "events"
    / video_path.stem
    )

    if events_dir.exists():
        shutil.rmtree(events_dir)

    event_writer = EventWriter(
    output_root="outputs",
    video_name=video,
    )

    cap = cv2.VideoCapture(str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    print(f"Video: {video}")
    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps:.2f}")
    print(f"ROI: {roi_polygon}")

    buffer_size = max(int(fps), 1)
    frame_buffer = deque(
    maxlen=buffer_size
    )
    
    after_delay_frames = max(
    int(fps),
    1
    )
    pending_events = []

    results = model.track(
        source=str(video_path),
        tracker="configs/bytetrack.yaml",
        persist=True,
        save=True,
        classes=[0, 2, 5, 7],  # 0: person, 2: car, 5: bus, 7: truck
        conf=0.5,             # minimum detection confidence
        iou=0.7,               # IoU threshold
        # max_det=300,           # maximum detections per frame
        # agnostic_nms=False,    # class-agnostic NMS
        # nms=False,
        # imgsz=640,             # inference image size
        # rect=True,             # minimum rectangular padding
        vid_stride=2,          # process every Nth frame
        # stream_buffer=False,   # buffer frames for live streams
        # batch=1,               # inference batch size
        device=0,
        # quantize=16,           # FP16 inference (newer replacement for half=True)
        # compile=False,         # torch.compile optimization
        # dnn=False,             # OpenCV DNN backend for ONNX
        # augment=False,         # test-time augmentation
        # visualize=False,       # save activation visualization
        project="outputs/tracking",
        name=output_name,
        exist_ok=True,
        # show=True,            # display video while processing
        verbose=False,          # print inference logs
        stream=True,          # return generator instead of storing all results
        # retina_masks=False,    # high-resolution masks (segmentation models)
        # embed=None,            # extract feature embeddings
    )

    for frame_index, result in enumerate(results):

        # frame = result.orig_img.copy()

        raw_frame = result.orig_img.copy()
        display_frame = raw_frame.copy()
        
        frame_buffer.append(
        raw_frame.copy()
        )

        # Draw ROI
        roi_points = event_engine.roi_polygon

        cv2.polylines(
            display_frame,
            [roi_points],
            isClosed=True,
            color=(0, 0, 255),
            thickness=3,
        )

        cv2.putText(
            display_frame,
            "RESTRICTED ZONE",
            tuple(roi_points[0][0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        # No tracked objects in this frame
        if result.boxes.id is None:
            cv2.imshow("VisionGuard AI", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for box, track_id, class_id, confidence in zip(
            boxes,
            track_ids,
            class_ids,
            confidences,
        ):
            class_id = int(class_id)
            track_id = int(track_id)

            class_name = result.names[class_id]

            event = event_engine.update(
                track_id=track_id,
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                box=box,
                frame_index=frame_index,
                fps=fps,
            )

            x1, y1, x2, y2 = map(int, box)

            center_x, bottom_y = event_engine.get_bottom_center(box)

            # Draw bounding box
            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            # Draw tracking ID
            label = f"{class_name} #{track_id} {confidence:.2f}"

            cv2.putText(
                display_frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # Draw bottom-center point
            cv2.circle(
                display_frame,
                (int(center_x), int(bottom_y)),
                5,
                (255, 0, 0),
                -1,
            )

            if event is not None:

                cv2.putText(
                    display_frame,
                    f"EVENT: {class_name} #{track_id} entered ROI",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3,
                )

                before_frame = None

                if len(frame_buffer) > 1:
                    before_frame = (
                        frame_buffer[0].copy()
                    )

                saved_event = (
                    event_writer.create_event(
                        event=event,
                        before_frame=before_frame,
                        keyframe=raw_frame.copy(),  # clean
                        annotated_frame=display_frame.copy(),    # annotated
                    )
                )

                pending_events.append(
                    {
                        "event_dir":
                            saved_event["event_dir"],

                        "target_frame":
                            frame_index
                            + after_delay_frames,
                    }
                )



            completed_events = []

        for pending in pending_events:
            if frame_index >= pending["target_frame"]:
                event_writer.save_after_frame(
                    event_dir=pending["event_dir"],
                    after_frame=raw_frame.copy(),
                )

                completed_events.append(pending)

        for completed in completed_events:
            pending_events.remove(completed)


        cv2.imshow("VisionGuard AI", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

    return (
    Path("outputs")
    / "events"
    / video_path.stem
)

if __name__ == "__main__":
    run_detection()
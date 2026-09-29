from pathlib import Path
import json

import cv2


video = "traffic_540_32fps.mp4"

video_path = Path("data/sample") / video
output_path = Path("configs") / "roi.json"

points = []


def mouse_callback(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Added point: ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            removed = points.pop()
            print(f"Removed point: {removed}")


def draw_points(frame):
    preview = frame.copy()

    for i, point in enumerate(points):
        cv2.circle(
            preview,
            point,
            6,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            preview,
            str(i + 1),
            (point[0] + 8, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    if len(points) >= 2:
        for i in range(len(points) - 1):
            cv2.line(
                preview,
                points[i],
                points[i + 1],
                (0, 0, 255),
                2,
            )

    if len(points) >= 3:
        cv2.line(
            preview,
            points[-1],
            points[0],
            (0, 0, 255),
            2,
        )

    return preview


def save_roi():
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    roi_data = {
        "video": video,
        "roi_polygon": points,
    }

    with open(output_path, "w") as file:
        json.dump(
            roi_data,
            file,
            indent=4,
        )

    print(f"\nROI saved to: {output_path}")
    print(f"ROI points: {points}")


def main():
    cap = cv2.VideoCapture(str(video_path))

    success, frame = cap.read()
    cap.release()

    if not success:
        raise RuntimeError(
            f"Could not read video: {video_path}"
        )

    window_name = "ROI Selector"

    cv2.namedWindow(window_name)

    cv2.setMouseCallback(
        window_name,
        mouse_callback,
    )

    print("Controls:")
    print("Left click  : add point")
    print("Right click : remove last point")
    print("R           : reset")
    print("Enter       : save ROI")
    print("Q / Esc     : quit")

    while True:
        preview = draw_points(frame)

        cv2.imshow(
            window_name,
            preview,
        )

        key = cv2.waitKey(20) & 0xFF

        if key == ord("r"):
            points.clear()
            print("ROI reset")

        elif key == 13:  # Enter
            if len(points) < 3:
                print(
                    "ROI needs at least 3 points."
                )
                continue

            save_roi()
            break

        elif key in (
            ord("q"),
            27,
        ):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
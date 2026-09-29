from pathlib import Path
import json
import sys

import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas

import streamlit as st


# Allow imports from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

def get_first_frame(video_path):
    cap = cv2.VideoCapture(str(video_path))

    success, frame = cap.read()

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    cap.release()

    if not success:
        raise RuntimeError(
            f"Could not read video: {video_path}"
        )

    # OpenCV BGR -> RGB
    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    image = Image.fromarray(
        frame_rgb
    )

    return image, width, height

sys.path.insert(0, str(SRC_DIR))


from utils import find_video
from pipeline import run_pipeline


st.set_page_config(
    page_title="VisionGuard AI",
    page_icon="🎥",
    layout="wide",
)


st.title("VisionGuard AI")

st.caption(
    "Event-driven video analytics with "
    "YOLO26, ByteTrack, and multimodal AI."
)


# -------------------------------------------------
# Video
# -------------------------------------------------

st.header("1. Video")

try:
    video_path = find_video()
except Exception as error:
    st.error(str(error))
    st.stop()


st.write(f"**Video:** `{video_path.name}`")

st.video(str(video_path))


# -------------------------------------------------
# ROI
# -------------------------------------------------

# -------------------------------------------------
# ROI
# -------------------------------------------------

st.header("2. Region of Interest")

roi_path = (
    PROJECT_ROOT
    / "configs"
    / "roi.json"
)

roi_valid = False
roi_data = None


# -----------------------------------------
# Check existing ROI
# -----------------------------------------

if roi_path.exists():

    try:
        with open(
            roi_path,
            "r",
            encoding="utf-8",
        ) as file:
            roi_data = json.load(file)

        if (
            roi_data.get("video")
            == video_path.name
        ):
            roi_valid = True

    except Exception as error:
        st.warning(
            f"Could not read ROI: {error}"
        )


# -----------------------------------------
# Existing ROI
# -----------------------------------------

if roi_valid:

    st.success(
        "ROI configuration found for this video."
    )

    with st.expander(
        "View current ROI"
    ):
        st.json(roi_data)


# -----------------------------------------
# ROI editor
# -----------------------------------------

show_roi_editor = st.checkbox(
    "Configure / change ROI",
    value=not roi_valid,
)


if show_roi_editor:

    st.write(
    "Left-click to add polygon points. "
    "Click the first point again to close the polygon."
    )

    try:
        (
            first_frame,
            video_width,
            video_height,
        ) = get_first_frame(
            video_path
        )

    except Exception as error:
        st.error(str(error))
        st.stop()


    # -------------------------------------
    # Canvas scaling
    # -------------------------------------

    MAX_CANVAS_WIDTH = 700

    scale = min(
        1.0,
        MAX_CANVAS_WIDTH
        / video_width,
    )

    canvas_width = int(
        video_width * scale
    )

    canvas_height = int(
        video_height * scale
    )


    st.caption(
        f"Video resolution: "
        f"{video_width} × {video_height}"
    )


    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.20)",
        stroke_width=3,
        stroke_color="#FF0000",

        background_image=first_frame,

        drawing_mode="polygon",

        update_streamlit=True,

        height=canvas_height,
        width=canvas_width,

        key="roi_canvas",
    )

    st.write("Canvas JSON:")
    st.json(canvas_result.json_data or {})

    # -------------------------------------
    # Read polygon
    # -------------------------------------

    polygon_points = None

    if (
        canvas_result.json_data
        and canvas_result.json_data.get("objects")
    ):
        objects = canvas_result.json_data["objects"]

        polygons = [
            obj
            for obj in objects
            if obj.get("type", "").lower() == "polygon"
        ]

        if polygons:
            polygon = polygons[-1]

            canvas_points = polygon.get(
                "points",
                []
            )

            if len(canvas_points) >= 3:
                polygon_points = [
                    [
                        int(point["x"] / scale),
                        int(point["y"] / scale),
                    ]
                    for point in canvas_points
                ]

    # -------------------------------------
    # Save ROI
    # -------------------------------------

    if polygon_points:

        st.write(
            "**Selected ROI coordinates:**"
        )

        st.write(
            polygon_points
        )

        if st.button(
            "Save ROI",
            type="primary",
        ):

            roi_data = {
                "video":
                    video_path.name,

                "resolution": {
                    "width":
                        video_width,

                    "height":
                        video_height,
                },

                "roi_polygon":
                    polygon_points,
            }

            roi_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with open(
                roi_path,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    roi_data,
                    file,
                    indent=4,
                )

            st.success(
                "ROI saved successfully."
            )

            st.rerun()

    else:

        st.info(
            "Draw and close a polygon "
            "to enable ROI saving."
        )

roi_path = (
    PROJECT_ROOT
    / "configs"
    / "roi.json"
)


roi_valid = False
roi_data = None


if roi_path.exists():

    try:
        with open(
            roi_path,
            "r",
            encoding="utf-8",
        ) as file:
            roi_data = json.load(file)

        roi_video = roi_data.get("video")

        if roi_video == video_path.name:
            roi_valid = True

    except Exception as error:
        st.warning(
            f"Could not read ROI configuration: {error}"
        )


if roi_valid:

    st.success(
        "ROI configuration found for this video."
    )

    with st.expander(
        "View ROI coordinates"
    ):
        st.json(roi_data)

else:

    if roi_path.exists():
        st.warning(
            "An ROI configuration exists, "
            "but it belongs to a different video."
        )

    else:
        st.warning(
            "No ROI has been configured yet."
        )

    st.info(
        "ROI configuration is required "
        "before video analysis can run."
    )


# -------------------------------------------------
# Pipeline
# -------------------------------------------------

st.header("3. Analysis")


if not roi_valid:

    st.button(
        "Run Analysis",
        disabled=True,
    )

    st.caption(
        "Configure an ROI before running analysis."
    )

else:

    if st.button(
        "Run Analysis",
        type="primary",
    ):

        with st.spinner(
            "Running VisionGuard pipeline..."
        ):

            try:

                events_dir = run_pipeline()

                st.success(
                    "Analysis complete."
                )

                st.session_state[
                    "events_dir"
                ] = str(events_dir)

            except Exception as error:

                st.exception(error)


# -------------------------------------------------
# Results
# -------------------------------------------------

st.header("4. Events")


events_dir = st.session_state.get(
    "events_dir"
)


if events_dir:

    events_dir = Path(events_dir)

elif roi_valid:

    possible_dir = (
        PROJECT_ROOT
        / "outputs"
        / "events"
        / video_path.stem
    )

    if possible_dir.exists():
        events_dir = possible_dir


if events_dir and Path(events_dir).exists():

    event_dirs = sorted(
        path
        for path in Path(events_dir).iterdir()
        if path.is_dir()
        and path.name.startswith("event_")
    )

    st.write(
        f"**Detected events:** {len(event_dirs)}"
    )

    for event_dir in event_dirs:

        st.subheader(
            event_dir.name
        )

        col1, col2 = st.columns(
            [1, 1]
        )

        annotated_image = (
            event_dir
            / "keyframe_annotated.jpg"
        )

        analysis_path = (
            event_dir
            / "analysis.json"
        )

        with col1:

            if annotated_image.exists():

                st.image(
                    str(annotated_image),
                    caption="Detected event",
                    use_container_width=True,
                )

        with col2:

            if analysis_path.exists():

                with open(
                    analysis_path,
                    "r",
                    encoding="utf-8",
                ) as file:
                    analysis = json.load(file)

                st.write(
                    "**AI Assessment**"
                )

                st.write(
                    analysis.get(
                        "description",
                        ""
                    )
                )

                st.write(
                    "**Valid Event:**",
                    analysis.get(
                        "event_valid"
                    ),
                )

                st.write(
                    "**Risk Level:**",
                    analysis.get(
                        "risk_level"
                    ),
                )

                st.write(
                    "**Recommended Action:**"
                )

                st.write(
                    analysis.get(
                        "recommended_action",
                        ""
                    )
                )

                with st.expander(
                    "Full AI Analysis"
                ):
                    st.json(
                        analysis
                    )

            else:

                st.info(
                    "VLM analysis not available."
                )

        st.divider()

else:

    st.info(
        "No event results are available yet."
    )
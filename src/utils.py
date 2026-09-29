from pathlib import Path


VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
}


def find_video(
    video_dir="data/sample",
):
    video_dir = Path(video_dir)

    video_files = sorted(
        path
        for path in video_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not video_files:
        raise FileNotFoundError(
            f"No video file found in: {video_dir}"
        )

    if len(video_files) > 1:
        raise RuntimeError(
            f"Multiple video files found in {video_dir}: "
            f"{[path.name for path in video_files]}"
        )

    return video_files[0]
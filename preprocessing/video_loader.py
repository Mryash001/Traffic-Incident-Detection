import cv2

from preprocessing.video_preprocessing import preprocess_frame


def load_video(
    video_path,
    window_seconds=3,
    target_fps=25
):

    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    original_fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    if original_fps <= 0:
        original_fps = target_fps

    required_frames = int(
        window_seconds * target_fps
    )

    frames = []

    for i in range(required_frames):

        timestamp_ms = (
            i / target_fps
        ) * 1000

        capture.set(
            cv2.CAP_PROP_POS_MSEC,
            timestamp_ms
        )

        success, frame = capture.read()

        if not success:
            break

        frame = preprocess_frame(frame)

        frames.append(frame)

    capture.release()

    return frames, original_fps
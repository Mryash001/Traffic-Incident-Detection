from preprocessing.video_loader import load_video


video_path = "data/raw/videos/traffic.mp4"


frames, fps = load_video(
    video_path,
    window_seconds=3,
    target_fps=25
)


print("Original FPS:", fps)
print("Frames extracted:", len(frames))

if len(frames) > 0:
    print("First frame shape:", frames[0].shape)
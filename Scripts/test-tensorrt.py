r"""
Run YOLO TensorRT segmentation inference on a video file.

This script:
1. Opens an input video with OpenCV
2. Runs YOLO predictions frame-by-frame
3. Displays the annotated video live
4. Saves an annotated output video

Run from project root:
    python .\Scripts\test-tensorrt.py

Controls:
    q = quit early
    space = pause/resume
"""

from pathlib import Path
import time

import cv2
from ultralytics import YOLO


# --------------------------------------------------
# Paths
# --------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

model_path = (
    PROJECT_ROOT
    / "Models"
    / "Combined Segmentation"
    / "yolov8s"
    / "1. 100 Epochs 640x640"
    / "best.engine"
)

video_path = (
    PROJECT_ROOT
    / "Videos"
    / "example.mp4"
)

output_dir = PROJECT_ROOT / "Runs" / "TensorRT-Video-Test"
output_dir.mkdir(parents=True, exist_ok=True)

output_video_path = output_dir / "example_predicted.mp4"


# --------------------------------------------------
# Settings
# --------------------------------------------------
IMG_SIZE = 640
CONFIDENCE = 0.25
WINDOW_NAME = "YOLO TensorRT Segmentation"

PLAY_AT_ORIGINAL_SPEED = True


# --------------------------------------------------
# Safety checks
# --------------------------------------------------
if not model_path.exists():
    raise FileNotFoundError(f"TensorRT model not found:\n{model_path}")

if not video_path.exists():
    raise FileNotFoundError(f"Input video not found:\n{video_path}")


# --------------------------------------------------
# Load TensorRT model
# IMPORTANT: task="segment" fixes your current error.
# --------------------------------------------------
print(f"Loading TensorRT model:\n{model_path}")
model = YOLO(str(model_path), task="segment")


# --------------------------------------------------
# Optional: set class names manually
# Replace these with your real Roboflow/YAML class names if needed.
# --------------------------------------------------
# Example:
# model.names = {
#     0: "class_0",
#     1: "class_1",
#     2: "class_2",
# }

print(f"Model task: {model.task}")
print(f"Model names: {model.names}")


# --------------------------------------------------
# Open input video
# --------------------------------------------------
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video:\n{video_path}")

input_fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if input_fps <= 0:
    input_fps = 30.0

print("\nInput video info:")
print(f"Resolution: {frame_width} x {frame_height}")
print(f"FPS: {input_fps:.2f}")
print(f"Total frames: {total_frames}")


# --------------------------------------------------
# Create output video writer
# --------------------------------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    str(output_video_path),
    fourcc,
    input_fps,
    (frame_width, frame_height),
)

if not out.isOpened():
    cap.release()
    raise RuntimeError(f"Could not create output video:\n{output_video_path}")


# --------------------------------------------------
# Process video
# --------------------------------------------------
print("\nStarting prediction.")
print("Press 'q' to quit early.")
print("Press SPACE to pause/resume.\n")

paused = False
frame_index = 0
prev_time = time.time()

while True:
    if not paused:
        ret, frame = cap.read()

        if not ret:
            print("Reached end of video.")
            break

        frame_index += 1

        results = model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            conf=CONFIDENCE,
            device=0,
            verbose=False,
        )

        annotated_frame = results[0].plot()

        if annotated_frame.shape[1] != frame_width or annotated_frame.shape[0] != frame_height:
            annotated_frame = cv2.resize(annotated_frame, (frame_width, frame_height))

        current_time = time.time()
        processing_fps = 1.0 / max(current_time - prev_time, 1e-6)
        prev_time = current_time

        cv2.putText(
            annotated_frame,
            f"Processing FPS: {processing_fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            annotated_frame,
            f"Frame: {frame_index}/{total_frames}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        out.write(annotated_frame)
        cv2.imshow(WINDOW_NAME, annotated_frame)

    delay_ms = int(1000 / input_fps) if PLAY_AT_ORIGINAL_SPEED else 1

    key = cv2.waitKey(delay_ms) & 0xFF

    if key == ord("q"):
        print("Stopped early by user.")
        break

    if key == ord(" "):
        paused = not paused
        print("Paused." if paused else "Resumed.")


cap.release()
out.release()
cv2.destroyAllWindows()

print("\nPrediction complete.")
print(f"Saved annotated video to:\n{output_video_path}")
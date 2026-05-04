"""
To export a YOLO model in the TensorRT format. 

Feel free to copy this code into a Google Colab notebook if you trained your model there.
"""

from pathlib import Path
from ultralytics import YOLO

# Model paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

model_path = PROJECT_ROOT / "Models" / "Object Detection" / "yolov8s" / "2. 100 Epochs 640x640" / "best.pt"

# Load a model
model = YOLO(model_path)

# Export model
model.export(
    format="engine",
    imgsz=640,
    half=True,
    device=0
)
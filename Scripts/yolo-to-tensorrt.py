"""
To export a YOLO model in the TensorRT format. 

Feel free to copy this code into a Google Colab notebook if you trained your model there.
"""

from ultralytics import YOLO

# Load a model
model = YOLO("example.pt")  # Load your model from a path

# Export the model with half precision (might want to play around with)
model.export(format="engine", half=True)
"""
To downsample Google Earth images to closer match the resolution of the camera.
"""

import os
import cv2

# Image load + output dirs
input_folder = "/Users/joelvalley/Desktop/MetRocketry-Humber-Valley/Original-Images"
output_folder = "/Users/joelvalley/Desktop/MetRocketry-Humber-Valley/Downsampled-Images"

# Downsampling factor
factor = 2

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        img = cv2.imread(os.path.join(input_folder, filename))

        h, w = img.shape[:2]
        small = cv2.resize(img, (w // factor, h // factor))

        cv2.imwrite(os.path.join(output_folder, filename), small)
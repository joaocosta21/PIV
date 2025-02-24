import os
import cv2
import numpy as np
from scipy.io import loadmat

# Paths
reference_image_path = "reference/img_ref.jpg"
mat_folder = "yolo_output/"

output_video_path = "detected_boxes.mp4"
fps = 30

# Load the reference image
reference_image = cv2.imread(reference_image_path)
if reference_image is None:
    raise FileNotFoundError(f"Reference image not found at {reference_image_path}")

# Get the frame size dynamically
frame_height, frame_width, _ = reference_image.shape
frame_size = (frame_width, frame_height)

# Debug: Print frame size
print(f"Frame size (width x height): {frame_width} x {frame_height}")

# Get list of .mat files
mat_files = sorted([f for f in os.listdir(mat_folder) if f.endswith('.mat')])

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

# Helper to draw bounding boxes
def draw_boxes(image, boxes, ids, classes):
    n_boxes = len(boxes)
    n_ids = len(ids)
    n_classes = len(classes)
    n_draw = min(n_boxes, n_ids, n_classes)  # Use the smallest size

    for i in range(n_draw):
        x1, y1, x2, y2 = map(int, boxes[i])  # Ensure coordinates are integers
        label = f"ID: {ids[i]}, Class: {classes[i]}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Process each .mat file
for mat_file in mat_files:
    # Load .mat data
    mat_data = loadmat(os.path.join(mat_folder, mat_file))
    xyxy = mat_data.get("xyxy", [])
    ids = mat_data.get("id", [])
    classes = mat_data.get("class", [])

    # Debug: Print the number of bounding boxes and other data
    print(f"Processing frame: {mat_file}")
    print(f"Number of boxes: {len(xyxy)}, IDs: {len(ids)}, Classes: {len(classes)}")

    # Clone the reference image
    frame = reference_image.copy()

    # Draw the boxes
    draw_boxes(frame, xyxy, ids, classes)

    # Write frame to video
    video_writer.write(frame)

# Release video writer
video_writer.release()
print(f"Video saved at {output_video_path}")

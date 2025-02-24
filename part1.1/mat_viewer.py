from scipy.io import loadmat

# Load the .mat file
file_path = '1kvvadvcmkm_vid.mat'
yolo_data = loadmat(file_path)

# Filter and process only the frame keys
for frame in yolo_data:
    if not frame.startswith('__'):  # Skip metadata keys
        print(f"Processing frame: {frame}")
        frame_data = yolo_data[frame]

        # Access and print specific parts of the frame data
        cls = frame_data[0]['cls'][0]  # Object classes
        conf = frame_data[0]['conf'][0]  # Confidence scores
        bboxes = frame_data[0]['xywh'][0]  # Bounding boxes (x1, y1, x2, y2)
        
        print("Classes:", cls)
        print("Confidences:", conf)
        print("Bounding boxes:", bboxes)

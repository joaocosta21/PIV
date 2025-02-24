import cv2
import torch
import numpy as np
import open3d as o3d
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from pathlib import Path
from PIL import Image

# Paths
video_paths = [
    "PIV_dataset1/2024-09-14_16-00-21-front.mp4"  # Replace with your video path(s)
]
output_dir = "output_pointclouds"
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Parameters
frame_limit = 10  # Number of frames to process per video
max_depth_value = 4350.0  # Maximum depth cutoff
fx, fy = 1000, 1000  # Focal lengths (adjust based on your camera setup)
cx, cy = 640, 360    # Principal point (adjust based on your camera setup)

# MiDaS Depth Estimation Model
print("Loading MiDaS model...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
device = torch.device("cpu")
midas.to(device)
midas.eval()

# MiDaS preprocessing
transform = Compose([
    Resize(384),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def estimate_depth(frame_rgb):
    """Estimate depth using MiDaS."""
    # Ensure the input is a PIL image
    if isinstance(frame_rgb, np.ndarray):  # If NumPy array, convert to PIL
        frame_pil = Image.fromarray(frame_rgb)
    else:  # Assume it is already a PIL image
        frame_pil = frame_rgb

    # Transform the PIL image for the MiDaS model
    input_batch = transform(frame_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        depth_map = midas(input_batch)
    depth_map = depth_map.squeeze().cpu().numpy()

    # Resize depth map to match the dimensions of the original frame
    frame_np = np.array(frame_pil)  # Convert PIL to NumPy for shape
    depth_map_resized = cv2.resize(depth_map, (frame_np.shape[1], frame_np.shape[0]))
    return depth_map_resized


def generate_point_cloud(rgb_image, depth_map, max_depth_value):
    """Generate a point cloud from an RGB image and a depth map."""
    # Create a mask for valid depth values
    mask = (depth_map > 0) & (depth_map < max_depth_value)

    # Ensure mask matches the RGB image dimensions
    if mask.shape != rgb_image.shape[:2]:
        raise ValueError("Mask shape does not match RGB image shape.")

    # Apply the mask to RGB and depth values
    rgb = rgb_image[mask]
    depth = depth_map[mask]

    # Generate point cloud (example using Open3D or other libraries)
    points = np.dstack(np.meshgrid(
        np.arange(depth_map.shape[1]),
        np.arange(depth_map.shape[0])
    ))[mask]

    points_3d = np.hstack((points, depth.reshape(-1, 1)))
    return points_3d, rgb

def save_point_cloud(pcd, output_path):
    """Save point cloud to a PLY file."""
    # Create an Open3D PointCloud object
    point_cloud = o3d.geometry.PointCloud()

    # Set points and colors
    point_cloud.points = o3d.utility.Vector3dVector(pcd[0])  # Point coordinates (Nx3)
    point_cloud.colors = o3d.utility.Vector3dVector(pcd[1] / 255.0)  # Colors (Nx3), normalize to [0, 1]

    # Save the point cloud
    o3d.io.write_point_cloud(output_path, point_cloud)


# Process Videos
for video_path in video_paths:
    video_name = Path(video_path).stem
    video_output_dir = Path(output_dir) / video_name
    video_output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        continue

    frame_idx = 0
    while cap.isOpened() and frame_idx < frame_limit:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Estimate depth
        frame_pil = Image.fromarray(frame_rgb)
        depth_map = estimate_depth(frame_pil)

        # Generate point cloud
        pcd = generate_point_cloud(frame_rgb, depth_map, max_depth_value)

        # Save point cloud as .ply file
        output_ply_path = video_output_dir / f"frame_{frame_idx}.ply"
        save_point_cloud(pcd, str(output_ply_path))

        frame_idx += 1

    cap.release()

print("Processing complete.")

import os
import open3d as o3d
import numpy as np

def save_point_cloud(pcd, output_path):
    """Save point cloud to a PLY file."""
    # Create an Open3D PointCloud object
    point_cloud = o3d.geometry.PointCloud()

    # Set points and colors
    point_cloud.points = o3d.utility.Vector3dVector(pcd[0])  # Point coordinates (Nx3)
    point_cloud.colors = o3d.utility.Vector3dVector(pcd[1] / 255.0)  # Colors (Nx3), normalize to [0, 1]

    # Save the point cloud
    o3d.io.write_point_cloud(output_path, point_cloud)

    # Print confirmation
    print(f"Point cloud saved to {output_path}")


def open_point_cloud_in_folder(folder_path):
    """Open the folder containing the point cloud and display it."""
    # List all files in the folder
    files = os.listdir(folder_path)
    ply_files = [f for f in files if f.endswith('.ply')]

    if ply_files:
        # Load and display the first point cloud file found
        ply_path = os.path.join(folder_path, ply_files[0])
        pcd = o3d.io.read_point_cloud(ply_path)

        # Visualize the point cloud
        o3d.visualization.draw_geometries([pcd])
    else:
        print("No .ply files found in the folder.")

# Specify the output directory and file name
output_dir = "output_pointclouds\\2024-09-14_16-00-21-front"
os.makedirs(output_dir, exist_ok=True)
output_ply_path = os.path.join(output_dir, "frame_1.ply")

# Open the folder and display the point cloud
open_point_cloud_in_folder(output_dir)

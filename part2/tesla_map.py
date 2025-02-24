import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
from PIL import Image
from transformers import pipeline
import matplotlib.pyplot as plt
from skimage import exposure, filters

# Set up the depth estimation pipeline
print("Loading depth estimation model...")
depth_pipeline = pipeline("depth-estimation")

# Function to normalize and enhance depth maps
def enhance_depth_map_smooth(depth_map, max_depth=10.0):
    # Clip depth values to max depth
    depth_map = np.clip(depth_map, 0, max_depth)
    
    # Normalize to [0, 1]
    depth_map_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    
    # Apply bilateral filter for smoothness (adjust parameters for your use case)
    depth_map_smoothed = cv2.bilateralFilter(depth_map_normalized.astype(np.float32), 
                                             d=5,   # Diameter of pixel neighborhood
                                             sigmaColor=0.1,  # Color standard deviation
                                             sigmaSpace=5)    # Space standard deviation
    
    # Scale back to original range
    depth_map_enhanced = depth_map_smoothed * max_depth
    return depth_map_enhanced

# Function to generate a point cloud from a depth map
def depth_to_point_cloud_with_rgb(depth_map, rgb_image, scale_factor=0.01):
    h, w = depth_map.shape
    fx, fy = w / 2, h / 2
    cx, cy = w / 2, h / 2

    # Create meshgrid
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = (x - cx) / fx
    y = (y - cy) / fy

    # Reshape
    z = depth_map * scale_factor
    x = -np.multiply(x, z)  # Flip x-axis
    y = -np.multiply(y, z)  # Flip y-axis

    # Stack into point cloud
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    # Flatten RGB image
    colors = rgb_image.reshape(-1, 3) / 255.0  # Normalize to [0, 1]

    # Filter out points with zero depth
    valid = (z.flatten() > 0)
    points = points[valid]
    colors = colors[valid]

    return points, colors

# Paths to videos
video_paths = [
    "PIV_dataset1/2024-09-14_16-00-21-front.mp4"
]

output_dir = "output_pointclouds"
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Process each video
frame_limit = 1  # Number of frames to process
max_depth_value = 8.0  # Maximum depth cutoff
camera_pcds = {}

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

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # Estimate depth
        depth_map = depth_pipeline(frame_pil)['depth']
        depth_array = np.array(depth_map)

        # Debug: Visualize original depth map
        plt.imshow(depth_array, cmap='inferno')
        plt.colorbar()
        plt.title(f"Original Depth Map Frame {frame_idx} - {video_name}")
        plt.show()

        # Enhance depth map
        depth_array_enhanced = enhance_depth_map_smooth(depth_array, max_depth=max_depth_value)

        # Visualize the enhanced depth map
        plt.imshow(depth_array_enhanced, cmap='inferno')
        plt.colorbar()
        plt.title(f"Enhanced Depth Map")
        plt.show()

        # Debug: Print depth stats
        print(f"Enhanced depth map stats - Min: {np.min(depth_array_enhanced)}, Max: {np.max(depth_array_enhanced)}, Mean: {np.mean(depth_array_enhanced)}")
        
        # Create point cloud
        points, colors = depth_to_point_cloud_with_rgb(depth_array_enhanced, frame_rgb)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Save point cloud to .npy
        pointcloud_data = {"points": points, "colors": colors}
        npy_file = video_output_dir / f"frame_{frame_idx:04d}.npy"
        np.save(npy_file, pointcloud_data)

        # Save point cloud in memory (grouped by camera)
        if video_name not in camera_pcds:
            camera_pcds[video_name] = []
        camera_pcds[video_name].append(pcd)

        # Print progress
        print(f"Processed frame {frame_idx + 1} from {video_name}.")

        frame_idx += 1

    cap.release()


def merge_point_clouds(pcd_list, voxel_downsample_size=None):
    combined_pcd = o3d.geometry.PointCloud()
    for pcd in pcd_list:
        combined_pcd += pcd
    if voxel_downsample_size:
        combined_pcd = combined_pcd.voxel_down_sample(voxel_downsample_size)
    return combined_pcd

# Combine all frames into one point cloud per video
merged_pcds = {}
for video_name, pcd_list in camera_pcds.items():
    print(f"Merging point clouds for video {video_name}...")
    merged_pcd = merge_point_clouds(pcd_list, voxel_downsample_size=0.00001)  # Adjust voxel size if needed
    merged_pcds[video_name] = merged_pcd

    # Save the merged point cloud as a .ply file
    merged_ply_file = Path(output_dir) / f"{video_name}_merged.ply"
    o3d.io.write_point_cloud(str(merged_ply_file), merged_pcd)
    print(f"Saved merged point cloud for {video_name} as {merged_ply_file}.")

# Visualize merged point clouds
for video_name, merged_pcd in merged_pcds.items():
    print(f"Displaying merged point cloud for {video_name}...")
    o3d.visualization.draw_geometries([merged_pcd])

print(f"Point clouds and merged files saved in {output_dir}.")


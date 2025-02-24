import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
from PIL import Image
from transformers import pipeline

# Set up the depth estimation pipeline
print("Loading depth estimation model...")
depth_pipeline = pipeline("depth-estimation")

# Function to generate a point cloud from a depth map
def depth_to_point_cloud_with_rgb(depth_map, rgb_image, scale_factor=0.001, max_depth=10.0):
    h, w = depth_map.shape
    fx, fy = w / 2, h / 2
    cx, cy = w / 2, h / 2

    # Remove invalid depth values
    depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
    depth_map = np.clip(depth_map, 0, max_depth)

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

# Function to load point clouds from .npy files
def load_point_clouds_from_dir(directory, downsample_voxel_size=0.02):
    pcds = []
    for npy_file in Path(directory).glob("*.npy"):
        data = np.load(npy_file, allow_pickle=True).item()
        points = data['points']
        colors = data['colors']
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd = pcd.voxel_down_sample(downsample_voxel_size)
        pcds.append(pcd)
    return pcds

# Approximate extrinsic matrices for Tesla cameras

R_x = np.array([[1,  0,       0,      0],
                [0,  0,      0,      0],
                [0,  1,       0,      0],
                [0,  0,       0,      1]])

extrinsics = {
    "front": R_x @ np.eye(4),
    "back": R_x @ np.array([[  -1,  0,  0,  0],
                      [   0, -1,  0, 0],
                      [   0,  0,  1,  0],
                      [   0,  0,  0,  1]]),
    "left": R_x @ np.array([[   0, -1,  0, -0.075],
                      [   1,  0,  0,    0],
                      [   0,  0,  1,    0],
                      [   0,  0,  0,    1]]),
    "right": R_x @ np.array([[   0,  1,  0,  0.075],
                       [  -1,  0,  0,     0],
                       [   0,  0,  1,     0],
                       [   0,  0,  0,     1]])
}

# Merge point clouds by applying approximate extrinsics
def merge_point_clouds_with_extrinsics(pcds, extrinsics):
    print("Merging point clouds using approximate extrinsics...")
    merged_points = []
    merged_colors = []

    print(f"Number of point clouds: {len(pcds)}")
    for i, (pcd, (camera, extrinsic)) in enumerate(zip(pcds, extrinsics.items())):
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        # Transform points using the extrinsic matrix
        ones = np.ones((points.shape[0], 1))
        transformed_points = (points @ extrinsic[:3, :3].T) + extrinsic[:3, 3]

        # Debug: Check transformed points
        print(f"Camera: {camera}, Transformed Points: {transformed_points[:, :3].shape}")
        pcd_transformed = o3d.geometry.PointCloud()
        pcd_transformed.points = o3d.utility.Vector3dVector(transformed_points[:, :3])
        pcd_transformed.colors = pcd.colors
        print(f"Transformed point cloud for {camera} displayed.")
        merged_points.append(transformed_points[:, :3])
        merged_colors.append(colors)

    # Concatenate all transformed point clouds
    final_points = np.vstack(merged_points)
    final_colors = np.vstack(merged_colors)

    # Debug: Final merged point cloud
    print(f"Final merged points count: {final_points.shape[0]}")
    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(final_points)
    merged_pcd.colors = o3d.utility.Vector3dVector(final_colors)

    return merged_pcd

# Paths to videos
video_paths = [
    "PIV_dataset1/2024-09-14_16-00-21-front.mp4",
    "PIV_dataset1/2024-09-14_16-00-21-back.mp4",
    "PIV_dataset1/2024-09-14_16-00-21-left_repeater.mp4",
    "PIV_dataset1/2024-09-14_16-00-21-right_repeater.mp4"
]

output_dir = "output_pointclouds"
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Process each video
frame_limit = 1  # Limit to first frame
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
    while cap.isOpened() and frame_idx < frame_limit:  # Process only the first frame
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # Estimate depth
        depth_map = depth_pipeline(frame_pil)['depth']
        depth_array = np.array(depth_map)

        # Create point cloud
        points, colors = depth_to_point_cloud_with_rgb(depth_array, frame_rgb, max_depth=max_depth_value)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
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
print(f"Point clouds saved in {output_dir}.")

# Merge all point clouds from memory
all_pcds = []
for camera, pcd_list in camera_pcds.items():
    all_pcds.extend(pcd_list)

final_pcd = merge_point_clouds_with_extrinsics(all_pcds, extrinsics)

# Save and visualize the merged point cloud
output_merged_pcd = Path(output_dir) / "merged_pointcloud.ply"
o3d.io.write_point_cloud(str(output_merged_pcd), final_pcd)
print(f"Merged point cloud saved to {output_merged_pcd}.")
o3d.visualization.draw_geometries([final_pcd], window_name="Merged Point Cloud")

import numpy as np
import os
import cv2
from scipy.io import loadmat, savemat
import open3d as o3d

def load_camera_data(file_path):
    """
    Load camera data from a .mat file.
    """
    try:
        data = loadmat(file_path)
        cams_info = data.get("cams_info", None)
        if cams_info is None:
            raise ValueError("cams_info not found in the provided .mat file.")
        return cams_info
    except Exception as e:
        raise IOError(f"Error loading camera data from {file_path}: {e}")

def unpack_nested_field(field):
    """
    Unpack a nested field, handling its shape and extracting the array.
    """
    if isinstance(field, np.ndarray) and field.size == 1:
        return field[0, 0]
    return field

def extract_camera_fields(camera_data):
    """
    Extract individual fields from a camera's structured array.
    """
    try:
        rgb_image = unpack_nested_field(camera_data["rgb"])
        depth_map = unpack_nested_field(camera_data["depth"])
        conf_map = unpack_nested_field(camera_data["conf"])
        focal_length = unpack_nested_field(camera_data["focal_lenght"])
        return rgb_image, depth_map, conf_map, focal_length
    except KeyError as e:
        raise KeyError(f"Missing required field in camera data: {e}")

def process_cameras(cams_info):
    """
    Process all cameras in cams_info and return their data.
    """
    num_cameras = cams_info.shape[0]  # Number of cameras

    cameras_data = []
    for i in range(num_cameras):
        try:
            camera_data = cams_info[i, 0]  # Adjust for (n, 1) shape
            cameras_data.append(extract_camera_fields(camera_data))
        except Exception as e:
            print(f"Error processing camera {i}: {e}")
    return cameras_data

def generate_point_cloud(depth_map, rgb_image, confidence_map, focal_length):
    """
    Generate 3D point cloud from depth map, confidence map, and camera intrinsics.
    """
    try:
        fx = focal_length[0, 0]
        fy = fx  # Assuming square pixels
        cx, cy = rgb_image.shape[1] / 2, rgb_image.shape[0] / 2

        h, w = depth_map.shape
        u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))

        valid_mask = (confidence_map > 0.7) & (depth_map > 0)
        Z = depth_map[valid_mask]
        X = Z * (u_coords[valid_mask] - cx) / fx
        Y = Z * (v_coords[valid_mask] - cy) / fy
        colors = rgb_image[valid_mask]

        return np.column_stack((X, Y, Z, colors.reshape(-1, 3)))
    except Exception as e:
        raise RuntimeError(f"Error generating point cloud: {e}")

def align_point_clouds_icp(source_cloud, target_cloud, threshold=0.001, max_iterations=500000):
    """
    Align two point clouds using the Iterative Closest Point (ICP) algorithm.
    """
    try:
        # Convert the source and target clouds to float64 and extract colors
        source_cloud = source_cloud.astype(np.float64)
        source_colors = source_cloud[:, 3:].astype(np.float64)

        target_cloud = target_cloud.astype(np.float64)
        target_colors = target_cloud[:, 3:].astype(np.float64)

        # Create Open3D PointCloud objects
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(source_cloud[:, :3])
        source.colors = o3d.utility.Vector3dVector(source_colors)

        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(target_cloud[:, :3])
        target.colors = o3d.utility.Vector3dVector(target_colors)

        # Optional: Visualize before ICP
        o3d.visualization.draw_geometries([source, target])  # Uncomment to visualize

        # ICP registration (single call with parameters)
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iterations,
            # relative_fitness=1e-6,  # Stop if fitness relative change is small
            # relative_rmse=1e-6  # Stop if RMSE relative change is small
        )
        
        icp_result = o3d.pipelines.registration.registration_icp(
            source, target, threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria
        )

        # Apply the transformation to align the source cloud
        aligned_cloud = np.asarray(icp_result.transformation @ np.column_stack((source_cloud[:, :3], np.ones(len(source_cloud)))).T).T[:, :3]

        # Combine the aligned point cloud with the original colors
        transformed_cloud = np.hstack((aligned_cloud, source_cloud[:, 3:]))  # Include original colors

        # Optional: Visualize after ICP
        # o3d.visualization.draw_geometries([source, target])  # Uncomment to visualize

        return transformed_cloud, icp_result.transformation

    except Exception as e:
        raise RuntimeError(f"Error aligning point clouds with ICP: {e}")

def merge_point_clouds(point_clouds):
    """
    Merge multiple point clouds into one by ensuring consistent dimensions.
    """
    max_columns = max(pc.shape[1] for pc in point_clouds)

    standardized_point_clouds = []
    for pc in point_clouds:
        if pc.shape[1] < max_columns:
            padding = np.zeros((pc.shape[0], max_columns - pc.shape[1]))
            standardized_pc = np.hstack((pc, padding))
        else:
            standardized_pc = pc[:, :max_columns]

        standardized_point_clouds.append(standardized_pc)

    return np.vstack(standardized_point_clouds)

def save_point_cloud(point_cloud, file_name):
    """
    Save the generated point cloud to a .mat file.
    """
    try:
        savemat(file_name, {"point_cloud": point_cloud})
        print(f"Point cloud saved to {file_name}")
    except Exception as e:
        raise IOError(f"Error saving point cloud: {e}")

def visualize_point_cloud(point_cloud):
    """
    Visualize a point cloud using Open3D.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    if point_cloud.shape[1] > 3:
        pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:])
    o3d.visualization.draw_geometries([pcd])

def save_transformations(transformations, file_name):
    """
    Save transformations to a .mat file.
    """
    savemat(file_name, {"transforms": transformations})

def main():
    current_dir = os.getcwd()
    cams_info_path = os.path.join(current_dir, "office", "cams_info_no_extr.mat")
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    cams_info = load_camera_data(cams_info_path)
    cameras_data = process_cameras(cams_info)
    # cameras_data = cameras_data[:2]  # Limit to first two cameras for testing
    all_point_clouds = []
    transformations = {}

    # Generate point clouds
    for i, (rgb_image, depth_map, conf_map, focal_length) in enumerate(cameras_data):
        print(f"Processing camera {i + 1}/{len(cameras_data)}")
        point_cloud = generate_point_cloud(depth_map, rgb_image, conf_map, focal_length)
        all_point_clouds.append(point_cloud)
        save_point_cloud(point_cloud, f"point_cloud_camera{i + 1}.mat")
        transformations[i] = {"R": np.eye(3), "T": np.zeros((3, 1))}

    # Align and merge point clouds
    reference_point_cloud = all_point_clouds[0]
    aligned_point_clouds = [reference_point_cloud]

    for i in range(1, len(all_point_clouds)):
        print(f"Aligning camera {i + 1} point cloud with the reference.")
        aligned_cloud, transformation = align_point_clouds_icp(all_point_clouds[i], reference_point_cloud)
        aligned_point_clouds.append(aligned_cloud)
        
        R = transformation[:3, :3]
        T = transformation[:3, 3]
        transformations[i] = {"R": R, "T": T}

    merged_point_cloud = merge_point_clouds(aligned_point_clouds)
    
    save_point_cloud(merged_point_cloud, os.path.join(output_dir, "output.mat"))
    save_transformations(transformations, os.path.join(output_dir, "transforms.mat"))

    # Visualize final merged point cloud
    visualize_point_cloud(merged_point_cloud)

if __name__ == "__main__":
    main()

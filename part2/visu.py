import numpy as np
from scipy.io import loadmat
import open3d as o3d

def load_point_cloud(file_path):
    """
    Load point cloud from a .mat file.
    """
    data = loadmat(file_path)
    print(data.keys())  # Check available keys
    print(data["point_cloud"].shape)  # Ensure it has the expected number of columns
    print(data["point_cloud"][:5]) 
    if "point_cloud" not in data:
        raise ValueError("No 'point_cloud' key found in the .mat file.")
    return data["point_cloud"]

def downsample_point_cloud(point_cloud, sample_size=1000):
    if len(point_cloud) > sample_size:
        indices = np.random.choice(len(point_cloud), sample_size, replace=False)
        return point_cloud[indices]
    return point_cloud

def visualize_point_cloud(file_path):
    """
    Visualize point cloud using Open3D.
    """
    # Load point cloud data
    point_cloud_data = load_point_cloud(file_path)
    downsampled_cloud = downsample_point_cloud(point_cloud_data, sample_size=len(point_cloud_data) - 1)

    # Extract XYZ and RGB
    xyz = downsampled_cloud[:, :3]
    rgb = downsampled_cloud[:, 3:6]
    xyz = xyz.astype(np.float64)
    rgb = rgb.astype(np.float64)
    
    # Create Open3D Point Cloud
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    pc.colors = o3d.utility.Vector3dVector(rgb)

    # Visualize
    o3d.visualization.draw_geometries([pc])

if __name__ == "__main__":
    # Replace with the path to your saved .mat file
    file_path = "point_cloud_camera1.mat"
    file_path = "output.mat"
    file_path = "point_cloud_camera1.mat"
    visualize_point_cloud(file_path)

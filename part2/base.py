import scipy.io
import open3d as o3d
import numpy as np

def load_wrld_data(file_path):
    """
    Load and extract point cloud data (xyz and color) from the wrld_info.mat file.
    """
    data = scipy.io.loadmat(file_path)
    if 'wrld' not in data:
        raise ValueError("No 'wrld' key found in the .mat file.")
    
    wrld_data = data['wrld']
    points = wrld_data['xyz'][0][0]
    colors = wrld_data['color'][0][0]
    
    # Ensure data types are float64 for Open3D compatibility
    points = points.astype(np.float64)
    colors = colors.astype(np.float64)
    
    return points, colors

def downsample_point_cloud(points, colors, sample_size=1000):
    """
    Downsample the point cloud to the specified number of points.
    """
    if len(points) > sample_size:
        indices = np.random.choice(len(points), sample_size, replace=False)
        return points[indices], colors[indices]
    return points, colors

def visualize_wrld_point_cloud(file_path, sample_size=None):
    """
    Load, downsample (optional), and visualize point cloud data from wrld_info.mat.
    """
    # Load point cloud data
    points, colors = load_wrld_data(file_path)
    
    # Downsample if sample_size is specified
    if sample_size:
        points, colors = downsample_point_cloud(points, colors, sample_size)
    
    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    # Replace with the path to your wrld_info.mat file
    file_path = "office/wrld_info.mat"
    
    # Specify the sample size for downsampling (set to None for no downsampling)
    sample_size = 2000000  # Adjust as needed
    visualize_wrld_point_cloud(file_path, sample_size)

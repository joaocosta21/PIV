import os
import numpy as np
from scipy.io import loadmat

def load_camera_data(file_path):
    """
    Load camera data from a .mat file and return cams_info.
    """
    data = loadmat(file_path)
    cams_info = data.get("cams_info", None)
    if cams_info is None:
        raise ValueError("cams_info not found in the provided .mat file.")
    return cams_info


def extract_camera_fields(camera_data):
    """
    Extract individual fields from a camera's structured array.
    """
    def unpack_nested_field(field):
        """
        Unpack a nested field, handling its shape and extracting the array.
        """
        if isinstance(field, np.ndarray) and field.size == 1:
            return field[0, 0]
        return field

    rgb_image = unpack_nested_field(camera_data["rgb"])
    depth_map = unpack_nested_field(camera_data["depth"])
    conf_map = unpack_nested_field(camera_data["conf"])
    focal_length = unpack_nested_field(camera_data["focal_lenght"])

    return rgb_image, depth_map, conf_map, focal_length


def process_cameras(cams_info):
    """
    Process all cameras in cams_info and print their data.
    """
    num_cameras = cams_info.shape[0]  # Number of cameras
    print(cams_info[0])
    for i in range(num_cameras):
        # Access structured array for the i-th camera
        camera_data = cams_info[i, 0]  # Adjust for (10, 1) shape

        try:
            rgb_image, depth_map, conf_map, focal_length = extract_camera_fields(camera_data)

            # Debug information
            print(f"Processing camera {i + 1}/{num_cameras}")
            print(f"RGB Image Shape: {rgb_image.shape}")
            print(f"Depth Map Shape: {depth_map.shape}")
            print(f"Confidence Map Shape: {conf_map.shape}")
            print(f"Focal Length: {focal_length}")

            # Add your processing logic here (e.g., point cloud generation, etc.)
        except Exception as e:
            print(f"Error processing camera {i + 1}: {e}")


def main():
    # File path setup
    current_dir = os.getcwd()
    cams_info_path = os.path.join(current_dir, "office", "cams_info_no_extr.mat")

    # Load and process data
    cams_info = load_camera_data(cams_info_path)
    process_cameras(cams_info)


if __name__ == "__main__":
    main()

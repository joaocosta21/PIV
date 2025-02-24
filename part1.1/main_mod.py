import os
import sys
import numpy as np
from scipy.io import savemat

# Add the parent directory to the sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from piv.functions import (
    load_keypoints_from_mat,
    compute_homography,
    import_yolo_data,
    process_yolo_data,
    point_homography_transformation,
    plot_side_by_side_images,
    create_video_from_images,
    transform_yolo_data
)

def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load keypoints from 'kp_gmaps.mat' and compute the homography matrix
    mat_file_path = os.path.join(script_dir, 'kp_gmaps.mat')
    print(f"Expected path for kp_gmaps.mat: {mat_file_path}")

    keypoints_image1, keypoints_image2 = load_keypoints_from_mat(mat_file_path)
    u1, v1 = keypoints_image1[:, 0], keypoints_image1[:, 1]
    u2, v2 = keypoints_image2[:, 0], keypoints_image2[:, 1]
    H = compute_homography(u1, v1, u2, v2)

    # Save the homography matrix
    homography_save_path = os.path.join(script_dir, 'homography.mat')
    savemat(homography_save_path, {'H': H})
    print(f"Homography matrix saved to {homography_save_path}")
    print("Computed Homography Matrix:")
    print(H)

    # Process YOLO data
    yolo_folder = os.path.join(script_dir, 'yolo')
    yolo_data = import_yolo_data(yolo_folder)
    processed_data = process_yolo_data(yolo_data)
    print("Processed all YOLO data.")

    # Transform YOLO data using the homography matrix
    output_yolo = os.path.join(script_dir, "yolo_output")
    transform_yolo_data(homography_save_path, yolo_data, output_yolo)
    print("Transformed YOLO data saved.")

    # Create output directory if it doesn't exist
    output_folder = os.path.join(script_dir, "aerial_images")
    os.makedirs(output_folder, exist_ok=True)

    # Transform points and generate images for each frame
    for frame_idx, frame_data in enumerate(processed_data):
        frame_points = np.array([item[1] for item in frame_data])
        transformed_points = np.array([
            point_homography_transformation(H, point) for point in frame_points
        ])

        camera_input_image = os.path.join(script_dir, f'images/img_{frame_idx+1:04d}.jpg')
        aerial_image = os.path.join(script_dir, 'airport_CapeTown_aerial.png')
        output_image_path = os.path.join(output_folder, f'aerial_img_{frame_idx+1:04d}.png')

        plot_side_by_side_images(
            camera_input_image,
            aerial_image,
            frame_points,
            transformed_points,
            f'Frame {frame_idx+1}',
            output_image_path
        )
        print(f"Processed frame {frame_idx+1}/{len(processed_data)}")

    print("All frames processed and images generated.")

    # Create a video from the generated images
    video_path = os.path.join(script_dir, 'aerial_video.mp4')
    create_video_from_images(output_folder, video_path, fps=10)
    print(f"Video saved to {video_path}")

if __name__ == "__main__":
    main()
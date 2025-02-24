import os
import sys
from pathlib import Path

# Add the parent directory to the sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from piv.functions import generate_panorama_lite
import scipy.io as sio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def get_transformed_corners(w, h, homography):
    """Get transformed corners using a homography."""
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    corners_homogeneous = np.hstack([corners, np.ones((4, 1))])
    transformed_corners = homography @ corners_homogeneous.T
    transformed_corners /= transformed_corners[2, :]  # Normalize by z
    return transformed_corners[:2, :].T


def display_homography(ref_dir, input_dir, output_dir):
    """Visualize and save homographies as a JPEG."""
    mat_file = os.path.join(output_dir, "homographies.mat")
    if not os.path.exists(mat_file):
        print(f"No homography file found in {output_dir}")
        return

    student_data = sio.loadmat(mat_file)
    homographies = student_data.get("H", None)
    if homographies is None:
        print(f"No 'H' matrix in {mat_file}")
        return

    # Get size of the first image
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
    if not image_files:
        print(f"No .jpg files in {input_dir}")
        return

    image_path = os.path.join(input_dir, image_files[0])
    image = Image.open(image_path)
    w, h = image.size

    # Plot transformed corners
    plt.figure()
    plt.title(f"Transformed Corners")
    for i in range(homographies.shape[2]):
        transformed_corners = get_transformed_corners(w, h, homographies[:, :, i])
        plt.plot(transformed_corners[[0, 1, 2, 3, 0], 0],
                 transformed_corners[[0, 1, 2, 3, 0], 1], '-')
    plt.gca().invert_yaxis()
    plt.axis('equal')

    # Save the visualization
    output_image = os.path.join(output_dir, "corners_to_ref.jpg")
    plt.savefig(output_image, format='jpg', dpi=300, bbox_inches='tight')
    print(f"Saved homography visualization: {output_image}")


def main():
    if len(sys.argv) < 4 or (len(sys.argv) - 2) % 2 != 0:
        print("Usage: python eval_single.py <ref_dir> <input_dir> <output_dir>")
        sys.exit(1)

    # Directories
    ref_dir = sys.argv[1]
    input_dir = sys.argv[2]
    output_dir = sys.argv[3]

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Run panorama generation
    print(f"Processing: ref_dir={ref_dir}, input_dir={input_dir}, output_dir={output_dir}")
    generate_panorama_lite(
        ref_dir,
        input_dir,
        output_dir,
        num_min_matches=31,
        inlier_threshold=2.0,
        ransac_iterations=2000,
        min_overlap=0.2,
        reference_index=0
    )

    # Display homographies
    display_homography(ref_dir, input_dir, output_dir)


if __name__ == "__main__":
    main()

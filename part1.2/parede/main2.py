import os
import sys
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio

def load_image_and_keypoints(image_path, keypoints_path):
    """Load image and its corresponding keypoints/descriptors."""
    img = cv.imread(image_path)
    if img is None:
        raise ValueError(f"Error loading image: {image_path}")

    kp_data = sio.loadmat(keypoints_path)
    keypoints = [cv.KeyPoint(x[0], x[1], 1) for x in kp_data['kp']]
    descriptors = kp_data['desc']

    return img, keypoints, descriptors

def find_matches(des1, des2):
    """Find matches between two sets of descriptors."""
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return good_matches

def compute_homography(kp1, kp2, good_matches):
    """Compute homography matrix between two sets of keypoints."""
    if len(good_matches) < 4:
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    
    return H

def combine_images(images, homographies, ref_img_shape):
    """Combine all images using precomputed homographies relative to reference image plane."""
    h1, w1 = ref_img_shape[:2]

    # Calculate panorama dimensions
    corners = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    all_corners = [corners]

    for H in homographies:
        img_corners = cv.perspectiveTransform(corners, H)
        all_corners.append(img_corners)

    all_points = np.concatenate(all_corners)
    min_x = int(np.floor(np.min(all_points[:, :, 0])))
    max_x = int(np.ceil(np.max(all_points[:, :, 0])))
    min_y = int(np.floor(np.min(all_points[:, :, 1])))
    max_y = int(np.ceil(np.max(all_points[:, :, 1])))

    # Create translation matrix
    translation = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ], dtype=np.float32)

    # Initialize result image
    result = np.zeros((max_y - min_y, max_x - min_x, 3), dtype=np.uint8)

    # Warp and place reference image
    cv.warpPerspective(images[0], translation, (max_x - min_x, max_y - min_y),
                       dst=result, borderMode=cv.BORDER_TRANSPARENT)

    # Warp and place each subsequent image
    for i, (img, H) in enumerate(zip(images[1:], homographies)):
        H_final = translation @ H
        cv.warpPerspective(img, H_final, (max_x - min_x, max_y - min_y),
                           dst=result, borderMode=cv.BORDER_TRANSPARENT)

    return result

def process_image_sequence(ref_dir, input_dir, output_dir):
    """Process a sequence of images and create a panorama."""
    # Load reference image and its keypoints
    ref_img_path = os.path.join(ref_dir, "img_ref.jpg")
    ref_kp_path = os.path.join(ref_dir, "kp_ref.mat")
    ref_img, ref_kp, ref_desc = load_image_and_keypoints(ref_img_path, ref_kp_path)

    # Initialize lists to store images, keypoints, and descriptors
        # Initialize lists to store images and homographies
    all_images = [ref_img]
    last_keypoints = ref_kp
    last_descriptors = ref_desc

    homographies = []  # H[i] transforms from image i to img1
    cumulative_homography = np.eye(3, dtype=np.float32)  # Cumulative homography for chaining

    # Get all image files from input directory
    input_files = sorted([f for f in os.listdir(input_dir) if f.startswith('img_') and f.endswith('.jpg')])

    for img_file in input_files:
        kp_file = f"kp_{img_file[4:-4]}.mat"
        img_path = os.path.join(input_dir, img_file)
        kp_path = os.path.join(input_dir, kp_file)

        print(f"Processing {img_file}...")

        try:
            # Load input image and its keypoints
            input_img, input_kp, input_desc = load_image_and_keypoints(img_path, kp_path)

            # Find matches only with the last added image
            good_matches = find_matches(last_descriptors, input_desc)

            # Compute homography relative to the most recent image
            src_pts = np.float32([last_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([input_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            H, _ = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)

            if H is not None:
                # Update cumulative homography to relate this image to img1
                cumulative_homography = cumulative_homography @ H
                homographies.append(cumulative_homography)

                # Update the reference keypoints and descriptors
                last_keypoints = input_kp
                last_descriptors = input_desc

                # Store image
                all_images.append(input_img)
            else:
                print(f"Not enough matches found for {img_file}")

        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")

    # Second pass: create panorama using all images and cumulative homographies
    if homographies:
        accumulated_img = combine_images(all_images, homographies, ref_img.shape)

        # Save final panorama
        final_output_path = os.path.join(output_dir, "final_panorama.jpg")
        cv.imwrite(final_output_path, accumulated_img)
        print(f"Saved final panorama to {final_output_path}")
    else:
        print("No valid homographies found")

    # Second pass: create panorama using all images and homographies
    if homographies:
        accumulated_img = combine_images(all_images, homographies, ref_img.shape)

        # Save final panorama
        final_output_path = os.path.join(output_dir, "final_panorama.jpg")
        cv.imwrite(final_output_path, accumulated_img)
        print(f"Saved final panorama to {final_output_path}")
    else:
        print("No valid homographies found")

def main():
    if len(sys.argv) != 4:
        print("Usage: python main.py reference_dir input_dir output_dir")
        sys.exit(1)

    ref_dir = sys.argv[1]
    input_dir = sys.argv[2]
    output_dir = sys.argv[3]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process the image sequence
    process_image_sequence(ref_dir, input_dir, output_dir)

if __name__ == "__main__":
    main()
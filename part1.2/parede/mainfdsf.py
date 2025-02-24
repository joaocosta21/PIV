import os
import sys

import cv2 as cv
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt


def load_image_and_keypoints(image_path, keypoints_path):
    """Load image and its corresponding keypoints/descriptors."""
    img = cv.imread(image_path)
    if img is None:
        raise ValueError(f"Error loading image: {image_path}")

    # Assuming keypoints are stored in MATLAB format
    kp_data = sio.loadmat(keypoints_path)
    # Ensure keypoints are loaded as cv.KeyPoint
    keypoints = [
        cv.KeyPoint(x[0], x[1], 1) for x in kp_data["kp"]
    ]  # Adjust indexing if necessary
    descriptors = kp_data["desc"]

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
    """
    Compute homography matrix using the Direct Linear Transform (DLT) method.
    """
    if len(good_matches) < 4:
        raise ValueError(
            "Not enough matches to compute a homography. Need at least 4.")

    # Prepare the A matrix
    A = []
    for match in good_matches:
        x, y = kp1[match.queryIdx].pt
        x_prime, y_prime = kp2[match.trainIdx].pt

        x, y = kp1[match.queryIdx].pt
        x_prime, y_prime = kp2[match.trainIdx].pt

        A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
        A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])

    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    H /= H[2, 2]  # Normalize the homography matrix
    return H


def compute_homography_ransac(src_pts, dst_pts):
    """
    Compute homography matrix using the Direct Linear Transform (DLT) method.
    Parameters:
        src_pts (numpy.ndarray): Source points of shape (N, 2).
        dst_pts (numpy.ndarray): Destination points of shape (N, 2).
    Returns:
        numpy.ndarray: Homography matrix of shape (3, 3).
    """
    if len(src_pts) < 4 or len(dst_pts) < 4:
        raise ValueError("Not enough points to compute a homography. Need at least 4.")

    # Prepare the A matrix
    A = []
    for (x, y), (x_prime, y_prime) in zip(src_pts, dst_pts):
        A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
        A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])

    # Solve for H using SVD
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)

    # Normalize the homography matrix
    H /= H[2, 2]

    return H



def ransac_homography(kp1, kp2, good_matches, threshold=1, max_iterations=10000):
    """Estimate a robust homography using RANSAC."""
    best_H = None
    best_inliers = 0
    best_inlier_mask = None

    # for match in good_matches[:10]:  # Print the first 10 matches
    #     print(
    #         f"QueryIdx: {match.queryIdx}, TrainIdx: {
    #             match.trainIdx}, Distance: {match.distance}"
    #     )
    # Extract points from matches
    # src_pts = np.float32(
    #     [kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # dst_pts = np.float32(
    #     [kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    #
    src_pts = []
    dst_pts = []
    for m in good_matches:
        src_pts.append(kp1[m.queryIdx].pt)
        dst_pts.append(kp2[m.trainIdx].pt)


    src_pts = np.float32(src_pts)
    dst_pts = np.float32(dst_pts) 

    for _ in range(max_iterations):
        # Randomly select 4 matches
        sample_indices = np.random.choice(len(good_matches), 4, replace=False)
        sample_src_pts = src_pts[sample_indices]
        sample_dst_pts = dst_pts[sample_indices]

        # Compute homography from the sample
        try:
            H = compute_homography_ransac(
                sample_src_pts, sample_dst_pts)

            # Compute projected points
            projected_pts = cv.perspectiveTransform(
                src_pts.reshape(-1, 1, 2), H
            ).reshape(-1, 2)

            # Compute distances between projected and actual points
            distances = np.linalg.norm(projected_pts - dst_pts, axis=1)

            # Count inliers
            inlier_mask = distances < threshold
            num_inliers = np.sum(inlier_mask)

            # Update the best model
            if num_inliers > best_inliers:
                best_H = H
                best_inliers = num_inliers
                best_inlier_mask = inlier_mask

        except np.linalg.LinAlgError:
            continue  # Skip invalid solutions

    if best_H is None:
        raise ValueError("RANSAC failed to find a valid homography.")

    return best_H, best_inlier_mask


def combine_images(images, homographies, ref_img_shape):
    """Combine all images using precomputed homographies relative to reference image plane."""
    h1, w1 = ref_img_shape[:2]

    # Calculate panorama dimensions
    corners = np.float32(
        [[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
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
    translation = np.array(
        [[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32
    )

    # Initialize result image
    result = np.zeros((max_y - min_y, max_x - min_x, 3), dtype=np.uint8)

    # Warp and place reference image
    cv.warpPerspective(
        images[0],
        translation,
        (max_x - min_x, max_y - min_y),
        dst=result,
        borderMode=cv.BORDER_TRANSPARENT,
    )

    # Warp and place each subsequent image
    for i, (img, H) in enumerate(zip(images[1:], homographies)):
        H_final = translation @ H
        cv.warpPerspective(
            img,
            H_final,
            (max_x - min_x, max_y - min_y),
            dst=result,
            borderMode=cv.BORDER_TRANSPARENT,
        )

    return result


def process_image_sequence(ref_dir, input_dir, output_dir):
    """Process a sequence of images and create a panorama."""
    # Load reference image and its keypoints
    ref_img_path = os.path.join(ref_dir, "img_ref.jpg")
    ref_kp_path = os.path.join(ref_dir, "kp_ref.mat")
    ref_img, ref_kp, ref_desc = load_image_and_keypoints(
        ref_img_path, ref_kp_path)

    # Initialize lists to store images, keypoints, and descriptors
    # Initialize lists to store images and homographies
    all_images = [ref_img]
    last_keypoints = ref_kp
    last_descriptors = ref_desc

    homographies = []  # H[i] transforms from image i to img1
    # Cumulative homography for chaining
    cumulative_homography = np.eye(3, dtype=np.float32)

    # Get all image files from input directory
    input_files = sorted(
        [
            f
            for f in os.listdir(input_dir)
            if f.startswith("img_") and f.endswith(".jpg")
        ]
    )

    for img_file in input_files:
        kp_file = f"kp_{img_file[4:-4]}.mat"
        img_path = os.path.join(input_dir, img_file)
        kp_path = os.path.join(input_dir, kp_file)

        print(f"Processing {img_file}...")

        try:
            # Load input image and its keypoints
            input_img, input_kp, input_desc = load_image_and_keypoints(
                img_path, kp_path
            )

            # Find matches only with the last added image
            good_matches = find_matches(last_descriptors, input_desc)
            print(f"good_matches size: {len(good_matches)}")
            print(f"k1 size: {len(last_keypoints)}")
            print(f"k2 size: {len(input_kp)}")

            # Compute homography relative to the most recent image
            H, _ = ransac_homography(last_keypoints, input_kp, good_matches)        
            # print("Hey, made it baby!")
            print(H)
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
            raise

    # Second pass: create panorama using all images and cumulative homographies
    if homographies:
        accumulated_img = combine_images(
            all_images, homographies, ref_img.shape)

        # Save final panorama
        final_output_path = os.path.join(output_dir, "final_panorama.jpg")
        cv.imwrite(final_output_path, accumulated_img)
        print(f"Saved final panorama to {final_output_path}")
    else:
        print("No valid homographies found")

    # Second pass: create panorama using all images and homographies
    if homographies:
        accumulated_img = combine_images(
            all_images, homographies, ref_img.shape)

        # Save final panorama
        final_output_path = os.path.join(output_dir, "final_panorama.jpg")
        cv.imwrite(final_output_path, accumulated_img)
        print(f"Saved final panorama to {final_output_path}")
    else:
        print("No valid homographies found")

    return homographies


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

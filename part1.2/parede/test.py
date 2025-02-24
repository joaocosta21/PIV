import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio

def display_matches(img1_path, img2_path, kp1_file, kp2_file, output_path, use_ransac=False):
    # Load the images in grayscale
    img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)  # Reference image
    img2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)  # Input image

    if img1 is None or img2 is None:
        print(f"Error loading images. Ensure {img1_path} and {img2_path} exist and are accessible.")
        return

    # Load precomputed keypoints and descriptors
    kp1_data = sio.loadmat(kp1_file)
    kp2_data = sio.loadmat(kp2_file)

    kp1 = [cv.KeyPoint(x[0], x[1], 1) for x in kp1_data['kp']]
    des1 = kp1_data['desc']

    kp2 = [cv.KeyPoint(x[0], x[1], 1) for x in kp2_data['kp']]
    des2 = kp2_data['desc']

    # Configure FLANN-based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    # Match descriptors using KNN
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    print(f"Number of good matches: {len(good_matches)}")

    if use_ransac and len(good_matches) > 4:
        # Extract location of good matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find homography using RANSAC
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        # Draw only inliers
        img_matches = cv.drawMatches(
            img1, kp1, img2, kp2, good_matches, None,
            matchColor=(0, 255, 0),
            singlePointColor=None,
            matchesMask=matches_mask,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        print(f"Number of inliers after RANSAC: {matches_mask.count(1)}")
    else:
        if use_ransac:
            print("Not enough matches for RANSAC.")
        img_matches = cv.drawMatches(
            img1, kp1, img2, kp2, good_matches, None,
            matchColor=(255, 0, 0),
            singlePointColor=None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

    # Save the matches to output
    plt.imshow(img_matches, 'gray')
    plt.title("Matched Keypoints")
    plt.savefig(output_path)
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    images_dir = "images"  # Directory containing images and keypoints
    output_dir = "output"  # Directory to save output matches
    os.makedirs(output_dir, exist_ok=True)

    # Get sorted list of images and keypoints
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    keypoint_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.mat')])

    for i in range(1, len(image_files)):
        img1_path = os.path.join(images_dir, image_files[i - 1])
        img2_path = os.path.join(images_dir, image_files[i])
        kp1_file = os.path.join(images_dir, keypoint_files[i - 1])
        kp2_file = os.path.join(images_dir, keypoint_files[i])
        output_path = os.path.join(output_dir, f"matches_{i-1}_to_{i}.png")

        print(f"Processing {image_files[i - 1]} and {image_files[i]}...")
        display_matches(img1_path, img2_path, kp1_file, kp2_file, output_path, use_ransac=True)

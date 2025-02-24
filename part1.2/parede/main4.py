import os
import sys

import cv2 as cv
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree
from scipy.io import loadmat, savemat

from collections import defaultdict
import heapq


def compute_graph(images, keypoints, descriptors, min_matches=137):
    """
    Build a graph where each image is a node, and edges represent valid homographies
    between image pairs with sufficient good matches.
    """
    num_images = len(images)
    graph = defaultdict(list)  # Adjacency list representation
    homographies = {}  # Store homographies between image pairs

    for i in range(num_images):
        for j in range(i + 1, num_images):
            # Find matches between images i and j
            good_matches = find_knn_matches_with_kdtree(descriptors[i], descriptors[j])

            if len(good_matches) >= min_matches:
                # Extract matching points
                src_pts = np.float32([keypoints[i][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints[j][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Compute homography
                H, mask = custom_find_homography(src_pts, dst_pts, "RANSAC", 5.0)
                if H is not None:
                    # Compute reprojection error as edge weight
                    inliers = mask.ravel().astype(bool)
                    error = np.mean(np.linalg.norm(src_pts[inliers] - dst_pts[inliers], axis=1))

                    # Add edge to the graph
                    graph[i].append((j, error))
                    graph[j].append((i, error))
                    homographies[(i, j)] = H
                    homographies[(j, i)] = np.linalg.inv(H)

    return graph, homographies

def compute_shortest_paths(graph, ref_idx=0):
    """
    Use Dijkstra's algorithm to compute the shortest paths from the reference image
    to all other images.
    """
    num_images = len(graph)
    distances = {i: float('inf') for i in range(num_images)}
    parents = {i: None for i in range(num_images)}
    distances[ref_idx] = 0

    priority_queue = [(0, ref_idx)]  # (distance, node)

    while priority_queue:
        current_dist, current_node = heapq.heappop(priority_queue)

        if current_dist > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node]:
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                parents[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    return parents

def save_homographies(homographies_list, homographies_output):
    H = np.stack(homographies_list, axis=-1)  # (3, 3, Nv) array

    # Create the dictionary with the key "H"
    homography_dict = {"H": H}

    # Ensure the output path exists
    os.makedirs(homographies_output, exist_ok=True)

    # Save the dictionary to a .mat file
    mat_file_path = os.path.join(homographies_output, "homographies.mat")
    savemat(mat_file_path, homography_dict)
    
    print(f"Saved homographies to {mat_file_path}")

def compute_cumulative_homographies(parents, homographies, ref_idx=0):
    """
    Compute cumulative homographies to transform each image into the reference image's plane.
    """
    num_images = len(parents)
    cumulative_homographies = {ref_idx: np.eye(3)}

    for i in range(num_images):
        if i == ref_idx:
            continue

        path = []
        current = i
        while current is not None:
            parent = parents[current]
            if parent is not None:
                path.append((parent, current))
            current = parent

        # Compute cumulative homography along the path
        H = np.eye(3)
        for src, dst in reversed(path):
            H = H @ homographies[(src, dst)]
    
        cumulative_homographies[i] = H

    return cumulative_homographies

def compute_homography_matrix(src_pts, dst_pts):
    """
    Compute the homography matrix H that maps src_pts to dst_pts.
    Uses Direct Linear Transform (DLT) method.
    """
    if len(src_pts) < 4:
        return None

    num_points = len(src_pts)
    A = np.zeros((2 * num_points, 9))

    for i in range(num_points):
        x, y = src_pts[i].ravel()  # Flatten to get x, y coordinates
        u, v = dst_pts[i].ravel()  # Flatten to get u, v coordinates

        A[2 * i] = [-x, -y, -1, 0, 0, 0, x * u, y * u, u]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, x * v, y * v, v]

    # Solve using SVD
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)

    # Normalize
    H = H / H[2, 2]
    return H



def compute_inliers(H, src_pts, dst_pts, threshold):
    """
    Compute inliers given a homography matrix and point correspondences.
    Returns a boolean mask of inliers.
    """
    num_points = len(src_pts)
    inliers = np.zeros(num_points, dtype=bool)

    src_pts = src_pts.reshape(-1, 2)  # Reshape to (N, 2)
    dst_pts = dst_pts.reshape(-1, 2)  # Reshape to (N, 2)

    # Convert points to homogeneous coordinates
    src_homogeneous = np.hstack([src_pts, np.ones((num_points, 1))])
    dst_homogeneous = np.hstack([dst_pts, np.ones((num_points, 1))])

    # Transform source points
    transformed_pts = (H @ src_homogeneous.T).T
    transformed_pts = transformed_pts[:, :2] / transformed_pts[:, 2:]

    # Compute distances
    distances = np.linalg.norm(transformed_pts - dst_pts, axis=1)
    inliers = distances < threshold

    return inliers


def custom_find_homography(
    src_pts, dst_pts, method=None, ransac_reprojThreshold=5.0, ransac_iterations=72
):
    """
    Custom implementation of RANSAC to find the best homography matrix.

    Parameters:
    - src_pts, dst_pts: Nx2 arrays of corresponding points
    - method: Not used, included for compatibility with OpenCV interface
    - ransac_reprojThreshold: Distance threshold for inlier classification
    - ransac_iterations: Number of RANSAC iterations

    Returns:
    - best_H: Best homography matrix found
    - mask: Binary mask indicating inliers (Nx1 uint8 array)
    """
    if len(src_pts) < 4:
        return None, None

    src_pts = np.asarray(src_pts)
    dst_pts = np.asarray(dst_pts)
    num_points = len(src_pts)
    best_H = None
    best_inliers = None
    max_inliers = 0

    for _ in range(ransac_iterations):
        # Randomly select 4 point correspondences
        idx = np.random.choice(num_points, 4, replace=False)
        sample_src = src_pts[idx]
        sample_dst = dst_pts[idx]

        # Compute homography for these points
        H = compute_homography_matrix(sample_src, sample_dst)
        if H is None:
            continue

        # Find inliers
        inliers = compute_inliers(H, src_pts, dst_pts, ransac_reprojThreshold)
        num_inliers = np.sum(inliers)

        # Update best model if we found more inliers
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_inliers = inliers
            best_H = H

    # Recompute homography using all inliers
    if best_H is not None and np.sum(best_inliers) > 4:
        final_src = src_pts[best_inliers]
        final_dst = dst_pts[best_inliers]
        best_H = compute_homography_matrix(final_src, final_dst)

    print(f"Matches depois de Ransac: {np.sum(best_inliers)}")

    # Convert inliers to match OpenCV's format
    if best_inliers is not None:
        mask = np.uint8(best_inliers)[:, np.newaxis]
    else:
        mask = None

    return best_H, mask


def load_image_and_keypoints(image_path, keypoints_path):
    """Load image and its corresponding keypoints/descriptors."""
    img = cv.imread(image_path)
    if img is None:
        raise ValueError(f"Error loading image: {image_path}")

    kp_data = sio.loadmat(keypoints_path)
    keypoints = [cv.KeyPoint(x[0], x[1], 1) for x in kp_data["kp"]]
    descriptors = kp_data["desc"]

    return img, keypoints, descriptors


def find_matches(des1, des2):

    """
    Find k-NN matches with KDTree.
    
    Args:
        des1 (np.ndarray): Descriptors of the first image.
        des2 (np.ndarray): Descriptors of the second image.
    
    Returns:
        good_matches (list): List of tuples with indices of good matches.
    """

    des1 = np.array(des1)
    des2 = np.array(des2)

    # Build KDTree for des2
    tree = KDTree(des2)

    # Query k-nearest neighbors for each descriptor in des1
    dists, indices = tree.query(des1, k=2)

    # Apply the ratio test
    good_matches = []
    for i, (d, idx) in enumerate(zip(dists, indices)):
        if len(d) > 1 and d[0] < 0.7 * d[1]:
            good_matches.append((i, idx[0]))

    return good_matches

class DMatch:
    def __init__(self, queryIdx, trainIdx, distance):
        self.queryIdx = queryIdx  # Index in des1
        self.trainIdx = trainIdx  # Index in des2
        self.distance = distance  # Distance between descriptors


def find_matches_by_hand(des1, des2):
    """Find mutual matches between two sets of descriptors."""
    des1 = np.array(des1)
    des2 = np.array(des2)

    # Compute distances from each descriptor in des1 to each descriptor in des2
    distances = np.linalg.norm(des1[:, np.newaxis] - des2[np.newaxis, :], axis=2)

    # Find the best match for each descriptor in des1
    des1_to_des2 = np.argmin(distances, axis=1)
    des1_to_des2_dist = np.min(distances, axis=1)

    # Find the best match for each descriptor in des2
    des2_to_des1 = np.argmin(distances, axis=0)
    des2_to_des1_dist = np.min(distances, axis=0)

    # Mutual nearest neighbors
    good_matches = []
    for i, j in enumerate(des1_to_des2):
        if des2_to_des1[j] == i:
            # if des1_to_des2_dist[i] < 0.7 * des2_to_des1_dist[j]:
            # print("yuh yuh, here are the matches")
            good_matches.append(
                DMatch(queryIdx=i, trainIdx=j, distance=des1_to_des2_dist[i])
            )

    print(f"Número de matches MNN: {len(good_matches)}")
    return good_matches


def find_knn_matches_with_kdtree(des1, des2, k=2, ratio_thresh=0.7):
    """Find k-NN matches with KDTree."""
    des1 = np.array(des1)
    des2 = np.array(des2)

    # Build KDTree for des2
    tree = KDTree(des2)

    # Query k-nearest neighbors for each descriptor in des1
    dists, indices = tree.query(des1, k=k)

    # Apply the ratio test
    good_matches = []
    for i, (d, idx) in enumerate(zip(dists, indices)):
        if len(d) > 1 and d[0] < ratio_thresh * d[1]:
            good_matches.append(DMatch(queryIdx=i, trainIdx=idx[0], distance=d[0]))

    print(f"Número de matches ANN: {len(good_matches)}")
    return good_matches


def combine_images(images, homographies, ref_img_shape):
    """Combine all images using precomputed homographies relative to the reference image plane."""
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
    translation = np.array(
        [[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32
    )

    # Initialize result image
    result = np.zeros((max_y - min_y, max_x - min_x, 3), dtype=np.uint8)

    # Warp and place reference (first) image
    ref_H = translation  # The first image has no homography transformation
    cv.warpPerspective(
        images[0],
        ref_H,
        (max_x - min_x, max_y - min_y),
        dst=result,
        borderMode=cv.BORDER_TRANSPARENT,
    )

    # Warp and place each subsequent image
    for (img, H) in zip(images[1:], homographies):
        H_final = translation @ H
        cv.warpPerspective(
            img,
            H_final,
            (max_x - min_x, max_y - min_y),
            dst=result,
            borderMode=cv.BORDER_TRANSPARENT,
        )
    return result

import heapq  # For priority queue in shortest path calculation

import os
import numpy as np
import cv2 as cv


def process_image_sequence_with_shortest_path(ref_dir, input_dir, output_dir):
    """Process a sequence of images and create a panorama using shortest path to the reference."""

    # Dynamically load reference image and its keypoints
    ref_img_file = next((f for f in os.listdir(ref_dir) if f.endswith(".jpg")), None)
    ref_kp_file = next((f for f in os.listdir(ref_dir) if f.endswith(".mat")), None)

    if not ref_img_file or not ref_kp_file:
        print("Error: Reference image or keypoints file not found in reference directory.")
        return

    ref_img_path = os.path.join(ref_dir, ref_img_file)
    ref_kp_path = os.path.join(ref_dir, ref_kp_file)
    ref_img, ref_kp, ref_desc = load_image_and_keypoints(ref_img_path, ref_kp_path)

    ref_name = os.path.splitext(ref_img_file)[0]

    # Initialize lists to store images and homographies
    all_images = {ref_name: ref_img}
    keypoints = {ref_name: ref_kp}
    descriptors = {ref_name: ref_desc}

    homographies = {ref_name: np.eye(3, dtype=np.float32)}  # Identity for reference image

    # Graph structure for shortest path tracking
    parents = {ref_name: None}  # Track the parent image for each image

    # Get all image files from input directory
    input_files = sorted(
        [f for f in os.listdir(input_dir) if f.startswith("img_") and f.endswith(".jpg")]
    )

    for img_file in input_files:
        kp_file = f"kp_{img_file[4:-4]}.mat"
        img_path = os.path.join(input_dir, img_file)
        kp_path = os.path.join(input_dir, kp_file)

        print(f"Processing {img_file}...")

        # Load input image and its keypoints
        input_img, input_kp, input_desc = load_image_and_keypoints(img_path, kp_path)

        # Try to match directly with reference image first
        matches = find_knn_matches_with_kdtree(descriptors[ref_name], input_desc)
        src_pts = np.float32([keypoints[ref_name][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([input_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = custom_find_homography(dst_pts, src_pts, "RANSAC", 5.0)

        if H is not None and mask is not None and np.sum(mask) > 137:
            # If sufficient matches with reference, add to graph
            parents[img_file] = ref_name
            homographies[img_file] = H
            all_images[img_file] = input_img
            keypoints[img_file] = input_kp
            descriptors[img_file] = input_desc
            print(f"{img_file} matches directly with reference ({ref_name}).")
        else:
            # If insufficient matches with reference, traverse the graph
            to_visit = [ref_name]  # Start traversal from the reference
            matched = False

            while to_visit:
                current_parent = to_visit.pop(0)  # Get the next node to try

                print(f"Attempting to match {img_file} with {current_parent}...")

                matches = find_knn_matches_with_kdtree(descriptors[current_parent], input_desc)
                src_pts = np.float32(
                    [keypoints[current_parent][m.queryIdx].pt for m in matches]
                ).reshape(-1, 1, 2)
                dst_pts = np.float32(
                    [input_kp[m.trainIdx].pt for m in matches]
                ).reshape(-1, 1, 2)

                H, mask = custom_find_homography(dst_pts, src_pts, "RANSAC", 5.0)

                if H is not None and mask is not None and np.sum(mask) > 137:
                    parents[img_file] = current_parent
                    cumulative_homography = homographies[current_parent] @ H
                    homographies[img_file] = cumulative_homography
                    all_images[img_file] = input_img
                    keypoints[img_file] = input_kp
                    descriptors[img_file] = input_desc
                    print(f"{img_file} matches with {current_parent}.")
                    matched = True
                    break

                # If match fails, add current_parent's parent to the queue
                if current_parent in parents:
                    to_visit.extend(
                        [child for child, parent in parents.items() if parent == current_parent]
                    )
                print(f"to_visit: {to_visit}")

            if not matched:
                print(f"Error: Could not find sufficient matches for {img_file}.")

    # Create panorama using all images and homographies
    if len(homographies) > 1:
        sorted_images = [ref_name] + [img for img in input_files if img in homographies]
        image_list = [all_images[img] for img in sorted_images]
        homography_list = [homographies[img] for img in sorted_images[1:]]

        accumulated_img = combine_images(image_list, homography_list, ref_img.shape)

        #save the homographies as a dict structure
        output_homography = "homographies.mat"
        save_homographies(homography_list, output_homography)

        # Save final panorama
        final_output_path = os.path.join(output_dir, "final_panorama.jpg")
        cv.imwrite(final_output_path, accumulated_img)
        print(f"Saved final panorama to {final_output_path}")
    else:
        print("No valid homographies found.")

    # Print the graph of homographies
    print("\nGraph of Homographies:")
    for img, parent in parents.items():
        if parent:
            print(f"{img} -> {parent}")
        else:
            print(f"{img} is the root.")

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
    process_image_sequence_with_shortest_path(ref_dir, input_dir, output_dir)


if __name__ == "__main__":
    main()

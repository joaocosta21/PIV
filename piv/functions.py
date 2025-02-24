import os
import numpy as np
from scipy.io import loadmat, savemat
from scipy.linalg import svd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import cv2
from sklearn.neighbors import KDTree
from collections import deque

def load_keypoints_from_mat(mat_file_path):

    """
    Load keypoints from a .mat file containing matched points.
    Assumes the .mat file contains an array 'kp_gmaps' with shape (N, 4),
    where each row corresponds to [x1, y1, x2, y2].

    Args:
        mat_file_path (str): Path to the .mat file.
    Returns:
        keypoints_image1 (np.ndarray): Array of keypoints of size (N, 2) in the first image.
        keypoints_image2 (np.ndarray): Array of keypoints of size (N, 2) in the second image.
    """

    # Read the .mat file and extract the keypoints
    data = loadmat(mat_file_path)
    matches = data['kp_gmaps']
    u1, v1 = matches[:, 0], matches[:, 1]
    u2, v2 = matches[:, 2], matches[:, 3]

    # Combine the keypoints into arrays
    keypoints_image1 = np.vstack((u1, v1)).T
    keypoints_image2 = np.vstack((u2, v2)).T
    return keypoints_image1, keypoints_image2


def compute_homography(u1, v1, u2, v2):

    """
    Compute homography matrix using the Direct Linear Transform (DLT) method.

    Args:
        u1 (np.ndarray): Array of x-coordinates in the first image.
        v1 (np.ndarray): Array of y-coordinates in the first image.
        u2 (np.ndarray): Array of x-coordinates in the second image.
        v2 (np.ndarray): Array of y-coordinates in the second image.
    
    Returns:
        H (np.ndarray): Homography matrix of shape (3, 3).
    """

    # Check if the input arrays have the same length
    if not (len(u1) == len(v1) == len(u2) == len(v2)):
        raise ValueError("Input arrays must have the same length.")
    
    # Check if the input arrays have at least 4 points
    if len(u1) < 4:
        raise ValueError("At least 4 points are required to compute the homography matrix.")
    
    A = []
    # Build matrix A for the DLT method
    for i in range(len(u1)):
        x1, y1 = u1[i], v1[i]
        x2, y2 = u2[i], v2[i]
        A.append([-x1, -y1, -1,  0,   0,   0, x2 * x1, x2 * y1, x2])
        A.append([  0,   0,  0, -x1, -y1, -1, y2 * x1, y2 * y1, y2])
    
    # Perform Singular Value Decomposition (SVD) on matrix A and extract the homography matrix
    A = np.array(A)
    _, _, Vt = svd(A)
    H = Vt[-1].reshape(3, 3)
    H /= H[2, 2]
    
    return H


def import_yolo_data(yolo_folder="yolo"):

    """
    Import YOLO data from .mat files in the specified folder.

    Args:
        yolo_folder (str): Path to the folder containing YOLO .mat files.
    
    Returns:
        yolo_data (list): List of YOLO data dictionaries.
    """

    yolo_data = []

    # Get all .mat files in the specified folder sorted by filename
    filenames = sorted([f for f in os.listdir(yolo_folder) if f.startswith("yolo_") and f.endswith(".mat")])

    # Load each .mat file and append the data to the list
    for filename in filenames:
        file_path = os.path.join(yolo_folder, filename)
        data = loadmat(file_path)
        yolo_data.append(data)

    return yolo_data


def process_yolo_data(yolo_data, y_pos=0.25):

    """
    Process YOLO data to extract object IDs and box coordinates.

    Args:
        yolo_data (list): List of YOLO data dictionaries.
        y_pos (float): Vertical position of the object center in the bounding box.

    Returns:
        processed_data (list): List of processed YOLO data. The format is as follows:
            [
                [(obj_id, [x_coordinate, y_coordinate]), ...],  # Frame 1
                [(obj_id, [x_coordinate, y_coordinate]), ...],  # Frame 2
                ...
            ]
    """

    processed_data = []

    for frame in yolo_data:

        # Extract object IDs and bounding box coordinates
        frame_data = []
        ids = frame['id'].flatten()
        xyxy = frame['xyxy']

        # Compute the item interesting coordinates (center on x and y_pos on y)
        for i in range(len(ids)):
            obj_id = ids[i]
            bbox = xyxy[i]
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = bbox[1] + y_pos * (bbox[3] - bbox[1])
            frame_data.append([obj_id, np.array([x_center, y_center])])

        processed_data.append(frame_data)

    return processed_data


def point_homography_transformation(H, point):

    """
    Apply homography transformation to a single point.

    Args:
        H (np.ndarray): Homography matrix of shape (3, 3).
        point (tuple): Point coordinates (x, y).

    Returns:
        transformed_point (np.ndarray): Transformed point coordinates.
    """

    # Change to homogeneous coordinates and apply the homography matrix
    x, y = point
    x, y, z = np.dot(H, [x, y, 1])

    # Get back to Cartesian coordinates
    return np.array([x / z, y / z])


def transform_yolo_data(homography_data, yolo_data_input, yolo_data_output_dir):
    """
    Transforms YOLO bounding box data using the provided homography matrix
    and saves the transformed data.

    Args:
        homography_data (str): Path to the .mat file containing the homography matrix.
        yolo_data_input (list): List of YOLO data dictionaries.
        yolo_data_output_dir (str): Path to the directory to save the transformed YOLO data.
    """
    # Load homography matrix
    H = loadmat(homography_data)['H']

    output_folder = yolo_data_output_dir
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over data frames
    for frame_idx, frame_data in enumerate(yolo_data_input):
        # Get bounding box and associated data
        xyxy = frame_data['xyxy']
        ids = frame_data['id']
        classes = frame_data['class']

        # Transform bounding box coordinates
        transformed_boxes = []
        for box in xyxy:
            blc = np.array([box[0], box[1], 1])  # Bottom-left corner
            trc = np.array([box[2], box[3], 1])  # Top-right corner

            # Apply homography
            blc_transformed = np.dot(H, blc)
            trc_transformed = np.dot(H, trc)

            # Normalize to convert from homogeneous coordinates
            blc_transformed /= blc_transformed[2]
            trc_transformed /= trc_transformed[2]

            # Store the transformed bounding box
            transformed_boxes.append([
                blc_transformed[0], blc_transformed[1],  # Transformed bottom-left
                trc_transformed[0], trc_transformed[1],  # Transformed top-right
            ])

        transformed_boxes = np.array(transformed_boxes)

        # Save transformed YOLO data to a new .mat file
        output_file = os.path.join(yolo_data_output_dir, f'yolooutput_{frame_idx+1:04d}.mat')
        transformed_data = {
            'xyxy': transformed_boxes,
            'id': ids,
            'class': classes,
        }
        savemat(output_file, transformed_data)
    print(f"Transformed YOLO data saved to {output_folder}")


def plot_points_on_image(image_path, points, title, output_path):

    """
    Plot points on the given image and save the output.

    Args:
        image_path (str): Path to the input image.
        points (np.ndarray): Array of points of shape (N, 2).
        title (str): Title of the plot.
        output_path (str): Path to save the output image.
    """

    img = plt.imread(image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.scatter(points[:, 0], points[:, 1], c='red', marker='o')
    plt.title(title)
    plt.axis('off')
    height, width = img.shape[:2]
    plt.xlim(0, width)
    plt.ylim(height, 0)

    # Ensure the directory for output_path exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_side_by_side_images(left_image_path, right_image_path, left_points, right_points, title, output_path):
    
    """
    Plot two images side by side with different points overlaid on each,
    excluding points outside the image bounds.

    Args:
        left_image_path (str): Path to the left image.
        right_image_path (str): Path to the right image.
        left_points (np.ndarray): Array of points in the left image of shape (N, 2).
        right_points (np.ndarray): Array of points in the right image of shape (N, 2).
        title (str): Title of the plot.
        output_path (str): Path to save the output image.
    """

    # Read images
    img_left = cv2.imread(left_image_path)
    img_right = cv2.imread(right_image_path)

    # Check if images were loaded successfully
    if img_left is None:
        raise FileNotFoundError(f"Could not load image at {left_image_path}")
    if img_right is None:
        raise FileNotFoundError(f"Could not load image at {right_image_path}")

    # Ensure both images have the same number of channels
    if img_left.shape[2] != img_right.shape[2]:
        if img_left.shape[2] == 1:
            img_left = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
        if img_right.shape[2] == 1:
            img_right = cv2.cvtColor(img_right, cv2.COLOR_GRAY2BGR)

    # Get dimensions of the images
    left_height, left_width = img_left.shape[:2]
    right_height, right_width = img_right.shape[:2]

    # Filter left_points within image bounds
    if len(left_points) > 0:
        left_points_filtered = left_points[
            (left_points[:, 0] >= 0) & (left_points[:, 0] < left_width) &
            (left_points[:, 1] >= 0) & (left_points[:, 1] < left_height)
        ]
    else:
        left_points_filtered = left_points

    # Filter right_points within image bounds
    if len(right_points) > 0:
        right_points_filtered = right_points[
            (right_points[:, 0] >= 0) & (right_points[:, 0] < right_width) &
            (right_points[:, 1] >= 0) & (right_points[:, 1] < right_height)
        ]
    else:
        right_points_filtered = right_points

    # Draw points on images
    for point in left_points_filtered:
        cv2.circle(img_left, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

    for point in right_points_filtered:
        cv2.circle(img_right, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

    # Pad images to have the same height
    max_height = max(left_height, right_height)
    if left_height < max_height:
        pad_top = (max_height - left_height) // 2
        pad_bottom = max_height - left_height - pad_top
        img_left = cv2.copyMakeBorder(img_left, pad_top, pad_bottom, 0, 0,
                                      cv2.BORDER_CONSTANT, value=[0, 0, 0])
    if right_height < max_height:
        pad_top = (max_height - right_height) // 2
        pad_bottom = max_height - right_height - pad_top
        img_right = cv2.copyMakeBorder(img_right, pad_top, pad_bottom, 0, 0,
                                       cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Concatenate images side by side
    combined_image = cv2.hconcat([img_left, img_right])

    # Add title to the combined image
    title_bar_height = 50
    title_bar = np.zeros((title_bar_height, combined_image.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar, title, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2, cv2.LINE_AA)

    # Stack the title bar and combined image vertically
    final_image = cv2.vconcat([title_bar, combined_image])

    # Ensure the directory for output_path exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the final image
    cv2.imwrite(output_path, final_image)


def create_video_from_images(image_folder, video_path, fps=10):

    """
    Create a video from a sequence of images in a folder using OpenCV.

    Args:
        image_folder (str): Path to the folder containing images.
        video_path (str): Path to save the output video.
        fps (int): Frames per second for the output video.
    """

    filenames = sorted([f for f in os.listdir(image_folder) if f.endswith(".png") or f.endswith(".jpg")])

    if not filenames:
        print("No images found in the folder.")
        return

    # Determine frame size from the first image
    first_image_path = os.path.join(image_folder, filenames[0])
    first_image = cv2.imread(first_image_path)

    if first_image is None:
        raise FileNotFoundError(f"Could not read the first image: {first_image_path}")
    
    # Get frame size (width, height)
    frame_size = (first_image.shape[1], first_image.shape[0])

    # Ensure the directory for video_path exists
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

    # Write each image to the video
    for filename in filenames:
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)

        if img is not None:
            if (img.shape[1], img.shape[0]) != frame_size:
                img = cv2.resize(img, frame_size)
            out.write(img)

        else:
            print(f"Warning: Could not read image {img_path}")

    out.release()
    print(f"Video saved to {video_path}")


def compute_inliers(H, src_pts, dst_pts, threshold):

    """
    Compute inliers given a homography matrix and point correspondences.

    Args:
        H (np.ndarray): Homography matrix of shape (3, 3).
        src_pts (np.ndarray): Source points of shape (N, 2).
        dst_pts (np.ndarray): Destination points of shape (N, 2).
        threshold (float): Threshold for reprojection error.
    
    Returns:
        inliers (np.ndarray): Boolean mask of inliers.
    """

    # Convert source points to homogeneous coordinates
    num_points = len(src_pts)
    src_pts_homogeneous = np.hstack([src_pts, np.ones((num_points, 1))])

    # Compute transformed source points
    transformed_src_pts = (H @ src_pts_homogeneous.T).T
    transformed_src_pts /= transformed_src_pts[:, [2]]

    # Compute Euclidean distance between transformed source points and destination points
    errors = np.linalg.norm(transformed_src_pts[:, :2] - dst_pts, axis=1)

    # Determine inliers based on the threshold
    inliers = errors < threshold
    return inliers


def ransac_find_homography(src_pts, dst_pts, ransac_threshold = 5.0, ransac_iterations = 1000):

    """
    Implementation of RANSAC to find the best homography matrix.

    Args:
        src_pts (np.ndarray): Source points of shape (N, 2).
        dst_pts (np.ndarray): Destination points of shape (N, 2).
        ransac_reprojThreshold (float): Threshold for reprojection error.
        ransac_iterations (int): Number of RANSAC iterations.

    Returns:
        best_H (np.ndarray): Best homography matrix of shape (3, 3).
        best_inliers_mask (np.ndarray): Boolean mask of inliers.
    """

    # Check if the input arrays have the same length and contain at least 4 points
    if len(src_pts) < 4:
        raise ValueError("At least 4 points are required to compute a homography.")
    
    if len(src_pts) != len(dst_pts):
        raise ValueError("Source and destination points must have the same length.")

    # Initialize variables to store the best homography and inliers
    best_H = None
    max_inliers = 0
    num_points = len(src_pts)
    best_inliers_mask = None

    # Perform RANSAC iterations
    for _ in range(ransac_iterations):
        
        # Randomly select 4 points
        indices = np.random.choice(num_points, 4)
        u1_sample = src_pts[indices, 0]
        v1_sample = src_pts[indices, 1]
        u2_sample = dst_pts[indices, 0]
        v2_sample = dst_pts[indices, 1]

        # Compute homography matrix using the 4 random points
        H_candidate = compute_homography(u1_sample, v1_sample, u2_sample, v2_sample)
        inliers_mask = compute_inliers(H_candidate, src_pts, dst_pts, ransac_threshold)
        num_inliers = np.sum(inliers_mask)

        # Update the best homography if the current one has more inliers
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_H = H_candidate
            best_inliers_mask = inliers_mask

    # Compute the final homography using all inliers found by RANSAC
    if best_inliers_mask is not None and max_inliers >= 4:
        inlier_src_pts = src_pts[best_inliers_mask]
        inlier_dst_pts = dst_pts[best_inliers_mask]
        best_H = compute_homography(inlier_src_pts[:, 0], inlier_src_pts[:, 1], inlier_dst_pts[:, 0], inlier_dst_pts[:, 1])

    return best_H, best_inliers_mask


def load_ref_image_and_keypoints(ref_dir):
    
    """
    Load the reference image and keypoints from the specified directory.

    Args:
        ref_dir (str): Path to the directory containing the reference image and keypoints.

    Returns:
        ref_image (np.ndarray): Reference image.
        keypoints (np.ndarray): Reference keypoints.
        descriptors (np.ndarray): Reference descriptors.
    """

    # Get paths to the reference image and keypoints
    ref_img_name = next((f for f in os.listdir(ref_dir) if f.endswith(".jpg")), None)
    ref_kp_name = next((f for f in os.listdir(ref_dir) if f.endswith(".mat")), None)

    # Check if the reference image and keypoints are found
    if ref_img_name is None:
        raise FileNotFoundError("Reference image file not found in the directory. Please provide a .jpg file.")
    
    if ref_kp_name is None:
        raise FileNotFoundError("Reference keypoints file not found in the directory. Please provide a .mat file.")

    # Construct full paths
    ref_img_path = os.path.join(ref_dir, ref_img_name)
    ref_kp_path = os.path.join(ref_dir, ref_kp_name)

    # Load the reference image
    ref_image = cv2.imread(ref_img_path)
    if ref_image is None:
        raise FileNotFoundError(f"Could not read the reference image at {ref_img_path}")

    # Load the reference keypoints data
    kp_data = loadmat(ref_kp_path)

    # Check if key points and descriptors are present in the .mat file
    if "kp" not in kp_data or "desc" not in kp_data:
        raise ValueError("Key points or descriptors not found in the .mat file.")
    
    # Extract keypoints and descriptors
    keypoints = kp_data["kp"]
    descriptors = kp_data["desc"]

    return ref_image, keypoints, descriptors


def load_image_and_keypoints(image_path, kp_path):
    
    """
    Load an image and keypoints from the specified paths.

    Args:
        image_path (str): Path to the image file.
        kp_path (str): Path to the keypoints file.

    Returns:
        img (np.ndarray): Image data.
        keypoints (np.ndarray): Keypoints.
        descriptors (np.ndarray): Descriptors.
    """

    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read the image at {image_path}")

    # Load the keypoints data
    kp_data = loadmat(kp_path)

    # Check if key points and descriptors are present in the .mat file
    if "kp" not in kp_data or "desc" not in kp_data:
        raise ValueError("Key points or descriptors not found in the .mat file.")

    # Extract keypoints and descriptors
    keypoints = kp_data["kp"]
    descriptors = kp_data["desc"]

    return img, keypoints, descriptors


def save_homographies(homographies_list, homographies_output):
    """
    Save a list of homography matrices in a single .mat file.
    homographies_list: list of 3x3 numpy arrays
    homographies_output: directory to store 'homographies.mat'
    """
    H = np.stack(homographies_list, axis=-1)  # shape (3, 3, N)
    homography_dict = {"H": H}

    os.makedirs(homographies_output, exist_ok=True)
    mat_file_path = os.path.join(homographies_output, "homographies.mat")
    savemat(mat_file_path, homography_dict)
    print(f"Saved homographies to {mat_file_path}")


def find_matches(des1, des2, k = 2, ratio_thresh = 0.7):

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
    dists, indices = tree.query(des1, k=k)

    # Apply the ratio test
    good_matches = []
    for i, (d, idx) in enumerate(zip(dists, indices)):
        if len(d) > 1 and d[0] < ratio_thresh * d[1]:
            good_matches.append((i, idx[0]))

    return good_matches


class Panorama_Img:

    """
    Class to store information about an image used in panorama generation.

    Attributes:
        image (np.ndarray): Image data.
        keypoints (np.ndarray): Keypoints detected in the image.
        descriptors (np.ndarray): Descriptors associated with the keypoints.
        homography (np.ndarray): Homography matrix relative to the reference image.
        parent (Panorama_Img): Parent image in the panorama graph.
    """

    def __init__(self, ref_image, ref_keypoints, ref_descriptors, homography = None, layer = None):
        self.image = ref_image
        self.keypoints = ref_keypoints
        self.descriptors = ref_descriptors
        self.homography = homography
        self.parent = None
        self.layer = layer


def build_panorama_list(ref_dir, input_dir):
    
    """
    Build a list of Panorama_Img objects from the reference image and input images.

    Args:
        ref_dir (str): Path to the directory containing the reference image and keypoints.
        input_dir (str): Path to the directory containing the input images.

    Returns:
        images (list): List of Panorama_Img objects.
    """

    # Load the reference image and keypoints
    ref_image, ref_keypoints, ref_descriptors = load_ref_image_and_keypoints(ref_dir)

    # Initialize the list of Panorama_Img objects
    images = [Panorama_Img(ref_image, ref_keypoints, ref_descriptors, np.eye(3), 0)]

    # Get paths to the input images
    input_image_names = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpg")])

    # Build Panorama_Img objects for each input image and create the list
    for img_name in input_image_names:

        # Get paths to the image and keypoints
        img_path = os.path.join(input_dir, img_name)
        kp_name = f"kp_{img_name[4:-4]}.mat"
        kp_path = os.path.join(input_dir, kp_name)

        # Build and append the Panorama_Img object to the list
        images.append(Panorama_Img(*load_image_and_keypoints(img_path, kp_path)))

    return images


def build_layered_tree(images, num_min_matches=100, inlier_threshold = 5.0, ransac_iterations = 1000, num_random_matches = np.inf):
    
    """
    Build a layered tree (rooted in the reference image, which is images[0])
    to find the best path from image[0] to all other images in 'images'. The structure is:
      - layer 0 is images[0]
      - layer 1 are images that directly match images[0]
      - layer 2 are images that match any image in layer 1
      - etc.

    Each Panorama_Img object should have (at minimum):
        images[i].descriptors       # NxD descriptor array
        images[i].keypoints         # Nx2 keypoint locations
        images[i].homography        # 3x3 transform to map from parent plane to this image's plane
        images[i].parent            # index of the parent image, or None if it's the root
        images[i].layer             # integer layer index (0 for the root, 1 for direct children, etc.)

    Args:
        images (list): List of Panorama_Img objects, where images[0] is the reference image.
        num_min_matches (int): Minimum number of mutual nearest-neighbor matches 
                               required to connect two images in the tree.

    Returns:
        images (list): Same list of Panorama_Img objects, but updated in place with:
            - .parent set to an integer index (or None for root)
            - .layer set to an integer denoting BFS level
            - .homography set to the estimated transform from the parent image
    """

    # Ensure root image (index 0) initialization as layer=0, no parent, identity homography
    images[0].parent = None
    images[0].layer = 0
    images[0].homography = np.eye(3, dtype=np.float32)

    # Initialize layer track variable
    layer = 0

    # Initialize best match in layer
    best_matching_images = {}

    # Keep track of visited nodes to avoid reprocessing
    visited = set([0])
    queue = [0]

    while len(visited) < len(images):

        # Check if it's time to move to the next layer
        if queue == []:

            # Increment layer
            layer += 1
            # print("\n----- Next Layer -----\n")

            # Update images and queue given the best matches
            for match in best_matching_images:
                images[match].parent = best_matching_images[match]["parent"]
                images[match].layer = layer
                images[match].homography = best_matching_images[match]["homography"]
                queue.append(match)
            
            # Reset best match in layer
            best_matching_images = {}

        # print("\nQueue:", queue)
        # print("Visited:", visited)

        # Get current image index and mark it as visited
        try:
            current = queue.pop(0)
            visited.add(current)
        except:
            break

        # Get current image descriptors and keypoints
        des_current = images[current].descriptors
        kpts_current = images[current].keypoints

        # Skip if no descriptors or too few
        if des_current is None or des_current.shape[0] < 4:
            # print(f"Skipping image {current} due to lack of descriptors.")
            continue

        # Try to match unvisited images
        for candidate in range(len(images)):
            if candidate in visited or candidate in queue or (best_matching_images.get(candidate, None) is not None and best_matching_images[candidate]["num_tries"] >= num_random_matches):
                continue

            # Get candidate image descriptors and keypoints
            des_candidate = images[candidate].descriptors
            kpts_candidate = images[candidate].keypoints

            if des_candidate is None or des_candidate.shape[0] < 4:
                # print(f"Skipping image {candidate} due to lack of descriptors.")
                continue

            # Find matches between current and candidate
            candidate_matches = find_matches(des_current, des_candidate)

            # Run RANSAC to find homography and number of inliers
            src_pts = []
            dst_pts = []
            for (idx_cur, idx_cand) in candidate_matches:
                src_pts.append(kpts_current[idx_cur])
                dst_pts.append(kpts_candidate[idx_cand])
            src_pts = np.float32(src_pts).reshape(-1, 2)
            dst_pts = np.float32(dst_pts).reshape(-1, 2)

            # print(f"\nRunning RANSAC for images {current} and {candidate}.")

            H, mask = ransac_find_homography(dst_pts, src_pts, ransac_threshold = inlier_threshold, ransac_iterations = ransac_iterations)
            num_matches = np.sum(mask)
            # print(f"Number of matches: {num_matches}")

            if H is None or mask is None:
                print(f"Error during RANSAC for images {current} and {candidate}.")
                continue

            if num_matches < num_min_matches:
                # print(f"Skipping image {candidate} due to lack of matches.")
                continue

            # Check if the match is better than the current best match and update if so
            if candidate not in best_matching_images or num_matches > best_matching_images[candidate]["num_matches"]:
                best_matching_images[candidate] = {
                    "parent": current,
                    "homography": H,
                    "num_matches": num_matches,
                    "num_tries": best_matching_images[candidate]["num_tries"] + 1 if candidate in best_matching_images else 1
                }
    
    for image in images:
        print(f"Image {images.index(image)}: Parent {image.parent}, Layer {image.layer}")

    return images


def compute_final_homographies(images):

    """
    Compute homography for each image to the reference image from the layered tree.

    Args:
        images (list): List of Panorama_Img objects.

    Returns:
        images (list): List of Panorama_Img objects with updated homographies.
    """

    # Create a copy of images to store the new homographies
    new_images = [Panorama_Img(image.image, image.keypoints, image.descriptors, homography = image.homography, layer = image.layer) for image in images]

    # Iterate over images to build homographies
    for i in range(1, len(images)):
        current = i
        homography = np.eye(3, dtype=np.float32)

        # Traverse the tree to build the homography
        while images[current].parent is not None:
            homography = images[current].homography @ homography
            current = images[current].parent

        # Store the homography for the current image in the new list
        new_images[i].homography = homography

    return new_images


def combine_images(images):
    """
    Combine all images using precomputed homographies relative to the reference image plane.
    
    Args:
        images (list): List of Panorama_Img objects. Each object has:
            - image: ndarray (H x W x 3)
            - homography: 3x3 ndarray mapping the reference plane to this image's plane.
        
    Returns:
        result (np.ndarray): Combined panorama image.
    """

    # Reference image is images[0]
    ref_image = images[0].image
    h_ref, w_ref = ref_image.shape[:2]

    # Reference-corner coordinates
    corners_ref = np.float32(
        [[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]]
    ).reshape(-1, 1, 2)

    # Collect transformed corners for all images to find bounding box
    all_corners = []
    for img_obj in images:
        H_ref_to_img = img_obj.homography
        transformed = cv2.perspectiveTransform(corners_ref, H_ref_to_img)
        all_corners.append(transformed)
    all_corners = np.concatenate(all_corners, axis=0)

    # Determine range of panorama
    min_x = int(np.floor(np.min(all_corners[:, :, 0])))
    max_x = int(np.ceil(np.max(all_corners[:, :, 0])))
    min_y = int(np.floor(np.min(all_corners[:, :, 1])))
    max_y = int(np.ceil(np.max(all_corners[:, :, 1])))

    width = max_x - min_x
    height = max_y - min_y

    # Create canvas
    panorama = np.zeros((height, width, 3), dtype=np.uint8)

    # Shift translation so everything fits on the canvas
    shift = np.array([[1, 0, -min_x],
                      [0, 1, -min_y],
                      [0, 0, 1]], dtype=np.float32)

    # Warp each image to panorama
    for img_obj in images:
        H_ref_to_img = img_obj.homography
        warp_matrix = shift @ H_ref_to_img

        print(f"\nWarping image {images.index(img_obj)}")

        warped = cv2.warpPerspective(img_obj.image, warp_matrix, (width, height))
        
        # Simple overlay blending: overwrite black pixels in panorama
        mask = np.any(warped > 0, axis=-1)
        panorama[mask] = warped[mask]

    return panorama


def transform_yolo_multiple_homographies(yolo_data, homographies_list, output_folder):
    """
    Transform YOLO bounding box data using the provided homography matrices.

    Args:
        yolo_data (list): List of YOLO data dictionaries.
        homographies_list (list): List of homography matrices.
        output_dir (str): Path to the directory to save the transformed YOLO data.
    """

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over data frames
    for frame_idx, frame_data in enumerate(yolo_data):
        # Get bounding box and associated data
        xyxy = frame_data['xyxy']
        ids = frame_data['id']
        classes = frame_data['class']

        # Transform bounding box coordinates
        transformed_boxes = []
        for box in xyxy:
            blc = np.array([box[0], box[1], 1])  # Bottom-left corner
            trc = np.array([box[2], box[3], 1])  # Top-right corner

            # Apply homography
            blc_transformed = np.dot(homographies_list[frame_idx], blc)
            trc_transformed = np.dot(homographies_list[frame_idx], trc)

            # Normalize to convert from homogeneous coordinates
            blc_transformed /= blc_transformed[2]
            trc_transformed /= trc_transformed[2]

            # Store the transformed bounding box
            transformed_boxes.append([
                blc_transformed[0], blc_transformed[1],  # Transformed bottom-left
                trc_transformed[0], trc_transformed[1],  # Transformed top-right
            ])

        transformed_boxes = np.array(transformed_boxes)

        # Save transformed YOLO data to a new .mat file
        output_file = os.path.join(output_folder, f'yolooutput_{frame_idx+1:04d}.mat')
        transformed_data = {
            'xyxy': transformed_boxes,
            'id': ids,
            'class': classes,
        }
        savemat(output_file, transformed_data)
    print(f"Transformed YOLO data saved to {output_folder}")


def transform_corners(corners, H):
    """
    Transforms an array of 2D corners using a homography matrix.
    corners: (N x 2) numpy array
    H: (3 x 3) homography
    Returns: (N x 2) transformed corners
    """
    # Convert corners to homogeneous form
    ones = np.ones((corners.shape[0], 1), dtype=np.float32)
    corners_h = np.hstack([corners, ones])  # (N x 3)
    
    # Apply homography
    transformed = (H @ corners_h.T).T  # (N x 3)
    
    # Normalize
    transformed[:, 0] /= transformed[:, 2]
    transformed[:, 1] /= transformed[:, 2]
    
    return transformed[:, :2]


def compute_polygon_area(vertices):
    """
    Compute area of a polygon using the shoelace formula.
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def is_inside(p, cp1, cp2):
    """
    Returns True if point p is on same side as cp2 relative to line cp1->cp2.
    """
    return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

def get_intersection(s, e, cp1, cp2):
    """
    Returns intersection point of line segment s->e with line cp1->cp2.
    """
    dc = np.array([cp1[0] - cp2[0], cp1[1] - cp2[1]])
    dp = np.array([s[0] - e[0], s[1] - e[1]])
    n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
    n2 = s[0] * e[1] - s[1] * e[0]
    n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
    return np.array([(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3])

def clip_polygon(subject_polygon, clip_polygon):
    """
    Sutherland-Hodgman polygon clipping algorithm.
    """
    output_list = subject_polygon
    cp1 = clip_polygon[-1]

    for i in range(len(clip_polygon)):
        cp2 = clip_polygon[i]
        input_list = output_list
        output_list = []
        if len(input_list) == 0:
            break
        s = input_list[-1]

        for j in range(len(input_list)):
            e = input_list[j]
            if is_inside(e, cp1, cp2):
                if not is_inside(s, cp1, cp2):
                    output_list.append(get_intersection(s, e, cp1, cp2))
                output_list.append(e)
            elif is_inside(s, cp1, cp2):
                output_list.append(get_intersection(s, e, cp1, cp2))
            s = e
        cp1 = cp2
    return np.array(output_list)

def compute_polygon_intersection_area(corners1, corners2):
    """
    Compute intersection area between two quadrilaterals and return IoU.
    """
    # Compute intersection polygon
    intersection_poly = clip_polygon(corners1, corners2)
    
    if len(intersection_poly) == 0:
        return 0.0
        
    # Compute areas
    area1 = compute_polygon_area(corners1)
    area2 = compute_polygon_area(corners2)
    intersection_area = compute_polygon_area(intersection_poly)
    union_area = area1 + area2 - intersection_area
    
    if union_area <= 0:
        return 0.0
        
    return intersection_area / union_area


def build_panorama_list_lite(ref_dir, input_dir):
    
    """
    Build a list of Panorama_Img objects from the reference image and input images.

    Args:
        ref_dir (str): Path to the directory containing the reference image and keypoints.
        input_dir (str): Path to the directory containing the input images.

    Returns:
        images (list): List of Panorama_Img objects.
    """

    # Initialize the list of Panorama_Img objects
    images = []

    # Get paths to the input images
    input_image_names = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpg")])

    # Build Panorama_Img objects for each input image and create the list
    for img_name in input_image_names:

        # Get paths to the image and keypoints
        img_path = os.path.join(input_dir, img_name)
        kp_name = f"kp_{img_name[4:-4]}.mat"
        kp_path = os.path.join(input_dir, kp_name)

        # Build and append the Panorama_Img object to the list
        images.append(Panorama_Img(*load_image_and_keypoints(img_path, kp_path)))

    return images


def homography_heuristic_match(images, num_min_matches=100, inlier_threshold=5.0, ransac_iterations=100, min_overlap=0.2):

    """
    1) Compute partial homographies only between adjacent images (i -> i+1).
       Compose them so each image has a final homography to the reference (index 0).
    2) Once each image is in the reference frame, compute bounding-box overlaps for
       *all* pairs (i, j). If IoU >= min_overlap, record the match.

    Args:
        images (List[Panorama_Img]): 
          Each has .image, .keypoints, .descriptors, and will store .homography to the reference.
        num_min_matches (int): Min inliers in RANSAC to accept a partial homography.
        inlier_threshold (float): RANSAC reprojection threshold.
        ransac_iterations (int): Number of RANSAC iterations.
        min_overlap (float): IoU threshold for deciding if two images overlap in the reference frame.

    Returns:
        List[Tuple[int, int]]: All pairs (i, j) whose projected bounding boxes overlap.
    """

    # 0) Initialize the reference image (index 0)
    images[0].parent = None
    images[0].homography = np.eye(3, dtype=np.float32)

    # 1) Compute partial homographies for adjacent pairs
    n = len(images)
    for i in range(n - 1):
        # Match descriptors between images[i] and images[i+1]
        pairs = find_matches(images[i].descriptors, images[i+1].descriptors)
        print(f"Found {len(pairs)} matches between images {i} and {i+1}")
        if len(pairs) < num_min_matches:
            continue
        
        # Extract corresponding keypoints
        src_pts = np.float32([images[i].keypoints[m[0]] for m in pairs])
        dst_pts = np.float32([images[i+1].keypoints[m[1]] for m in pairs])

        # RANSAC homography: (i+1) -> i
        H_partial, mask = ransac_find_homography(dst_pts, src_pts,
                                                 ransac_threshold=inlier_threshold,
                                                 ransac_iterations=ransac_iterations)
        if H_partial is None or mask is None or np.sum(mask) < num_min_matches:
            continue
        
        # Compose final homography to reference
        images[i+1].homography = images[i].homography @ H_partial

    for image in range(len(images)):
        if image % 1 == 0 and image < 20:
            print(f"Computed homography for image {image}: {images[image].homography}")

    # 2) Transform corners to reference frame
    transformed_corners_list = []
    for i in range(n):
        if not hasattr(images[i], "homography"):
            images[i].homography = np.eye(3, dtype=np.float32)

        h_img, w_img = images[i].image.shape[:2]
        corners = np.float32([[0, 0],
                            [w_img, 0],
                            [w_img, h_img],
                            [0, h_img]])
        transformed = transform_corners(corners, images[i].homography)
        transformed_corners_list.append(transformed)

    # 3) Compare actual polygon overlaps for all pairs
    matches = []
    for i in range(n):
        for j in range(i + 1, n):
            overlap = compute_polygon_intersection_area(
                transformed_corners_list[i],
                transformed_corners_list[j]
            )
            if overlap >= min_overlap:
                matches.append((i, j))

    return matches


def generate_panorama(ref_dir, input_dir, output_dir, num_min_matches = 100, inlier_threshold = 5.0, ransac_iterations = 1000, num_random_matches = np.inf):

    """
    Generate a panorama using the reference image and images in the input directory.

    Args:
        ref_dir (str): Path to the directory containing the reference image and keypoints.
        input_dir (str): Path to the directory containing the input images.
        output_dir (str): Path to save the output panorama.
        inlier_threshold (float): Threshold for error in RANSAC.
        num_min_matches (int): Minimum number of matches needed to link two images.
        ransac_iterations (int): Number of RANSAC iterations.
    """

    # Create panorama list
    images = build_panorama_list(ref_dir, input_dir)
    print("Panorama list created.")

    # Build layered tree with homographies between images
    images = build_layered_tree(images, num_min_matches = num_min_matches, inlier_threshold = inlier_threshold, ransac_iterations = ransac_iterations, num_random_matches = num_random_matches)
    print("Layered tree built.")

    # Compute homography for each image to the reference image from the layered tree
    images = compute_final_homographies(images)
    print("Final homographies computed.")

    for image in images:
        print(f"\nImage {images.index(image)} homography:\n {image.homography}")

    # Combine images using homographies
    panorama = combine_images(images)
    print("Images combined.")

    # Save the homographies to a .mat file
    homographies_list = [img.homography for img in images]
    save_homographies(homographies_list, output_dir)

    # Save the panorama
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "panorama.jpg")
    cv2.imwrite(output_path, panorama)
    print(f"Panorama saved to {output_path}")


def compute_homographies_and_yolo_transform(ref_dir, input_dir, yolo_dir, output_dir, num_min_matches = 100, inlier_threshold = 5.0, ransac_iterations = 1000, num_random_matches = np.inf):

    """
    Compute homographies between images and transform YOLO bounding boxes.

    Args:
        ref_dir (str): Path to the directory containing the reference image and keypoints.
        input_dir (str): Path to the directory containing the input images.
        output_dir (str): Path to save the output panorama.
        inlier_threshold (float): Threshold for error in RANSAC.
        num_min_matches (int): Minimum number of matches needed to link two images.
        ransac_iterations (int): Number of RANSAC iterations.
    """

    # Build panorama list
    images = build_panorama_list(ref_dir, input_dir)
    print("Panorama list created.")

    # Build layered tree with homographies between images
    images = build_layered_tree(images, num_min_matches = num_min_matches, inlier_threshold = inlier_threshold, ransac_iterations = ransac_iterations, num_random_matches = num_random_matches)  
    print("Layered tree built.")

    # Compute homography for each image to the reference image from the layered tree
    images = compute_final_homographies(images)
    print("Final homographies computed.")

    # Load YOLO data
    yolo_data = import_yolo_data(yolo_dir)

    # Transform YOLO data using the computed homographies
    homographies_list = [img.homography for img in images]
    transform_yolo_multiple_homographies(yolo_data, homographies_list, output_dir)

    

def search_best_path(images, matches, reference_index):

    """
    Discover the best path from the reference image to all other images.

    Args:
        images (list): List of Panorama_Img objects.
        matches (list): List of tuples representing edges between images (i, j).
        reference_index (int): Index of the reference image.

    Returns:
        list: Updated list of Panorama_Img objects with parent attribute set.
    """

    # Initialize graph
    n = len(images)
    graph = {i: [] for i in range(n)}
    for i, j in matches:
        graph[i].append(j)
        graph[j].append(i)

    # BFS to find the best path
    queue = deque([reference_index])
    visited = set()
    visited.add(reference_index)

    while queue:
        current = queue.popleft()
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                images[neighbor].parent = current

    return images

def compute_homographies_from_best_path(images):

    """
    Compute both partial and cumulative homographies for all images relative to the reference image.

    Args:
        images (list): List of Panorama_Img objects.

    Returns:
        images (list): List of Panorama_Img objects with updated homographies.
    """

    # Build the dependency graph as an adjacency list
    graph = {i: [] for i in range(len(images))}
    for i, image in enumerate(images):
        if image.parent is not None:
            graph[image.parent].append(i)

    # Compute a topological order using Depth First Search
    def topological_sort(graph):
        visited = set()
        order = []

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for child in graph[node]:
                dfs(child)
            order.append(node)

        for node in graph:
            if node not in visited:
                dfs(node)

        return order[::-1]  # Reverse to get the correct order

    topo_order = topological_sort(graph)

    # Compute homographies in topological order
    for current in topo_order:
        if images[current].parent is not None:
            parent_idx = images[current].parent
            print(f"Finding matches and homography for Image {current} (Parent: {parent_idx})")
            matches = find_matches(images[current].descriptors, images[parent_idx].descriptors)

            # Extract matched points
            src_pts = np.float32([images[current].keypoints[m[0]] for m in matches])
            dst_pts = np.float32([images[parent_idx].keypoints[m[1]] for m in matches])

            # Compute the partial homography using RANSAC
            H, _ = ransac_find_homography(src_pts, dst_pts)

            # Compute cumulative homography
            images[current].homography = images[parent_idx].homography @ H
        
        else:
            # For the root image (reference), the cumulative homography is identity
            images[current].homography = np.eye(3, dtype=np.float32)

    return images

def check_yolo_data(yolo_folder):
    """
    Check if the specified folder contains any YOLO .mat files.

    Args:
        yolo_folder (str): Path to the folder containing YOLO .mat files.

    Returns:
        bool: True if there are YOLO .mat files, False otherwise.
        list: List of YOLO .mat filenames.
    """
    # Get a list of YOLO .mat files in the folder
    yolo_files = [f for f in os.listdir(yolo_folder) if f.startswith("yolo_") and f.endswith(".mat")]

    # Return True if any .mat files are found, otherwise False
    return len(yolo_files) > 0

def find_reference_image(ref_img, images):
    """
    Given a reference image (ref_img), loop through the list of Panorama_Img objects
    (images) and compare ref_img to each object's .image.

    Returns the index of the first matching image based on identical pixel data.
    Returns -1 if no match is found.

    Args:
        ref_img (np.ndarray): The reference image array to compare against.
        images (List[Panorama_Img]): Each has a .image attribute (np.ndarray).

    Returns:
        int: Index of the matching image, or -1 if none match.
    """

    for i, pano_img in enumerate(images):
        # Check shape first for efficiency
        if ref_img.shape == pano_img.image.shape:
            # If shapes match, compare pixel data
            if np.array_equal(ref_img, pano_img.image):
                return i
    return -1


def generate_panorama_lite(ref_dir, input_dir, output_dir, num_min_matches = 100, inlier_threshold = 5.0, ransac_iterations = 1000, min_overlap = 0.2, reference_index = 0):

    """
    Generate a panorama using the reference image and images in the input directory.

    Args:
        ref_dir (str): Path to the directory containing the reference image and keypoints.
        input_dir (str): Path to the directory containing the input images.
        output_dir (str): Path to save the output panorama.
        inlier_threshold (float): Threshold for error in RANSAC.
        num_min_matches (int): Minimum number of matches needed to link two images.
        ransac_iterations (int): Number of RANSAC iterations.
    """

    # Create panorama list
    if check_yolo_data(input_dir):
        print("YOLO data found. Computing homographies and transforming YOLO data.")
        compute_homographies_and_yolo_transform_lite(ref_dir, input_dir, output_dir, num_min_matches = num_min_matches, inlier_threshold = inlier_threshold, ransac_iterations = ransac_iterations, min_overlap = min_overlap, reference_index = reference_index)
        return
    print("No YOLO data found. Proceeding with panorama generation.")

    images = build_panorama_list_lite(ref_dir, input_dir)
    print("Panorama list created.")

    ref_image, _, _ = load_ref_image_and_keypoints(ref_dir)

    # Find reference image in the list of Panorama_Img objects
    ref_index = find_reference_image(ref_image, images)

    # Match images using homography heuristic
    matches = homography_heuristic_match(images, num_min_matches = num_min_matches, inlier_threshold = inlier_threshold, ransac_iterations = ransac_iterations, min_overlap = min_overlap)

    # Discover the best path from the reference image to all other images
    images = search_best_path(images, matches, ref_index)

    for idx, image in enumerate(images):
        print(f"Image {idx}, Parent: {image.parent}")
        
    # Compute homography for each image to the reference image from the matched pairs
    images = compute_homographies_from_best_path(images)

    for image in images:
        print(f"\nImage {images.index(image)} homography:\n {image.homography}")

    # Combine images using homographies
    panorama = combine_images(images)
    print("Images combined.")

    # Save the homographies to a .mat file
    homographies_list = [img.homography for img in images]
    save_homographies(homographies_list, output_dir)

    # Save the panorama
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "panorama.jpg")
    cv2.imwrite(output_path, panorama)
    print(f"Panorama saved to {output_path}")


def compute_homographies_and_yolo_transform_lite(ref_dir, input_dir, output_dir, num_min_matches = 100, inlier_threshold = 5.0, ransac_iterations = 1000, min_overlap = 0.2, reference_index = 0):

    """
    Compute homographies between images and transform YOLO bounding boxes.

    Args:
        ref_dir (str): Path to the directory containing the reference image and keypoints.
        input_dir (str): Path to the directory containing the input images.
        output_dir (str): Path to save the output panorama.
        inlier_threshold (float): Threshold for error in RANSAC.
        num_min_matches (int): Minimum number of matches needed to link two images.
        ransac_iterations (int): Number of RANSAC iterations.
    """

    # Build panorama list
    images = build_panorama_list_lite(ref_dir, input_dir)
    print("Panorama list created.")

    # Load reference image
    ref_image, _, _ = load_ref_image_and_keypoints(ref_dir)

    # Find reference image in the list of Panorama_Img objects
    ref_index = find_reference_image(ref_image, images)

    ref_index = reference_index
    # Match images using homography heuristic
    matches = homography_heuristic_match(images, num_min_matches = num_min_matches, inlier_threshold = inlier_threshold, ransac_iterations = ransac_iterations, min_overlap = min_overlap)

    # Discover the best path from the reference image to all other images
    images = search_best_path(images, matches, ref_index)

    for idx, image in enumerate(images):
        print(f"Image {idx}, Parent: {image.parent}")
        
    # Compute homography for each image to the reference image from the matched pairs
    images = compute_homographies_from_best_path(images)

    # Load YOLO data
    yolo_data = import_yolo_data(input_dir)

    # Transform YOLO data using the computed homographies
    homographies_list = [img.homography for img in images]
    save_homographies(homographies_list, output_dir)
    transform_yolo_multiple_homographies(yolo_data, homographies_list, output_dir)
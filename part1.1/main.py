import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat, savemat
from scipy.linalg import svd

def main():
    # Load the homography matrix from kp_gmaps.mat
    mat_file_path = 'kp_gmaps.mat'
    keypoints_image1, keypoints_image2 = load_keypoints_from_mat(mat_file_path)
    u1, v1 = keypoints_image1[:, 0], keypoints_image1[:, 1]
    u2, v2 = keypoints_image2[:, 0], keypoints_image2[:, 1]
    H = compute_homography(u1, v1, u2, v2)

    # Save the homography matrix
    homography_save_path = 'output_homography.mat'
    savemat(homography_save_path, {'H': H})
    print(f"Homography matrix saved to {homography_save_path}")
    print("Computed Homography Matrix:")
    print(H)

    # Process YOLO data
    yolo_data = import_yolo_data()
    processed_data = process_yolo_data(yolo_data)
    
    # Process our YOLO data
    # yolo_data = loadmat("1kvvadvcmkm_vid.mat")
    # processed_data = process_our_yolo_data(yolo_data)

    print("Processed all data!")
    print("Transforming points")
    

    #save yolo_transformed points

    output_yolo = "Yolo_ouput"

    transform_yolo_data(homography_save_path, yolo_data, output_yolo)

    output_folder = "aerial_images"
    os.makedirs(output_folder, exist_ok=True)
    """
    for frame_idx, frame_data in enumerate(processed_data):
        frame_points = np.array([item[1] for item in frame_data])
        transformed_points = np.array([point_homography_transformation(H, point) for point in frame_points])

        input_image_path = f'airport_CapeTown_aerial.png'
        camera_input_image = f'images/img_{frame_idx+1:04d}.jpg'
        output_image_path = os.path.join(output_folder, f'aerial_img_{frame_idx+1:04d}.png')

        # plot_points_on_image_cv2(input_image_path, transformed_points, f'Frame {frame_idx+1}', output_image_path)
        plot_side_by_side_images(camera_input_image, input_image_path, frame_points, transformed_points, f'Frame {frame_idx+1}', output_image_path)
        print(f'Finished frame {frame_idx}')
    
    print("Points Transformed!")

    print("Making video points!")

    # Create a video from the aerial images
    video_path = 'aerial_video.mp4'

    # Determine frame size from the first image
    first_image_path = os.path.join(output_folder, 'aerial_img_0001.png')
    first_image = cv2.imread(first_image_path)
    frame_size = (first_image.shape[1], first_image.shape[0])  # (width, height)

    fps = 10
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

    for frame_idx in range(len(processed_data)):
        img_path = os.path.join(output_folder, f'aerial_img_{frame_idx+1:04d}.png')
        img = cv2.imread(img_path)
        if img is not None:
            out.write(img)
        else:
            print(f"Warning: Could not read image {img_path}")

    out.release()
    print("Video finished!")
    print(f"Video saved to {video_path}")
    """

def load_keypoints_from_mat(mat_file_path):
    """
    Load keypoints from kp_gmaps.mat file.
    """
    data = loadmat(mat_file_path)
    matches = data['kp_gmaps']  # Assuming 'matches' is the key in the .mat file.
    u1, v1 = matches[:, 0], matches[:, 1]  # Keypoints in the first image
    u2, v2 = matches[:, 2], matches[:, 3]  # Corresponding keypoints in the second image
    return np.stack([u1, v1], axis=1), np.stack([u2, v2], axis=1)


def compute_homography(u1, v1, u2, v2):
    """
    Compute homography matrix using the Direct Linear Transform (DLT) method.
    """
    A = []
    for i in range(len(u1)):
        x1, y1 = u1[i], v1[i]
        x2, y2 = u2[i], v2[i]
        A.append([-x1, -y1, -1, 0, 0, 0, x2 * x1, x2 * y1, x2])
        A.append([0, 0, 0, -x1, -y1, -1, y2 * x1, y2 * y1, y2])
    A = np.array(A)
    _, _, Vt = svd(A)
    H = Vt[-1].reshape(3, 3)
    H /= H[2, 2]  # Normalize the homography matrix
    return H


def import_yolo_data():
    """
    Import YOLO data from .mat files.
    """
    yolo_folder = "yolo"
    yolo_data = []
    filenames = sorted([f for f in os.listdir(yolo_folder) if f.endswith(".mat")])
    for filename in filenames:
        file_path = os.path.join(yolo_folder, filename)
        data = loadmat(file_path)
        yolo_data.append(data)
    return yolo_data


def process_yolo_data(yolo_data, y_pos=0.25):
    """
    Process YOLO data to extract object IDs and box coordinates.
    """
    processed_data = []
    for frame in yolo_data:
        frame_data = []
        ids = frame['id'].flatten()
        xyxy = frame['xyxy']
        for i in range(len(ids)):
            xy = [(xyxy[i][0] + xyxy[i][2]) / 2, ((1 - y_pos) * xyxy[i][1] + y_pos * xyxy[i][3])]
            frame_data.append((ids[i], xy))
        processed_data.append(frame_data)
    return processed_data

def process_our_yolo_data(yolo_data, y_pos=1):
    """
    Process YOLO data to extract object IDs and box coordinates.
    """
    processed_data = []
    for frame in yolo_data:   
        if not frame.startswith('__'):  # Skip metadata keys:
            frame_data = []
            frame_values = yolo_data[frame]
            ids = frame_values[0]['id'][0].flatten()
            xyxy = frame_values[0]['xyxy'][0]
            for i in range(len(ids)):
                xy = [(xyxy[i][0] + xyxy[i][2]) / 2, ((1 - y_pos) * xyxy[i][1] + y_pos * xyxy[i][3])]
                frame_data.append((ids[i], xy))
            processed_data.append(frame_data)
    return processed_data


def point_homography_transformation(H, point):
    """
    Apply homography transformation to a single point.
    """
    x, y = point
    x, y, z = np.dot(H, [x, y, 1])
    return np.array([x / z, y / z])


def plot_points_on_image(image_path, points, title, output_path):
    """
    Plot points on the given image and save the output.
    """
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.scatter(points[:, 0], points[:, 1], c='red', marker='o')
    plt.title(title)
    plt.axis('off')
    height, width, _ = img.shape
    plt.xlim(0, width)
    plt.ylim(height, 0)
    plt.savefig(output_path)
    plt.close()


def plot_points_on_image_cv2(image_path, points, title, output_path):
    """
    Plot points on the given image using OpenCV and save the output.
    """
    # Read the image
    img = cv2.imread(image_path)

    # Ensure the points are integers for OpenCV
    points = points.astype(int)

    # Draw points on the image
    for point in points:
        cv2.circle(img, (point[0], point[1]), radius=5, color=(0, 0, 255), thickness=-1)  # Red filled circle

    # Add the title to the image (optional, OpenCV doesn't handle titles directly)
    # To include text, adjust this if needed, e.g., using cv2.putText.
    if title:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = 50
        cv2.putText(img, title, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

    # Save the modified image
    cv2.imwrite(output_path, img)

def transform_yolo_data(homography_data, yolo_data_input, yolo_data_output_dir):
    """
    Transforms YOLO bounding box data using the provided homography matrix
    and saves the transformed data.
    """

    # Load homography matrix
    H = loadmat(homography_data)['H']

    output_folder = yolo_data_output_dir
    os.makedirs(output_folder, exist_ok=True)

    #iterate over data frames
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


def plot_side_by_side_images(left_image_path, right_image_path, left_points, right_points, title, output_path):
    """
    Plot two images side by side with different points overlaid on each.
    
    Args:
        left_image_path (str): Path to the left image.
        right_image_path (str): Path to the right image.
        left_points (np.ndarray): Points to plot on the left image.
        right_points (np.ndarray): Points to plot on the right image.
        title (str): Title for the combined image.
        output_path (str): Path to save the output image.
    """
    # Read both images
    img_left = cv2.imread(left_image_path)
    img_right = cv2.imread(right_image_path)

    if img_left is None:
        raise FileNotFoundError(f"Image at path {left_image_path} not found.")
    if img_right is None:
        raise FileNotFoundError(f"Image at path {right_image_path} not found.")

    # Ensure points are integers for OpenCV
    left_points = left_points.astype(int)
    right_points = right_points.astype(int)

    # Draw points on the left image
    for point in left_points:
        cv2.circle(img_left, (point[0], point[1]), radius=5, color=(255, 0, 0), thickness=-1)  # Blue filled circle

    # Draw points on the right image
    for point in right_points:
        cv2.circle(img_right, (point[0], point[1]), radius=5, color=(0, 0, 255), thickness=-1)  # Red filled circle

    # Resize images to the same height for side-by-side concatenation
    height = max(img_left.shape[0], img_right.shape[0])
    img_left = cv2.resize(img_left, (int(img_left.shape[1] * height / img_left.shape[0]), height))
    img_right = cv2.resize(img_right, (int(img_right.shape[1] * height / img_right.shape[0]), height))

    # Combine the left and right images side by side
    combined_image = cv2.hconcat([img_left, img_right])

    # Optionally add a title
    if title:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
        text_x = (combined_image.shape[1] - text_size[0]) // 2
        text_y = 50
        cv2.putText(combined_image, title, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

    # Save the combined image
    cv2.imwrite(output_path, combined_image)

    # Display the combined image
    # cv2.imshow('Side-by-Side Display', combined_image)

    # Wait for a key press
    # key = cv2.waitKey(10)  # Press any key to proceed, or use a timeout for real-time use (e.g., `cv2.waitKey(30)`)
    # if key == 27:  # Escape key
    #     cv2.destroyAllWindows()
    #     return 'exit'

if __name__ == "__main__":
    main()

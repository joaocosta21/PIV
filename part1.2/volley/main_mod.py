import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from piv.functions import compute_homographies_and_yolo_transform

def main():
    if len(sys.argv) != 5:
        print("Usage: python main_mod.py <ref_dir> <input_dir> <yolo_dir> <output_dir>")
        sys.exit(1)

    ref_dir = sys.argv[1]
    input_dir = sys.argv[2]
    yolo_dir = sys.argv[3]
    output_dir = sys.argv[4]

    os.makedirs(output_dir, exist_ok=True)
    compute_homographies_and_yolo_transform(ref_dir, input_dir, yolo_dir, output_dir, num_min_matches = 40, inlier_threshold = 5.0, ransac_iterations = 200, num_random_matches = 1)

if __name__ == "__main__":
    main()
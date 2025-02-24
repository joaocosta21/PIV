import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from piv.functions import compute_homographies_and_yolo_transform_lite

def main():
    if len(sys.argv) != 4:
        print("Usage: python main_mod_lite.py <ref_dir> <input_dir> <output_dir>")
        sys.exit(1)

    ref_dir = sys.argv[1]
    input_dir = sys.argv[2]
    output_dir = sys.argv[3]

    os.makedirs(output_dir, exist_ok=True)
    compute_homographies_and_yolo_transform_lite(ref_dir, input_dir, output_dir, num_min_matches = 40, inlier_threshold = 1.7, ransac_iterations = 200, min_overlap = 0.1, reference_index = 243)

if __name__ == "__main__":
    main()
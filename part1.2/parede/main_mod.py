import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from piv.functions import generate_panorama

def main():
    if len(sys.argv) != 4:
        print("Usage: python main_mod.py <ref_dir> <input_dir> <output_dir>")
        sys.exit(1)

    ref_dir = sys.argv[1]
    input_dir = sys.argv[2]
    output_dir = sys.argv[3]

    os.makedirs(output_dir, exist_ok=True)
    generate_panorama(ref_dir, input_dir, output_dir, num_min_matches = 50, inlier_threshold = 5.0, ransac_iterations = 1000)

if __name__ == "__main__":
    main()
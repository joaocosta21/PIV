import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from piv.functions import generate_panorama_lite

def main():
    if len(sys.argv) < 4 or (len(sys.argv) - 2) % 2 != 0:
        print("Usage: python main_mod_lite.py <ref_dir> <input_dir1> <output_dir1> [<input_dir2> <output_dir2> ...]")
        sys.exit(1)

    # First argument is the reference directory
    ref_dir = sys.argv[1]

    # Remaining arguments are input-output pairs
    input_output_pairs = sys.argv[2:]

    # Validate the number of arguments after the ref_dir
    if len(input_output_pairs) % 2 != 0:
        print("Error: Input and output directories must be provided in pairs.")
        sys.exit(1)

    for i in range(0, len(input_output_pairs), 2):
        input_dir = input_output_pairs[i]
        output_dir = input_output_pairs[i + 1]

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        print(f"Processing: ref_dir={ref_dir}, input_dir={input_dir}, output_dir={output_dir}")
        generate_panorama_lite(
            ref_dir,
            input_dir,
            output_dir,
            num_min_matches=100,
            inlier_threshold=1.7,
            ransac_iterations=200,
            min_overlap=0.1,
            reference_index=0
        )

if __name__ == "__main__":
    main()
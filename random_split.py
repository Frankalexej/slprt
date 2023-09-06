import os
import random
import shutil

from paths import src_dir

def random_split(parent_dir, ratios, output_dirs):
    """
    Randomly separate subdirectories within a parent directory and move them to
    separate output directories according to the given ratios.

    Parameters:
    - parent_dir: The parent directory containing subdirectories to be split.
    - ratios: A list of ratios (e.g., [0.7, 0.3]) specifying the split ratio.
    - output_dirs: A list of output directory names for the separated data.

    Returns:
    - None
    """

    if sum(ratios) != 1.0:
        raise ValueError("Ratios must sum to 1.0")

    if len(ratios) != len(output_dirs):
        raise ValueError("The number of output directories must match the number of ratios")

    subdirectories = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]

    # Create output directories if they don't exist
    for output_dir in output_dirs:
        os.makedirs(output_dir, exist_ok=True)

    random.shuffle(subdirectories)

    total_dirs = len(subdirectories)
    split_points = [int(total_dirs * ratio) for ratio in ratios]

    for i, output_dir in enumerate(output_dirs):
        move_dirs = subdirectories[:split_points[i]]
        for move_dir in move_dirs:
            source = os.path.join(parent_dir, move_dir)
            destination = os.path.join(output_dir, move_dir)
            shutil.move(source, destination)

        subdirectories = subdirectories[split_points[i]:]


if __name__ == '__main__':
    random_split(os.path.join(src_dir, "try/one/"), [0.3, 0.7], [os.path.join(src_dir, "try/first/"), os.path.join(src_dir, "try/second/")])
import os
import random
import shutil

from paths import det_dir
from model_configs import suffixes

def add_suffix_to_path(path, suffix): 
    # Check if the path already ends with a slash
    if path.endswith('/'):
        new_path = path.rstrip('/') + suffix + '/'
    else:
        new_path = path + suffix + '/'
    return new_path


def random_split(parent_dir, ratio_train=0.8):
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
    # define the suffixes of the different data chunks

    # adding suffixes to form output directories
    output_dirs = [add_suffix_to_path(parent_dir, suffix) for suffix in suffixes]

    # count the subdirectories that will be moved
    subdirectories = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]

    # Create output directories if they don't exist
    for output_dir in output_dirs:
        os.makedirs(output_dir, exist_ok=True)

    random.shuffle(subdirectories)

    # calculate the position of split
    total_dirs = len(subdirectories)
    split_point = int(ratio_train * total_dirs)

    train_dirs = subdirectories[:split_point]
    test_dirs = subdirectories[split_point:]
    both_dirs = [train_dirs, test_dirs]

    for move_idx in range(len(both_dirs)): 
        move_dirs = both_dirs[move_idx]
        output_dir = output_dirs[move_idx]

        for move_dir in move_dirs: 
            source = os.path.join(parent_dir, move_dir)
            destination = os.path.join(output_dir, move_dir)
            shutil.move(source, destination)


if __name__ == '__main__':
    random_split(os.path.join(det_dir, "Cynthia_full"), 0.8)    # if the name is not Cynthia_full, then must change name here. 
import os
import shutil

def get_filenames(directory):
    """Returns a set of filenames in the given directory."""
    return set(os.listdir(directory))

def copy_unique_files(dir1, dir2, dir3):
    """Copies files from dir2 to dir3 that are not in dir1 and returns the number of files copied."""
    # Get the filenames in each directory
    filenames_dir1 = get_filenames(dir1)
    filenames_dir2 = get_filenames(dir2)

    # Ensure dir3 exists
    if not os.path.exists(dir3):
        os.makedirs(dir3)

    # Counter for the number of files copied
    files_copied = 0

    # Iterate through files in dir2
    for filename in filenames_dir2:
        if filename not in filenames_dir1:
            # Copy file from dir2 to dir3
            src_file = os.path.join(dir2, filename)
            dst_file = os.path.join(dir3, filename)
            shutil.copy2(src_file, dst_file)
            print(f"Copied {filename} to {dir3}")
            files_copied += 1

    return files_copied

if __name__ == "__main__":
    dir1 = "path_to_dir1"
    dir2 = "path_to_dir2"
    dir3 = "path_to_dir3"

    files_copied = copy_unique_files(dir1, dir2, dir3)

    print(f"\nSummary: {files_copied} file(s) were copied to {dir3}.")

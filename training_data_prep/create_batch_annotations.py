import os
import shutil
import multiprocessing
import tqdm

def copy_files_to_batch(batch_files, batch_dir, input_dir):
    """
    Copies files to a specific batch directory, preserving their relative paths.
    """
    for file_path in batch_files:
        # Create a relative path for the file based on the input directory
        image_file_path = file_path.replace('.pkl', '.png')
        relative_path = os.path.relpath(file_path, input_dir)
        unique_file_name = relative_path.replace('/', '--')
        unique_image_file_name = unique_file_name.replace('.pkl', '.png')
        dest_file_path_pkl = os.path.join(batch_dir, unique_file_name)
        dest_file_path_png = os.path.join(batch_dir, unique_image_file_name)

        # Ensure the directory for the destination file exists
        os.makedirs(os.path.dirname(dest_file_path_pkl), exist_ok=True)

        # Copy the file
        shutil.copy(file_path, dest_file_path_pkl)
        shutil.copy(image_file_path, dest_file_path_png)

def create_batches(input_dir, output_dir, batch_size=100):
    """
    Organizes files into batches and copies them, using multiprocessing to enhance performance.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Collect all files recursively from the input directory
    files = []
    for root, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith(('.pkl')):
                files.append(os.path.join(root, filename))

    # Sort files to keep images and .pkl files ordered
    files.sort()

    # Split files into batches and process each batch in parallel
    processes = []
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        batch_dir = os.path.join(output_dir, f'batch_{i // batch_size + 1}')
        
        # Spawn a new process for copying files in this batch
        process = multiprocessing.Process(target=copy_files_to_batch, args=(batch_files, batch_dir, input_dir))
        processes.append(process)
        process.start()

        # Display progress bar
        for process in tqdm.tqdm(processes, desc="Processing batches"):
            process.join()

    # Wait for all processes to complete
    for process in processes:
        process.join()

if __name__ == "__main__":
    input_directory = '/ihub/homedirs/am_cse/pramay/work/Dataset/cropped_png'
    output_directory = '/ihub/homedirs/am_cse/pramay/work/Dataset/cropped_png_batched'
    batch_size = 100

    create_batches(input_directory, output_directory, batch_size)

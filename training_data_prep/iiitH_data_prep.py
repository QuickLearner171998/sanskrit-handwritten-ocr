import os
import tqdm
import multiprocessing
from functools import partial

def process_line(dataset_dir, root_dir, line):
    rel_image_path, annotation = line.strip().split(' ', 1)
    full_image_path = os.path.abspath(os.path.join(dataset_dir, rel_image_path))
    new_rel_path = os.path.relpath(full_image_path, root_dir)

    # Update the path and delimiter
    updated_line = f"{new_rel_path}\t{annotation}"

    # Create annotation .txt file
    annotation_file_path = f"{full_image_path}.txt"
    with open(annotation_file_path, 'w') as af:
        af.write(annotation)

    return updated_line

def update_txt_file(input_file, dataset_dir, output_file, root_dir):
    """
    Process the train, val, test files to update the path to full path and adjust the delimiter.
    Create .txt annotation files for each image.
    """
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Use multiprocessing to process lines in parallel
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    func = partial(process_line, dataset_dir, root_dir)
    
    updated_lines = list(tqdm.tqdm(pool.imap(func, lines), total=len(lines), desc=f"Processing {os.path.basename(input_file)}"))
    
    pool.close()
    pool.join()
    
    # Write the updated lines back to the output file
    with open(output_file, 'w') as file:
        file.write('\n'.join(updated_lines) + '\n')

def main(root_dir):
    dataset_dir = os.path.join(root_dir, 'IIITH_data')
    
    # Paths to the input files
    train_file = os.path.join(dataset_dir, 'train.txt')
    val_file = os.path.join(dataset_dir, 'val.txt')
    test_file = os.path.join(dataset_dir, 'test.txt')
    
    # Process each file
    update_txt_file(train_file, dataset_dir, train_file, root_dir)
    update_txt_file(val_file, dataset_dir, val_file, root_dir)
    update_txt_file(test_file, dataset_dir, test_file, root_dir)

if __name__ == "__main__":
    root_dir = "/ihub/homedirs/am_cse/pramay/work/Dataset/"
    main(root_dir)
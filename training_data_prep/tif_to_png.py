import os
from PIL import Image
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def resize_image(img, max_size):
    """Resize an image to have a maximum dimension of `max_size` while maintaining the aspect ratio."""
    original_width, original_height = img.size
    if max(original_width, original_height) <= max_size:
        return img  # No resizing needed

    # Calculate the new dimensions
    if original_width > original_height:
        new_width = max_size
        new_height = int((max_size / original_width) * original_height)
    else:
        new_height = max_size
        new_width = int((max_size / original_height) * original_width)

    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

def process_file(file_info):
    """Convert a single TIFF file to a resized PNG."""
    input_path, output_path = file_info
    try:
        # Open the TIFF image
        with Image.open(input_path) as img:
            # Resize the image
            img = resize_image(img, max_size=2048)
            # Save the resized image as PNG
            img.save(output_path, format="PNG", optimize=True)
        return f"Converted and resized: {input_path} -> {output_path}"
    except Exception as e:
        return f"Error processing {input_path}: {e}"

def collect_tiff_files(input_dir, output_dir):
    """Collect TIFF files and their corresponding output paths."""
    file_list = []
    for root, _, files in os.walk(input_dir):
        # Calculate the relative path for subdirectory preservation
        relative_path = os.path.relpath(root, input_dir)
        target_dir = os.path.join(output_dir, relative_path)

        # Create target directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)

        for file in files:
            if file.lower().endswith(".tif") or file.lower().endswith(".tiff"):
                input_path = os.path.join(root, file)
                output_path = os.path.join(target_dir, os.path.splitext(file)[0] + ".png")
                file_list.append((input_path, output_path))
    return file_list

def convert_tiff_to_png_recursive(input_dir, output_dir):
    """Main function to convert TIFF to resized PNG using multiprocessing."""
    file_list = collect_tiff_files(input_dir, output_dir)

    # Show progress with tqdm
    print(f"Found {len(file_list)} files to process.")
    with Pool(cpu_count()) as pool:
        for result in tqdm(pool.imap(process_file, file_list), total=len(file_list)):
            if "Error" in result:  # Print errors inline for debugging
                print(result)

# Example usage
input_directory = "/home/pramay/myStuff/ai_apps/IITJodhpur/work/data_for_annotation"
output_directory = "/home/pramay/myStuff/ai_apps/IITJodhpur/work/data_for_annotation_png"

convert_tiff_to_png_recursive(input_directory, output_directory)

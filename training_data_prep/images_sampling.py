import os
import random
from PIL import Image
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings


# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

def resize_image(img, max_size):
    """
    Resize an image to have a maximum dimension of `max_size` 
    while maintaining the aspect ratio.
    """
    original_width, original_height = img.size
    if max(original_width, original_height) <= max_size:
        return img  # No resizing needed

    if original_width > original_height:
        new_width = max_size
        new_height = int((max_size / original_width) * original_height)
    else:
        new_height = max_size
        new_width = int((max_size / original_height) * original_width)

    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

def process_file(args):
    """
    Process a single image: resize and convert to PNG.
    """
    src_path, dest_path = args
    try:
        with Image.open(src_path) as img:
            img = resize_image(img, max_size=2048)
            img.save(dest_path, format="PNG", optimize=True)
        return f"Processed: {src_path} -> {dest_path}"
    except Exception as e:
        return f"Error processing {src_path}: {e}"

def sample_and_process_images(src_dir, dest_dir, percentage=5):
    """
    Sample images, resize them, and convert them to PNG.
    """
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.tif'}
    tasks = []

    # Create the destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    for root, _, files in os.walk(src_dir):
        if 'Mysore' in root:  # Skip directories containing 'Mysore'
            continue

        # Filter image files
        images = [file for file in files if os.path.splitext(file)[1].lower() in VALID_EXTENSIONS]
        if images:
            # Sample 70% of the images randomly
            sampled_images = random.sample(images, int(len(images) * (percentage / 100)))

            for image in sampled_images:
                src_path = os.path.join(root, image)

                # Generate relative path-based unique filename
                relative_path = os.path.relpath(root, src_dir)
                new_filename = f"{relative_path.replace(os.sep, '_')}_{os.path.splitext(image)[0]}.png"

                dest_path = os.path.join(dest_dir, new_filename)
                tasks.append((src_path, dest_path))

    # Process the images using multiprocessing
    with Pool(cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(process_file, tasks), total=len(tasks)):
            if "Error" in result:  # Print errors for debugging
                print(result)

# Example usage
if __name__ == "__main__":
    source_directory = "/home/pramay/myStuff/ai_apps/IITJodhpur/work/Dataset/REAL"
    destination_directory = "/home/pramay/myStuff/ai_apps/IITJodhpur/work/data_for_annotation_png_5_percent"
    
    print("Sampling, resizing, and converting images...")
    sample_and_process_images(source_directory, destination_directory)

from os import walk, makedirs
from os.path import join, splitext, exists, dirname
from dotenv import load_dotenv
from google.cloud import vision_v1p4beta1 as vision
import pickle
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import traceback

load_dotenv()

def detect_text(path):
    try:
        client = vision.ImageAnnotatorClient()
        with open(path, "rb") as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        image_context = vision.ImageContext(language_hints=["sa", 'en', 'hi'])
        response = client.text_detection(image=image, image_context=image_context)

        # Save the pickle file with the same name as the image
        output_dir = dirname(path)  # Use the same directory as the image
        image_name = splitext(path.split("/")[-1])[0]  # Get image name without extension
        pickle_path = join(output_dir, f"{image_name}.pkl")
        pickle.dump(response, open(pickle_path, "wb"))
        return f"Processed {path}"
    except Exception as e:
        return f"Error processing {path}: {traceback.format_exc()}"

def process_file(file_path):
    return detect_text(file_path)

def main():
    input_dir = "/ihub/homedirs/am_cse/pramay/work/Dataset/cropped_png/"

    # Collect all files to process
    files_to_process = []
    for root, _, files in walk(input_dir):
        if 'Mysore' in root:  # Skip directories containing 'Mysore'
            continue
        for file in files:
            if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png') or file.endswith('.tiff') or file.endswith('.tif'):
                files_to_process.append(join(root, file))

    # Use multiprocessing with a pool of workers
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_file, files_to_process), total=len(files_to_process)))

    # Print results
    for result in results:
        print(result)

if __name__ == "__main__":
    main()

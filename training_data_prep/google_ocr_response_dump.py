from os import walk, makedirs
from os.path import join, splitext, exists
from dotenv import load_dotenv
from google.cloud import vision_v1p4beta1 as vision
import pickle
from tqdm import tqdm

load_dotenv()

def detect_text(path, output_dir):
    client = vision.ImageAnnotatorClient()
    with open(path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    image_context = vision.ImageContext(language_hints=["sa", 'en', 'hi'])
    response = client.text_detection(image=image, image_context=image_context)
    # texts = response.text_annotations

    # Save the pickle file with the same name as the image
    image_name = splitext(path.split("/")[-1])[0]  # Get image name without extension
    pickle_path = join(output_dir, f"{image_name}.pkl")
    pickle.dump(response, open(pickle_path, "wb"))

    # ocr_text = []
    # for text in texts:
    #     ocr_text.append(f"\r\n{text.description}")
    # if response.error.message:
    #     raise Exception(
    #         "{}\nFor more info on error messages, check: "
    #         "https://cloud.google.com/apis/design/errors".format(response.error.message)
    #     )
    # return texts[0].description if texts else None

def main():
    input_dir = "/home/pramay/myStuff/ai_apps/IITJodhpur/work/data_for_annotation_png_5_percent"
    output_dir = "/home/pramay/myStuff/ai_apps/IITJodhpur/work/5_percent_gocr_out_pkl/"

    # Create the output directory if it doesn't exist
    if not exists(output_dir):
        makedirs(output_dir)

    # Walk through the input directory recursively
    for root, _, files in walk(input_dir):
        for file in tqdm(files):
            file_path = join(root, file)
            try:
                text = detect_text(file_path, output_dir)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    main()

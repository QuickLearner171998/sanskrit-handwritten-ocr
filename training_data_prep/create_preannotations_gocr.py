import os
import pickle
import json
from tqdm import tqdm
from PIL import Image

def get_vertices(bounding_box):
    return [[vertex.x, vertex.y] for vertex in bounding_box.vertices]

def create_labelme_annotation(response, image_filename, img_size):
    # Initial configuration for the LabelMe annotation
    labelme_annotation = {
        "version": "4.5.6",
        "flags": {},
        "shapes": [],
        "imagePath": image_filename,
        "imageData": None,
        "imageHeight": img_size[1],
        "imageWidth": img_size[0]
    }

    # If the response contains "fullTextAnnotation"
    if response.full_text_annotation is not None:
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                block_points = get_vertices(block.bounding_box)
                block_label = "block"
                labelme_annotation['shapes'].append({
                    "label": block_label,
                    "points": block_points,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                })

                for paragraph in block.paragraphs:
                    paragraph_points = get_vertices(paragraph.bounding_box)
                    paragraph_label = "paragraph"
                    labelme_annotation['shapes'].append({
                        "label": paragraph_label,
                        "points": paragraph_points,
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {}
                    })

                    for word in paragraph.words:
                        word_text = ''.join([symbol.text for symbol in word.symbols])
                        word_points = get_vertices(word.bounding_box)
                        word_label = word_text
                        labelme_annotation['shapes'].append({
                            "label": word_label,
                            "points": word_points,
                            "group_id": None,
                            "shape_type": "polygon",
                            "flags": {}
                        })

    return labelme_annotation

def process_image(image_filepath, response_filepath):
    output_filepath = f"{os.path.splitext(image_filepath)[0]}.json"

    if os.path.exists(response_filepath):
        with open(response_filepath, 'rb') as f:
            response = pickle.load(f)
        
        # Open image to get dimensions
        with Image.open(image_filepath) as img:
            img_size = img.size

        # Generate LabelMe annotation
        labelme_annotation = create_labelme_annotation(response, os.path.basename(image_filepath), img_size)

        # Save to a JSON file with UTF-8 encoding
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(labelme_annotation, f, ensure_ascii=False, indent=4)
        return f"Saved LabelMe annotation for {image_filepath}"
    return f"Response file not found for {image_filepath}"

def convert_responses_to_labelme(image_dir, response_dir):
    results = []
    for root, _, files in os.walk(image_dir):
        image_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))]
        for image_file in tqdm(image_files, desc=f'Processing images in {root}', unit='image'):
            image_filepath = os.path.join(root, image_file)
            response_filepath = image_filepath.replace('.png', '.pkl')
            # response_filepath = os.path.join(response_dir, f"{os.path.splitext(image_file)[0]}.pkl")
            result = process_image(image_filepath, response_filepath)
            results.append(result)

    for result in results:
        print(result)

if __name__ == "__main__":
    image_dir = "/home/pramay/myStuff/ai_apps/IITJodhpur/work/Dataset/REAL_PNG"  
    response_dir = "/home/pramay/myStuff/ai_apps/IITJodhpur/work/Dataset/REAL_PNG"  
    convert_responses_to_labelme(image_dir, response_dir)

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

def process_image(image_filename, image_dir, response_dir, output_dir):
    image_filepath = os.path.join(image_dir, image_filename)
    response_filepath = os.path.join(response_dir, f"{os.path.splitext(image_filename)[0]}.pkl")
    output_filepath = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}.json")

    if os.path.exists(response_filepath):
        with open(response_filepath, 'rb') as f:
            response = pickle.load(f)
        
        # Open image to get dimensions
        with Image.open(image_filepath) as img:
            img_size = img.size

        # Generate LabelMe annotation
        labelme_annotation = create_labelme_annotation(response, image_filename, img_size)

        # Save to a JSON file with UTF-8 encoding
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(labelme_annotation, f, ensure_ascii=False, indent=4)
        return f"Saved LabelMe annotation for {image_filename}"
    return f"Response file not found for {image_filename}"

def convert_responses_to_labelme(image_dir, response_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_filenames = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    results = []
    for image_filename in tqdm(image_filenames, desc='Processing images', unit='image'):
        result = process_image(image_filename, image_dir, response_dir, output_dir)
        results.append(result)

    for result in results:
        print(result)


if __name__ == "__main__":
    image_dir = "/home/pramay/myStuff/ai_apps/IITJodhpur/work/data_for_annotation_png_5_percent"  
    response_dir = "/home/pramay/myStuff/ai_apps/IITJodhpur/work/5_percent_gocr_out_pkl"  
    output_dir = "/home/pramay/myStuff/ai_apps/IITJodhpur/work/5_percent_labelme_annotations" 
    convert_responses_to_labelme(image_dir, response_dir, output_dir)
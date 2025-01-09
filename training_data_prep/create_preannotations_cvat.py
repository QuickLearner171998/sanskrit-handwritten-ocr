import os
import pickle
from tqdm import tqdm
from PIL import Image
from lxml import etree as ET
from multiprocessing import Pool, cpu_count
from datetime import datetime

def get_vertices(bounding_box):
    return [[vertex.x, vertex.y] for vertex in bounding_box.vertices]

def create_cvat_image_annotation(image_id, image_filename, img_size, shapes):
    image_element = ET.Element('image', {
        'id': str(image_id),
        'name': image_filename,
        'width': str(img_size[0]),
        'height': str(img_size[1])
    })

    for label, points, text_value in shapes:
        points_str = ";".join([f"{x},{y}" for x, y in points])
        polygon = ET.SubElement(image_element, 'polygon', {
            'label': label,
            'source': 'manual',
            'occluded': '0',
            'points': points_str,
            'z_order': '0'
        })
        if label == 'text':
            attribute = ET.SubElement(polygon, 'attribute', {'name': 'value'})
            attribute.text = text_value

    return image_element

def process_image(args):
    image_id, image_filepath, response_filepath = args
    results = []
    if os.path.exists(response_filepath):
        with open(response_filepath, 'rb') as f:
            response = pickle.load(f)

        with Image.open(image_filepath) as img:
            img_size = img.size

        shapes = []
        if response.full_text_annotation is not None:
            for page in response.full_text_annotation.pages:
                for block in page.blocks:
                    block_points = get_vertices(block.bounding_box)
                    shapes.append(('block', block_points, ''))

                    for paragraph in block.paragraphs:
                        paragraph_points = get_vertices(paragraph.bounding_box)
                        shapes.append(('paragraph', paragraph_points, ''))

                        for word in paragraph.words:
                            word_text = ''.join([symbol.text for symbol in word.symbols])
                            word_points = get_vertices(word.bounding_box)
                            shapes.append(('text', word_points, word_text))

        image_element = create_cvat_image_annotation(image_id, os.path.basename(image_filepath), img_size, shapes)
        results.append((image_id, ET.tostring(image_element, pretty_print=True).decode()))

    return results

def convert_responses_to_cvat(image_dir):
    results = []
    input_data = []

    # Collect files and sort them
    image_files = []
    for root, _, files in os.walk(image_dir):
        image_files.extend([os.path.join(root, f) for f in files if f.endswith(('.jpg', '.jpeg', '.png'))])
    image_files.sort()  # Sort by filename

    # Assign IDs and create input data for multiprocessing
    for image_id, image_filepath in enumerate(image_files):
        response_filepath = image_filepath.replace('.png', '.pkl')
        if os.path.exists(response_filepath):
            input_data.append((image_id, image_filepath, response_filepath))

    # Use multiprocessing to process images
    with Pool(cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(process_image, input_data), total=len(input_data), desc='Processing images', unit='image'):
            results.extend(result)

    # Sort results by image_id
    results.sort(key=lambda x: x[0])

    # Create the XML tree and fill it with the collected results
    annotation = ET.Element("annotations")
    ET.SubElement(annotation, 'version').text = "1.1"

    meta = ET.SubElement(annotation, 'meta')
    job = ET.SubElement(meta, 'job')
    ET.SubElement(job, 'id').text = "1914846"
    ET.SubElement(job, 'size').text = str(len(results))
    ET.SubElement(job, 'mode').text = "annotation"
    ET.SubElement(job, 'overlap').text = "0"
    ET.SubElement(job, 'bugtracker')
    ET.SubElement(job, 'created').text = datetime.utcnow().isoformat()
    ET.SubElement(job, 'updated').text = datetime.utcnow().isoformat()
    ET.SubElement(job, 'subset').text = "default"
    ET.SubElement(job, 'start_frame').text = "0"
    ET.SubElement(job, 'stop_frame').text = str(len(results) - 1)
    ET.SubElement(job, 'frame_filter')
    segment = ET.SubElement(ET.SubElement(job, 'segments'), 'segment')
    ET.SubElement(segment, 'start').text = "0"
    ET.SubElement(segment, 'stop').text = str(len(results) - 1)
    owner = ET.SubElement(job, 'owner')
    ET.SubElement(owner, 'username').text = "pramay8"
    ET.SubElement(owner, 'email').text = "pramay.singhvi@gmail.com"
    ET.SubElement(job, 'assignee')

    labels = ET.SubElement(job, 'labels')
    for label_name, color in [("text", "#7ba0c1"), ("block", "#a8df23"), ("paragraph", "#6f35fe")]:
        label = ET.SubElement(labels, 'label')
        ET.SubElement(label, 'name').text = label_name
        ET.SubElement(label, 'color').text = color
        ET.SubElement(label, 'type').text = "polygon"
        if label_name == "text":
            attr = ET.SubElement(ET.SubElement(label, 'attributes'), 'attribute')
            ET.SubElement(attr, 'name').text = "value"
            ET.SubElement(attr, 'mutable').text = "True"
            ET.SubElement(attr, 'input_type').text = "text"
            ET.SubElement(attr, 'default_value')
            ET.SubElement(attr, 'values')

    ET.SubElement(meta, 'dumped').text = datetime.utcnow().isoformat()

    # Append all collected image elements
    for _, result in results:
        annotation.append(ET.fromstring(result))

    # Save the final CVAT XML file
    output_filepath = os.path.join(image_dir, 'annotations.xml')
    tree = ET.ElementTree(annotation)
    tree.write(output_filepath, pretty_print=True, xml_declaration=True, encoding="utf-8")
    print(f"Saved CVAT annotation to {output_filepath}")

if __name__ == "__main__":
    image_dir = "/ihub/homedirs/am_cse/pramay/work/Dataset/cropped_png_batched"
    
    for batch_name in os.listdir(image_dir):
        convert_responses_to_cvat(os.path.join(image_dir, batch_name))
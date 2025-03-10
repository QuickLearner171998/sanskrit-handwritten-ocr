import os
import json
import xml.etree.ElementTree as ET
from multiprocessing import Pool, current_process

def load_corrected_image_ids(txt_file):
    with open(txt_file, 'r') as file:
        corrected_image_ids = {line.strip() for line in file}
    return corrected_image_ids

def process_image(image_elem, dir_prefix, corrected_image_ids, output_path):
    image_id = image_elem.get('id')
    image_name = image_elem.get('name')
    if image_name not in corrected_image_ids:
        return
    width = image_elem.get('width')
    height = image_elem.get('height')
    
    annotations = []

    for polygon in image_elem.findall('polygon'):
        attrs = {attr.get('name'): attr.text for attr in polygon.findall('attribute')}
        annotations.append({
            'label': polygon.get('label'),
            'source': polygon.get('source'),
            'occluded': polygon.get('occluded'),
            'points': polygon.get('points'),
            'z_order': polygon.get('z_order'),
            'attributes': attrs
        })
    
    image_path = os.path.join(dir_prefix, image_name)

    json_data = {
        'id': image_id,
        'name': image_path,
        'width': width,
        'height': height,
        'annotations': annotations
    }

    # Define the path to save the JSON file
    json_name = image_name.replace('/', '_').replace('\\', '_')
    if '.png' in json_name or '.jpg' in json_name or '.jpeg' in json_name:
        json_name = json_name.split('.')[0] + '.json'
    
    json_file_path = os.path.join(output_path, json_name)
    
    # Write JSON data to file
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, ensure_ascii=False, indent=4)

    print(f"Process {current_process().name} processed image: {image_id} -> {json_file_path}")

def process_xml(xml_file, dir_prefix, corrected_image_ids, output_path):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    images = root.findall('image')

    pool = Pool()  # Using multiprocessing pool
    pool.starmap(process_image, [(img, dir_prefix, corrected_image_ids, output_path) for img in images])

    pool.close()
    pool.join()

if __name__ == "__main__":
    xml_file_paths = [
        "/ihub/homedirs/am_cse/pramay/work/annotation/corrections/XMLS/job_1971423_annotations_2025_02_14_09_54_51_cvat for images 1.1/annotations.xml",
        "/ihub/homedirs/am_cse/pramay/work/annotation/corrections/XMLS/job_1971431_annotations_2025_02_09_10_30_26_cvat for images 1.1/annotations.xml",
        "/ihub/homedirs/am_cse/pramay/work/annotation/corrections/XMLS/job_1971437_annotations_2025_02_11_06_31_02_cvat for images 1.1/annotations.xml",
        "/ihub/homedirs/am_cse/pramay/work/annotation/corrections/XMLS/job_2007277_annotations_2025_02_27_07_28_25_cvat for images 1.1/annotations.xml"
    ]
    dir_prefixes = [
        "/ihub/homedirs/am_cse/pramay/work/annotation/v1/all_images_batched",
        "/ihub/homedirs/am_cse/pramay/work/annotation/v1/all_images_batched",
        "/ihub/homedirs/am_cse/pramay/work/annotation/v1/all_images_batched",
        "/ihub/homedirs/am_cse/pramay/work/"
    ]

    corrected_images_txt = "/ihub/homedirs/am_cse/pramay/work/annotation/corrections/corrected_images.txt"
    corrected_image_ids = load_corrected_image_ids(corrected_images_txt)
    
    output_path = "/ihub/homedirs/am_cse/pramay/work/annotation/corrections/JSONS/corrected_jsons"
    
    if os.path.exists(output_path):
        print(f"Output path {output_path} already exists. Deleting...")
        os.system(f"rm -r {output_path}")
    
    os.makedirs(output_path, exist_ok=True)

    for xml_file, dir_prefix in zip(xml_file_paths, dir_prefixes):
        process_xml(xml_file, dir_prefix, corrected_image_ids, output_path)

    print("Processing complete.")
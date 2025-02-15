import os
import pickle
from tqdm import tqdm
from PIL import Image
from lxml import etree as ET
from multiprocessing import Pool, cpu_count
from datetime import datetime
from collections import defaultdict

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

def create_text_file(text_blocks, output_filepath):
    """
    Creates a text file maintaining relative positioning of text blocks.
    Text blocks should be a list of dictionaries with 'text' and 'position' (y, x).
    """
    if not text_blocks:
        return
        
    # Sort blocks by vertical position first, then horizontal
    text_blocks.sort(key=lambda x: (x['position'][0], x['position'][1]))
    
    # Group text blocks by their vertical position (rounded to help group nearby lines)
    lines = defaultdict(list)
    current_y = None
    y_threshold = 10  # Threshold for considering text to be on the same line
    
    for block in text_blocks:
        y_pos = block['position'][0]
        
        # If this is the first block or significantly different from current y
        if current_y is None or abs(y_pos - current_y) > y_threshold:
            current_y = y_pos
            
        # Add text to the appropriate line
        lines[current_y].append(block)
    
    # Sort each line horizontally
    for y in lines:
        lines[y].sort(key=lambda x: x['position'][1])
    
    # Write to file
    with open(output_filepath, 'w', encoding='utf-8') as f:
        # Process lines in vertical order
        for y in sorted(lines.keys()):
            line_text = ' '.join(block['text'] for block in lines[y])
            f.write(line_text + '\n')
            
            # Add an extra newline if there's a significant gap to the next line
            next_lines = [ny for ny in lines.keys() if ny > y]
            if next_lines and min(next_lines) - y > y_threshold * 2:
                f.write('\n')

def process_image(args):
    image_id, image_filepath, response_filepath = args
    results = []
    if os.path.exists(response_filepath):
        with open(response_filepath, 'rb') as f:
            response = pickle.load(f)

        with Image.open(image_filepath) as img:
            img_size = img.size

        shapes = []
        text_blocks = []
        
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
                            
                            # Calculate center position for relative positioning
                            vertices = word_points
                            y_center = sum(v[1] for v in vertices) / len(vertices)
                            x_center = sum(v[0] for v in vertices) / len(vertices)
                            
                            text_blocks.append({
                                'text': word_text,
                                'position': (y_center, x_center)
                            })

        # Create CVAT annotation
        image_element = create_cvat_image_annotation(image_id, os.path.basename(image_filepath), img_size, shapes)
        results.append((image_id, ET.tostring(image_element, pretty_print=True).decode()))
        
        # Create text file
        output_txt_filepath = image_filepath.replace('.png', '.txt')
        create_text_file(text_blocks, output_txt_filepath)

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
    image_dir = "/home/pramay/myStuff/ai_apps/IITJodhpur/work/Data_extended_PNG"
    
    batch_mode = False #True
    
    if batch_mode:
        for batch_name in os.listdir(image_dir):
            convert_responses_to_cvat(os.path.join(image_dir, batch_name))
            
    else:
        convert_responses_to_cvat(image_dir)
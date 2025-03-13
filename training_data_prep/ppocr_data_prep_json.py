import os
import json
import csv
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
import shutil
from multiprocessing import Pool, cpu_count
import cv2
import numpy as np

def create_directory_structure(train_dir, val_dir):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

def order_points(pts):
    """Orders points in a consistent order: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = np.sum(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def warp_image(image, points):
    """Warp image using block or paragraph coordinates and return the warped image and homography matrix."""
    points = np.array(points, dtype="float32")
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.int64(box)
    box = order_points(box)

    # Compute width and height of the bounding box
    (tl, tr, br, bl) = box
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # Compute the perspective transform matrix and warp the image
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(box, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped, M

def append_to_report(crop_name, issue, report_path):
    with open(report_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([crop_name, issue])

def transform_and_crop_polygon(image_info, small_crop_dir, small_area_crop_dir, report_path):
    """Transform polygon points using the homography matrix, crop the image, and apply dilation."""
    warped, M, points, output_path, label = image_info

    # Transform the points
    points = np.array(points, dtype="float32")
    transformed_points = cv2.perspectiveTransform(np.array([points]), M)[0]
    
    # Calculate the bounding box dimensions
    min_x = int(np.min(transformed_points[:, 0]))
    max_x = int(np.max(transformed_points[:, 0]))
    min_y = int(np.min(transformed_points[:, 1]))
    max_y = int(np.max(transformed_points[:, 1]))

    # Apply dilation
    dilation_width = int(0.05 * (max_x - min_x))
    dilation_height = int(0.01 * (max_y - min_y))
    min_x = max(0, min_x - dilation_width)
    min_y = max(0, min_y - dilation_height)
    max_x = min(warped.shape[1], max_x + dilation_width)
    max_y = min(warped.shape[0], max_y + dilation_height)

    # Ensure coordinates are within the image boundaries
    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(warped.shape[1], max_x)
    max_y = min(warped.shape[0], max_y)

    # Crop the transformed image
    cropped_img = warped[min_y:max_y, min_x:max_x]
    
    min_width = 5  # Minimum width for valid crops
    min_height = 30  # Minimum height for valid crops
    min_area = 100  # Minimum area (width * height) to avoid tiny crops

    if cropped_img.size == 0:
        return

    if cropped_img.shape[1] < min_width or cropped_img.shape[0] < min_height:
        small_crop_output_path = os.path.join(small_crop_dir, os.path.basename(output_path))
        cv2.imwrite(small_crop_output_path, cropped_img)
        with open(small_crop_output_path.replace('.jpg', '.txt'), 'w', encoding='utf-8') as f:
            f.write(label)
        append_to_report(os.path.basename(output_path), 'small crop dimensions', report_path)
        return

    if (cropped_img.shape[1] * cropped_img.shape[0]) < min_area:
        small_area_crop_output_path = os.path.join(small_area_crop_dir, os.path.basename(output_path))
        cv2.imwrite(small_area_crop_output_path, cropped_img)
        with open(small_area_crop_output_path.replace('.jpg', '.txt'), 'w', encoding='utf-8') as f:
            f.write(label)
        append_to_report(os.path.basename(output_path), 'small area', report_path)
        return

    # Save the cropped and straightened image
    cropped_image = Image.fromarray(cropped_img)
    cropped_image.save(output_path)
    with open(output_path.replace('.jpg', '.txt'), 'w', encoding='utf-8') as f:
        f.write(label)
    return output_path

def process_image(data):
    image_info, small_crop_dir, small_area_crop_dir, report_path = data
    image_path, image_file, polygons, all_crops_dir = image_info
    image_name = '--'.join(image_path.split(os.sep)[-3:]).replace('.jpg', '').replace('.png', '')

    with Image.open(image_path) as img:
        img = np.array(img.convert('RGB'))

        # Find the block coordinates to warp the image if available
        block_coords = None
        for polygon in polygons:
            label = polygon.get('label')
            if label == "block":
                block_coords = [(int(float(x)), int(float(y))) for x, y in [point.split(',') for point in polygon.get('points').split(';')]]
                break

        if block_coords:
            # Warp the entire image
            warped, M = warp_image(img, block_coords)
        else:
            # If no block coordinates, use the original image and identity matrix
            warped, M = img, np.eye(3)

        annotations = []
        crop_count = 0
        for polygon in polygons:
            label = polygon.get('label')
            
            if label == "text":
                text_value = polygon['attributes'].get('value')
                if text_value is None:
                    continue
                
                if '\n' in text_value:
                    print(f"Newline character: {image_file} -- {text_value}")
                    continue
                points = [(int(float(x)), int(float(y))) for x, y in [point.split(',') for point in polygon.get('points').split(';')]]
                # Generate output filename for cropped word image
                output_filename = os.path.join(all_crops_dir, f'{image_name}_{crop_count:04d}.jpg')
                if transform_and_crop_polygon((warped, M, points, output_filename, text_value), small_crop_dir, small_area_crop_dir, report_path):
                    annotations.append((output_filename, text_value))
                    crop_count += 1

    return annotations

def process_annotations(json_path, output_dir, small_crop_dir, small_area_crop_dir, report_path):
    all_crops_dir = os.path.join(output_dir, 'all_crops')
    os.makedirs(all_crops_dir, exist_ok=True)

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error processing file {json_path}: {e}")
        return []

    tasks = []

    image_file = data['name']
    image_path = os.path.join(os.path.dirname(json_path), data['name'])  # Ensure the correct path
    polygons = data['annotations']
    tasks.append((image_path, image_file, polygons, all_crops_dir))

    all_annotations = []
    
    with Pool(cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(process_image, [(task, small_crop_dir, small_area_crop_dir, report_path) for task in tasks]), total=len(tasks), desc='Processing images', unit='image'):
            all_annotations.extend(result)

    return all_annotations

def split_data(annotations, output_dir):
    train_dir = os.path.join(output_dir, 'train_data', 'train')
    val_dir = os.path.join(output_dir, 'train_data', 'val')

    create_directory_structure(train_dir, val_dir)

    train_data, val_data = train_test_split(annotations, test_size=0.2, random_state=42)

    train_gt_path = os.path.join(output_dir, 'train_data', 'rec_gt_train.txt')
    val_gt_path = os.path.join(output_dir, 'train_data', 'rec_gt_val.txt')

    with open(train_gt_path, 'w', encoding='utf-8') as train_f:
        for image_path, label in train_data:
            # Copy the file to the train directory
            shutil.copy(image_path, train_dir)
            rel = os.path.relpath(image_path, root_dir)
            train_f.write(f'{rel}\t{label}\n')

    print(f"Total crops in training: {len(train_data)}")

    with open(val_gt_path, 'w', encoding='utf-8') as val_f:
        for image_path, label in val_data:
            # Copy the file to the validation directory
            shutil.copy(image_path, val_dir)
            rel = os.path.relpath(image_path, root_dir)
            val_f.write(f'{rel}\t{label}\n')
            
    print(f"Total crops in validation: {len(val_data)}")

    print(f"Train annotations written to {train_gt_path}")
    print(f"Validation annotations written to {val_gt_path}")

def main(input_dir, output_dir):
    json_files = [os.path.join(input_dir, json_file) for json_file in os.listdir(input_dir) if json_file.endswith('.json')]
    all_annotations = []

    small_crop_dir = os.path.join(output_dir, 'small_crops')
    small_area_crop_dir = os.path.join(output_dir, 'small_area_crops')
    os.makedirs(small_crop_dir, exist_ok=True)
    os.makedirs(small_area_crop_dir, exist_ok=True)

    report_path = os.path.join(output_dir, 'issue_report.csv')
    with open(report_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['crop_name', 'issue'])

    print("Processing annotations and cropping word images...")
    for json_file in json_files:
        annotations = process_annotations(json_file, output_dir, small_crop_dir, small_area_crop_dir, report_path)
        all_annotations.extend(annotations)
    
    print("Splitting data into training and validation sets and saving annotations...")
    split_data(all_annotations, output_dir)
    print(f"Issue report written to {report_path}")

if __name__ == '__main__':
    input_dir = "/ihub/homedirs/am_cse/pramay/work/annotation/corrections/JSONS/corrected_jsons/"
    output_dir = "/ihub/homedirs/am_cse/pramay/work/Dataset/real_training_v1"
    root_dir = "/ihub/homedirs/am_cse/pramay/work/Dataset/"
    main(input_dir, output_dir)
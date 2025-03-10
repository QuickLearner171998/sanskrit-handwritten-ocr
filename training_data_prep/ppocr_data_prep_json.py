import os
import json
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

def transform_and_crop_polygon(image_info):
    """Transform polygon points using the homography matrix, crop the image, and apply dilation."""
    warped, M, points, output_path = image_info

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
    
    min_width = 50  # Minimum width for valid crops
    min_height = 20  # Minimum height for valid crops
    min_area = 1000  # Minimum area (width * height) to avoid tiny crops

    if cropped_img.size == 0 or cropped_img.shape[1] < min_width or cropped_img.shape[0] < min_height:
        return

    # Check if the area is too small
    if (cropped_img.shape[1] * cropped_img.shape[0]) < min_area:
        return

    # Save the cropped and straightened image
    cropped_image = Image.fromarray(cropped_img)
    cropped_image.save(output_path)

    return output_path

def process_image(image_info):
    image_path, image_file, polygons, all_crops_dir = image_info
    image_name = os.path.splitext(image_file)[0]

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
                points = [(int(float(x)), int(float(y))) for x, y in [point.split(',') for point in polygon.get('points').split(';')]]
                # Generate output filename for cropped word image
                output_filename = os.path.join(all_crops_dir, f'{image_name}_{crop_count:04d}.jpg')
                if transform_and_crop_polygon((warped, M, points, output_filename)):
                    annotations.append((output_filename, text_value))
                    crop_count += 1

    return annotations

def process_annotations(json_path, output_dir):
    # Directory for saving all cropped images before split
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
    image_path = data['name']
    polygons = data['annotations']
    tasks.append((image_path, image_file, polygons, all_crops_dir))

    # Using multiprocessing to process images in parallel
    with Pool(cpu_count()) as pool:
        all_annotations = list(tqdm(pool.imap_unordered(process_image, tasks), total=len(tasks), desc='Processing images', unit='image'))

    # Flattening the list of annotations
    annotations = [item for sublist in all_annotations for item in sublist]

    return annotations

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
            cos_file_path = os.path.join('train_data', 'train', os.path.basename(image_path))
            train_f.write(f'{cos_file_path}\t{label}\n')

    print(f"Total crops in training: {len(train_data)}")

    with open(val_gt_path, 'w', encoding='utf-8') as val_f:
        for image_path, label in val_data:
            # Copy the file to the validation directory
            shutil.copy(image_path, val_dir)
            cos_file_path = os.path.join('train_data', 'val', os.path.basename(image_path))
            val_f.write(f'{cos_file_path}\t{label}\n')
            
    print(f"Total crops in validation: {len(val_data)}")

    print(f"Train annotations written to {train_gt_path}")
    print(f"Validation annotations written to {val_gt_path}")

def main(input_dir, output_dir):
    json_files = [os.path.join(input_dir, json_file) for json_file in os.listdir(input_dir) if json_file.endswith('.json')]
    all_annotations = []

    print("Processing annotations and cropping word images...")
    for json_file in json_files:
        annotations = process_annotations(json_file, output_dir)
        all_annotations.extend(annotations)
    
    print("Splitting data into training and validation sets and saving annotations...")
    split_data(all_annotations, output_dir)

if __name__ == '__main__':
    input_dir = "/ihub/homedirs/am_cse/pramay/work/annotation/corrections/JSONS/corrected_jsons/"
    output_dir = "/ihub/homedirs/am_cse/pramay/work/Dataset/real_training"
    main(input_dir, output_dir)
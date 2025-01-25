import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

def bresenham_line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points

def random_shift(image, w, h, shift_factor=0.02):
    rows, cols = image.shape
    result_image = np.zeros_like(image)
    for y in range(rows):
        for x in range(cols):
            if image[y, x] != 0:
                r = np.random.uniform(-1, 1)
                x_shift = int(x + shift_factor * w * r)
                y_shift = int(y + shift_factor * h * r)
                if 0 <= x_shift < cols and 0 <= y_shift < rows:
                    result_image[y_shift, x_shift] = image[y, x]

                # Euclidean distance calculation
                d = np.sqrt((x - x_shift)**2 + (y - y_shift)**2)
                print(f'Shifted point ({x}, {y}) to ({x_shift}, {y_shift}) with distance {d:.2f}')
    return result_image

def gaussian_distortion(image, std_dev):
    distorted_image = gaussian_filter(image, sigma=std_dev)
    return distorted_image

def curved_distortion(image, rainbow_type='normal'):
    rows, cols = image.shape
    result_image = np.zeros_like(image)
    factor = cols // 3

    for col in range(cols):
        if rainbow_type == 'normal':
            offset = int(factor * np.sin(np.pi * col / cols))
        elif rainbow_type == 'inverted':
            offset = int(-factor * np.sin(np.pi * col / cols))
        result_image[:, col] = np.roll(image[:, col], offset)

    return result_image

def sinusoidal_distortion(image):
    rows, cols = image.shape
    result_image = np.zeros_like(image)
    factor = rows // 10
  
    for row in range(rows):
        offset = int(factor * np.sin(2 * np.pi * row / rows))
        result_image[row, :] = np.roll(image[row, :], offset)
    
    return result_image

def elliptical_distortion(image, a, b):
    rows, cols = image.shape
    result_image = np.zeros_like(image)

    xc, yc = cols // 2, rows // 2
    
    for row in range(rows):
        for col in range(cols):
            x_dist = (col - xc) / a
            y_dist = (row - yc) / b
            
            if x_dist**2 + y_dist**2 <= 1:
                new_col = int(x_dist * a + xc)
                new_row = int(y_dist * b + yc)
                if 0 <= new_col < cols and 0 <= new_row < rows:
                    result_image[row, col] = image[new_row, new_col]
                
    return result_image

def find_bounding_boxes(img):
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    bounding_boxes = []
    for stat in stats[1:]: # skip the first component (background)
        x, y, w, h = stat[:4]
        bounding_boxes.append((x, y, w, h))
    return bounding_boxes

def apply_distortion_on_bbox(image, bbox, distortion_func, **kwargs):
    x, y, w, h = bbox
    sub_image = image[y:y+h, x:x+w]
    distorted_sub_image = distortion_func(sub_image, **kwargs)
    result_image = np.copy(image)
    result_image[y:y+h, x:x+w] = np.where(distorted_sub_image != 0, distorted_sub_image, result_image[y:y+h, x:x+w])
    return result_image

# Function to test individual or combined distortions
def test_methods(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Display original image
    cv2.imshow('Original', image)
    cv2.waitKey(0)

    w, h = image.shape[1], image.shape[0]

    # Find bounding boxes for characters/words
    bounding_boxes = find_bounding_boxes(image)

    random_shift_image = np.copy(image)
    gaussian_distorted_image = np.copy(image)
    curved_image = np.copy(image)
    sinusoidal_image = np.copy(image)
    elliptical_image = np.copy(image)

    # Apply various distortions within bounding boxes and visualize each step
    for bbox in bounding_boxes:
        random_shift_image = apply_distortion_on_bbox(random_shift_image, bbox, random_shift, w=w, h=h)
    cv2.imshow('Random Shift', random_shift_image)
    cv2.waitKey(0)

    for bbox in bounding_boxes:
        gaussian_distorted_image = apply_distortion_on_bbox(gaussian_distorted_image, bbox, gaussian_distortion, std_dev=0.02 * min(w, h))
    cv2.imshow('Gaussian Distortion', gaussian_distorted_image)
    cv2.waitKey(0)

    for bbox in bounding_boxes:
        curved_image = apply_distortion_on_bbox(curved_image, bbox, curved_distortion, rainbow_type='normal')
    cv2.imshow('Curved Distortion', curved_image)
    cv2.waitKey(0)

    for bbox in bounding_boxes:
        sinusoidal_image = apply_distortion_on_bbox(sinusoidal_image, bbox, sinusoidal_distortion)
    cv2.imshow('Sinusoidal Distortion', sinusoidal_image)
    cv2.waitKey(0)

    for bbox in bounding_boxes:
        elliptical_image = apply_distortion_on_bbox(elliptical_image, bbox, elliptical_distortion, a=w//2, b=h//2)
    cv2.imshow('Elliptical Distortion', elliptical_image)
    cv2.waitKey(0)

    combined_image = np.copy(curved_image)
    for bbox in bounding_boxes:
        combined_image = apply_distortion_on_bbox(combined_image, bbox, random_shift, w=w, h=h)
        combined_image = apply_distortion_on_bbox(combined_image, bbox, gaussian_distortion, std_dev=0.02 * min(w, h))
    cv2.imshow('Combined', combined_image)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "/home/pramay/myStuff/ai_apps/IITJodhpur/work/Dataset/SYNTH/train_val/AksharUnicode_2.png"
    test_methods(image_path)
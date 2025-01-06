import cv2
import numpy as np
from skimage.morphology import remove_small_objects, remove_small_holes, closing, opening, square
import os

class DocumentImageProcessor:
    def __init__(self, image_path, output_directory):
        self.image_path = image_path
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.image_name = os.path.splitext(os.path.basename(image_path))[0]
        self.output_directory = output_directory

    def save_image(self, step_name):
        output_path = os.path.join(self.output_directory, f"{self.image_name}_{step_name}.jpg")
        cv2.imwrite(output_path, self.image)

    def resize_image(self, image, max_size=2048):
        if image.shape[0] > max_size or image.shape[1] > max_size:
            # maintain aspect ratio
            if image.shape[0] > image.shape[1]:
                new_height = max_size
                new_width = int(image.shape[1] * (max_size / image.shape[0]))
            else:
                new_width = max_size
                new_height = int(image.shape[0] * (max_size / image.shape[1]))
            return cv2.resize(image, (new_width, new_height))
        return image

    def gaussian_blur(self, kernel_size=(1, 1)):
        self.image = cv2.GaussianBlur(self.image, kernel_size, 0)
        self.save_image('gaussian_blur')
        return self

    def median_blur(self, kernel_size=1):
        self.image = cv2.medianBlur(self.image, kernel_size)
        self.save_image('median_blur')
        return self

    def bilateral_filter(self, d=9, sigma_color=75, sigma_space=75):
        self.image = cv2.bilateralFilter(self.image, d, sigma_color, sigma_space)
        self.save_image('bilateral_filter')
        return self

    def otsu_threshold(self):
        _, self.image = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.save_image('otsu_threshold')
        return self

    def adaptive_threshold(self, block_size=11, C=2):
        self.image = cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, block_size, C)
        self.save_image('adaptive_threshold')
        return self

    def morphological_operation(self, operation, kernel_size=(1, 1)):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        self.image = cv2.morphologyEx(self.image, operation, kernel)
        self.save_image('morphological_operation')
        return self

    def histogram_equalization(self):
        self.image = cv2.equalizeHist(self.image)
        self.save_image('histogram_equalization')
        return self

    def clahe(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        self.image = clahe.apply(self.image)
        self.save_image('clahe')
        return self

    def sharpen(self):
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        self.image = cv2.filter2D(self.image, -1, kernel)
        self.save_image('sharpen')
        return self

    def blackfilter(self, min_black_area_size=100):
        """Remove large black areas from the image."""
        _, black_and_white_image = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(black_and_white_image, connectivity=4)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= min_black_area_size:
                self.image[labels == label] = 255
        self.save_image('blackfilter')
        return self
    
    def better_noisefilter(self, min_size=30, max_size=150):
        """Remove small clusters of pixels (noise) while preserving text."""
        # clean = self.image > 0
        clean = opening(self.image, square(2))
        clean= closing(clean, square(2))
        # clean = remove_small_objects(clean, min_size=min_size, connectivity=2)
        # clean = remove_small_holes(clean, area_threshold=max_size)
        # invert image
        # clean = ~clean
        self.image = (clean * 255).astype(np.uint8)
        self.save_image('better_noisefilter')
        return self

    def blurfilter(self, lonely_cluster_threshold=15):
        """Remove isolated clusters of pixels with limited dark pixels in the neighborhood."""
        kernel_size = 1
        for y in range(kernel_size // 2, self.image.shape[0] - kernel_size // 2):
            for x in range(kernel_size // 2, self.image.shape[1] - kernel_size // 2):
                if self.image[y, x] > 0:
                    neighbors = self.image[y - kernel_size // 2 : y + kernel_size // 2 + 1,
                                           x - kernel_size // 2 : x + kernel_size // 2 + 1]
                    if np.sum(neighbors) <= lonely_cluster_threshold:
                        self.image[y, x] = 0
        self.save_image('blurfilter')
        return self

    def crop_white_region(self):
        """Crop the largest white region that is surrounded by a black region."""
        # otsu thresholding
        _, binary = cv2.threshold(self.image, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)

        # draw all contours on image and save image
        
        x, y, w, h = cv2.boundingRect(max_contour)
        
        self.image = self.image[y:y+h, x:x+w]
        self.save_image('crop_white_region')
        return self

    def preprocess_image(self):
        self.image = self.resize_image(self.image)  # Resize after cropping

        self.crop_white_region()

        self.median_blur()  # You can choose other methods as well
        self.clahe()
        # self.blackfilter()
        self.better_noisefilter()  # Enhanced noise filter
        self.sharpen()

        return self.image

# Test the class-based preprocessing
if __name__ == "__main__":
    import tqdm

    input_directory = '/ihub/homedirs/am_cse/pramay/work/Dataset/Sample_test_1'  # Replace with your input directory path
    output_directory = '/ihub/homedirs/am_cse/pramay/work/outputs/image_enh'  # Replace with your output directory path

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for root, _, files in os.walk(input_directory):
        for file in tqdm.tqdm(files):
            if file.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                input_image_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_directory)
                output_image_path = os.path.join(output_directory, relative_path, file)

                if not os.path.exists(os.path.dirname(output_image_path)):
                    os.makedirs(os.path.dirname(output_image_path))

                processor = DocumentImageProcessor(input_image_path, output_directory)
                processed_image = processor.preprocess_image()
                cv2.imwrite(output_image_path, processed_image)
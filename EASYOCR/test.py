import cv2
from PIL import Image, ImageDraw, ImageFont
import easyocr
import os
import argparse
import logging
import numpy as np
from preproc import DocumentImageProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def cleanup_text(text):
    """Function to clean up the text for better formatting."""
    return text.strip()

class OCRProcessor:
    def __init__(self, input_dir, output_dir, width=1024, font_path="NotoSansDevanagari-Regular.ttf", preprocess=False):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.width = width
        self.font_path = font_path  # Path to Sanskrit font
        self.preprocess = preprocess

        # Initialize EasyOCR Reader for Hindi (Sanskrit-compatible)
        logging.info("Initializing EasyOCR Reader for Hindi")
        self.reader = easyocr.Reader(['hi'])

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Output directory created: {output_dir}")

    # Function to resize the image while keeping the aspect ratio using OpenCV
    def resize_image(self, image):
        h, w = image.shape[:2]
        aspect_ratio = self.width / w
        new_dimensions = (self.width, int(h * aspect_ratio))
        resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
        logging.info(f"Image resized to width: {self.width}, aspect ratio maintained.")
        return resized_image


    # Function to create a blank template with the image, text, and confidence scores using Pillow
    def visualize_text_on_template(self, image, result, title_tag = 'ocr'):
        # Convert OpenCV image to PIL image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        w, h = pil_image.size
        # Set a max width for the template, and adjust if it's too large
        max_template_width = min(w * 3, 4000)

        # Calculate the height of the template based on the number of text predictions
        line_height = 60  # Height for each line of text
        num_lines = len(result)
        template_height = max(h, (num_lines * line_height))+ 60  # Add extra space for padding

        # Create a blank white template
        template = Image.new("RGB", (max_template_width, template_height), (255, 255, 255))

        # Paste the resized image on the left side of the template
        template.paste(pil_image, (0, 0))

        # Prepare to write the text in the middle and confidence scores on the right
        draw = ImageDraw.Draw(template)
        font = ImageFont.truetype(self.font_path, 48)

        y_offset = 30  # Start drawing below the image
        for i, (bbox, text, score) in enumerate(result):
            # Cleanup the text
            text = cleanup_text(text)

            # Draw text and confidence score
            text_position = (w + 20, y_offset)
            confidence_position = (max_template_width - 300, y_offset)
            draw.text(text_position, text, font=font, fill=(0, 0, 0))
            draw.text(confidence_position, f"{score:.2f}", font=font, fill=(0, 0, 0))

            # Draw bounding box on the template
            # Unpack the bounding box coordinates and convert them to integers
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            br = (int(br[0]), int(br[1]))

            # Draw the bounding box
            # draw.rectangle([tl, br], outline="red", width=2)  # Draw the bounding box in red

            y_offset += line_height  # Line spacing


        # resize the template to 1024
        template = template.resize((2048, int(template_height * 2048 / max_template_width)))


        return template

    def process_image(self, image_path):
        # Derive the relative path from the input directory
        relative_path = os.path.relpath(image_path, self.input_dir)
        output_dir = os.path.join(self.output_dir, os.path.dirname(relative_path))
        filename = os.path.basename(image_path)
        file_output_dir = os.path.join(output_dir, filename)
        
        # Create the output subdirectory if it doesn't exist
        if not os.path.exists(file_output_dir):
            os.makedirs(file_output_dir)
            logging.info(f"Output subdirectory created: {file_output_dir}")

        # Prepare output filenames
        
        output_txt_filename = f"{os.path.splitext(filename)[0]}.txt"
        output_template_filename = f"{os.path.splitext(filename)[0]}_template.png"
        output_file = os.path.join(file_output_dir, output_txt_filename)
        output_template_file = os.path.join(file_output_dir, output_template_filename)

        # Read the image using OpenCV
        logging.info(f"Processing image: {filename}")
        processed_image = cv2.imread(image_path)
        if processed_image is None:
            logging.error(f"Failed to load image: {filename}")
            return
        

        resized_img = self.resize_image(processed_image) # only for viz

        if self.preprocess:
            processor = DocumentImageProcessor(image_path, file_output_dir)
            processed_image = processor.preprocess_image()
        else:
            processed_image = resized_img  


        # Perform OCR on the original resized image
        result_original = self.reader.readtext(processed_image)

        # # Perform OCR on the preprocessed image
        # result_preprocessed = self.reader.readtext(preprocessed_img)

        # Visualize text on blank template and save
        template_img_original = self.visualize_text_on_template(resized_img, result_original, title_tag="Original Image OCR")
        output_template_file_original = os.path.join(file_output_dir, f"{os.path.splitext(filename)[0]}_original_template.png")
        template_img_original.save(output_template_file_original)
        logging.info(f"Original template image saved to: {output_template_file_original}")

        # Save the text results in .txt files
        logging.info(f"Saving OCR results to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Original Image OCR Results:\n\n")
            for bbox, text, score in result_original:
                text = cleanup_text(text)
                f.write(f"{text} (Confidence: {score:.2f})\n")
                logging.info(f"Original Image - Extracted Text: {text}, Confidence: {score:.2f}")


    def process_images(self):
        # Use os.walk to traverse the directory structure
        for root, _, files in os.walk(self.input_dir):
            for filename in files:
                if filename.endswith((".tif", ".tiff", '.jpg', '.png')):  # Process only image files
                    # Construct the full path to the image file
                    image_path = os.path.join(root, filename)
                    # if 'Scan_0649' not in image_path:
                    #     continue
                    # Process the image file
                    self.process_image(image_path)


# Main function to parse arguments and run the processor
def main():
    parser = argparse.ArgumentParser(description="Process images for OCR and save results as text files.")
    parser.add_argument('--input_dir', type=str, help='Path to the input directory containing images')
    parser.add_argument('--output_dir', type=str, help='Path to the output directory for saving text files and images')
    parser.add_argument('--width', type=int, default=2048, help='Width to resize the image (default: 1024)')
    parser.add_argument('--font_path', type=str, help='Path to the Sanskrit font .ttf file')
    # add strore true preprocess flag
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the image before OCR')

    args = parser.parse_args()

    logging.info(f"Input directory: {args.input_dir}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Image width set to: {args.width}")
    logging.info(f"Using font: {args.font_path}")
    logging.info(f"Preprocess flag: {args.preprocess}")

    # Create an instance of OCRProcessor and process images
    ocr_processor = OCRProcessor(input_dir=args.input_dir, output_dir=args.output_dir, width=args.width, font_path=args.font_path, preprocess=args.preprocess)
    ocr_processor.process_images()

if __name__ == "__main__":
    main()
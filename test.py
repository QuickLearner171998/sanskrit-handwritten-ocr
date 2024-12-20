import cv2
from PIL import Image, ImageDraw, ImageFont
import easyocr
import os
import argparse
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def cleanup_text(text):
    """Function to clean up the text for better formatting."""
    return text.strip()

class OCRProcessor:
    def __init__(self, input_dir, output_dir, width=1024, font_path="NotoSansDevanagari-Regular.ttf"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.width = width
        self.font_path = font_path  # Path to Sanskrit font

        # Initialize EasyOCR Reader for Hindi (Sanskrit-compatible)
        logging.info("Initializing EasyOCR Reader for Hindi...")
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
    def visualize_text_on_template(self, image, result):
        # Convert OpenCV image to PIL image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        w, h = pil_image.size
        # Set a max width for the template, and adjust if it's too large
        max_template_width = min(w * 3, 2048)

        # Calculate the height of the template based on the number of text predictions
        line_height = 40  # Height for each line of text
        num_lines = len(result)
        template_height = h + (num_lines * line_height) + 60  # Add extra space for padding

        # Create a blank white template
        template = Image.new("RGB", (max_template_width, template_height), (255, 255, 255))

        # Paste the resized image on the left side of the template
        template.paste(pil_image, (0, 0))

        # Prepare to write the text in the middle and confidence scores on the right
        draw = ImageDraw.Draw(template)
        font = ImageFont.truetype(self.font_path, 24)

        y_offset = h + 30  # Start drawing below the image
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
            draw.rectangle([tl, br], outline="red", width=2)  # Draw the bounding box in red

            y_offset += line_height  # Line spacing

        return template


    # Function to process individual images
    def process_image(self, image_path):
        # Derive the relative path from the input directory
        relative_path = os.path.relpath(image_path, self.input_dir)
        output_dir = os.path.join(self.output_dir, os.path.dirname(relative_path))
        
        # Create the output subdirectory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Output subdirectory created: {output_dir}")

        # Prepare output filenames
        filename = os.path.basename(image_path)
        output_txt_filename = f"{os.path.splitext(filename)[0]}.txt"
        output_template_filename = f"{os.path.splitext(filename)[0]}_template.png"
        output_file = os.path.join(output_dir, output_txt_filename)
        output_template_file = os.path.join(output_dir, output_template_filename)

        # Read the image using OpenCV
        logging.info(f"Processing image: {filename}")
        img = cv2.imread(image_path)
        if img is None:
            logging.error(f"Failed to load image: {filename}")
            return

        # Resize the image while keeping the aspect ratio
        resized_img = self.resize_image(img)

        # Perform OCR on the resized image
        result = self.reader.readtext(resized_img)

        # Save the text results in a .txt file
        logging.info(f"Saving OCR results to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for bbox, text, score in result:
                text = cleanup_text(text)
                f.write(f"{text} (Confidence: {score:.2f})\n")
                logging.info(f"Extracted Text: {text}, Confidence: {score:.2f}")

        # Visualize text on blank template and save
        template_img = self.visualize_text_on_template(resized_img, result)
        template_img.save(output_template_file)
        logging.info(f"Template image saved to: {output_template_file}")

    def process_images(self):
        # Use os.walk to traverse the directory structure
        for root, _, files in os.walk(self.input_dir):
            for filename in files:
                if filename.endswith((".png", ".jpg", ".jpeg", ".tiff", ".tif")):  # Process only image files
                    # Construct the full path to the image file
                    image_path = os.path.join(root, filename)
                    # Process the image file
                    self.process_image(image_path)


# Main function to parse arguments and run the processor
def main():
    parser = argparse.ArgumentParser(description="Process images for OCR and save results as text files.")
    parser.add_argument('--input_dir', type=str, help='Path to the input directory containing images')
    parser.add_argument('--output_dir', type=str, help='Path to the output directory for saving text files and images')
    parser.add_argument('--width', type=int, default=1024, help='Width to resize the image (default: 1024)')
    parser.add_argument('--font_path', type=str, help='Path to the Sanskrit font .ttf file')

    args = parser.parse_args()

    logging.info(f"Input directory: {args.input_dir}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Image width set to: {args.width}")
    logging.info(f"Using font: {args.font_path}")

    # Create an instance of OCRProcessor and process images
    ocr_processor = OCRProcessor(input_dir=args.input_dir, output_dir=args.output_dir, width=args.width, font_path=args.font_path)
    ocr_processor.process_images()

if __name__ == "__main__":
    main()

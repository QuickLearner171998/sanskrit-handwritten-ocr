import argparse
import random
import os
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

def add_noise(img):
    """
    Add random noise to the image.
    """
    np_img = np.array(img)
    noise = np.random.normal(loc=0, scale=25, size=np_img.shape)
    noisy_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def apply_distortion(img):
    """
    Apply a slight perspective distortion to the image without cutting text.
    """
    width, height = img.size
    max_shift = width * 0.01  # Limit the maximum shift to 1% of the width
    xshift = random.uniform(-max_shift, max_shift)
    yshift = random.uniform(-max_shift, max_shift)

    coeffs = (
        1 + random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01), xshift,
        random.uniform(-0.01, 0.01), 1 + random.uniform(-0.01, 0.01), yshift
    )

    img = img.transform((width, height), Image.AFFINE, coeffs, Image.BICUBIC)
    return img

def blend_with_background(img, bg_path):
    """
    Blend the image with a random background texture.
    """
    if not bg_path:
        return img

    available_backgrounds = [os.path.join(bg_path, f) for f in os.listdir(bg_path) if os.path.isfile(os.path.join(bg_path, f))]
    if available_backgrounds:
        background = Image.open(random.choice(available_backgrounds)).resize(img.size)
        blended_img = Image.blend(background, img, alpha=0.7)
        return blended_img
    else:
        return img

def adjust_font_size(text, font_path, max_width, max_height, min_font_size=10, max_font_size=200):
    """
    Adjust the font size so that the text fits within the given dimensions.
    """
    for font_size in range(max_font_size, min_font_size, -5):
        font = ImageFont.truetype(font_path, font_size)
        bbox = ImageDraw.Draw(Image.new('RGB', (10, 10))).textbbox((0, 0), text, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if text_width <= max_width and text_height <= max_height:
            return font_size
    return min_font_size

def render_text_as_image(args):
    """
    Render text as an image and apply augmentations.
    """
    text, font_path, output_path, max_height, max_width, bg_path = args

    font_size = adjust_font_size(text, font_path, max_width, max_height)
    font = ImageFont.truetype(font_path, font_size)
    bbox = ImageDraw.Draw(Image.new('RGB', (10, 10))).textbbox((0, 0), text, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

    image = Image.new("RGB", (text_width + 40, text_height + 40), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), text, font=font, fill=(0, 0, 0))

    image = add_noise(image)
    image = apply_distortion(image)
    image = blend_with_background(image, bg_path)

    image.save(output_path)

def get_valid_words(lines, max_char):
    """
    Extract words from lines and filter by maximum character length.
    """
    words = []
    for line in lines:
        # Split the line into words and filter by length
        line_words = [word.strip() for word in line.split() if word.strip()]
        valid_words = [word for word in line_words if len(word) <= max_char]
        words.extend(valid_words)
    return list(set(words))  # Remove duplicates

def process_lines(args, fontfile, words, output_font_dir, label_file, bg_path, split='train'):
    """
    Process the words for a single font and save the images and their labels.
    """
    fontname = os.path.splitext(fontfile)[0]
    fontpath = os.path.join(args.font, fontfile)

    tasks = []
    image_id = 1
    for word in words:
        output_image_path = os.path.join(output_font_dir, f"train_data/{split}_synth", f"{split}_{fontname}_{image_id}.png")
        tasks.append((word.strip(), fontpath, output_image_path, args.max_height, args.max_width, bg_path))
        image_id += 1

    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(render_text_as_image, tasks)

    with open(label_file, "a", encoding='utf-8') as ft:
        image_id = 1
        for word in words:
            output_image_path = os.path.join(output_font_dir, f"train_data/{split}_synth", f"{split}_{fontname}_{image_id}.png")
            relative_path = os.path.relpath(output_image_path, start=output_font_dir)
            sanitized_word = word.strip().upper()
            ft.write(f"{relative_path}\t{sanitized_word}\n")
            image_id += 1

def process_font(args, fontfile, words):
    output_font_dir = args.output_dir

    if not os.path.exists(os.path.join(output_font_dir, "train_data", "train_synth")):
        os.makedirs(os.path.join(output_font_dir, "train_data", "train_synth"))

    if not os.path.exists(os.path.join(output_font_dir, "train_data", "val_synth")):
        os.makedirs(os.path.join(output_font_dir, "train_data", "val_synth"))

    if not os.path.exists(os.path.dirname(os.path.join(output_font_dir, "train_data", args.train_label_file))):
        os.makedirs(os.path.dirname(os.path.join(output_font_dir, "train_data", args.train_label_file)))

    if not os.path.exists(os.path.dirname(os.path.join(output_font_dir, "train_data", args.val_label_file))):
        os.makedirs(os.path.dirname(os.path.join(output_font_dir, "train_data", args.val_label_file)))

    sampled_words = random.sample(words, min(args.num_samples, len(words)))
    
    # Print statistics
    max_len = max([len(word) for word in sampled_words])
    avg_len = sum([len(word) for word in sampled_words]) / len(sampled_words)
    print(f"Processing font {fontfile}:")
    print(f"- Number of samples: {len(sampled_words)}")
    print(f"- Max word length: {max_len}")
    print(f"- Average word length: {avg_len:.2f}")

    split_index = int(0.8 * len(sampled_words))
    train_words = sampled_words[:split_index]
    val_words = sampled_words[split_index:]

    process_lines(args, fontfile, train_words, output_font_dir, os.path.join(output_font_dir, "train_data", args.train_label_file), args.background_dir, split='train')
    process_lines(args, fontfile, val_words, output_font_dir, os.path.join(output_font_dir, "train_data", args.val_label_file), args.background_dir, split='val')

def process_single_font(arg_tuple):
    args, fontfile, words = arg_tuple
    process_font(args, fontfile, words)

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic image crops of Sanskrit words.")
    parser.add_argument("-f", "--font", type=str, required=True, help="Path to the directory containing TrueType font files.")
    parser.add_argument("-n", "--num_samples", type=int, default=500, help="Number of samples to generate per font.")
    parser.add_argument("-o", "--output_dir", type=str, default="output", help="Directory to save generated images.")
    parser.add_argument("-b", "--background_dir", type=str, help="Directory of background textures to blend.")
    parser.add_argument("-t", "--text_file", type=str, required=True, help="Text file containing lines of Sanskrit text.")
    parser.add_argument("-tl", "--train_label_file", type=str, default="train_gt_synth.txt", help="Output train label file.")
    parser.add_argument("-vl", "--val_label_file", type=str, default="val_gt_synth.txt", help="Output validation label file.")
    parser.add_argument("--max_height", type=int, default=80, help="Maximum height of the generated images.")
    parser.add_argument("--max_width", type=int, default=800, help="Maximum width of the generated images.")
    parser.add_argument("--max_char", type=int, default=25, help="Maximum number of characters in a word.")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.text_file, encoding='utf-8') as f:
        lines = f.readlines()

    # Get valid words based on max_char limit
    words = get_valid_words(lines, args.max_char)
    print(f"Total unique words (after filtering by max_char={args.max_char}): {len(words)}")

    # Ensure random sampling consistency
    random.seed(42)

    font_files = [f for f in os.listdir(args.font) if f.endswith('.ttf')]
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        list(tqdm(executor.map(process_single_font, [(args, fontfile, words) for fontfile in font_files]), 
                 total=len(font_files), desc="Processing Fonts"))

if __name__ == "__main__":
    main()
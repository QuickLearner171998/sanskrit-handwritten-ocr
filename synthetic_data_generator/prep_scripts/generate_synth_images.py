import argparse
import random
import os
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

def add_noise(img):
    np_img = np.array(img)
    noise = np.random.normal(loc=0, scale=25, size=np_img.shape)
    noisy_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def apply_distortion(img):
    width, height = img.size
    max_shift = width * 0.01
    xshift = random.uniform(-max_shift, max_shift)
    yshift = random.uniform(-max_shift, max_shift)
    
    coeffs = (
        1 + random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01), xshift,
        random.uniform(-0.01, 0.01), 1 + random.uniform(-0.01, 0.01), yshift
    )
    
    return img.transform((width, height), Image.AFFINE, coeffs, Image.BICUBIC)

def blend_with_background(img, bg_path):
    if not bg_path:
        return img
        
    available_backgrounds = [f for f in os.listdir(bg_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if available_backgrounds:
        bg_file = os.path.join(bg_path, random.choice(available_backgrounds))
        background = Image.open(bg_file).convert('RGB').resize(img.size)
        return Image.blend(background, img, 0.7)
    return img

def get_text_size(text, font):
    dummy = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]

def adjust_font_size(text, font_path, max_width, max_height):
    min_size, max_size = 10, 200
    current_size = max_size
    
    while current_size > min_size:
        font = ImageFont.truetype(font_path, current_size)
        width, height = get_text_size(text, font)
        
        if width <= max_width * 0.8 and height <= max_height * 0.8:
            return current_size
        current_size -= 5
        
    return min_size

def render_text_as_image(args):
    text, font_path, output_path, max_height, max_width, bg_path = args

    # Calculate font size and create font
    font_size = adjust_font_size(text, font_path, max_width, max_height)
    font = ImageFont.truetype(font_path, font_size)
    
    # Get text dimensions
    width, height = get_text_size(text, font)
    
    # Add padding
    padding = 40
    img_width = width + 2 * padding
    img_height = height + 2 * padding
    
    # Create image with white background
    image = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Draw text centered
    text_x = (img_width - width) // 2
    text_y = (img_height - height) // 2
    draw.text((text_x, text_y), text, font=font, fill='black')
    
    # Apply effects
    image = add_noise(image)
    image = apply_distortion(image)
    image = blend_with_background(image, bg_path)
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
    except Exception as e:
        print(f"Error saving image {output_path}: {e}")

def get_valid_words(lines, max_char):
    words = set()
    for line in lines:
        for word in line.strip().split():
            word = word.strip()
            if word and len(word) <= max_char and not word.isspace():
                words.add(word)
    return list(words)

def process_lines(args, fontfile, words, output_dir, label_file, bg_path, split='train'):
    fontname = os.path.splitext(fontfile)[0]
    fontpath = os.path.join(args.font, fontfile)
    
    tasks = []
    for i, word in enumerate(words, 1):
        output_path = os.path.join(
            output_dir, 
            f"train_data/{split}_synth", 
            f"{split}_{fontname}_{i}.png"
        )
        tasks.append((word, fontpath, output_path, args.max_height, args.max_width, bg_path))
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(render_text_as_image, tasks)
    
    os.makedirs(os.path.dirname(label_file), exist_ok=True)
    with open(label_file, "a", encoding='utf-8') as f:
        for i, word in enumerate(words, 1):
            img_path = f"train_data/{split}_synth/{split}_{fontname}_{i}.png"
            f.write(f"{img_path}\t{word.strip().upper()}\n")

def process_font(args, fontfile, words):
    os.makedirs(os.path.join(args.output_dir, "train_data/train_synth"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "train_data/val_synth"), exist_ok=True)
    
    sampled_words = random.sample(words, min(args.num_samples, len(words)))
    split_idx = int(0.8 * len(sampled_words))
    
    train_words = sampled_words[:split_idx]
    val_words = sampled_words[split_idx:]
    
    process_lines(
        args, fontfile, train_words, 
        args.output_dir,
        os.path.join(args.output_dir, "train_data", args.train_label_file),
        args.background_dir, 
        'train'
    )
    
    process_lines(
        args, fontfile, val_words,
        args.output_dir,
        os.path.join(args.output_dir, "train_data", args.val_label_file),
        args.background_dir,
        'val'
    )

def process_single_font(arg_tuple):
    args, fontfile, words = arg_tuple
    try:
        process_font(args, fontfile, words)
    except Exception as e:
        print(f"Error processing font {fontfile}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic image crops of Sanskrit words.")
    parser.add_argument("-f", "--font", type=str, required=True, help="Font directory path")
    parser.add_argument("-n", "--num_samples", type=int, default=500, help="Samples per font")
    parser.add_argument("-o", "--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("-b", "--background_dir", type=str, help="Background textures directory")
    parser.add_argument("-t", "--text_file", type=str, required=True, help="Input text file")
    parser.add_argument("-tl", "--train_label_file", type=str, default="train_gt_synth.txt", help="Train labels file")
    parser.add_argument("-vl", "--val_label_file", type=str, default="val_gt_synth.txt", help="Validation labels file")
    parser.add_argument("--max_height", type=int, default=80, help="Maximum image height")
    parser.add_argument("--max_width", type=int, default=800, help="Maximum image width")
    parser.add_argument("--max_char", type=int, default=25, help="Maximum characters per word")
    args = parser.parse_args()

    with open(args.text_file, encoding='utf-8') as f:
        words = get_valid_words(f.readlines(), args.max_char)
    
    print(f"Found {len(words)} valid unique words")
    
    random.seed(42)
    font_files = [f for f in os.listdir(args.font) if f.endswith('.ttf')]
    
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        list(tqdm(
            executor.map(process_single_font, [(args, font, words) for font in font_files]),
            total=len(font_files),
            desc="Processing fonts"
        ))

if __name__ == "__main__":
    main()
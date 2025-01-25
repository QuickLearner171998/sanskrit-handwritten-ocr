#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import sys
import os
from tqdm import tqdm
from PIL import Image
import cairocffi as cairo
import pangocffi
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

width, height = 320, 120

y = list(',.0123456789-_|')
CHARMAP = [chr(i) for i in range(2304, 2432)] + [chr(i) for i in range(65, 90)] + y

def bgra_surf_to_rgba_string(cairo_surface):
    img = Image.frombuffer('RGBA', (cairo_surface.get_width(), cairo_surface.get_height()), cairo_surface.get_data(), 'raw', 'BGRA', 0, 1)
    return img.tobytes('raw', 'RGBA', 0, 1)

fontname = sys.argv[1] if len(sys.argv) >= 2 else "Sans"

pathf = "line_images/"
FONT_TYPE = ["Dekko", "Shobhika", "Yatra One", "Yantramanav", "Kalam", "Utsaah", "Tillana", "Teko", "Sura", "Siddhanta", "Sarpanch", "Sarala", "Sarai", "Sanskrit 2003", "Sanskrit Text", "Samyak Devanagari", "Samanata", "SakalBharati", "Sahadeva", "Rozha One", "Rhodium Libre", "Rajdhani", "Poppins", "Nirmala UI", "Nakula", "Modak", "Lohit Devanagari", "Kokila", "Khand", "Karma", "Hind", "Halant", "GIST-DVOTMohini", "GIST-DVOTKishor", "GISTOT-BRXVinit", "GISTOT-DGRDhruv", "Eczar", "Ek Mukta", "Gargi", "Chandas", "Biryani", "Asar", "Arya", "Amiko", "Amita", "Aparajita", "Akshar Unicode", "Laila", "Kurale", "Noto Sans", "Mukta", "Gotu", "Pragati Narrow", "Baloo 2", "Baloo", "Martel Sans", "Khula", "Jaldi", "Glegoo", "Palanquin", "Palanquin Dark", "Cambay", "Kadwa", "Vesper Libre", "Sumana", "Ranga", "Sahitya"]
print(sorted(FONT_TYPE))
print(len(FONT_TYPE))

if not os.path.exists(os.getcwd() + "/label_data"):
    os.mkdir(os.getcwd() + "/label_data")

with open("label_data/annot_synthetic.txt", "w") as ft:
    for fontname in tqdm(FONT_TYPE, desc="Processing Fonts"):
        with open('data_preparation/synthetic/sanskritdoc.txt') as f:
            lines = random.sample(f.readlines(), 5000)
        url_name = 1
        for line in tqdm(lines, desc=f"Rendering lines for '{fontname}'", leave=False):
            if not os.path.exists(pathf + fontname.replace(" ", "")):
                os.makedirs(pathf + fontname.replace(" ", ""))
            surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, 1800, 80)
            context = cairo.Context(surf)

            context.rectangle(0, 0, 1800, 80)
            context.set_source_rgb(1, 1, 1)
            context.fill()
            context.translate(20, 20)

            layout = pangocffi.create_layout(context)
            font = pangocffi.FontDescription(f"{fontname} 25")
            layout.set_font_description(font)
            layout.set_text(line.strip())
            context.set_source_rgb(0, 0, 0)
            layout.show_in_cairo_context(context)

            image_path = os.path.join(pathf + fontname.replace(" ", ""), f"{url_name}.png")
            with open(image_path, "wb") as image_file:
                surf.write_to_png(image_file)
                line = line.upper()
                for i, c in enumerate(line):
                    if ord(c) >= 65 and ord(c) <= 90:
                        line = line.replace(c, "#")
                line = line.strip()
                ft.write(f"{os.getcwd()}/line_images/{fontname.replace(' ', '')}/{url_name}.png {line}\n")
                url_name += 1
        print(fontname)
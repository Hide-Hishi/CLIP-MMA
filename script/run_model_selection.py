import yaml
import clip
import torch
import pickle
import os
from PIL import Image, ImageFilter
import glob
import sys
from pathlib import Path
import argparse


base_dir = Path(os.path.abspath(__file__)).parents[1]
sys.path.insert(0, base_dir)

from clip_mma.selection_func import measure_similarity


SAVE_DIR = "results"


def main():
    parser = argparse.ArgumentParser(description="Process an image.")
    parser.add_argument("cfg_filename", type=str, default= "config.yaml", help="Path to the config file you want to use")
    parser.add_argument("img_path", type=str, help="Path to the image file")

    args = parser.parse_args()

    cfg_name = args.cfg_filename
    yaml_file_path = os.path.join(base_dir, 'config',cfg_name)

    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)

    img_path = args.img_path

    pil_img = Image.open(img_path)
    blur_img = pil_img.filter(ImageFilter.GaussianBlur(radius=3))

    os.makedirs(SAVE_DIR,exist_ok=True)
    measure_similarity(blur_img, config, SAVE_DIR, auto_crop=True)


if __name__ == "__main__":
    main()

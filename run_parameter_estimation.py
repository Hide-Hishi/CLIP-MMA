import yaml
import clip
import torch
import pickle
import os
from PIL import Image
import glob
import sys
import numpy as np
from PIL import Image
import glob
import argparse

from clip_mma.estimation import parameter_estimation, show_estimation2D

def target_param_extractor(img_path):
    filename = os.path.basename(img_path)
    fv = float(filename.split("_")[1])
    gv = float(filename.split("_")[3])
    return [fv,gv]


def main():
    parser = argparse.ArgumentParser(description="Process an image.")
    parser.add_argument("cfg_filename", type=str, default= "config.yaml", help="Path to the config file you want to use")
    parser.add_argument("target_images_path", type=str)

    args = parser.parse_args()

    cfg_name = args.cfg_filename
    yaml_file_path = os.path.join('config',cfg_name)

    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)

    #custom how to extract ground truth parameters of the target
    target_images_path = args.target_images_path
    gt_param = target_param_extractor(os.path.basename(target_images_path))

    pil_images = []
    for img_path in glob.glob(os.path.join(target_images_path,"*.png")):
        pil_img = Image.open(img_path)
        pil_images.append(pil_img)

    param_pred = parameter_estimation(config,pil_images)
    show_estimation2D(config,param_pred,gt_param)


if __name__ == "__main__":
    main()

import yaml
import clip
import torch
import pickle
import os
from PIL import Image
import glob
import sys
import argparse

from clip_mma.estimation import parameter_estimation_preprocessing

def main():
    parser = argparse.ArgumentParser(description="Process an image.")
    parser.add_argument("cfg_filename", type=str, default= "config.yaml", help="Path to the config file you want to use")

    args = parser.parse_args()

    cfg_name = args.cfg_filename
    yaml_file_path = os.path.join('config',cfg_name)

    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)

    parameter_estimation_preprocessing(config)
    
if __name__ == "__main__":
    main()

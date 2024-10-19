import pickle
import os
import glob
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import ImageFilter
import cv2
import clip

from . import utils
from .pil_iterator import PILImageIterator
from .handle_data import MathModelData



class ImageEncoder():
    """
    A class to encode images into embedding vectors using a CLIP model and manage the storage and retrieval of these embeddings.

    Attributes
    ----------
    config : dict
        Configuration settings for the encoder including model specifics and data paths.
    device : torch.device
        The computation device (CPU or GPU) that the model will utilize.
    model : torch.nn.Module
        The visual part of the CLIP model used for generating image embeddings.
    preprocess : function
        A function to preprocess images suitable for the CLIP model input.
    """
    
    def __init__(self,config):
        self.config = config
        self.device = torch.device("cpu")
        self.model = None
        self.preprocess = None

        #gpu setting
        if config["use_gpu"]:
            self.device = utils.find_gpu()

        if self.device.type == "cpu":
            self.dtype = torch.float
        else:
            self.dtype = torch.half

        #load architecture 
        if config["use_model"]=="clip":
            clip_cfg = config["model"]["clip"]
            base_model = clip_cfg["base_model"]
            clip_model,self.preprocess = clip.load(base_model, device=self.device)
            self.model = clip_model.visual
        else:
            assert()


    def save_embedding_vectors(self,save_dir,image_features,basename_list):
        features_save_path = os.path.join(save_dir,"image_features.pt")
        label_save_path = os.path.join(save_dir,"basename_list.pickle")

        image_features = image_features.to("cpu")
        torch.save(image_features,features_save_path)

        with open(label_save_path, 'wb') as f:
            pickle.dump(basename_list, f)

    def embed_model_data(self,image_dir):
        math_model = os.path.basename(image_dir)
        image_path_list = glob.glob(os.path.join(image_dir,"*.png"))
        image_path_list.sort()

        use_model = self.config["use_model"]
        data_dir = Path(image_dir).parents[1]
        save_dir = os.path.join(data_dir,use_model,"image_features",math_model)
        os.makedirs(save_dir,exist_ok=True)

        self.model = self.model.to(self.device)
        total = []
        for files in tqdm(PILImageIterator(image_path_list,batch=100)):
            if files == []:
                break
            lst = []
            for image in files:
                blur_img = image.filter(ImageFilter.GaussianBlur(radius=3))
                processed_images = self.preprocess(blur_img)
                lst.append(processed_images)
            processed_images = torch.stack(lst,axis=0).to(self.dtype).to(self.device)
            self.model.eval()
            with torch.no_grad():
                image_features = self.model(processed_images)
                image_features = image_features.float()
            total.append(image_features)
            del processed_images,image_features
            torch.cuda.empty_cache()

        image_features = torch.cat(total,axis=0).to("cpu")

        lst = []
        for path in image_path_list:
            lst.append(os.path.basename(path))

        self.save_embedding_vectors(save_dir,image_features,lst)

    def embed_whole_data(self,mode=1):
        data_dir = self.config["data"]["data_dir"]
        if mode == 1:
            print("start embedding for model selection")
            mathmodel_lst = self.config["data"]["mathematical_model_list"]
        elif mode == 2:
            print("start embedding for parameter estimation")
            mathmodel_lst = [
                self.config["data"]["parameter_estimation_data"]["train"],
                self.config["data"]["parameter_estimation_data"]["val"]
            ]
        n = len(mathmodel_lst)
        for i, mathmodel_name in enumerate(mathmodel_lst):
            image_dir = os.path.join(data_dir,"model_image",mathmodel_name)
            print("start embedding images in",image_dir)
            self.embed_model_data(image_dir)
            print(str(i+1)+"/"+str(n)+" models have encoded.")
            
    def embed_single_image(self,pil_img):
        pil_img = pil_img.convert("RGB")
        blur_img = pil_img.filter(ImageFilter.GaussianBlur(radius=3))
        processed_images = self.preprocess(blur_img)
        processed_images = processed_images.unsqueeze(0).to(self.dtype).to(self.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            embedding_vector = self.model(processed_images)
            embedding_vector = embedding_vector.squeeze(0).float().to("cpu")
        return embedding_vector
        
    def load_embedding_vectors(self,save_dir):
        image_features = torch.load(os.path.join(save_dir,"image_features.pt"))
        math_model = os.path.basename(save_dir)

        data_dir = Path(save_dir).parents[2]
        if "model_image" not in os.listdir(data_dir):
            print("The directory named 'model_image' does not exist")
            assert()
        image_dir = os.path.join(data_dir,"model_image",math_model)

        with open(os.path.join(save_dir,"basename_list.pickle"),"rb") as f:
            basename_list = pickle.load(f)

        image_path_list = []
        for basename in basename_list:
            image_path_list.append(os.path.join(image_dir,basename))
        return image_features, image_path_list

    def load_all_data(self):
        use_model = self.config["use_model"]
        data_cfg = self.config["data"]
        data_dir = data_cfg["data_dir"]

        model_name_list = data_cfg["mathematical_model_list"]

        total_features = []
        total_path = []
        category_label = []
        modelname_dict = {}

        for i, model_name in enumerate(model_name_list):
            model_data_path = os.path.join(data_dir,use_model,"image_features",model_name)
            image_features, image_path_list = self.load_embedding_vectors(model_data_path)

            category_label.extend([i]*len(image_path_list))
            modelname_dict[i]=model_name
            total_features.append(image_features)
            total_path.extend(image_path_list)

        total_features = torch.cat(total_features,axis=0)
        category_label = np.array(category_label)
        return MathModelData(total_features,total_path,category_label,modelname_dict)
    
import yaml
import torch
import pickle
import os
from PIL import Image
import glob
import sys
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from handle_data import embedding_datadict_path,parameter_domain_path
from ngb.ngboost_trainer import ParameterEstimator, ngboost_train, NGBoostTrainer
from encoder import ImageEncoder
from dml.dml_module import Dimension_Reducer
from dml_train import dml_train
from ngboost_dataset import generate_ngboost_dataset
from dml_func import save_dml_model

def parameter_estimation_preprocessing(config):
    encoder = ImageEncoder(config)
    encoder.embed_whole_data(mode=2)

    dml_model = dml_train(config)
    save_dml_model(config,dml_model)
    generate_ngboost_dataset(config)
    ngboost_train(config)

def parameter_estimation(config,pil_images):
    encoder = ImageEncoder(config)
    reducer = Dimension_Reducer(config)
    embeddings = []
    for pil_img in pil_images:
        embedding = encoder.embed_single_image(pil_img)
        embeddings.append(embedding)
    embeddings = torch.stack(embeddings,axis=0)
    reduced_embeddings = reducer(embeddings)
    estimator = ParameterEstimator(config)
    param_pred = estimator.estimate(reduced_embeddings)
    return param_pred

    
def show_estimation2D(config,param_pred,gt_param):
    param_domain_path = parameter_domain_path(config)
    parameter_domain_array = np.load(param_domain_path)

    grid_num = config["ngboost"]["grid_num"]
    param_names = config["ngboost"]["parameter_list"]
    x = np.linspace(parameter_domain_array[0,0],parameter_domain_array[0,1],grid_num)
    y = np.linspace(parameter_domain_array[1,0],parameter_domain_array[1,1],grid_num)
    X, Y = np.meshgrid(x, y)
    plt.contourf(X, Y, param_pred, cmap='viridis')
    plt.scatter(gt_param[0],gt_param[1],label="target",color="red")
    plt.colorbar()
    plt.title("predicted posterior distribution")
    plt.xlabel(param_names[0])
    plt.ylabel(param_names[1])
    plt.show()
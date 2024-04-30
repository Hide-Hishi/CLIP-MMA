import dataclasses
from typing import List, Dict, Any
import os
import torch
import numpy as np


@dataclasses.dataclass(frozen=True)
class MathModelData:
    embeddings: torch.Tensor
    image_path: List[str]
    category_label: np.ndarray
    modelname_dict: Dict[int,str]

@dataclasses.dataclass(frozen=True)
class NGBData:
    model: Any
    x_m: np.ndarray
    x_s: np.ndarray
    domain: np.ndarray
    prior_D: np.ndarray


def Turing_filename_parser(filename):
    return [float(t) for t in filename.split("_")[1:6]]

def parameter_extractor(img_path):
    filename = os.path.basename(img_path)
    param_list = Turing_filename_parser(filename)
    return [param_list[1],param_list[3]]

def parameter_extractor_fufv(img_path):
    filename = os.path.basename(img_path)
    param_list = Turing_filename_parser(filename)
    return [param_list[0],param_list[1]]


def embedding_datadict_path(config):
    return os.path.join(
        config["data"]["data_dir"],
        config["use_model"],
        "image_features",
        config["data"]["parameter_estimation_data"]["train"],
        "emb_dict.pickle"
        )


def ngb_model_path(config):
    return os.path.join(
        config["data"]["data_dir"],
        config["use_model"],
        "ngboost",
        "model",
        config["ngboost"]["modelname"]+".pickle"
        )

def parameter_domain_path(config):
    return os.path.join(
        config["data"]["data_dir"],
        "parameter_domain",
        config["data"]["parameter_estimation_data"]["train"],
        "parameter_domain_array.npy"
        )
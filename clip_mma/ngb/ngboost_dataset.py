import torch
import pickle
import os
import sys
import numpy as np
from pathlib import Path

from dml.dml_dataset import BaseDataset
from handle_data import embedding_datadict_path
from dml.dml_module import Dimension_Reducer

def save_emb_dict(config,emb_dict):
    emb_save_path = embedding_datadict_path(config)
    with open(emb_save_path, "wb") as f:
        pickle.dump(emb_dict,f)

def generate_ngboost_dataset(config):
    dataset = BaseDataset(config)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=16)

    reducer = Dimension_Reducer(config)

    embeddings = []
    target_param = []
    for feature,target in dataloader:
        embedding = reducer(feature)
        embeddings.append(embedding)
        target_param.append(target)

    embeddings = torch.cat(embeddings,axis=0).cpu().numpy()
    target_param = torch.cat(target_param,axis=0).cpu().numpy()

    test_rate = 0.1
    test_split = np.zeros(target_param.shape[0])
    flg_array = np.random.rand(target_param.shape[0])
    test_split[flg_array<test_rate]=1

    emb_dict = {}
    emb_dict["embeddings"]=embeddings
    emb_dict["params"]=target_param
    emb_dict["test_split"]=test_split

    save_emb_dict(config,emb_dict)
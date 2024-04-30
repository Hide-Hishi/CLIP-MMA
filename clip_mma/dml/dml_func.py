import logging
from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

import pytorch_metric_learning


logging.getLogger().setLevel(logging.INFO)
logging.info("VERSION %s" % pytorch_metric_learning.__version__)


def visualizer_hook(umapper, umap_embeddings, labels, split_name, keyname, *args):
    logging.info(
        "UMA plot for the {} split and label set {}".format(split_name, keyname)
    )
    label_set = np.unique(labels)
    num_classes = len(label_set)
    plt.figure(figsize=(20, 15))
    plt.gca().set_prop_cycle(
        cycler(
            "color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]
        )
    )
    for i in range(num_classes):
        idx = labels == label_set[i]
        plt.plot(umap_embeddings[idx, 0], umap_embeddings[idx, 1], ".", markersize=1)
    plt.show()

def save_dml_model(config,dml_model):
    save_dir = os.path.join(config["data"]["data_dir"],config["use_model"],"dml_model")
    torch.save(dml_model["trunk"].cpu().state_dict(),os.path.join(save_dir,"trunk_state_dict.pth"))
    torch.save(dml_model["embedder"].cpu().state_dict(),os.path.join(save_dir,"embedder_state_dict.pth"))
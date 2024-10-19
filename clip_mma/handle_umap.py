import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap


class UMAP_Module():
    """
    A module for dimensionality reduction using UMAP.
    This class provides methods to fit UMAP to data, save, and load the UMAP mapper object.

    Attributes
    ----------
    config : dict
        Configuration dictionary specifying parameters for UMAP and data paths.

    Methods
    -------
    fit(all_model_data, display=True)
        Fits UMAP to the embeddings from all_model_data and optionally displays a scatter plot.
    save_umap_mapper(mapper)
        Saves the UMAP mapper object to a file.
    load_umap_mapper()
        Loads the UMAP mapper object from a file.
    """

    def __init__(self,config):
        self.config = config

    def fit(self,all_model_data,display=True):
        umap_cfg = self.config["umap"]

        embeddings = all_model_data.embeddings
        category_label = all_model_data.category_label
        modelname_dict = all_model_data.modelname_dict

        if isinstance(embeddings, np.ndarray):
            pass
        elif isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        else:
            print("invalid data type")
            assert()

        print("n_neighbors: ",umap_cfg["n_neighbors"])
        print("min_dist: ",umap_cfg["min_dist"])
        mapper = umap.UMAP(n_neighbors=umap_cfg["n_neighbors"],
                           min_dist=umap_cfg["min_dist"],
                           random_state=umap_cfg["randomseed"])
        print("umap fitting has started")
        feature = mapper.fit_transform(embeddings)
        print("done")

        if display:
            fig, ax = plt.subplots(figsize=(10,10))
            for i, model_name in modelname_dict.items():
                ax.scatter(feature[category_label==i,0],feature[category_label==i,1],label=model_name)
                ax.autoscale()
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
            plt.legend()
            plt.show()

        return mapper

    def save_umap_mapper(self,mapper):
        data_dir = self.config["data"]["data_dir"]
        use_model = self.config["use_model"]
        save_dir = os.path.join(data_dir,use_model,"mapper")
        umap_cfg = self.config["umap"]
        filename = "_".join(["mapper","nn",str(umap_cfg["n_neighbors"]),"md",str(umap_cfg["min_dist"])])+".pickle"
        save_path = os.path.join(save_dir,filename)
        with open(save_path,"wb") as f:
            pickle.dump(mapper,f)

    def load_umap_mapper(self):
        data_dir = self.config["data"]["data_dir"]
        use_model = self.config["use_model"]
        save_dir = os.path.join(data_dir,use_model,"mapper")
        umap_cfg = self.config["umap"]
        mapper_path = os.path.join(save_dir,umap_cfg["load_mapper_name"])
        with open(mapper_path,"rb") as f:
            mapper = pickle.load(f)
        return mapper
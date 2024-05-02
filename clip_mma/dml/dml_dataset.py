import os
import torch
import numpy as np

from ..encoder import ImageEncoder
from ..handle_data import parameter_extractor, parameter_extractor_fufv


def assign_to_grid(param_array,split_n=None):
    assert(param_array.ndim==2)
    n,d = param_array.shape
    if split_n is None:
        split_n = np.ones(d)*10
    
    assert(np.min(split_n)>0)

    min_array = np.min(param_array, axis=0)
    max_array = np.max(param_array, axis=0)

    bins_lst = []
    for i in range(d):
        bins = np.linspace(min_array[i],max_array[i],split_n[i]+1)
        bins_lst.append(bins)

    int_labels = np.zeros(n)
    for i in range(d):
        indices = np.digitize(param_array[:, i], bins_lst[i]) - 1
        int_labels += indices * 10**i

    return int_labels

class BaseDataset(torch.utils.data.Dataset):
    """
    A dataset class for loading and handling image features and corresponding raw parameters as labels.

    Attributes
    ----------
    data : torch.Tensor
        The tensor containing all the pre-computed image features loaded from the specified directory.
    targets : torch.Tensor
        The tensor containing all the target parameters associated with the images, extracted from filenames.
    """

    def __init__(self, config,mode="train"):
        if mode not in ["train","val","test"]:
            print("Only 'train' or 'val' are acceptable")
            assert()

        image_dir = os.path.join(
            config["data"]["data_dir"],
            config["use_model"],
            "image_features",
            config["data"]["parameter_estimation_data"][mode]
        )
        encoder = ImageEncoder(config)
        image_features, image_path_list = encoder.load_embedding_vectors(image_dir)

        param_lst = []
        for path in image_path_list:
            param = parameter_extractor(path)
            param_lst.append(param)

        self.data = image_features
        self.targets = torch.tensor(param_lst)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feature, target = self.data[index], self.targets[index]
        return feature, target


class DMLDataset(BaseDataset):
    """
    A subclass of BaseDataset designed specifically for Contrastive Learning tasks. 
    This class modifies the target parameters of the dataset by assigning integer labels based on a grid split configuration.
    """
    def __init__(self, config,mode="train"):
        super().__init__(config,mode)
        split_array = config["data"]["param_space_split_n"]
        int_labels = assign_to_grid(self.targets.detach().cpu().numpy(),split_n=split_array)
        self.targets = torch.tensor(int_labels,dtype=torch.int16)

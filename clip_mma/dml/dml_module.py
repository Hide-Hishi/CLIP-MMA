import torch
import torch.nn as nn
import os

class Dimension_Reducer():
    """
    A class that encapsulates a two-stage deep metric learning model for dimensionality reduction. 
    The model comprises a feature extractor (trunk) and an embedding layer (embedder), both of which are loaded from specified state dictionaries.

    Attributes
    ----------
    trunk : torch.nn.Module
        The trunk model, a pre-trained network used for feature extraction.
    embedder : torch.nn.Module
        The embedding layer, a linear layer used for reducing the dimensionality of the features extracted by the trunk.

    Methods
    -------
    __call__(feature)
        Processes the input feature through the trunk and embedder to produce a lower-dimensional embedding.

    """
    def __init__(self,config):
        model_dir = os.path.join(config["data"]["data_dir"],config["use_model"],"dml_model")
        trunk = torch.load(os.path.join(model_dir,"trunk_state_dict.pth"))
        embedder = torch.load(os.path.join(model_dir,"embedder_state_dict.pth"))


        trunk = DenseNet()
        embedder = nn.Linear(16, config["dml"]["output_dim"])

        model_dir = os.path.join(config["data"]["data_dir"],config["use_model"],"dml_model")
        trunk.load_state_dict(torch.load(os.path.join(model_dir,"trunk_state_dict.pth")))
        embedder.load_state_dict(torch.load(os.path.join(model_dir,"embedder_state_dict.pth")))

        trunk = trunk.cpu()
        embedder = embedder.cpu()
        
        trunk.eval()
        embedder.eval()

        self.trunk = trunk
        self.embedder = embedder

    def __call__(self,feature):
        feature = feature.cpu()
        with torch.no_grad():
            embedding = self.embedder(self.trunk(feature))
        return embedding

class DenseNet(nn.Module):
    def __init__(self, dropout_prob=0.2):
        super(DenseNet, self).__init__()

        self.layers = nn.Sequential(
            # Input 512
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
        )

    def forward(self, x):
        return self.layers(x)

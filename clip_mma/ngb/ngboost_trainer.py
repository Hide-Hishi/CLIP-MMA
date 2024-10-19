import os
import sys
import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import copy

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import ngboost.distns.multivariate_normal as mn
from scipy.stats import multivariate_normal

from .ngboost_model import Regressor
from ..handle_data import embedding_datadict_path, parameter_domain_path, ngb_model_path, NGBData

def ngboost_train(config):
    param_domain_path = parameter_domain_path(config)
    parameter_domain_array = np.load(param_domain_path)

    emb_save_dir = embedding_datadict_path(config)
    with open(emb_save_dir, "rb") as f:
        emb_dict = pickle.load(f)

    test_split = emb_dict["test_split"]
    x_train = emb_dict["embeddings"][test_split==False]
    y_train = emb_dict["params"][test_split==False]
    x_test = emb_dict["embeddings"][test_split==True]
    y_test = emb_dict["params"][test_split==True]
    
    ngb_trainer = NGBoostTrainer(config)
    ngb_trainer.fit(x_train,y_train,parameter_domain_array)
    ngb_trainer.evaluate(x_test,y_test)
    ngb_trainer.save()

class NGB_Base():
    """
    A base class for managing a NGBoost model, providing functionalities to normalize and denormalize data, make predictions,
    and save and load the model configuration and parameters.

    Attributes
    ----------
    x_m : float or None
        The mean of the dataset used for normalization. None until set.
    x_s : float or None
        The standard deviation of the dataset used for normalization. None until set.
    config : dict
        Configuration settings for the model.
    model : NGBoost model or None
        The NGBoost model object. None until a model is trained or loaded.
    domain : array-like or None
        The domain for the model predictions. None until set.
    prior_D : distribution or None
        The prior distribution for the NGBoost model. None until set.
    """

    def __init__(self,config):
        self.x_m = None
        self.x_s = None
        self.config = config
        self.model = None
        self.domain = None
        self.prior_D = None

    def normalize(self,x):
        if self.x_m is None:
            print("mean and std has not been calculated yet.")
            assert()
        return (x - self.x_m)/self.x_s

    def denormalize(self,n_x):
        if self.x_m is None:
            print("mean and std has not been calculated yet.")
            assert()
        return self.x_m + n_x*self.x_s

    def predict(self,x):
        n_x = self.normalize(x)
        return self.model.pred_dist(n_x)

    def save(self,overwrite = False):
        if self.model is None:
            print("Model object does not exist.")
            return 

        save_path = ngb_model_path(self.config)
        if not overwrite:
            if os.path.isfile(save_path):
                print("the same file already exists. Saving was canceled")
                return 
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        with open(save_path,"wb") as f:
            pickle.dump(NGBData(self.model,self.x_m,self.x_s,self.domain,self.prior_D),f)

    def load(self):
        save_path = ngb_model_path(self.config)
        with open(save_path,"rb") as f:
            ngb_data = pickle.load(f)
        return ngb_data


class NGBoostTrainer(NGB_Base):
    """
    A subclass of NGB_Base that facilitates training of NGBoost models using customized settings and evaluating
    their performance.

    Attributes
    ----------
    lattice_size : float
        The size parameter used for defining the precision of the lattice in the model's probabilistic predictions.
    grid_num : int
        The number of grid points used in the parameter space for generating prior distributions.
    domain : array-like or None
        The domain of the parameter space for which the model will make predictions.
    """

    def __init__(self,config):
        super().__init__(config)
        ngb_cfg = self.config["ngboost"]
        learner = DecisionTreeRegressor(criterion=ngb_cfg["criterion"],
                                        max_depth=ngb_cfg["tree_max_depth"])

        self.lattice_size = ngb_cfg["lattice_size"]
        self.grid_num = ngb_cfg["grid_num"]

        # Model training
        MVN = mn.MultivariateNormal(ngb_cfg["parameter_n"])

        self.model = Regressor(Dist=MVN,
                                Base=learner,
                                learning_rate=ngb_cfg["learning_rate"],
                                n_estimators=ngb_cfg["n_estimators"]
                                )
        self.domain = None
        self.prior_D = None


    def calc_statistics(self,x):
        self.x_m = np.mean(x,axis=0)
        self.x_s = np.std(x,axis=0)

    def fit(self,x,y,parameter_domain_array):
        self.calc_statistics(x)
        n_x = self.normalize(x)
        self.model.fit(n_x, y, self.lattice_size)

        self.domain = parameter_domain_array
        self.prior_D = self.get_prior_distribution(x)


    def evaluate(self,x,y):
        n_x = self.normalize(x)
        pred_ngb = self.model.predict(n_x)
        # Coefficient of determination
        r2_ngb = r2_score(y, pred_ngb)
        # MAE
        mae_ngb = mean_absolute_error(y, pred_ngb)
        print("R2 : %.3f" % r2_ngb)
        print("MAE : %.3f" % mae_ngb)
        print("feature_importances = ", self.model.feature_importances_)

    def get_prior_distribution(self,x):
        n_x = self.normalize(x)
        pred = self.model.pred_dist(n_x)
        pred_loc = pred.params["loc"]
        pred_scale = pred.params["scale"]

        # Create grids on the XY plane
        x = np.linspace(self.domain[0,0],self.domain[0,1],self.grid_num)
        y = np.linspace(self.domain[1,0],self.domain[1,1],self.grid_num)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))

        pdf_lst = []
        for i in tqdm(range(pred_loc.shape[0])):
            # Calculate the probability density at each point on the grids
            pdf = multivariate_normal(pred_loc[i], pred_scale[i]).pdf(pos)
            pdf_lst.append(pdf)

        pdfs = np.stack(pdf_lst,axis=0)
        prior_D = np.mean(pdfs,axis=0)
        return prior_D



class ParameterEstimator(NGB_Base):
    """
    A subclass of NGB_Base for parameter estimation tasks. This class loads an NGBoost model
    and its associated data to perform probabilistic predictions over a specified domain, incorporating prior
    distributions and returning the posterior distribution for parameter estimation.

    Attributes
    ----------
    grid_num : int
        The number of divisions in each dimension of the grid used for parameter estimation.
    """
    def __init__(self,config):
        super().__init__(config)
        ngb_data = self.load()
        self.model = ngb_data.model
        self.x_m = ngb_data.x_m
        self.x_s = ngb_data.x_s
        self.domain = ngb_data.domain
        self.prior_D = ngb_data.prior_D
        self.grid_num = config["ngboost"]["grid_num"]

    def estimate(self,x):
        pred = self.predict(x)
        pred_loc = pred.params["loc"]
        pred_scale = pred.params["scale"]

        pdf_lst = []
        for i in range(pred_loc.shape[0]):

            # Create grids on the XY plane
            x = np.linspace(self.domain[0,0],self.domain[0,1],self.grid_num)
            y = np.linspace(self.domain[1,0],self.domain[1,1],self.grid_num)
            X, Y = np.meshgrid(x, y)
            pos = np.dstack((X, Y))

            # Calculate the probability density at each point on the grids
            pdf = multivariate_normal(pred_loc[i], pred_scale[i]).pdf(pos)
            pdf_lst.append(pdf)

        p_theta_x = copy.deepcopy(self.prior_D)
        for pdf in pdf_lst:
            p_theta_x *= pdf/self.prior_D
            p_theta_x /= np.sum(p_theta_x)

        #Set the sum of the points within the range to 1 to find the normalization constant.
        p_theta_x /= np.sum(p_theta_x)
        return p_theta_x


    
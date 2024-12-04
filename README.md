# CLIP-MMA
CLIP-MMA is a Python library designed for the analysis of mathematical models, implementing two functionalities: selection of mathematical models and parameter estimation of a mathematical model. By preparing datasets related to mathematical models as described in our paper, this library enables the search for mathematical models that generate patterns similar to a target image containing patterns of interest, and approximate Bayesian estimation of parameters similar to several target images.

## Installation
```
$git clone https://github.com/Hide-Hishi/CLIP-MMA.git
```

## Initial Setup
```
$cd ./CLIP-MMA
$pip install -r requirements.txt
```

### Preparing the Dataset
1. **For Selection of Mathematical Models:**
   - Organize the pattern images by the type of mathematical model into separate folders under `data/model_image/`.
   - Place a target image of interest into any subdirectory under　`data/target_image/`.
   
2. **For Parameter Estimation:**
   - Store pattern images based on different parameters within the considered parameter space into a single folder and place it in `data/model_image/`.
   - Place folders containing the target images of interest into any subdirectory under　`data/target_image/`.
   - Our code is designed to estimate two parameters. If you wish to change the number of variables, please modify the relevant part of the script. Additionally, the range of the parameters to be predicted must be specified in matrix form in `data/clip/parameter_domain/<your-training-dataset-name>/parameter_domain_array.npy`. This is necessary for proper discretization of the parameter space using a grid. Note that the directory name must match the name of the training dataset directory.

3. **Configuration Files:**
   - Write the necessary configurations in a YAML file and store it in the `config/` directory.

## Running the Code

### Embedding and Model Training
- After preparing the dataset, you can perform image embedding with the image encoder and train the machine learning models. Execute the appropriate script depending on your task:
  - **For Selection of Mathematical Models:**
    ```
    %run script/prepare_model_selection.py config.yaml
    ```
  - **For Parameter Estimation:**
    ```
    %run script/prepare_parameter_estimation.py config.yaml
    ```

### Prediction
- To run predictions after setting up and training, use the following commands according to your specific task:
  - **For Selection of Mathematical Models:**
    ```
    %run script/run_model_selection.py config.yaml <one target image path>
    ```
  - **For Parameter Estimation:**
    ```
    %run script/run_parameter_estimation.py config.yaml <target images directory path>
    ```

## Mathematical Models Data for Training Datasets

You can generate pattern images from each mathematical model by setting the parameters within the `.ipynb` files located in the `model_notebooks` directory. To construct a dataset, please follow the instructions provided in the **Preparing the Dataset** section.
Our dataset can be obtained from the following repository.
```
https://doi.org/10.6084/m9.figshare.27263601.v1
```
Please place both the `model_image` and `target_image` directories under `CLIP-MMA/data`.

## Trained Models Data
The pre-trained models can be obtained from the following repository.
```
https://doi.org/10.6084/m9.figshare.27263634.v1
```
To use this data, please place the extracted `clip` directory directly under `CLIP-MMA/data`.



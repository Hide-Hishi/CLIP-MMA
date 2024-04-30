import os
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning import losses, samplers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

import utils
from dml_module import DenseNet
from dml_func import visualizer_hook
from dml_dataset import DMLDataset
from custom_trainer import TrainerWithCLIP
from custom_miner import SimpleCLIPMiner


def dml_train(config):
    device = torch.device("cpu")

    #gpu setting
    if config["use_gpu"]:
        device = utils.find_gpu()

    dml_cfg = config["dml"]
    # Set trunk model and replace the softmax layer with an identity function
    trunk = DenseNet()
    trunk = trunk.to(device)

    # Set embedder model. This takes in the output of the trunk and outputs 64 dimensional embeddings
    embedder = nn.Linear(16, dml_cfg["output_dim"])
    embedder = embedder.to(device)

    # Set optimizers
    trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=dml_cfg["learning_rate"], weight_decay=dml_cfg["weight_decay"])
    embedder_optimizer = torch.optim.Adam(embedder.parameters(), lr=dml_cfg["learning_rate"], weight_decay=dml_cfg["weight_decay"])

    train_dataset = DMLDataset(config)
    val_dataset = DMLDataset(config,mode="val")

    # Set the loss function
    loss = losses.TripletMarginLoss(margin=dml_cfg["loss_margin"])

    # Set the mining function
    miner = miners.MultiSimilarityMiner(epsilon=dml_cfg["miner_epsilon"])
    #miner = SimpleCLIPMiner()
    #miner = BatchEasyHardMinerWithCLIP()

    # Set the dataloader sampler
    sampler = samplers.MPerClassSampler(
        train_dataset.targets, m=dml_cfg["sampler_m"], length_before_new_iter=len(train_dataset)
    )

    # Set other training parameters
    batch_size = config["dml"]["batch_size"]
    num_epochs = config["dml"]["epoch"]

    # Package the above stuff into dictionaries.
    models = {"trunk": trunk, "embedder": embedder}
    optimizers = {
        "trunk_optimizer": trunk_optimizer,
        "embedder_optimizer": embedder_optimizer,
    }
    loss_funcs = {"metric_loss": loss}
    
    #mining_funcs = {"tuple_miner": miner}
    mining_funcs = {"rawdata_distance_miner": miner}


    record_keeper, _, _ = logging_presets.get_record_keeper(
        "example_logs", "example_tensorboard"
    )
    hooks = logging_presets.get_hook_container(record_keeper)
    dataset_dict = {"val": val_dataset}
    model_folder = os.path.join(config["data"]["data_dir"],config["use_model"],"dml_model")

    # Create the tester
    tester = testers.GlobalEmbeddingSpaceTester(
        end_of_testing_hook=hooks.end_of_testing_hook,
        visualizer=PCA(),
        visualizer_hook=visualizer_hook,
        dataloader_num_workers=dml_cfg["num_workers"],
        accuracy_calculator=AccuracyCalculator(k="max_bin_count"),
    )

    end_of_epoch_hook = hooks.end_of_epoch_hook(
        tester, dataset_dict, model_folder, test_interval=5, patience=1
    )

    trainer = TrainerWithCLIP(
        models,
        optimizers,
        batch_size,
        loss_funcs,
        train_dataset,
        mining_funcs=mining_funcs,
        sampler=sampler,
        dataloader_num_workers=2,
        end_of_iteration_hook=hooks.end_of_iteration_hook,
        end_of_epoch_hook=end_of_epoch_hook,
    )

    trainer.train(num_epochs=num_epochs)

    return {"trunk":trunk,"embedder":embedder}
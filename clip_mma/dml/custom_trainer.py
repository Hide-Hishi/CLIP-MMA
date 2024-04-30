import torch
import tqdm

from pytorch_metric_learning import trainers
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_tracker as l_t
from pytorch_metric_learning.utils.key_checker import KeyChecker, KeyCheckerDict

class TrainerWithCLIP(trainers.MetricLossOnly):

    def maybe_mine_embeddings_with_external_distance(self, data, embeddings, labels):
        #If you implement the customized miner "rawdata_distance_miner","tuple_miner" is ignored.
        if "rawdata_distance_miner" in self.mining_funcs:
            return self.mining_funcs["rawdata_distance_miner"](data, embeddings, labels)
        elif "tuple_miner" in self.mining_funcs:
            return self.mining_funcs["tuple_miner"](embeddings, labels)
        return None

    def calculate_loss(self, curr_batch):
        data, labels = curr_batch
        embeddings = self.compute_embeddings(data)
        indices_tuple = self.maybe_mine_embeddings_with_external_distance(data,embeddings, labels)
        self.losses["metric_loss"] = self.maybe_get_metric_loss(
            embeddings, labels, indices_tuple
        )
    
    #List of candidates for mining_funcs keys was changed.
    def set_schema(self):
        self.schema = KeyCheckerDict(
            {
                "models": KeyChecker(["trunk", "embedder"], essential=["trunk"]),
                "loss_funcs": KeyChecker(["metric_loss"]),
                "mining_funcs": KeyChecker(
                    ["subset_batch_miner", "tuple_miner","rawdata_distance_miner"],
                    warn_empty=False,
                    important=[],
                ),
                "loss_weights": KeyChecker(
                    self.loss_names, warn_empty=False, essential=self.loss_names
                ),
                "optimizers": KeyChecker(
                    lambda s, d: c_f.append_map(
                        d["models"].keys + d["loss_funcs"].keys, "_optimizer"
                    ),
                    important=c_f.append_map(self.models.keys(), "_optimizer"),
                ),
                "lr_schedulers": KeyChecker(
                    lambda s, d: [
                        x + y
                        for y in self.allowed_lr_scheduler_key_suffixes.values()
                        for x in d["models"].keys + d["loss_funcs"].keys
                    ],
                    warn_empty=False,
                    important=[],
                ),
                "gradient_clippers": KeyChecker(
                    lambda s, d: c_f.append_map(
                        d["models"].keys + d["loss_funcs"].keys, "_grad_clipper"
                    ),
                    warn_empty=False,
                    important=[],
                ),
            }
        )
        self.modify_schema()
import torch

from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.miners import BaseMiner
from pytorch_metric_learning.miners.batch_easy_hard_miner import BatchEasyHardMiner
from pytorch_metric_learning.miners.multi_similarity_miner import MultiSimilarityMiner
from pytorch_metric_learning.distances import CosineSimilarity

class SimpleCLIPMiner(BaseMiner):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def forward(self, data, embeddings, labels, ref_emb=None, ref_labels=None):
        self.reset_stats()
        with torch.no_grad():
            mining_output = self.conditional_mine(data,labels)
        self.output_assertion(mining_output)
        return mining_output

    def conditional_mine(self,data,labels):
        ref_labels = labels
        a_idx, p_idx, n_idx = lmu.get_all_triplets_indices(labels, ref_labels)
        p_similarity = self.cos_sim(data,a_idx,p_idx)
        n_similarity = self.cos_sim(data,a_idx,n_idx)
        not_overlapped = p_similarity > n_similarity
        return a_idx[not_overlapped], p_idx[not_overlapped],n_idx[not_overlapped]

    def cos_sim(self,data, a_idx, other_idx):
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        return cos(data[a_idx],data[other_idx])







class BatchEasyHardMinerWithCLIP(BatchEasyHardMiner):
    HARD = "hard"
    SEMIHARD = "semihard"
    EASY = "easy"
    ALL = "all"
    all_batch_mining_strategies = [HARD, SEMIHARD, EASY, ALL]

    def __init__(self,pos_strategy=EASY,neg_strategy=SEMIHARD,allowed_pos_range=None,allowed_neg_range=None,**kwargs):
        super().__init__(
            pos_strategy=pos_strategy,
            neg_strategy=neg_strategy,
            allowed_pos_range=allowed_pos_range,
            allowed_neg_range=allowed_neg_range,
            **kwargs
            )

    def forward(self, data, embeddings, labels, ref_emb=None, ref_labels=None):
        self.reset_stats()
        with torch.no_grad():
            c_f.check_shapes(embeddings, labels)
            labels = c_f.to_device(labels, embeddings)
            data = c_f.to_device(data,embeddings)
            ref_emb, ref_labels = c_f.set_ref_emb(
                embeddings, labels, ref_emb, ref_labels
            )
            mining_output = self.mine(embeddings, labels, ref_emb, ref_labels)
            mining_output = self.conditional_mine(data,mining_output)
        self.output_assertion(mining_output)
        return mining_output

    def conditional_mine(self,data,mining_output):
        a1_idx, p_idx, a2_idx, n_idx = mining_output
        p_similarity = self.cos_sim(data,a1_idx,p_idx)
        n_similarity = self.cos_sim(data,a2_idx,n_idx)
        not_overlapped = p_similarity > n_similarity
        return a1_idx[not_overlapped], p_idx[not_overlapped], a2_idx[not_overlapped], n_idx[not_overlapped]

    def cos_sim(self,data, a_idx, other_idx):
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        return cos(data[a_idx],data[other_idx])


class MultiSimilarityMinerWithCLIP(MultiSimilarityMiner):
    def __init__(self, epsilon=0.1, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.add_to_recordable_attributes(name="epsilon", is_stat=False)

    def forward(self, data, embeddings, labels, ref_emb=None, ref_labels=None):
        self.reset_stats()
        with torch.no_grad():
            c_f.check_shapes(embeddings, labels)
            labels = c_f.to_device(labels, embeddings)
            data = c_f.to_device(data,embeddings)
            ref_emb, ref_labels = c_f.set_ref_emb(
                embeddings, labels, ref_emb, ref_labels
            )
            mining_output = self.mine(embeddings, labels, ref_emb, ref_labels)
            #The output tuple of MultiSimilarityMiner is (a1,p,a2,n).
            #Get a1==a2 tuples
            mining_output = lmu.convert_to_triplets(mining_output, labels)
            mining_output = self.conditional_mine(data,mining_output)
        self.output_assertion(mining_output)
        return mining_output

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        mat = self.distance(embeddings, ref_emb)
        a1, p, a2, n = lmu.get_all_pairs_indices(labels, ref_labels)

        if len(a1) == 0 or len(a2) == 0:
            empty = torch.tensor([], device=labels.device, dtype=torch.long)
            return empty.clone(), empty.clone(), empty.clone(), empty.clone()

        mat_neg_sorting = mat
        mat_pos_sorting = mat.clone()

        dtype = mat.dtype
        pos_ignore = (
            c_f.pos_inf(dtype) if self.distance.is_inverted else c_f.neg_inf(dtype)
        )
        neg_ignore = (
            c_f.neg_inf(dtype) if self.distance.is_inverted else c_f.pos_inf(dtype)
        )

        mat_pos_sorting[a2, n] = pos_ignore
        mat_neg_sorting[a1, p] = neg_ignore
        if embeddings is ref_emb:
            mat_pos_sorting.fill_diagonal_(pos_ignore)
            mat_neg_sorting.fill_diagonal_(neg_ignore)

        pos_sorted, pos_sorted_idx = torch.sort(mat_pos_sorting, dim=1)
        neg_sorted, neg_sorted_idx = torch.sort(mat_neg_sorting, dim=1)

        if self.distance.is_inverted:
            hard_pos_idx = torch.where(
                pos_sorted - self.epsilon < neg_sorted[:, -1].unsqueeze(1)
            )
            hard_neg_idx = torch.where(
                neg_sorted + self.epsilon > pos_sorted[:, 0].unsqueeze(1)
            )
        else:
            hard_pos_idx = torch.where(
                pos_sorted + self.epsilon > neg_sorted[:, 0].unsqueeze(1)
            )
            hard_neg_idx = torch.where(
                neg_sorted - self.epsilon < pos_sorted[:, -1].unsqueeze(1)
            )

        a1 = hard_pos_idx[0]
        p = pos_sorted_idx[a1, hard_pos_idx[1]]
        a2 = hard_neg_idx[0]
        n = neg_sorted_idx[a2, hard_neg_idx[1]]

        return a1, p, a2, n

    def get_default_distance(self):
        return CosineSimilarity()

    def conditional_mine(self,data,mining_output):
        a_idx, p_idx, n_idx = mining_output
        p_similarity = self.cos_sim(data,a_idx,p_idx)
        n_similarity = self.cos_sim(data,a_idx,n_idx)
        not_overlapped = p_similarity > n_similarity
        return a_idx[not_overlapped], p_idx[not_overlapped], n_idx[not_overlapped]

    def cos_sim(self,data, a_idx, other_idx):
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        return cos(data[a_idx],data[other_idx])

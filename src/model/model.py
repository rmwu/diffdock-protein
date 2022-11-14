import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_cluster import knn_graph

from .diffusion import TensorProductScoreModel
from .losses import DiffusionLoss


class BaseModel(nn.Module):
    """
        enc(receptor) -> R^(dxL)
        enc(ligand)  -> R^(dxL)
    """
    def __init__(self, args, params):
        super(BaseModel, self).__init__()

        ######## unpack model parameters
        self.model_type = args.model_type
        self.knn_size = args.knn_size
        self.args = args

        ######## initialize (shared) modules
        # raw encoders
        self.encoder = TensorProductScoreModel(args, params)

        self._init()

    def _init(self):
        for name, param in self.named_parameters():
            # NOTE must name parameter "bert"
            if "bert" in name:
                continue
            # bias terms
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            # weight terms
            else:
                nn.init.xavier_normal_(param)

    def forward(self, batch):
        raise Exception("Extend me")

    def prepare_batch(self, batch):
        """
            Move everything to CUDA
        """
        for key in ["receptor", "ligand"]:
            batch[key] = batch_graph(batch[key])
            batch[key] = batch[key].cuda()
            batch[key].label = batch[key].x  # save copy for labels
            batch[key].x = self.res_ebd(batch[key])

        for key in ["pose_new"]:
            if key in batch:
                pos_new = [item for item in batch[key]]
                batch[key] = torch.cat(pos_new).cuda()
        return batch

    def dist(self, x, y):
        if len(x.size()) > 1:
            return ((x-y)**2).sum(-1)
        return ((x-y)**2)


class ScoreModel(BaseModel):
    def __init__(self, args, params):
        super(ScoreModel, self).__init__(args, params)
        # loss function
        self.loss = DiffusionLoss(args)

        self._init()

    def forward(self, batch):
        # move graphs to cuda
        tr_pred, rot_pred, tor_pred = self.encoder(batch)

        outputs = {}
        outputs["tr_pred"] = tr_pred
        outputs["rot_pred"] = rot_pred
        outputs["tor_pred"] = tor_pred

        return outputs

    def compute_loss(self, batch, outputs):
        losses = self.loss(batch, outputs)
        return losses


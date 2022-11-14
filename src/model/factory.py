import os
import sys
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.data_parallel import DataParallel

from utils import printt, get_model_path
from .model import ScoreModel


def load_model(args, model_params, fold):
    """
        Model factory
    """
    # load model with specified arguments
    kwargs = {}
    if args.model_type == "e3nn":
        model = ScoreModel(args, model_params, **kwargs)
    else:
        raise Exception(f"invalid model type {args.model_type}")
    printt("loaded model with kwargs:", " ".join(kwargs.keys()))

    # (optional) load checkpoint if provided
    if args.checkpoint_path is not None:
        fold_dir = os.path.join(args.checkpoint_path, f"fold_{fold}")
        checkpoint = get_model_path(fold_dir)
        if checkpoint is not None:
            # extract current model
            state_dict = model.state_dict()
            # load onto CPU, transfer to proper GPU
            pretrain_dict = torch.load(checkpoint, map_location="cpu")["model"]
            pretrain_dict = {k:v for k,v in pretrain_dict.items() if k in state_dict}
            # update current model
            state_dict.update(pretrain_dict)
            # >>>
            for k,v in state_dict.items():
                if k not in pretrain_dict:
                    print(k, "not saved")
            model.load_state_dict(state_dict)
            printt("loaded checkpoint from", checkpoint)
        else:
            printt("no checkpoint found")
    return model


def to_cuda(model, args):
    """
        move model to cuda
    """
    # specify number in case test =/= train GPU
    if args.gpu >= 0:
        model = model.cuda(args.gpu)
        if args.num_gpu > 1:
            device_ids = [args.gpu + i for i in range(args.num_gpu)]
            model = DataParallel(model, device_ids=device_ids)
    return model



import os
import sys
import copy
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader, DataListLoader

from . import utils as data_utils
from utils import compute_rmsd
from geom_utils import NoiseTransform


def load_data(args):
    """
        Load and minimally process data
    """
    # add others like db5 or sabdab if we run on those
    if args.dataset == "dips":
        data = data_utils.DIPSLoader(args)
    elif args.dataset == "db5":
        data = data_utils.DB5Loader(args)
    elif args.dataset == "toy":
        data = data_utils.ToyLoader(args)
    else:
        raise Exception("invalid --dataset", args.dataset)
    return data


def get_data(dataset, fold_num, args):
    """
        Convert raw data into DataLoaders for training.
    """
    use_pose = type(dataset) is tuple
    if use_pose:
        dataset, poses = dataset
    # smush folds and convert to Dataset object
    # or extract val and rest are train
    splits = dataset.crossval_split(fold_num)
    for split, pdb_ids in splits.items():
        # debug mode: only load small dataset
        if args.debug:
            pdb_ids = pdb_ids[:10]
        splits[split] = BindingDataset(args, dataset.data, pdb_ids)
    # convert to DataLoader
    data_loaders = _get_loader(splits, args)
    return data_loaders


def _get_loader(splits, args):
    """
        Convert lists into DataLoader
    """
    # current reverse diffusion does NOT use DataLoader
    if args.mode == "test":
        return splits
    # convert to DataLoader
    loaders = {}
    for split, data in splits.items():
        # account for test-only datasets
        if len(data) == 0:
            loaders[split] = []
            continue
        # do not shuffle val/test
        shuffle = (split == "train")
        # set proper DataLoader object (PyG)
        if torch.cuda.is_available() and args.num_gpu > 1:
            loader = DataListLoader
        else:
            loader = DataLoader
        loaders[split] = loader(data,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=False,
                                pin_memory=True,
                                shuffle=shuffle)
    return loaders

# ------ DATASET -------


class BindingDataset(Dataset):
    """
        Protein-protein binding dataset
    """
    def __init__(self, args, data, pdb_ids=None):
        super(BindingDataset, self).__init__(
            transform = NoiseTransform(args)
        )
        self.args = args
        # select subset for given split
        if pdb_ids is not None:
            data = {k:v for k,v in data.items() if k in pdb_ids}
            self.pdb_ids = [k for k in data if k in pdb_ids]
        else:
            self.pdb_ids = list(data)
        # convert to PyTorch geometric objects upon GET not INIT
        self.data = list(data.values())
        self.length = len(self.data)

    def len(self):
        return self.length

    def __delitem__(self, idx):
        """
            Easier deletion interface. MUST update length.
        """
        del self.data[idx]
        self.len = len(self.data)

    def get(self, idx):
        """
            Create graph object to keep original object intact,
            so we can modify pos, etc.
        """
        item = self.data[idx]["graph"]
        # >>> fix this later no need to copy tensors only references
        data = copy.deepcopy(item)
        return data


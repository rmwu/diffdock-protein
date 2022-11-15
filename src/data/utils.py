import os
import sys
import dill
import pickle
import random
import warnings
import itertools
from collections import defaultdict

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import knn_graph
from torch_geometric.data import HeteroData

from scipy.spatial.transform import Rotation

import Bio
# These are just annoying :')
import Bio.PDB
warnings.filterwarnings("ignore",
    category=Bio.PDB.PDBExceptions.PDBConstructionWarning)

from utils import load_csv, printt
from utils import compute_rmsd


# -------- DATA LOADING -------


class Loader:
    """
        Raw data loader with shared utilities
        @param (bool) data  default=True to load complexes, not poses
    """
    def __init__(self, args, data=True):
        self.args = args
        if data:
            self.root = args.data_path
            self.data_file = args.data_file
        else:
            self.root = args.pose_path
            self.data_file = args.pose_file
        # read in file paths
        self.data_to_load = load_csv(self.data_file)
        self.data_cache = self.data_file.replace(".csv", "_cache.pkl")
        # cache for post-processed data, optional
        self.graph_cache = self.data_file.replace(".csv", "_graph.pkl")
        if args.debug:
            self.data_cache = self.data_cache.replace(".pkl", "_debug.pkl")
            self.graph_cache = self.data_cache.replace(".pkl", "_debug.pkl")
        # biopython PDB parser
        self.parser = Bio.PDB.PDBParser()

    def read_files(self):
        raise Exception("implement me")

    def parse_pdb(self, fp, models=None, chains=None):
        """
            Parse PDB file via Biopython
            (instead of my hacky implementation based on the spec)
            @return  (residue sequence, coordinates, atom sequence)
        """
        all_res, all_atom, all_pos = [], [], []
        structure = self.parser.get_structure(fp, fp)  # name, path
        # (optional) select subset of models
        if models is not None:
            models = [structure[m] for m in models]
        else:
            models = structure
        for model in models:
            # (optional) select subset of chains
            if chains is not None:
                chains = [model[c] for c in chains if c in model]
            else:
                chains = model
            # retrieve all atoms in desired chains
            for chain in chains:
                res = chain.get_residues()
                for res in chain:
                    # ignore HETATM records
                    if res.id[0] != " ":
                        continue
                    for atom in res:
                        all_res.append(res.get_resname())
                        all_pos.append(torch.from_numpy(atom.get_coord()))
                        all_atom.append((atom.get_name(), atom.element))
        all_pos = torch.stack(all_pos, dim=0)
        return all_res, all_atom, all_pos

    def _to_dict(self, pdb, p1, p2):
        item = {
            "pdb_id": pdb,
            "receptor": p1,
            "ligand": p2
        }
        return item

    def convert_pdb(self, all_res, all_atom, all_pos):
        """
            Unify PDB representation across different dataset formats.
            Given all residues, coordinates, and atoms, select subset
            of all atoms to keep.

            @return tuple(list, torch.Tensor) seq, pos
        """
        if self.args.resolution == "atom":
            atoms_to_keep = None
        elif self.args.resolution == "backbone":
            atoms_to_keep = ["N", "CA", "O", "CB"]
        elif self.args.resolution == "residue":
            atoms_to_keep = ["CA"]
        else:
            raise Exception(f"invalid resolution {args.resolution}")

        if atoms_to_keep is not None:
            to_keep, atoms = [], []
            for i,a in enumerate(all_atom):
                if a[0] in atoms_to_keep:  # atom name
                    to_keep.append(i)
                    atoms.append(a[1])  # element symbol
            seq = [all_res[i] for i in to_keep]
            pos = all_pos[torch.tensor(to_keep)]
        else:
            atoms = all_atom
            seq = all_res
            pos = all_pos
        assert pos.shape[0] == len(seq) == len(atoms)
        return atoms, seq, pos

    def process_data(self, data, args):
        """
            Tokenize, etc.
        """
        # check if cache exists
        if not args.no_cache and os.path.exists(self.graph_cache):
            with open(self.graph_cache, "rb") as f:
                data, data_params = pickle.load(f)
                return data, data_params

        # non-data parameters
        data_params = {}
        #### select subset of residues that match desired
        # resolution (e.g. residue-level vs. atom-level)
        for item in data.values():
            for k in ["receptor", "ligand"]:
                # >>> in what case would this be false?
                if k in item:
                    subset = self.convert_pdb(*item[k])
                    item[f"{k}_atom"] = subset[0]
                    item[f"{k}_seq"] = subset[1]
                    item[f"{k}_xyz"] = subset[2]

        #### protein sequence tokenization
        # tokenize residues
        tokenizer = tokenize(data.values(), "receptor_seq")
        tokenize(data.values(), "ligand_seq", tokenizer)
        data_params["tokenizer"] = tokenizer
        # tokenize atoms
        atom_tokenizer = tokenize(data.values(), "receptor_atom")
        tokenize(data.values(), "ligand_atom", atom_tokenizer)
        data_params["atom_tokenizer"] = atom_tokenizer
        printt("finished tokenizing all inputs")

        #### convert to HeteroData graph objects
        for item in data.values():
            item["graph"] = self.to_graph(item)

        if not args.no_cache:
            with open(self.graph_cache, "wb+") as f:
                pickle.dump([data, data_params], f)

        return data, data_params

    def to_graph(self, item):
        """
            Convert raw dictionary to PyTorch geometric object
        """
        data = HeteroData()
        data["name"] = item["path"]
        # retrieve position and compute kNN
        for key in ["receptor", "ligand"]:
            data[key].pos = item[f"{key}_xyz"].float()
            data[key].x = item[f"{key}_atom"]
            # kNN graph
            edge_index = knn_graph(data[key].pos, self.args.knn_size)
            data[key, "contact", key].edge_index = edge_index
        # center receptor at origin
        center = data["receptor"].pos.mean(dim=0, keepdim=True)
        for key in ["receptor", "ligand"]:
            data[key].pos = data[key].pos - center
        data.center = center  # save old center
        return data

    def split_data(self, raw_data, args):
        # separate out train/test
        data = {
            "train": {},
            "val":   {},
            "test":  {}
        }
        ## if test split is pre-specified, split into train/test
        # otherwise, allocate all data to train for cross-validation
        for k, item in raw_data.items():
            if "split" in item:
                data[item["split"]][k] = item
            else:
                data["train"] = raw_data
                break
        # split train into separate folds
        data["train"] = split_into_folds(data["train"], args)
        return data

    def crossval_split(self, fold_num):
        """
            number of folds-way cross validation
            @return  dict: split -> [pdb_ids]
        """
        splits = { "train": [] }
        # split into train/val/test
        folds = [list(sorted(f)) for f in self.splits["train"]]
        val_data = list(sorted(self.splits["val"]))
        test_data = list(sorted(self.splits["test"]))
        # if val split is pre-specified, do not take fold
        if len(val_data) == 0:
            val_fold = fold_num
            splits["val"] = folds[val_fold]
        else:
            val_fold = -1  # all remaining folds go to train
            splits["val"] = val_data
        # if test split is pre-specified, do not take fold
        if len(test_data) > 0:
            test_fold = val_fold  # must specify to allocate train folds
            splits["test"] = test_data
        # otherwise both val/test labels depend on fold_num
        else:
            test_fold = (fold_num+1) % self.args.num_folds
            splits["test"] = folds[test_fold]
        # add remaining to train
        for idx in range(self.args.num_folds):
            if idx in [val_fold, test_fold]:
                continue
            splits["train"].extend(folds[idx])
        return splits


class DIPSLoader(Loader):
    def __init__(self, args):
        super(DIPSLoader, self).__init__(args)
        # load and process files
        data = self.read_files()
        data = self.assign_receptor(data)
        data, data_params = self.process_data(data, args)
        self.data = data
        self.data_params = data_params
        printt(len(self.data), "entries loaded")
        # split into folds
        self.splits = self.split_data(data, args)

    def read_files(self):
        data = {}
        # check if loaded previously
        if os.path.exists(self.data_cache):
            with open(self.data_cache, "rb") as f:
                path_to_data = pickle.load(f)
        else:
            path_to_data = {}
        for line in tqdm(self.data_to_load, desc="data loading", ncols=50):
            path = line["path"]
            item = path_to_data.get(path)
            if item is None:
                item = self.parse_dill(os.path.join(self.root, path), path)
                path_to_data[path] = item
            item.update(line)  # add meta-data
            data[path] = item
            if self.args.debug:
                if len(data) >= 1000:
                    break
        # write to cache
        if not os.path.exists(self.data_cache):
            with open(self.data_cache, "wb+") as f:
                pickle.dump(path_to_data, f)
        return data

    def assign_receptor(self, data):
        """
            For docking, we assigned smaller protein as ligand
            for this dataset (since no canonical receptor/ligand
            assignments)
        """
        for item in data.values():
            rec = item["receptor"]
            lig = item["ligand"]
            if len(rec[0]) < len(lig[0]):
                item["receptor"] = lig
                item["ligand"] = rec
        return data

    def parse_dill(self, fp, pdb_id):
        with open(fp, "rb") as f:
            data = dill.load(f)
        p1, p2 = data[1], data[2]
        p1, p2 = self.parse_df(p1), self.parse_df(p2)
        return self._to_dict(pdb_id, p1, p2)

    def parse_df(self, df):
        """
            Parse PDB DataFrame
        """
        # extract dataframe values
        all_res = df["resname"]
        all_pos = torch.tensor([df["x"], df["y"], df["z"]]).t()
        all_atom = list(zip(df["atom_name"], df["element"]))
        # convert to seq, pos
        return all_res, all_atom, all_pos


class DB5Loader(Loader):
    """
        Docking benchmark 5.5
    """
    def __init__(self, args):
        super(DB5Loader, self).__init__(args)
        # load and process files
        self.root = os.path.join(self.root, "structures")
        # cache file dependent on use_unbound
        if args.use_unbound:
            self.data_cache = self.data_cache.replace(".pkl", "_u.pkl")
        else:
            self.data_cache = self.data_cache.replace(".pkl", "_b.pkl")
        data = self.read_files()
        data, data_params = self.process_data(data, args)
        self.data = data
        self.data_params = data_params
        printt(len(self.data), "entries loaded")
        # split into folds
        self.splits = self.split_data(data, args)

    def read_files(self):
        data = {}
        # check if loaded previously
        if os.path.exists(self.data_cache):
            with open(self.data_cache, "rb") as f:
                path_to_data = pickle.load(f)
        else:
            path_to_data = {}
        for line in tqdm(self.data_to_load, desc="data loading", ncols=50):
            pdb = line["path"]
            item = path_to_data.get(pdb)
            if item is None:
                item = self.parse_path(pdb, os.path.join(self.root, pdb))
                path_to_data[pdb] = item
            item.update(line)  # add meta-data
            data[pdb] = item
        # write to cache
        if not os.path.exists(self.data_cache):
            with open(self.data_cache, "wb+") as f:
                pickle.dump(path_to_data, f)
        return data

    def parse_path(self, pdb, path):
        if self.args.use_unbound:
            fp_rec = f"{path}_r_u.pdb"
            fp_lig = f"{path}_l_u.pdb"
        else:
            fp_rec = f"{path}_r_b.pdb"
            fp_lig = f"{path}_l_b.pdb"
        p1, p2 = self.parse_pdb(fp_rec), self.parse_pdb(fp_lig)
        return self._to_dict(pdb, p1, p2)


class SabDabLoader(Loader):
    """
        Structure Antibody Database
        Downloaded May 2, 2022.
    """
    def __init__(self, args):
        super(SabDabLoader, self).__init__(args)
        # standard workflow
        data = self.read_files()
        data, data_params = self.process_data(data, args)
        self.data = data
        self.data_params = data_params
        printt(len(self.data), "entries loaded")
        # split into folds
        self.splits = self.split_data(data, args)

    def read_files(self):
        data = {}
        # check if loaded previously
        if os.path.exists(self.data_cache):
            with open(self.data_cache, "rb") as f:
                path_to_data = pickle.load(f)
        else:
            path_to_data = {}

        for line in tqdm(self.data_to_load, desc="data loading", ncols=50):
            pdb = line["pdb"]
            item = path_to_data.get(pdb)
            if item is None:
                # parse sabdab entry
                pdb = line["pdb"]
                rec_chains = [
                    c.strip() for c in line["antigen_chain"].split("|")]
                lig_chains = set([c.upper() for c in [
                    line["Hchain"], line["Lchain"]] if c is not None])
                # parse pdb object
                path = os.path.join(self.root, f"{pdb}.pdb")
                rec = self.parse_pdb(path, line["model"], rec_chains)
                lig = self.parse_pdb(path, line["model"], lig_chains)
                # TODO SabDab pdb_id needs to be de-duped
                item = self._to_dict(pdb, rec, lig)
                path_to_data[pdb] = item
            item.update(line)  # add meta-data
            data[pdb] = item
        # write to cache
        if not os.path.exists(self.data_cache):
            with open(self.data_cache, "wb+") as f:
                pickle.dump(path_to_data, f)
        return data


class ToyLoader(Loader):
    def __init__(self, args):
        super(ToyLoader, self).__init__(args)
        self.size = 10  # number of residues
        self.args = args
        data = self.read_files()
        self.data, self.data_params = self.process_data(data, args)
        self.splits = self.split_data(self.data, args)

        # fake pose data for testing
        self.poses = self.read_poses()

    def read_files(self):
        # separated by 1A
        receptor_xyz = torch.zeros(self.size, 3)
        ligand_xyz = torch.zeros(self.size, 3)

        # be careful to span all of R3 for FA
        # zig zag shape noise. separated by approx 1A.
        receptor_xyz[:,0] = torch.arange(self.size)
        receptor_xyz[:,1] = (torch.arange(self.size) % 2 - 1) * 0.1
        ligand_xyz[:,0] = torch.arange(self.size)
        ligand_xyz[:,2] = 1

        met_seq = ["MET"] * self.size
        gly_seq = ["GLY"] * self.size
        item = {
            "receptor_xyz": receptor_xyz,
            "ligand_xyz": ligand_xyz,
        }

        data = []
        sizes = {
            "train": 2,
            "val": 2,
            "test": 2
        }
        for split, max_size in sizes.items():
            for i in range(max_size):
                new_item = item.copy()
                new_item["split"] = split
                new_item["pdb_id"] = len(data)
                new_item["path"] = len(data)
                xyz, xyz2, rot = self.rotate(item["ligand_xyz"],
                                             item["receptor_xyz"],
                                             seed=len(data))
                # sequence classification sub-task
                if i % 2 == 0:
                    new_item["receptor_seq"] = met_seq
                    new_item["ligand_seq"] = met_seq
                    new_item["label"] = 0
                else:
                    new_item["receptor_seq"] = gly_seq
                    new_item["ligand_seq"] = gly_seq
                    new_item["label"] = 1
                # assign coordinates
                new_item["ligand_xyz"] = xyz
                new_item["receptor_xyz"] = xyz2
                new_item["rot"] = rot
                new_item["old_ligand_xyz"] = item["ligand_xyz"]
                data.append(new_item)
        data = {i:x for i,x in enumerate(data)}
        return data

    def rotate(self, xyz, xyz2, seed):
        # get random rotation
        rot = Rotation.random(random_state=seed)
        rot = torch.tensor(rot.as_matrix().squeeze()).to(xyz)
        # apply rotation
        c1, c2 = xyz.mean(1, keepdim=True), xyz2.mean(1, keepdim=True)
        xyz = (xyz - c1) @ rot + c1
        xyz2 = (xyz2 - c2) @ rot + c2
        return xyz, xyz2, rot

    def read_poses(self):
        data = defaultdict(dict)
        # technically original sigmas are standard deviations
        # but we can reuse them here. convert to A from nm
        noise = torch.linspace(self.args.sigma_begin * 10, 0,
                               self.args.num_noise_level)
        for k,v in self.data.items():
            lig_xyz = v["old_ligand_xyz"]
            poses = []
            for delta in noise:
                xyz = lig_xyz.clone()
                xyz[:,2] = xyz[:,2] + delta
                c = xyz.mean(1, keepdim=True)
                xyz = (xyz - c) @ v["rot"] + c
                poses.append(xyz)
            data[k]["poses"] = poses
            data[k]["rmsds"] = [compute_rmsd(lig_xyz, xyz) for xyz in poses]
        return data


def apply_transforms(xyz, rot, trans, com):
    """
        Apply rotation and translations
        @param xyz  (N, L, 3) coordinates of interest
        @param rot  (N, 3) Euler angles for rotation
        @param trans  (N, 3) translation
        @param com  (N, 3) center of mass
    """
    Q = get_matrix(rot)
    Q = Q.expand(len(xyz), -1, -1)
    x = torch.bmm(Q, (xyz - com)[:,:,None]).squeeze()
    x = x + com + trans
    return x


def get_matrix(euler):
    """
        Thank you Prof. Huang ah ha ha ha
        @param (torch.Tensor) euler  (N, 3)
    """
    cos = torch.cos(euler).t()  # (3, N)
    sin = torch.sin(euler).t()

    rot = torch.zeros(1, 3, 3)

    rot[:,0,0] = cos[2]*cos[0]-cos[1]*sin[0]*sin[2]
    rot[:,0,1] = cos[2]*sin[0]+cos[1]*cos[0]*sin[2]
    rot[:,0,2] = sin[2]*sin[1]

    rot[:,1,0] = -sin[2]*cos[0]-cos[1]*sin[0]*cos[2]
    rot[:,1,1] = -sin[2]*sin[0]+cos[1]*cos[0]*cos[2]
    rot[:,1,2] =  cos[2]*sin[1]

    rot[:,2,0] =  sin[1]*sin[0]
    rot[:,2,1] = -sin[1]*cos[0]
    rot[:,2,2] =  cos[1]

    return rot


# ------ DATA PROCESSING ------

def tokenize(data, key, tokenizer=None):
    """
        Tokenize every item[key] in data.
        Modifies item[key] and copies original value to item[key_raw].
        @param (list) data
        @param (str) key  item[key] is iterable
    """
    if len(data) == 0:  # sometimes no val split, etc.
        return
    # if tokenizer is not provided, create index
    all_values = [item[key] for item in data]
    all_values = set(itertools.chain(*all_values))
    if tokenizer is None:
        tokenizer = {}  # never used
    for item in sorted(all_values):
        if item not in tokenizer:
            tokenizer[item] = len(tokenizer) + 1  # 1-index
    f_token = lambda seq: [tokenizer[x] for x in seq]
    # tokenize items and modify data in place
    raw_key = f"{key}_raw"
    for item in data:
        raw_item = item[key]
        item[raw_key] = raw_item  # save old
        item[key] = torch.tensor(f_token(raw_item))  # tokenize
    return tokenizer


def split_into_folds(data, args):
    """
        Split into train/val/test folds
        @param (list) data
    """
    # split by cluster
    for item in data.values():
        by_cluster = ("cluster" in item)
        break
    if by_cluster:
        clst_to_pdb = defaultdict(list)
        for k in data:
            clst_to_pdb[data[k]["cluster"]].append(k)
    else:
        clst_to_pdb = {k:[k] for k in data}
    keys = list(sorted(clst_to_pdb))
    random.shuffle(keys)
    # split into folds
    cur_clst = 0
    folds = [{} for _ in range(args.num_folds)]
    fold_ratio = 1. / args.num_folds
    max_fold_size = int(len(data) * fold_ratio)
    for fold_num in range(args.num_folds):
        fold = []
        while (len(fold) < max_fold_size) and (cur_clst < len(clst_to_pdb)):
            fold.extend(clst_to_pdb[keys[cur_clst]])
            cur_clst += 1
        for k in fold:
            folds[fold_num][k] = data[k]
    return folds

# ------ DATA COLLATION -------

def get_mask(lens):
    """ torch.MHA style mask (False = attend, True = mask) """
    mask = torch.arange(max(lens))[None, :] >= lens[:, None]
    return mask


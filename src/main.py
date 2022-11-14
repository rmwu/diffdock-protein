import os
import sys
import yaml
import random
import resource
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from args import parse_args
from data import load_data, get_data
from train import train, evaluate
from model import load_model, to_cuda
from utils import printt, print_res, log


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    # he he he
    with open("data/goodluck.txt") as f:
        for line in f:
            print(line, end="")

    args = parse_args()
    torch.cuda.set_device(args.gpu)

    if args.mode != "test":
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        # save args
        with open(args.args_file, "w+") as f:
            yaml.dump(args.__dict__, f)

    # load raw data
    data = load_data(args)
    data_params = data.data_params
    printt("finished loading raw data")

    # needs to be set if DataLoader does heavy lifting
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    # needs to be set if sharing resources
    if args.num_workers >= 1:
        torch.multiprocessing.set_sharing_strategy("file_system")

    if args.mode == "train":
        # save scores
        test_scores = defaultdict(list)
        for fold in range(args.num_folds):
            log_dir = os.path.join(args.tensorboard_path,
                                   args.run_name, str(fold))
            writer = SummaryWriter(log_dir=log_dir)
            ## set up fold
            set_seed(args.seed)
            # make save folder
            fold_dir = os.path.join(args.save_path, f"fold_{fold}")
            args.fold_dir = fold_dir
            if not os.path.exists(fold_dir):
                os.makedirs(fold_dir)
            printt("fold {} seed {}\nsaved to {}".format(fold, args.seed, fold_dir))
            # load and convert data to DataLoaders
            loaders = get_data(data, fold, args)
            printt("finished creating data splits")
            # get model and load checkpoint, if relevant
            model = load_model(args, data_params, fold)
            model = to_cuda(model, args)
            printt("finished loading model")

            ## run training loop
            train_loader = loaders["train"]
            val_loader = loaders["val"]
            best_score, best_epoch, best_path = train(
                    train_loader, val_loader,
                    model, writer, fold_dir, args)
            printt("finished training best epoch {} loss {:.3f}".format(
                    best_epoch, best_score))

            ## run eval loop
            if best_path is not None:
                model = load_model(args, data_params, fold)
                model.load_state_dict(torch.load(best_path,
                    map_location="cpu")["model"])
                model = to_cuda(model, args)
                printt(f"loaded model from {best_path}")
            # test is actual score
            test_score = evaluate(loaders["test"], model, writer, args)
            test_score["fold"] = fold
            # add val for hyperparameter search
            val_score = evaluate(loaders["val"], model, writer, args)
            for key, val in val_score.items():
                test_score[f"val_{key}"] = val
            # print and save
            for key, val in test_score.items():
                test_scores[key].append(val)
            printt("fold {}".format(fold))
            print_res(test_score)
            # set next seed
            args.seed += 1
            # end of fold ========

        printt(f"{args.num_folds} folds average")
        print_res(test_scores)
        log(test_scores, args.log_file)
        # end of all folds ========

    elif args.mode == "test":
        set_seed(args.seed)
        printt("running inference")
        writer = None  # no need to tensorboard
        ## set up
        # load and convert data to DataLoaders
        loaders = get_data(data, args.test_fold, args)
        printt("finished creating data splits")
        # get model and load checkpoint, if relevant
        model = load_model(args, data_params, args.test_fold)
        model = to_cuda(model, args)
        printt("finished loading model")

        test_score = evaluate(loaders["test"], model, writer, args)
        # add val for hyperparameter search
        val_score = evaluate(loaders["val"], model, writer, args)

        for key, val in val_score.items():
            test_score[f"val_{key}"] = val
        print_res(test_score)
        log(test_score, args.log_file, reduction=False)


if __name__ == "__main__":
    main()


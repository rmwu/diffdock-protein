import os
import sys
import yaml
import random
import resource
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn


from args import parse_args
from data import load_data, get_data
from model import load_model, to_cuda
from utils import printt, print_res, log, get_unixtime
from train import train, evaluate, evaluate_pose
from helpers import WandbLogger, TensorboardLogger
from sample import sample


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
    torch.hub.set_dir(args.torchhub_path)

    # training mode, dump args for reproducibility
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

    # train mode: train model for args.fold different seeds
    # and evaluate at the end
    if args.mode == "train":
        # save scores
        test_scores = defaultdict(list)
        # try different seeds
        for fold in range(args.num_folds):
            log_dir = os.path.join(args.tensorboard_path,
                                   args.run_name, str(fold),
                                   get_unixtime())
            if args.logger == "tensorboard":
                writer = TensorboardLogger(log_dir=log_dir)
            elif args.logger == "wandb":
                writer = WandbLogger(project=f"{args.project}", entity=args.entity, name=f"{args.run_name}-fold-{fold}",
                                     group=f"{args.group}")
            else:
                raise Exception("Improper logger.")
            #### set up fold experiment
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

            #### run training loop
            train_loader = loaders["train"]
            val_loader = loaders["val"]
            best_score, best_epoch, best_path = train(
                    train_loader, val_loader,
                    model, writer, fold_dir, args)
            printt("finished training best epoch {} loss {:.3f}".format(
                    best_epoch, best_score))

            #### run eval loop
            if best_path is not None:
                model = load_model(args, data_params, fold)
                model.load_state_dict(torch.load(best_path,
                    map_location="cpu")["model"])
                model = to_cuda(model, args)
                printt(f"loaded model from {best_path}")
            # eval on test set
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

            # Finish writing for this fold
            writer.finish()
            # end of fold ========

        printt(f"{args.num_folds} folds average")
        print_res(test_scores)
        log(test_scores, args.log_file)
        # end of all folds ========

    # test mode: load up all replicates from checkpoint directory
    # and evaluate by sampling from reverse diffusion process
    elif args.mode == "test":
        set_seed(args.seed)
        printt("running inference")
        test_scores = defaultdict(list)
        for fold_dir in os.listdir(args.save_path):
            if "fold_" not in fold_dir:
                continue
            fold = int(fold_dir[5:])
            # load and convert data to DataLoaders
            loaders = get_data(data, fold, args)
            printt("finished creating data splits")
            # get model and load checkpoint, if relevant
            model = load_model(args, data_params, fold)
            model = to_cuda(model, args)
            printt("finished loading model")

            # run reverse diffusion process
            samples_test = sample(loaders["test"], model, args)
            samples_val = sample(loaders["val"], model, args)

            # test fold
            test_score = evaluate_pose(loaders["test"], samples_test)

            # add val for hyperparameter search
            val_score = evaluate_pose(loaders["val"], samples_val)
            for key, val in val_score.items():
                test_score[f"val_{key}"] = val

            # print and save
            for key, val in test_score.items():
                test_scores[key].append(val)
            # end of fold ========

        printt(f"Average test/val performance")
        print_res(test_scores)
        log(test_scores, args.log_file, reduction=False)
        # end of all folds ========


if __name__ == "__main__":
    main()


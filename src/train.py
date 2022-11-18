"""
    Training script

    TODO: add in sampling to evaluation every k epochs
    This can be accomplished by importing sample from sample
"""

import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import printt, print_res, get_optimizer
from utils import log, compute_rmsd


def train(train_loader, val_loader, model,
          writer, fold_dir, args):

    # optimizer
    start_epoch, optimizer = get_optimizer(model, args)

    # validation
    best_loss = float("inf")
    best_path = None
    best_epoch = start_epoch
    # logging
    log_path = os.path.join(fold_dir, "log.yaml")

    num_batches = start_epoch * (len(train_loader) // args.batch_size)
    ep_iterator = range(start_epoch, start_epoch+args.epochs)
    if not args.no_tqdm:
        ep_iterator = tqdm(ep_iterator,
                           initial=start_epoch,
                           desc="train epoch", ncols=50)
    for epoch in ep_iterator:
        # start epoch!
        iterator = enumerate(train_loader)
        if not args.no_tqdm:
            iterator = tqdm(iterator,
                            total=len(train_loader),
                            desc="train batch",
                            leave=False, ncols=50)

        for batch_num, batch in iterator:
            # eval before training and start numbering epochs at 1
            if epoch == 0:
                break

            # reset optimizer gradients
            model.train()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            # forward pass
            try:
                output = model(batch)
                # compute loss (modifies output in place)
                if args.num_gpu > 1:
                    losses = model.module.compute_loss(batch, output)
                else:
                    losses = model.compute_loss(batch, output)

                # backpropagate
                loss = losses["loss"]
                for name, param in model.named_parameters():
                    if torch.any(torch.isnan(param)):
                        print(name, "nan")
                loss.backward()

            # catch OOM this is amazing !!!
            except RuntimeError as e:
                printt("RuntimeError", e)
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue

            # log gradients
            grads = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    #print(name, param.grad.norm())
                    grads.append(param.grad.norm().item())
            grad_mean = torch.tensor(grads).mean()
            if writer is not None:
                writer.add_scalar("gradient", grad_mean, num_batches)
            #nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            # write to tensorboard
            num_batches += 1
            loss = loss.cpu().item()  # CPU for logging
            if num_batches % args.log_frequency == 0:
                for loss_type, loss in losses.items():
                    if loss is None:
                        continue
                    log_key = f"train_{loss_type}"
                    if writer is not None:
                        writer.add_scalar(log_key, loss.mean(), num_batches)

            # end of batch ========
            # force checkpoint saving + evaluation
            if num_batches > 5000:
                break

        # evaluate (end of epoch)
        avg_train_loss, avg_train_score = train_epoch_end(num_batches,
                train_loader, model,
                log_path, fold_dir, args,
                max_batch=20, split="train")  # adjust based on batch size
        val_loss, avg_val_score = train_epoch_end(num_batches,
                val_loader, model, log_path, fold_dir, args, writer)
        printt('\nepoch', epoch)
        printt('train loss', avg_train_loss, args.metric, avg_train_score)
        printt('val loss', val_loss, args.metric, avg_val_score)

        # save model
        path_suffix = f"{num_batches}_{epoch}_{avg_val_score:.3f}_{val_loss:.3f}.pth"
        # >>>
        #if avg_val_score < best_loss:
        #    best_loss = avg_val_score
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            # save model ONLY IF best
            best_path = os.path.join(fold_dir, f"model_best_{path_suffix}")
            if args.num_gpu > 1:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save({"model": state_dict,
                        "optimizer": optimizer.state_dict()}, best_path)
            printt(f"\nsaved model to {best_path}")

        # check if out of patience
        if epoch - best_epoch >= args.patience:
            break

        # end of epoch ========
    # end of all epochs ========

    return best_loss, best_epoch, best_path
    #return best_score, best_epoch, best_path


def train_epoch_end(num_batches, val_loader,
                    model, log_path, fold_dir, args,
                    writer=None,
                    max_batch=None, split="val"):
    """
        Evaluate at end of training epoch and write to log file
    """
    log_item = { "batch": num_batches }
    val_loss, avg_val_score = float("inf"), 0
    if len(val_loader) == 0:
        return val_loss, avg_val_score

    # run inference
    val_scores = evaluate(val_loader, model, writer, args,
                          max_batch=max_batch)
    metric = args.metric
    if metric not in val_scores:
        metric = list(val_scores)[0]
    avg_val_score = np.nanmean(val_scores[metric])
    for key, value in val_scores.items():
        try:
            log_item[f"{split}_{key}"] = value.mean()
        except:
            log_item[f"{split}_{key}"] = value
    print_res(log_item)

    # tensorboard
    if writer is not None:
        for key, value in log_item.items():
            if key not in ["batch"]:
                writer.add_scalar(key, value, num_batches)
    # append to YAML log
    log(log_item, log_path)
    # early stopping
    val_loss = val_scores["loss"]
    return val_loss, avg_val_score


def evaluate(val_loader, model, writer, args,
             max_batch=None):
    """
        @param (int) max_batch       number of batches to sample
    """
    all_output = defaultdict(list)
    all_losses = defaultdict(list)
    with torch.no_grad():
        model.eval()
        torch.cuda.empty_cache()
        # loop through all batches
        iterator = enumerate(val_loader)
        if not args.no_tqdm:
            iterator = tqdm(iterator,
                            total=len(val_loader),
                            desc="evaluation",
                            leave=False, ncols=50)
        for batch_num, batch in iterator:
            # if we evaluate via poses, no need to compute fake loss
            if args.mode == "test" and not args.debug:
                printt("Skipping to pose")
                break
            # model predictions
            output = model(batch)
            if args.num_gpu > 1:
                losses = model.module.compute_loss(batch, output)
            else:
                losses = model.compute_loss(batch, output)

            for loss_type, loss in losses.items():
                all_losses[loss_type].append(loss.mean().item())
            for key in []:
                if key in output:
                    all_output[key].append(output[key].detach().cpu())

            if max_batch is not None and batch_num >= max_batch:
                break

    ######## evaluate metrics
    scores = {}
    # average loss over batches
    for loss_type, losses in all_losses.items():
        scores[loss_type] = torch.mean(torch.tensor(losses)).item()
    print_res(all_losses)

    # save outputs
    for key, value in all_output.items():
        try:
            value = torch.tensor(value).float()
        except ValueError:
            value = torch.cat(value)
        scores[key] = torch.mean(value).item()

    return scores


def evaluate_pose(data_list, samples_list):
    """
        Evaluate sampled pose vs. ground truth
    """
    all_rmsds = []
    assert len(data_list) == len(samples_list)
    for true_graph, pred_graph in zip(data_list, samples_list):
        true_xyz = true_graph["ligand"].pos
        pred_xyz = pred_graph["ligand"].pos
        assert true_xyz.shape == pred_xyz.shape
        rmsd = compute_rmsd(true_xyz, pred_xyz)
        all_rmsds.append(rmsd)

    scores = {
        "rmsd": all_rmsds,
    }

    return scores


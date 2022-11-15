"""
    Inference script
"""

import os
import sys
import copy

import numpy as np
import torch
from torch_geometric.loader import DataLoader, DataListLoader
from scipy.spatial.transform import Rotation as R

from utils import printt
from geom_utils import set_time, NoiseTransform


def sample(data_list, model, args):
    """
        Run reverse process
    """

    # stores various noise-related utils
    transform = NoiseTransform(args)

    # diffusion timesteps
    timesteps = get_timesteps(args.num_steps)

    # randomize original position and COPY data_list
    data_list = randomize_position(data_list, args)

    # sample
    for t_idx in range(args.num_steps):
        # create new loader with current step graphs
        if torch.cuda.is_available() and args.num_gpu > 1:
            loader = DataListLoader
        else:
            loader = DataLoader
        test_loader = loader(data_list, batch_size=args.batch_size)
        new_data_list = []  # updated every step
        # DiffDock uses same schedule for all noise
        cur_t = timesteps[t_idx]
        if t_idx == args.num_steps - 1:
            dt = cur_t
        else:
            dt = cur_t - timesteps[t_idx+1]

        for complex_graphs in test_loader:
            # move to CUDA
            #complex_graphs = complex_graphs.cuda()

            # this MAY differ from args.batch_size
            # based on # GPUs and last batch
            if type(complex_graphs) is list:
                batch_size = len(complex_graphs)
            else:
                batch_size = complex_graphs.num_graphs

            # convert to sigma space and save time
            tr_s, rot_s, tor_s = transform.noise_schedule(
                cur_t, cur_t, cur_t)
            if type(complex_graphs) is list:
                for g in complex_graphs:
                    set_time(g, cur_t, cur_t, cur_t, 1)
            else:
                set_time(complex_graphs, cur_t, cur_t, cur_t, batch_size)

            with torch.no_grad():
                outputs = model(complex_graphs)
            tr_score = outputs["tr_pred"].cpu()
            rot_score = outputs["rot_pred"].cpu()
            tor_score = outputs["tor_pred"].cpu()

            # translation gradient (?)
            tr_scale = torch.sqrt(
                2 * torch.log(torch.tensor(args.tr_s_max /
                                           args.tr_s_min)))
            tr_g = tr_s * tr_scale

            # rotation gradient (?)
            rot_scale = torch.sqrt(
                    torch.log(torch.tensor(args.rot_s_max /
                                           args.rot_s_min)))
            rot_g = 2 * rot_s * rot_scale

            # actual update
            if args.ode:
                tr_update = (0.5 * tr_g**2 * dt * tr_score)
                rot_update = (0.5 * rot_score * dt * rot_g**2)
            else:
                if args.no_final_noise and t_idx == args.num_steps-1:
                    tr_z = torch.zeros((batch_size, 3))
                    rot_z = torch.zeros((batch_size, 3))
                elif args.no_random:
                    tr_z = torch.zeros((batch_size, 3))
                    rot_z = torch.zeros((batch_size, 3))
                else:
                    tr_z = torch.normal(0, 1, size=(batch_size, 3))
                    rot_z = torch.normal(0, 1, size=(batch_size, 3))

                tr_update = (tr_g**2 * dt * tr_score)
                tr_update = tr_update + (tr_g * np.sqrt(dt) * tr_z)

                rot_update = (rot_score * dt * rot_g**2)
                rot_update = rot_update + (rot_g * np.sqrt(dt) * rot_z)

            # >>> no torsion for now
            if not args.no_torsion:
                tor_g = tor_sigma * torch.sqrt(
                    torch.tensor(
                        2 * np.log(model_args.tor_sigma_max / model_args.tor_sigma_min)
                    )
                )
                if ode:
                    tor_update = (0.5 * tor_g**2 * dt_tor * tor_score.cpu()).numpy()
                else:
                    tor_z = (
                        torch.zeros(tor_score.shape)
                        if args.no_random
                        or (no_final_step_noise and t_idx == num_steps - 1)
                        else torch.normal(mean=0, std=1, size=tor_score.shape)
                    )
                    tor_update = (
                        tor_g**2 * dt_tor * tor_score.cpu()
                        + tor_g * np.sqrt(dt_tor) * tor_z
                    ).numpy()
                torsions_per_mol = tor_update.shape[0] // b
            else:
                tor_update = None

            # apply transformations
            if type(complex_graphs) is not list:
                complex_graphs = complex_graphs.to("cpu").to_data_list()
            for i, data in enumerate(complex_graphs):
                if not args.no_torsion:
                    tor_update_i = tor_update[i*torsions_per_mol:
                                              (i+1)*torsions_per_mol]
                else:
                    tor_update_i = None
                new_graph = transform.apply_updates(data,
                        tr_update[i:i+1],
                        rot_update[i:i+1].squeeze(0),
                        tor_update_i)

                new_data_list.append(new_graph)
            # === end of batch ===

        # update starting point for next step
        assert len(new_data_list) == len(data_list)
        data_list = new_data_list
        printt(f"Completed {t_idx} out of {args.num_steps} steps")
        # === end of timestep ===

    return data_list


def get_timesteps(inference_steps):
    return np.linspace(1, 0, inference_steps + 1)[:-1]


def randomize_position(data_list, args):
    """
        Modify COPY of data_list objects
    """
    data_list = copy.deepcopy(data_list)

    if not args.no_torsion:
        raise Exception("not yet implemented")
        # randomize torsion angles
        for complex_graph in data_list:
            torsion_updates = np.random.uniform(
                low=-np.pi, high=np.pi,
                size=complex_graph["ligand"].edge_mask.sum()
            )
            complex_graph["ligand"].pos = modify_conformer_torsion_angles(
                complex_graph["ligand"].pos,
                complex_graph["ligand", "ligand"].edge_index.T[
                    complex_graph["ligand"].edge_mask
                ],
                complex_graph["ligand"].mask_rotate[0],
                torsion_updates,
            )

    for complex_graph in data_list:
        # randomize rotation
        pos = complex_graph["ligand"].pos
        center = torch.mean(pos, dim=0, keepdim=True)
        random_rotation = torch.from_numpy(R.random().as_matrix())
        pos = (pos - center) @ random_rotation.T.float()

        # random translation
        tr_update = torch.normal(0, args.tr_s_max, size=(1, 3))
        pos = pos + tr_update
        complex_graph["ligand"].pos = pos

    return data_list


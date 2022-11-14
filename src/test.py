import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import printt, print_res, get_model_path
from utils import compute_metrics, log


def generate(loader, model, writer, args):
    all_outputs = defaultdict(list)
    with torch.no_grad():
        model.eval()
        # loop through all batches
        iterator = enumerate(loader)
        if not args.no_tqdm:
            iterator = tqdm(iterator,
                            total=len(loader),
                            desc="generation",
                            leave=False, ncols=50)
        for batch_num, batch in iterator:
            output = model(batch)
            for k, v in output.items():
                if "raw" in k:
                    continue
                if type(v) is torch.Tensor:
                    all_outputs[k].extend(v.detach().cpu().tolist())
                else:
                    all_outputs[k].extend(v)
            if len(all_outputs[k]) > 5000:
                #break
                pass
        # end of all batches ========
    return all_outputs


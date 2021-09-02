import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List
from pmsmodel.laplacian import *
from .module import image_differentiable_warping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WarpLoss(nn.Module):
    def __init__(self):
        super(WarpLoss, self).__init__()
        self.stage = 4
        self.lap = LapLoss(0)

    def forward(self, depth_patchmatch,
                refined_depth,
                imgs,
                proj_matrices
    ):
        img_ref = torch.unbind(imgs[f"stage_0"], 1)[0]
        loss = torch.mean(self.lap(img_ref, img_ref))
        print('=====loss in loss=======',loss.grad)
        # loss.requires_grad_(True)
        refined_warpedsrc = []
        for l in reversed(range(0, self.stage)):
            self.lap = LapLoss(3-l)
            img_ref = torch.unbind(imgs[f"stage_{l}"], 1)[0]
            imgs_src = torch.unbind(imgs[f"stage_{l}"], 1)[1:]
            proj_matrice_ref =  torch.unbind(proj_matrices[f"stage_{l}"].float(), 1)[0]
            proj_matrice_srcs =  torch.unbind(proj_matrices[f"stage_{l}"].float(), 1)[1:]
            if (l == 0):
                depth = refined_depth[f"stage_{l}"]
            else:
                depth = depth_patchmatch[f"stage_{l}"][-1]
            for j in range(len(imgs_src)):        #should be two
                img_src = imgs_src[j]
                proj_matrice_src = proj_matrice_srcs[j]
                warped_src = image_differentiable_warping(img_src, proj_matrice_src, proj_matrice_ref, depth)
                if (l == 0):
                    refined_warpedsrc.append(warped_src)
                loss = loss + torch.mean(self.lap(warped_src, img_ref))
                # loss.requires_grad_(True)
        del imgs
        del proj_matrices
        return loss




import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from model.warplayer import inverse_warp
from torchstat import stat
from torch.nn.parallel import DistributedDataParallel as DDP
from model.IDNet import *
import torch.nn.functional as F
from model.loss import *
from model.laplacian import *
from model.refine import *

device = torch.device("cuda")

class Model:
    def __init__(self, local_rank=-1):
        self.depthnet = IDNet()
        self.device()
        self.optimG = AdamW(self.depthnet.parameters(), lr=1e-5, weight_decay=1e-4)
        self.epe = EPE()
        self.lap = LapLoss()
        self.sobel = SOBEL()
        if local_rank != -1:
            self.depthnet = DDP(self.depthnet, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.depthnet.train()

    def eval(self):
        self.depthnet.eval()

    def device(self):
        self.depthnet.to(device)

    def load_model(self, path, rank=0):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }

        if rank <= 0:
            self.depthnet.load_state_dict(convert(torch.load('{}/depthnet_999.pkl'.format(path))['model']))

    def save_model(self, path, rank=0, epoch=0):
        if rank == 0:
            torch.save({'epoch': epoch,
                        'model': self.depthnet.state_dict(),
                        'optimizer': self.optimG.state_dict()},
                        '{}/depthnet_{}.pkl'.format(path, epoch))

    '''
    def predict(self, imgs, depth, merged, training=True, depth_gt=None):
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        c0 = self.contextnet(img0, depth[:, :2])
        c1 = self.contextnet(img1, depth[:, 2:4])
        refine_output = self.unet(img0, img1, depth, merged, c0, c1, depth_gt)
        res = torch.sigmoid(refine_output[:, :3]) * 2 - 1
        pred = merged + res
        pred = torch.clamp(pred, 0, 1)
        if training:
            return pred, merged
        else:
            return pred
    '''

    def inference(self, img0, img1, RT1, RT2, scale_list=[4, 2, 1], TTA=False):
        imgs = torch.cat((img0, img1), 1)
        RT = torch.cat((RT1, RT2), 0)
        depth, mask, merged, depth_teacher, merged_teacher, loss_distill = self.depthnet(imgs, RT, scale_list)
        if TTA == False:
            return merged[2]
        else:
            depth2, mask2, merged2, depth_teacher2, merged_teacher2, loss_distill2 = self.depthnet(imgs.flip(2), RT, scale_list)
            return (merged[2] + merged2[2].flip(2)) / 2

    def update(self, imgs, gt, RT, learning_rate=0, mul=1, training=True, depth_gt=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        RT0 = RT[0]
        RT1 = RT[1]
        if training:
            self.train()
        else:
            self.eval()
        depth, mask, merged, depth_teacher, merged_teacher, loss_distill = self.depthnet(torch.cat((imgs, gt), 1), RT)
        loss_l1 = (self.lap(merged[2], gt)).mean()
        loss_tea = (self.lap(merged_teacher, gt)).mean()
        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1 + loss_tea + loss_distill * 0.01
            loss_G.backward()
            self.optimG.step()
        else:
            depth_teacher = depth[2]
            merged_teacher = merged[2]
        return merged[2], {
            'merged_tea': merged_teacher,
            'mask': mask,
            'mask_tea': mask,
            'depth': depth[2],
            'depth_tea': depth_teacher,
            'loss_l1': loss_l1,
            'loss_tea': loss_tea,
            'loss_distill': loss_distill,
            }

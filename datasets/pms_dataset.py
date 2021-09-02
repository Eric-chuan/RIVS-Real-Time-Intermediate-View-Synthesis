from sys import meta_path
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
from collections import defaultdict
from PIL import Image
import torch
from torchvision import transforms as T
import math
from torch.nn import functional as F

class PMSDataset(Dataset):
    def __init__(self, datapath, split='test', img_wh=(1920,1080)):

        self.stages = 4
        self.dataroot = datapath
        self.img_wh = img_wh
        self.split = split

        self.build_metas()


    def build_metas(self):
        with open(f"{self.dataroot}/trainlist.txt", "r") as f1:
            self.train_path = f1.readlines()
        with open(f"{self.dataroot}/testlist.txt", "r") as f2:
            self.val_path = f2.readlines()

        if self.split == 'train':
            self.meta_data = self.train_path
        else:
            self.meta_data = self.val_path
        self.nr_sample = len(self.meta_data)


        # for scan in self.scans:
        #     with open(os.path.join(self.datapath, scan, 'pair.txt')) as f:
        #         num_viewpoint = int(f.readline())
        #         for view_idx in range(num_viewpoint):
        #             ref_view = int(f.readline().rstrip())
        #             src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
        #             if len(src_views) != 0:
        #                 self.metas += [(scan, -1, ref_view, src_views)]


    def read_cam_file(self, np_file):
        extrinsics = np_file['extrinsic']
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np_file['intrinsic']

        depth_min = np_file['depth_min']
        depth_min = float(depth_min)
        if depth_min < 0: # for botanical garden, exist incorrect value
            depth_min = 1
        depth_max = np_file['depth_max']
        depth_max = float(depth_max)

        return intrinsics, extrinsics, depth_min, depth_max

    def read_img(self, np_img):
        # scale 0~255 to 0~1
        np_img = np_img.astype(np.float32) / 255.
        np_img = np_img[:,:,::-1]
        original_h, original_w, _ = np_img.shape

        ph = ((original_h - 1) // 64 + 1) * 64
        pw = ((original_w - 1) // 64 + 1) * 64
        # padding = ((0, ph - original_h), (0, pw - original_w),(0,0))
        # np_img = np.pad(np_img, padding, 'constant', constant_values=0)
        # np_img = cv2.resize(np_img, (pw, ph), interpolation=cv2.INTER_LINEAR),

        np_img_ms = {
            "stage_3": cv2.resize(np_img, (pw//16, ph//16), interpolation=cv2.INTER_LINEAR),
            "stage_2": cv2.resize(np_img, (pw//8, ph//8), interpolation=cv2.INTER_LINEAR),
            "stage_1": cv2.resize(np_img, (pw//4, ph//4), interpolation=cv2.INTER_LINEAR),
            "stage_0": cv2.resize(np_img, (pw//2, ph//2), interpolation=cv2.INTER_LINEAR),
        }
        return np_img_ms, ph//2, pw//2



    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, index):
        data_path = f'{self.dataroot}/{self.meta_data[index].strip()}/data.npz'
        f = np.load(data_path)
        data = f['i0i1gt']
        channel_index = [2, 0, 1]

        imgs_0 = []
        imgs_1 = []
        imgs_2 = []
        imgs_3 = []

        # depth = None
        depth_min = None
        depth_max = None

        proj_matrices_0 = []
        proj_matrices_1 = []
        proj_matrices_2 = []
        proj_matrices_3 = []

        for i, cdx in enumerate(channel_index):
            imgs, original_h, original_w = self.read_img(data[3*cdx:3*cdx+3].transpose(1, 2, 0))
            # imgs, original_h, original_w = self.read_img(data[3*cdx:3*cdx+3])
            imgs_0.append(imgs['stage_0'])
            imgs_1.append(imgs['stage_1'])
            imgs_2.append(imgs['stage_2'])
            imgs_3.append(imgs['stage_3'])

            intrinsics, extrinsics, depth_min_, depth_max_ = self.read_cam_file(f)
            intrinsics[0] *= original_w / self.img_wh[0]
            intrinsics[1] *= original_h / self.img_wh[1]
            proj_mat = extrinsics[cdx].copy()
            intrinsics[:2,:] *= 0.125
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_3.append(proj_mat)

            proj_mat = extrinsics[cdx].copy()
            intrinsics[:2,:] *= 2
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_2.append(proj_mat)

            proj_mat = extrinsics[cdx].copy()
            intrinsics[:2,:] *= 2
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_1.append(proj_mat)

            proj_mat = extrinsics[cdx].copy()
            intrinsics[:2,:] *= 2
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_0.append(proj_mat)


            if i == 0:  # reference view

                depth_min = depth_min_
                depth_max = depth_max_


        # imgs: N*3*H0*W0, N is number of images
        imgs_0 = np.stack(imgs_0).transpose([0, 3, 1, 2])
        imgs_1 = np.stack(imgs_1).transpose([0, 3, 1, 2])
        imgs_2 = np.stack(imgs_2).transpose([0, 3, 1, 2])
        imgs_3 = np.stack(imgs_3).transpose([0, 3, 1, 2])
        imgs = {}
        imgs['stage_0'] = imgs_0
        imgs['stage_1'] = imgs_1
        imgs['stage_2'] = imgs_2
        imgs['stage_3'] = imgs_3
        # proj_matrices: N*4*4
        proj_matrices_0 = np.stack(proj_matrices_0)
        proj_matrices_1 = np.stack(proj_matrices_1)
        proj_matrices_2 = np.stack(proj_matrices_2)
        proj_matrices_3 = np.stack(proj_matrices_3)
        proj={}
        proj['stage_3']=proj_matrices_3
        proj['stage_2']=proj_matrices_2
        proj['stage_1']=proj_matrices_1
        proj['stage_0']=proj_matrices_0

        return {"imgs": imgs,                   # N*3*H0*W0
                "proj_matrices": proj, # N*4*4
                "depth_min": depth_min,         # scalar
                "depth_max": depth_max,
                "filename": data_path
                }

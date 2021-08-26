import cv2
import ast
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
import csv
from scipy.spatial.transform import Rotation as ROT

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class VimeoDataset(Dataset):
    def __init__(self, dataset_name, batch_size=32):
        self.batch_size = batch_size
        self.data_root = '/home/Eric-chuan/workspace/dataset/dataset_kitti'
        self.dataset_name = dataset_name
        self.load_data()
        self.h = 256
        self.w = 448
        xx = np.arange(0, self.w).reshape(1,-1).repeat(self.h,0)
        yy = np.arange(0, self.h).reshape(-1,1).repeat(self.w,1)
        self.grid = np.stack((xx,yy),2).copy()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        with open(f"{self.data_root}/trainlist.txt", "r") as f1:
            self.train_path = f1.readlines()
        with open(f"{self.data_root}/testlist.txt", "r") as f2:
            self.val_path = f2.readlines()
        if self.dataset_name == 'train':
            self.meta_data = self.train_path
        else:
            self.meta_data = self.val_path
        self.nr_sample = len(self.meta_data)

    def aug(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1

    def getdata(self, index):
        data = self.meta_data[index]
        f = np.load(f'{self.data_root}/sequences/{self.meta_data[index].strip()}/data.npz')
        data = f['i0i1gt']
        img0 = data[0:3].transpose(1, 2, 0)
        img1 = data[3:6].transpose(1, 2, 0)
        gt = data[6:9].transpose(1, 2, 0)
        pose = f['pose']  #
        poseLeft, poseRight, poseTarget = pose[0:6], pose[6:12], pose[12:18]
        RLeft,   TLeft    = ROT.from_euler('xyz',poseLeft[0:3]).as_matrix(), poseLeft[3:].reshape(3, 1)
        RRight,  TRight   = ROT.from_euler('xyz',poseRight[0:3]).as_matrix(), poseRight[3:].reshape(3, 1)
        RTarget, TTarget  = ROT.from_euler('xyz',poseTarget[0:3]).as_matrix(), poseTarget[3:].reshape(3, 1)
        T1 = RLeft.T.dot(TTarget - TLeft)/50.     #Target to Left
        T2 = RRight.T.dot(TTarget - TRight)/50.   #Target to Right

        mat1 = np.block(
            [ [RLeft.T@RTarget, T1],
                [np.zeros((1,3)), 1] ] )
        mat2 = np.block(
            [ [RRight.T@RTarget, T2],
              [np.zeros((1,3)), 1] ] )

        return {'Left': img0, 'Right': img1, 'Target': gt, 'RT1': mat1.astype(np.float32), 'RT2': mat2.astype(np.float32)}

    def __getitem__(self, index):
        item = self.getdata(index)
        img0, gt, img1 = item['Left'], item['Target'], item['Right']
        RT1, RT2 = item['RT1'], item['RT2']
        if self.dataset_name == 'train':
            img0, gt, img1 = self.aug(img0, gt, img1, 224, 224)
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            # if random.uniform(0, 1) < 0.5:
            #     img0 = img0[::-1]
            #     img1 = img1[::-1]
            #     gt = gt[::-1]
            # if random.uniform(0, 1) < 0.5:
            #     img0 = img0[:, ::-1]
            #     img1 = img1[:, ::-1]
            #     gt = gt[:, ::-1]
            if random.uniform(0, 1) < 0.5:
                tmp = img1
                img1 = img0
                img0 = tmp
                tmp_RT = RT2
                RT2 = RT1
                RT1 = tmp_RT
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        img = torch.cat((img0, img1, gt), 0)
        RT1 = torch.from_numpy(RT1).unsqueeze(0)
        RT2 = torch.from_numpy(RT2).unsqueeze(0)
        RT = torch.cat((RT1, RT2), 0)
        return {'img': img, 'RT': RT}
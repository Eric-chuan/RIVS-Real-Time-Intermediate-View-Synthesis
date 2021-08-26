import torch
import torch.nn as nn
import torch.nn.functional as F
from model.warplayer import inverse_warp
from model.refine import *

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        nn.PReLU(out_planes)
    )

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )

class IDBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IDBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 3, 4, 2, 1)

    def forward(self, x, depth, RT, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor = 1. / scale, mode="bilinear", align_corners=False)
        if depth != None:
            # depth[:, :1] = depth[:, :1].bmm(RT[0][:,:3,:3])
            # depth[:, 1:2] = depth[:, 1:2].bmm(RT[1][:,:3,:3])
            depth = F.interpolate(depth, scale_factor = 1. / scale, mode="bilinear", align_corners=False)
            x = torch.cat((x, depth), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        # x = self.transform(x, c, RT)
        tmp = self.lastconv(x)
        tmp = F.interpolate(tmp, scale_factor = scale * 2, mode="bilinear", align_corners=False)
        depth = tmp[:, :2]
        mask = tmp[:, 2:3]
        return depth, mask

    def transform(self, z, nz, RT):  #just a frame
        z_tf1 = z[0].view(-1,nz,3).bmm(RT[0].inverse()[:,:3,:3])
        z_tf2 = z[1].view(-1,nz,3).bmm(RT[1].inverse()[:,:3,:3])
        z_tf1 = z_tf1 + RT[0].inverse()[:,:3,3].unsqueeze(1).expand((-1,nz,3))
        z_tf2 = z_tf2 + RT[1].inverse()[:,:3,3].unsqueeze(1).expand((-1,nz,3))
        z_tf = torch.cat((z_tf1, z_tf2), 1)
        return z_tf.view(-1, nz * 3)


class IDNet(nn.Module):
    def __init__(self):
        super(IDNet, self).__init__()
        self.block0 = IDBlock(6, c=240)
        self.block1 = IDBlock(13+2, c=150)
        self.block2 = IDBlock(13+2, c=90)
        self.block_tea = IDBlock(16+2, c=90)
        self.contextnet = Contextnet()
        self.unet = Unet()

    def forward(self, x, RT, scale=[4,2,1]):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:] # In inference time, gt is None
        RT1, RT2 = RT[:,:1], RT[:,1:2]
        depth_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        depth = None
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if depth != None:
                depth_d, mask_d = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), depth, RT, scale=scale[i])
                depth = depth + depth_d
                mask = mask + mask_d
            else:
                depth, mask = stu[i](torch.cat((img0, img1), 1), None, RT, scale=scale[i])
            mask_list.append(torch.sigmoid(mask))
            depth_list.append(depth)
            warped_img0 = self.warp(img0, depth[:, :1], RT1)
            warped_img1 = self.warp(img1, depth[:, 1:2], RT2)
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        if gt.shape[1] == 3:
            depth_d, mask_d = self.block_tea(torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), depth, RT, scale=1)
            depth_teacher = depth + depth_d
            warped_img0_teacher = self.warp(img0, depth_teacher[:, :1], RT1)
            warped_img1_teacher = self.warp(img1, depth_teacher[:, 1:2], RT2)
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            depth_teacher = None
            merged_teacher = None
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            if gt.shape[1] == 3:
                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1, True) + 0.01).float().detach()
                loss_distill += ((depth_teacher.detach() - depth_list[i]).abs() * loss_mask).mean()
        c0 = self.contextnet(img0, depth[:, :1], RT1)
        c1 = self.contextnet(img1, depth[:, 1:2], RT2)
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, depth, c0, c1)
        res = tmp[:, :3] * 2 - 1
        merged[2] = torch.clamp(merged[2] + res, 0, 1)
        return depth_list, mask_list[2], merged, depth_teacher, merged_teacher, loss_distill

    def warp(self, image, depth, RT):
        depth = 1 / (10 * torch.sigmoid(depth) + 0.01)
        warped_img, _, _ =  inverse_warp(image, depth, RT)
        return warped_img

if __name__ == '__main__':
    i = 0
import os
import cv2
import math
import time
import torch
import numpy as np
import random
import argparse
import torch.distributed as dist

from model.RIDE import Model
from datasets.my_dataset import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
import warnings
warnings.filterwarnings("ignore")

# def get_learning_rate(step):
#     if step < 2000:
#         mul = step / 2000.
#     else:
#         mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
#     return 1e-4 * mul

def get_learning_rate(step):
    max_lr, min_lr = 1e-4, 6e-5
    if step < 6000:
        mul = step / 6000.
    else:
        mul = np.cos((step - 6000) / (args.epoch * args.step_per_epoch - 6000.) * math.pi) * 0.5 + 0.5
    return (max_lr - min_lr) * mul + min_lr

def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())

    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

def depth2rgb(depth_map_np):
    h, w, _ = depth_map_np.shape
    depth_map_np = depth_map_np.squeeze()
    # normalized_depth_map = depth_map_np / (np.abs(depth_map_np).max())
    depth_map_np = (depth_map_np + 1) / 2
    depth_map_np = 1 / depth_map_np * 255
    # 1 / (depth_map_np + 0.01)
    rgb_map = cv2.applyColorMap(cv2.convertScaleAbs(depth_map_np,alpha=10), cv2.COLORMAP_JET)
    # rgb_map = rgb_map / (rgb_map).max()
    return rgb_map.clip(0, 255)

def train(model, local_rank, start_epoch=0):
    print('==========global start=============\n')
    log_path = 'train_log'
    if local_rank == 0:
        writer = SummaryWriter(log_path + '/train')
        writer_val = SummaryWriter(log_path + '/validate')
    else:
        writer, writer_val = None, None
    step = 0
    nr_eval = 0
    dataset = VimeoDataset('train')
    sampler = DistributedSampler(dataset)
    train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True, sampler=sampler)
    args.step_per_epoch = train_data.__len__()
    dataset_val = VimeoDataset('validation')
    val_data = DataLoader(dataset_val, batch_size=16, pin_memory=True, num_workers=8)
    evaluate(model, val_data, nr_eval, local_rank, writer_val)
    # model.save_model(log_path, local_rank, -1)
    print('training...')
    time_stamp = time.time()
    for epoch in range(start_epoch, args.epoch):
        sampler.set_epoch(epoch)
        for i, item in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            # data_gpu, flow_gt = data
            data_gpu = item['img'].to(device, non_blocking=True) / 255.
            RT = item['RT'].to(device, non_blocking=True)
            # flow_gt = flow_gt.to(device, non_blocking=True)
            imgs = data_gpu[:, :6]
            gt = data_gpu[:, 6:9]
            # mul = np.cos(step / (args.epoch * args.step_per_epoch) * math.pi) * 0.5 + 0.5
            learning_rate = get_learning_rate(step)
            pred, info = model.update(imgs, gt, RT, learning_rate, training=True)
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            if step % 200 == 1 and local_rank == 0:
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss/l1', info['loss_l1'], step)
                writer.add_scalar('loss/tea', info['loss_tea'], step)
                writer.add_scalar('loss/distill', info['loss_distill'], step)
            if step % 1000 == 1 and local_rank == 0:
                gt = (gt.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                mask = (torch.cat((info['mask'], info['mask_tea']), 3).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                pred = (pred.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                merged_img = (info['merged_tea'].permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                depth0 = info['depth'].permute(0, 2, 3, 1).detach().cpu().numpy()
                depth1 = info['depth_tea'].permute(0, 2, 3, 1).detach().cpu().numpy()
                # flow_gt = flow_gt.permute(0, 2, 3, 1).detach().cpu().numpy()
                for i in range(5):
                    imgs = np.concatenate((merged_img[i], pred[i], gt[i]), 1)[:, :, ::-1]
                    writer.add_image(str(i) + '/img', imgs, step, dataformats='HWC')
                    writer.add_image(str(i) + '/depth', np.concatenate((depth2rgb(depth0[i][:, :, :1]), depth2rgb(depth1[i][:, :, :1])), 1), step, dataformats='HWC')
                    writer.add_image(str(i) + '/mask', mask[i], step, dataformats='HWC')
                writer.flush()
            if local_rank == 0:
                print('epoch:{} {}/{} time:{:.2f}+{:.2f} loss_l1:{:.4e}'.format(epoch, i, args.step_per_epoch, data_time_interval, train_time_interval, info['loss_l1']))
            step = args.step_per_epoch * epoch + i
        nr_eval += 1
        if nr_eval % 5 == 0:
            evaluate(model, val_data, step, local_rank, writer_val)
        os.system('rm -rf train_log/*.pkl')
        model.save_model(log_path, local_rank, epoch)
        dist.barrier()

def evaluate(model, val_data, nr_eval, local_rank, writer_val):
    loss_l1_list = []
    loss_distill_list = []
    loss_tea_list = []
    psnr_list = []
    psnr_list_teacher = []
    time_stamp = time.time()
    for i, item in enumerate(val_data):
        # data_gpu, flow_gt = data
        data_gpu = item['img'].to(device, non_blocking=True) / 255.
        RT = item['RT'].to(device, non_blocking=True)
        # flow_gt = flow_gt.to(device, non_blocking=True)
        imgs = data_gpu[:, :6]
        gt = data_gpu[:, 6:9]
        with torch.no_grad():
            pred, info = model.update(imgs, gt, RT, training=False)
            merged_img = info['merged_tea']
        loss_l1_list.append(info['loss_l1'].cpu().numpy())
        loss_tea_list.append(info['loss_tea'].cpu().numpy())
        loss_distill_list.append(info['loss_distill'].cpu().numpy())
        for j in range(gt.shape[0]):
            psnr = -10 * math.log10(torch.mean((gt[j] - pred[j]) * (gt[j] - pred[j])).cpu().data)
            psnr_list.append(psnr)
            psnr = -10 * math.log10(torch.mean((merged_img[j] - gt[j]) * (merged_img[j] - gt[j])).cpu().data)
            psnr_list_teacher.append(psnr)
        gt = (gt.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        pred = (pred.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        merged_img = (merged_img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        depth0 = info['depth'].permute(0, 2, 3, 1).cpu().numpy()
        depth1 = info['depth_tea'].permute(0, 2, 3, 1).cpu().numpy()
        if i == 0 and local_rank == 0:
            for j in range(10):
                imgs = np.concatenate((merged_img[j], pred[j], gt[j]), 1)[:, :, ::-1]
                writer_val.add_image(str(j) + '/img', imgs.copy(), nr_eval, dataformats='HWC')
                writer_val.add_image(str(j) + '/flow', depth2rgb(depth0[j][:, :, :1]), nr_eval, dataformats='HWC')
                writer_val.add_image(str(j) + '/flow_gt', depth2rgb(depth1[j][:, :, :1]), nr_eval, dataformats='HWC')

    eval_time_interval = time.time() - time_stamp
    print('eval time: {}'.format(eval_time_interval))

    if local_rank != 0:
        return
    writer_val.add_scalar('psnr', np.array(psnr_list).mean(), nr_eval)
    writer_val.add_scalar('psnr_teacher', np.array(psnr_list_teacher).mean(), nr_eval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='slomo')
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--batch_size', default=8, type=int, help='minibatch size') # 4 * 12 = 48
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--world_size', default=4, type=int, help='world size')
    parser.add_argument('--resume', action='store_true', help='continue to train the model')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
    parser.add_argument('--logdir', default='train_log', help='the directory to save checkpoints/logs')
    args = parser.parse_args()
    num_gpus = torch.cuda.device_count()
    rank = int(os.environ['RANK'])
    print(f'{num_gpus} gpu available and rank is {rank}')
    torch.distributed.init_process_group(backend="nccl", world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    model = Model(args.local_rank)
    # if args.resume:
    #     model.load_model(args.logdir, args.local_rank, args.start_epoch, args.resume)
    train(model, args.local_rank, args.start_epoch)


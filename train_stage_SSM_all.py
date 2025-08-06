import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import sys
from comparison_methods.TransUNet import TransUNet
sys.path.append('comparison_methods')

from Network.U3Net import *
from my_utils.dataloader_alone_folder import *
from my_utils.post_process import post_process, alone2sequence_npy

import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data.dataset import Dataset

# from utils.dice_score import dice_loss
from my_utils.metrics import *
from my_utils.loss import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")   # use CPU or GPU
print('GPU num: ', torch.cuda.device_count())
from torch.utils.data import DataLoader
from torch import optim
from Network.stage_SSM import Stage_SSM

def shift9pos(input, h_shift_unit=1,  w_shift_unit=1):
    # input should be padding as (c, 1+ height+1, 1+width+1)
    input_pd = np.pad(input, ((h_shift_unit, h_shift_unit), (w_shift_unit, w_shift_unit)), mode='edge')
    input_pd = np.expand_dims(input_pd, axis=0)

    # assign to ...
    top     = input_pd[:, :-2 * h_shift_unit,          w_shift_unit:-w_shift_unit]
    bottom  = input_pd[:, 2 * h_shift_unit:,           w_shift_unit:-w_shift_unit]
    left    = input_pd[:, h_shift_unit:-h_shift_unit,  :-2 * w_shift_unit]
    right   = input_pd[:, h_shift_unit:-h_shift_unit,  2 * w_shift_unit:]

    center = input_pd[:,h_shift_unit:-h_shift_unit,w_shift_unit:-w_shift_unit]

    bottom_right    = input_pd[:, 2 * h_shift_unit:,   2 * w_shift_unit:]
    bottom_left     = input_pd[:, 2 * h_shift_unit:,   :-2 * w_shift_unit]
    top_right       = input_pd[:, :-2 * h_shift_unit,  2 * w_shift_unit:]
    top_left        = input_pd[:, :-2 * h_shift_unit,  :-2 * w_shift_unit]

    shift_tensor = np.concatenate([     top_left,    top,      top_right,
                                        left,        center,      right,
                                        bottom_left, bottom,    bottom_right], axis=0)
    return shift_tensor

def init_spixel_grid(img_height, img_width, batch_size):   # torch.Size([2, 9, 384, 384])
    curr_img_height = int(np.floor(img_height))
    curr_img_width = int(np.floor(img_width))

    all_h_coords = np.arange(0, curr_img_height, 1)  # (384,)
    all_w_coords = np.arange(0, curr_img_width, 1)  # (384,)
    curr_pxl_coord = np.array(np.meshgrid(all_h_coords, all_w_coords, indexing='ij'))  # curr_pxl_coord shape (2, 384, 384)
    coord_tensor = np.concatenate([curr_pxl_coord[1:2, :, :], curr_pxl_coord[:1, :, :]])  # coord_tensor shape (2, 384, 384)
    all_XY_feat = (torch.from_numpy(np.tile(coord_tensor, (batch_size, 1, 1, 1)).astype(np.float32)).cuda())  # torch.Size([2, 2, 384, 384])

    return all_XY_feat

def build_LABXY_feat(label_in, XY_feat):
    img_lab = label_in.clone().type(torch.float)
    b, _, curr_img_height, curr_img_width = XY_feat.shape
    scale_img =  F.interpolate(img_lab, size=(curr_img_height,curr_img_width), mode='nearest')
    LABXY_feat = torch.cat([scale_img, XY_feat],dim=1)

    return LABXY_feat

def poolfeat(input, prob, sp_h=2, sp_w=2):

    def feat_prob_sum(feat_sum, prob_sum, shift_feat):
        feat_sum += shift_feat[:, :-1, :, :]
        prob_sum += shift_feat[:, -1:, :, :]
        return feat_sum, prob_sum

    b, _, h, w = input.shape

    h_shift_unit = 1
    w_shift_unit = 1
    p2d = (w_shift_unit, w_shift_unit, h_shift_unit, h_shift_unit)                          #prob: torch.Size([2, 9, 24, 24])
    feat_ = torch.cat([input, torch.ones([b, 1, h, w]).cuda()], dim=1)  # b* (n+1) *h*w         feat_shape:  torch.Size([2, 4+1, 384, 384])
    # prob narrow shape: torch.Size([2, 1, 384, 384]) # test = feat_ * prob.narrow(1, 0, 1)   # print('test.shape', test.shape) test.shape torch.Size([2, 5, 384, 384])
    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 0, 1), kernel_size=(sp_h, sp_w),stride=(sp_h, sp_w)) #prob_feat shape:  torch.Size([2, 5, 24, 24])  sp_h 16
    # prob shape torch.Size([2, 9, 384, 384])   # prob.narrow(1, 0, 1) shape   torch.Size([2, 1, 384, 384])
    temp = F.pad(prob_feat, p2d, mode='constant', value=0)   # temp shape torch.Size([2, 5, 26, 26])
    send_to_top_left = temp[:, :, 2 * h_shift_unit:, 2 * w_shift_unit:]  # send_to_top_left.shape torch.Size([2, 5, 24, 24])
    feat_sum = send_to_top_left[:, :-1, :, :].clone()  # feat_sum.shape torch.Size([2, 4, 24, 24])
    prob_sum = send_to_top_left[:, -1:, :, :].clone()  # prob_sum.shape torch.Size([2, 1, 24, 24])

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 1, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    top = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, 2 * h_shift_unit:, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum,prob_sum,top) # prob_sum.shape torch.Size([2, 1, 24, 24])  prob_sum.shape torch.Size([2, 1, 24, 24])

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 2, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    top_right = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, 2 * h_shift_unit:, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, top_right)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 3, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    left = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, h_shift_unit:-h_shift_unit, 2 * w_shift_unit:]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, left)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 4, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    center = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, h_shift_unit:-h_shift_unit, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, center)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 5, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    right = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, h_shift_unit:-h_shift_unit, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, right)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 6, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    bottom_left = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, :-2 * h_shift_unit, 2 * w_shift_unit:]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom_left)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 7, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    bottom = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 8, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    bottom_right = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom_right)


    pooled_feat = feat_sum / (prob_sum + 1e-8)

    return pooled_feat

def upfeat(input, prob, up_h=2, up_w=2):
    # input b*n*H*W  downsampled
    # prob b*9*h*w
    b, c, h, w = input.shape

    h_shift = 1
    w_shift = 1

    p2d = (w_shift, w_shift, h_shift, h_shift)
    feat_pd = F.pad(input, p2d, mode='constant', value=0)

    gt_frm_top_left = F.interpolate(feat_pd[:, :, :-2 * h_shift, :-2 * w_shift], size=(h * up_h, w * up_w),mode='nearest')
    feat_sum = gt_frm_top_left * prob.narrow(1,0,1)

    top = F.interpolate(feat_pd[:, :, :-2 * h_shift, w_shift:-w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += top * prob.narrow(1, 1, 1)

    top_right = F.interpolate(feat_pd[:, :, :-2 * h_shift, 2 * w_shift:], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += top_right * prob.narrow(1,2,1)

    left = F.interpolate(feat_pd[:, :, h_shift:-w_shift, :-2 * w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += left * prob.narrow(1, 3, 1)

    center = F.interpolate(input, (h * up_h, w * up_w), mode='nearest')
    feat_sum += center * prob.narrow(1, 4, 1)

    right = F.interpolate(feat_pd[:, :, h_shift:-w_shift, 2 * w_shift:], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += right * prob.narrow(1, 5, 1)

    bottom_left = F.interpolate(feat_pd[:, :, 2 * h_shift:, :-2 * w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += bottom_left * prob.narrow(1, 6, 1)

    bottom = F.interpolate(feat_pd[:, :, 2 * h_shift:, w_shift:-w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += bottom * prob.narrow(1, 7, 1)

    bottom_right =  F.interpolate(feat_pd[:, :, 2 * h_shift:, 2 * w_shift:], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += bottom_right * prob.narrow(1, 8, 1)

    return feat_sum

def compute_semantic_pos_loss(prob_in, labxy_feat,  pos_weight = 0.003,  kernel_size=16):
    # todo: currently we assume the downsize scale in x,y direction are always same

    S = kernel_size
    m = pos_weight
    prob = prob_in.clone()

    b, c, h, w = labxy_feat.shape   #torch.Size([8, 52, 384, 384])
    pooled_labxy = poolfeat(labxy_feat, prob, kernel_size, kernel_size)  #  pooled_labxy.shape   torch.Size([2, 52, 24, 24])

    reconstr_feat = upfeat(pooled_labxy, prob, kernel_size, kernel_size)  # reconstr_feat.shape torch.Size([2, 52, 384, 384])

    loss_map = reconstr_feat[:,-2:,:,:] - labxy_feat[:,-2:,:,:]   #loss_map.shape torch.Size([2, 2, 384, 384])

    # self def cross entropy  -- the official one combined softmax
    logit = torch.log(reconstr_feat[:, :-2, :, :] + 1e-8)
    loss_sem = - torch.sum(logit * labxy_feat[:, :-2, :, :]) / b
    loss_pos = torch.norm(loss_map, p=2, dim=1).sum() / b * m / S

    # empirically we find timing 0.005 tend to better performance
    loss_sum = 0.005 * (loss_sem + loss_pos)
    loss_sem_sum = 0.005 * loss_sem
    loss_pos_sum = 0.005 * loss_pos

    return loss_sum, loss_sem_sum,  loss_pos_sum

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_model(model, criterion, optimizer, scheduler, train_loader, num_epochs, total_epoch, model_name,
                state_save_path, state_load_path=None):
    #     torch.backends.cudnn.benchmark = True

    if state_load_path is not None:
        model.load_state_dict(torch.load(state_load_path))

    num_step = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        dice_score = 0

        xy_feat1 = init_spixel_grid(64, 64, batch_size)
        xy_feat2 = init_spixel_grid(32, 32, batch_size)
        xy_feat3 = init_spixel_grid(16, 16, batch_size)
        xy_feat4 = init_spixel_grid(8, 8, batch_size)

        model.train()
        with tqdm(total=num_step, desc=f'Epoch {epoch + 1}/{num_epochs}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(train_loader):
                images, masks = batch[0], batch[1]
                batch_step = images.shape[0]
                images = images.to(device)
                masks = masks.to(device)

                if iteration == num_step - 1:
                    xy_feat1 = init_spixel_grid(64, 64, batch_step)
                    xy_feat2 = init_spixel_grid(32, 32, batch_step)
                    xy_feat3 = init_spixel_grid(16, 16, batch_step)
                    xy_feat4 = init_spixel_grid(8, 8, batch_step)

                masks1 = F.interpolate(masks, size=(64, 64), mode='nearest')
                masks2 = F.interpolate(masks, size=(32, 32), mode='nearest')
                masks3 = F.interpolate(masks, size=(16, 16), mode='nearest')
                masks4 = F.interpolate(masks, size=(8, 8), mode='nearest')

                LABXY_feat_tensor1 = build_LABXY_feat(masks1, xy_feat1)
                LABXY_feat_tensor2 = build_LABXY_feat(masks2, xy_feat2)
                LABXY_feat_tensor3 = build_LABXY_feat(masks3, xy_feat3)
                LABXY_feat_tensor4 = build_LABXY_feat(masks4, xy_feat4)

                masks_pred, Q_prob_collect = model(images)

                slic_loss1, loss_sem, loss_pos = compute_semantic_pos_loss(Q_prob_collect[0], LABXY_feat_tensor1, pos_weight=0.003, kernel_size=2)
                slic_loss2, loss_sem, loss_pos = compute_semantic_pos_loss(Q_prob_collect[1], LABXY_feat_tensor2, pos_weight=0.003, kernel_size=2)
                slic_loss3, loss_sem, loss_pos = compute_semantic_pos_loss(Q_prob_collect[2], LABXY_feat_tensor3, pos_weight=0.003, kernel_size=2)
                slic_loss4, loss_sem, loss_pos = compute_semantic_pos_loss(Q_prob_collect[3], LABXY_feat_tensor4, pos_weight=0.003, kernel_size=1)

                masks_pred = nn.Sigmoid()(masks_pred)

                loss_value = criterion(masks_pred, masks) + dice_loss_binary(masks_pred, masks)
                loss_sum = loss_value + (slic_loss1 + slic_loss2 + slic_loss3 + slic_loss4) * 0.2
                optimizer.zero_grad()
                loss_sum.backward()
                optimizer.step()
                epoch_loss += loss_sum.item()

                masks_pred = (masks_pred > 0.5).float()
                dice_score += dice_metric_binary(masks_pred, masks)

                pbar.set_postfix(**{'loss': epoch_loss / (iteration + 1),
                                    'lr': get_lr(optimizer),
                                    'dice': dice_score.item() / (iteration + 1)})
                pbar.update(1)
        #             epoch_avg_loss = epoch_loss / sample_num

        scheduler.step()
    #         print('epoch_avg_loss: ', epoch_avg_loss)
    torch.save(model.state_dict(), state_save_path + model_name + '_{}.pth'.format(total_epoch))

def train_model_mc(model, criterion, optimizer, scheduler, train_loader, num_epochs, total_epoch, model_name,
                state_save_path, state_load_path=None):
    #     torch.backends.cudnn.benchmark = True

    if state_load_path is not None:
        model.load_state_dict(torch.load(state_load_path))

    num_step = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        dice_score = 0

        xy_feat1 = init_spixel_grid(64, 64, batch_size)
        xy_feat2 = init_spixel_grid(32, 32, batch_size)
        xy_feat3 = init_spixel_grid(16, 16, batch_size)
        xy_feat4 = init_spixel_grid(8, 8, batch_size)

        model.train()
        with tqdm(total=num_step, desc=f'Epoch {epoch + 1}/{num_epochs}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(train_loader):
                images, masks = batch[0], batch[1]
                batch_step = images.shape[0]
                images = images.to(device)
                masks = masks.to(device)
                masks_hot = F.one_hot(masks, 3).permute(0, 3, 1, 2).float()  # 我添加了
                masks_unsqueeze = torch.unsqueeze(masks, dim=1).float()

                if iteration == num_step - 1:
                    xy_feat1 = init_spixel_grid(64, 64, batch_step)
                    xy_feat2 = init_spixel_grid(32, 32, batch_step)
                    xy_feat3 = init_spixel_grid(16, 16, batch_step)
                    xy_feat4 = init_spixel_grid(8, 8, batch_step)

                masks1 = F.interpolate(masks_unsqueeze, size=(64, 64), mode='nearest')
                masks2 = F.interpolate(masks_unsqueeze, size=(32, 32), mode='nearest')
                masks3 = F.interpolate(masks_unsqueeze, size=(16, 16), mode='nearest')
                masks4 = F.interpolate(masks_unsqueeze, size=(8, 8), mode='nearest')

                LABXY_feat_tensor1 = build_LABXY_feat(masks1, xy_feat1)
                LABXY_feat_tensor2 = build_LABXY_feat(masks2, xy_feat2)
                LABXY_feat_tensor3 = build_LABXY_feat(masks3, xy_feat3)
                LABXY_feat_tensor4 = build_LABXY_feat(masks4, xy_feat4)

                masks_pred, Q_prob_collect = model(images)

                slic_loss1, loss_sem, loss_pos = compute_semantic_pos_loss(Q_prob_collect[0], LABXY_feat_tensor1, pos_weight=0.003, kernel_size=2)
                slic_loss2, loss_sem, loss_pos = compute_semantic_pos_loss(Q_prob_collect[1], LABXY_feat_tensor2, pos_weight=0.003, kernel_size=2)
                slic_loss3, loss_sem, loss_pos = compute_semantic_pos_loss(Q_prob_collect[2], LABXY_feat_tensor3, pos_weight=0.003, kernel_size=2)
                slic_loss4, loss_sem, loss_pos = compute_semantic_pos_loss(Q_prob_collect[3], LABXY_feat_tensor4, pos_weight=0.003, kernel_size=1)

                criterion_loss = criterion(masks_pred, masks)
                loss_value = criterion_loss + dice_loss_multiclass(F.softmax(masks_pred, dim=1).float(), masks_hot)

                loss_sum = loss_value + (slic_loss1 + slic_loss2 + slic_loss3 + slic_loss4) * 0.2
                optimizer.zero_grad()
                loss_sum.backward()
                optimizer.step()
                epoch_loss += loss_sum.item()

                masks_pred = F.softmax(masks_pred, dim=1)
                masks_pred = F.one_hot(masks_pred.argmax(dim=1), 3).permute(0, 3, 1, 2).float()
                dice_score += dice_metric_multiclass(masks_pred[:, 1:, ...], masks_hot[:, 1:, ...])

                pbar.set_postfix(**{'loss': epoch_loss / (iteration + 1),
                                    'lr': get_lr(optimizer),
                                    'dice': dice_score.item() / (iteration + 1)})
                pbar.update(1)
        #             epoch_avg_loss = epoch_loss / sample_num

        scheduler.step()
    #         print('epoch_avg_loss: ', epoch_avg_loss)
    torch.save(model.state_dict(), state_save_path + model_name + '_{}.pth'.format(total_epoch))

class Alone_Dataset(Dataset):
    def __init__(self, alone_image_mask_npy_path_list, new_shape, del_view=None, is_predict=False):
        super(Alone_Dataset, self).__init__()
        self.new_shape = new_shape
        self.is_predict = is_predict
        self.datalist_line = []
        for alone_image_mask_npy_path in alone_image_mask_npy_path_list:
            with open(alone_image_mask_npy_path) as f:
                self.datalist_line = self.datalist_line + f.readlines()

        self.datalist_line_copy = self.datalist_line.copy()
        if del_view is not None:
            for data_element in self.datalist_line_copy:
                # print(data_element.rstrip('\n').split(' ')[2])
                if data_element.rstrip('\n').split(' ')[2] in del_view:
                    self.datalist_line.remove(data_element)
                    # break

    def __len__(self):
        return len(self.datalist_line)

    def __getitem__(self, index):
        mask_path = self.datalist_line[index].rstrip('\n').split(' ')[-1]
        # mask_view = self.datalist_line[index].rstrip('\n').split(' ')[2]
        mask_npy = np.load(mask_path)
        mask_npy = np.where(mask_npy > 1, 0, mask_npy)
        mask_npy = np.squeeze(mask_npy)
        mask_npy = mask_npy[..., np.newaxis]
        mask_npy = cv2.resize(mask_npy, self.new_shape, interpolation=cv2.INTER_NEAREST)
        # mask_npy = np.array(mask_npy, dtype=np.float32)
        mask_npy = np.array(mask_npy, dtype=np.float32)
        # if mask_view == 'psaxin':
        #     mask_npy = np.where(mask_npy == 1, 3, mask_npy)
        mask_npy = mask_npy[np.newaxis, ...]
        mask_npy = torch.from_numpy(mask_npy)

        image_path = self.datalist_line[index].rstrip('\n').split(' ')[-2]
        image_npy = np.load(image_path)
        image_npy = np.array(image_npy, dtype=np.float32)
        image_npy = image_npy / 255
        image_npy = image_npy[..., np.newaxis]
        image_npy = cv2.resize(image_npy, self.new_shape)
        image_npy = image_npy[np.newaxis, ...]
        image_npy = torch.from_numpy(image_npy)

        # image_file = image_path.split('/')[-1]
        # mask_file = mask_path.split('/')[-1]
        # assert image_file == mask_file

        if not self.is_predict:
            return image_npy, mask_npy
        else:
            image_file = image_path.split('/')[-1]
            return image_npy, mask_npy, image_file

class Alone_Dataset_mc_previous(Dataset):
    def __init__(self, alone_image_mask_npy_path_list, new_shape, del_view=None, is_predict=False):
        super(Alone_Dataset_mc, self).__init__()
        self.new_shape = new_shape
        self.is_predict = is_predict
        self.datalist_line = []
        for alone_image_mask_npy_path in alone_image_mask_npy_path_list:
            with open(alone_image_mask_npy_path) as f:
                self.datalist_line = self.datalist_line + f.readlines()

        self.datalist_line_copy = self.datalist_line.copy()
        if del_view is not None:
            for data_element in self.datalist_line_copy:
                # print(data_element.rstrip('\n').split(' ')[2])
                if data_element.rstrip('\n').split(' ')[2] in del_view:
                    self.datalist_line.remove(data_element)
                    # break

    def __len__(self):
        return len(self.datalist_line)

    def __getitem__(self, index):
        mask_path = self.datalist_line[index].rstrip('\n').split(' ')[-1]
        mask_view = self.datalist_line[index].rstrip('\n').split(' ')[2]
        if mask_view == 'psaxmass':
            mask_path = mask_path.replace('psaxmass', 'psaxnwm')
        mask_npy = np.load(mask_path)
        mask_npy = np.where(mask_npy >= 3, 0, mask_npy)
        # print(np.unique(mask_npy))
        mask_npy = np.squeeze(mask_npy)
        mask_npy = mask_npy[..., np.newaxis]
        mask_npy = cv2.resize(mask_npy, self.new_shape, interpolation=cv2.INTER_NEAREST)
        mask_npy = np.array(mask_npy, dtype=np.int64)
        # mask_npy = np.array(mask_npy, dtype=np.float32)
        # if mask_view == 'psaxin':
        #     mask_npy = np.where(mask_npy == 1, 3, mask_npy)
        # mask_npy = mask_npy[np.newaxis, ...]
        mask_npy = torch.from_numpy(mask_npy)

        image_path = self.datalist_line[index].rstrip('\n').split(' ')[-2]
        image_npy = np.load(image_path)
        image_npy = np.array(image_npy, dtype=np.float32)
        image_npy = image_npy / 255
        image_npy = image_npy[..., np.newaxis]
        image_npy = cv2.resize(image_npy, self.new_shape)
        image_npy = image_npy[np.newaxis, ...]
        image_npy = torch.from_numpy(image_npy)

        # image_file = image_path.split('/')[-1]
        # mask_file = mask_path.split('/')[-1]
        # assert image_file == mask_file

        if not self.is_predict:
            return image_npy, mask_npy
        else:
            image_file = image_path.split('/')[-1]
            return image_npy, mask_npy, image_file

class Alone_Dataset_mc(Dataset):
    def __init__(self, alone_image_mask_npy_path_list, new_shape, del_view=None, is_predict=False, nature=False):
        super(Alone_Dataset_mc, self).__init__()
        self.new_shape = new_shape
        self.is_predict = is_predict
        self.datalist_line = []
        self.nature = nature
        for alone_image_mask_npy_path in alone_image_mask_npy_path_list:
            with open(alone_image_mask_npy_path) as f:
                self.datalist_line = self.datalist_line + f.readlines()

        self.datalist_line_copy = self.datalist_line.copy()
        if del_view is not None:
            for data_element in self.datalist_line_copy:
                # print(data_element.rstrip('\n').split(' ')[2])
                if data_element.rstrip('\n').split(' ')[2] in del_view:
                    self.datalist_line.remove(data_element)
                    # break

    def __len__(self):
        return len(self.datalist_line)

    def __getitem__(self, index):
        mask_path = self.datalist_line[index].rstrip('\n').split(' ')[-1]
        mask_view = self.datalist_line[index].rstrip('\n').split(' ')[2]
        # data_type = self.datalist_line[index].rstrip('\n').split(' ')[1]
        if mask_view == 'psaxmass':
            mask_path = mask_path.replace('psaxmass', 'psaxnwm')
        mask_npy = np.load(mask_path)
        if self.nature:
            mask_npy = mask_npy[..., 0]
        mask_npy = np.where(mask_npy >= 3, 0, mask_npy)
        # print(np.unique(mask_npy))
        mask_npy = np.squeeze(mask_npy)
        mask_npy = mask_npy[..., np.newaxis]
        # mask_npy = np.array(mask_npy, dtype=np.float32)
        mask_npy = cv2.resize(mask_npy, self.new_shape, interpolation=cv2.INTER_NEAREST)
        mask_npy = np.array(mask_npy, dtype=np.int64)
        # mask_npy = np.array(mask_npy, dtype=np.float32)
        # if mask_view == 'psaxin':
        #     mask_npy = np.where(mask_npy == 1, 3, mask_npy)
        # mask_npy = mask_npy[np.newaxis, ...]
        mask_npy = torch.from_numpy(mask_npy)

        image_path = self.datalist_line[index].rstrip('\n').split(' ')[-2]
        image_npy = np.load(image_path)
        if self.nature:
            image_npy = image_npy[..., 0]
        image_npy = np.array(image_npy, dtype=np.float32)
        image_npy = image_npy / 255
        image_npy = image_npy[..., np.newaxis]
        image_npy = cv2.resize(image_npy, self.new_shape)
        image_npy = image_npy[np.newaxis, ...]
        image_npy = torch.from_numpy(image_npy)

        # image_file = image_path.split('/')[-1]
        # mask_file = mask_path.split('/')[-1]
        # assert image_file == mask_file

        if not self.is_predict:
            return image_npy, mask_npy
        else:
            image_file = image_path.split('/')[-1]
            return image_npy, mask_npy, image_file

def predict_images(net, dataloader, device, output_path):
    net.eval()
    num_step = len(dataloader)

    with tqdm(total=num_step, desc=f'predict images:', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(dataloader):
            image, mask_true, alone_image_file = batch[0], batch[1], batch[2]
            image = image.to(device=device, dtype=torch.float32)

            #             print(alone_image_file)
            #             alone_image_file_prefix = os.path.splitext(alone_image_file)[0]
            with torch.no_grad():
                mask_pred = net(image)

                #                 print('mask_pred shape:', mask_pred.shape)   #shape: torch.Size([4, 3, 852, 1136])
                mask_pred = torch.sigmoid(mask_pred)
                mask_pred = torch.gt(mask_pred, 0.5)
                mask_pred = mask_pred.type(torch.float32)
                #                 print(mask_pred)
                #                 print('mask_pred shape:', mask_pred.shape)    # shape: torch.Size([4, 852, 1136])
                for s in range(mask_pred.shape[0]):
                    mask_pred_s = mask_pred[s]
                    mask_pred_s = mask_pred_s.cpu().numpy()
                    mask_pred_s = np.squeeze(mask_pred_s)
                    mask_pred_s = post_process(mask_pred_s)
                    #                     rr = np.unique(mask_pred_s)
                    #                     print(rr)

                    alone_image_file_s = alone_image_file[s]
                    alone_image_file_s_prefix = os.path.splitext(alone_image_file_s)[0]

                    np.save(output_path + alone_image_file_s_prefix + '.npy', mask_pred_s)

            pbar.set_postfix(**{'mask_pred': mask_pred.shape})
            pbar.update(1)




if __name__ == '__main__':
    model = Stage_SSM(num_class=3)
    model = model.to(device)
    # model = nn.DataParallel(model)

    batch_size = 64
    learning_rate = 0.000075

    train_fetal = '/data/dwl/all_age_data/datalist_txt/fetal/fetal_train_image_mask_npy_datalist-1.txt'
    train_children14 = '/data/dwl/all_age_data/datalist_txt/14children/14children_train_image_mask_npy_datalist-1.txt'
    train_children86 = '/data/dwl/all_age_data/datalist_txt/86children/86children_train_image_mask_npy_datalist-1.txt'
    train_children120 = '/data/dwl/all_age_data/datalist_txt/children120/children120_train_image_mask_npy_datalist-1.txt'
    train_children110 = '/data/dwl/all_age_data/datalist_txt/110children/110children_train_image_mask_npy_datalist-1.txt'
    train_adultmultiperiod = '/data/dwl/all_age_data/datalist_txt/adultmultiperiod/adultmultiperiod_train_image_mask_npy_datalist-1.txt'
    train_adultsingleperiod = '/data/dwl/all_age_data/datalist_txt/adultsingleperiod/adultsingleperiod_train_image_mask_npy_datalist-1.txt'
    train_hmcqu = '/data/dwl/all_age_data/datalist_txt/HMCQU/HMCQU_train_image_mask_npy_datalist-1.txt'
    train_camus = '/data/dwl/all_age_data/datalist_txt/camus/camus_train_image_mask_npy_datalist-1.txt'
    # train_nature = '/data/dwl/all_age_data/datalist_txt/nature/nature_subdataset/nature_subdataset_train_image_mask_npy_datalist-1.txt'

    test_fetal = '/data/dwl/all_age_data/datalist_txt/fetal/fetal_test_image_mask_npy_datalist-1.txt'
    test_children14 = '/data/dwl/all_age_data/datalist_txt/14children/14children_test_image_mask_npy_datalist-1.txt'
    test_children86 = '/data/dwl/all_age_data/datalist_txt/86children/86children_test_image_mask_npy_datalist-1.txt'
    test_chilren120 = '/data/dwl/all_age_data/datalist_txt/children120/children120_test_image_mask_npy_datalist-1.txt'
    test_children110 = '/data/dwl/all_age_data/datalist_txt/110children/110children_test_image_mask_npy_datalist-1.txt'
    test_adultmultiperiod = '/data/dwl/all_age_data/datalist_txt/adultmultiperiod/adultmultiperiod_test_image_mask_npy_datalist-1.txt'
    test_adultsingleperiod = '/data/dwl/all_age_data/datalist_txt/adultsingleperiod/adultsingleperiod_test_image_mask_npy_datalist-1.txt'
    test_hmcqu = '/data/dwl/all_age_data/datalist_txt/HMCQU/HMCQU_test_image_mask_npy_datalist-1.txt'
    test_camus = '/data/dwl/all_age_data/datalist_txt/camus/camus_test_image_mask_npy_datalist-1.txt'
    # test_nature = '/data/dwl/all_age_data/datalist_txt/nature/nature_subdataset/nature_subdataset_test_image_mask_npy_datalist-1.txt'

    state_save_path = '/data/dwl/all_age_data/model/train_SSM_all/'
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)

    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.91)
    # grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    global_step = 0

############################################################################  以下是训练的相关设置代码

    train_image_mask_npy_path_list = [train_children86, train_children120, train_children110, train_adultmultiperiod, train_adultsingleperiod,
                                      train_hmcqu, train_camus]
    train_set = Alone_Dataset_mc(train_image_mask_npy_path_list, new_shape=(128, 128), del_view=['psaxin', 'psaxout'])
    train_loader = DataLoader(train_set, shuffle=True, drop_last=False, **loader_args)

    # state_load_path = '/data/dwl/all_age_data/model/atrial_dataset_baseline_SP/Baseline-SP-atrial_50.pth'
    train_model(model=model, criterion=criterion, optimizer=optimizer, scheduler=lr_scheduler, \
                train_loader=train_loader, num_epochs=3, total_epoch=3, model_name='Stage_SSM_all',
                state_save_path=state_save_path, state_load_path=None)




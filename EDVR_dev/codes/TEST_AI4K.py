import os.path as osp
import cv2
import os
import numpy as np
import torch
import models.modules.EDVR_arch as EDVR_arch
import math
from torchvision.utils import make_grid

testpath = '/input/testpngs'
out_path = '../results'
index = 0
def read_img(img_path):
    """Read an image from a given image path
    Args:
        img_path (str): image path

    Returns:
        img (Numpy): size (H, W, C), BGR, [0, 1]
    """
    img = cv2.imread(img_path)
    img = img.astype(np.float32) / 255.
    return img

def read_img_seq(img_list_l):
    """Read a sequence of images from a given folder path
    Args:
        img_folder_path (str): image folder path

    Returns:
        imgs (Tensor): size (T, C, H, W), RGB, [0, 1]
    """
    img_l = [read_img(v) for v in img_list_l]
    # stack to Torch tensor
    imgs = np.stack(img_l, axis=0)
    imgs = imgs[:, :, :, [2, 1, 0]]
    imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()
    return imgs
list_testpath = [[] for _ in range(len(os.listdir(testpath)))]

for clip_name in sorted(os.listdir(testpath)):

    out_path_clip = osp.join(out_path,clip_name)
    testpath_clips = osp.join(testpath,clip_name)
    for i in range(100):
        if i == 0:
            for x in range(3):
                list_testpath[index].append(osp.join(testpath_clips, sorted(os.listdir(testpath_clips))[i]))
        elif i == 99:
            for x in range(3):
                list_testpath[index].append(osp.join(testpath_clips, sorted(os.listdir(testpath_clips))[i]))
        else:
            list_testpath[index].append(osp.join(testpath_clips, sorted(os.listdir(testpath_clips))[i]))

    index += 1
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)

model_path = '../experiments/AI4K_TEST/models/latest_G.pth'
device = torch.device('cuda')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
N_in = 5
model = EDVR_arch.EDVR(64, N_in, 4, 5, 20, predeblur=False, HR_in=False,w_TSA=True)
model.load_state_dict(torch.load(model_path), strict=True)
for t in range(len(list_testpath)):
    list_testpath = sorted(list_testpath)
    save_name = list_testpath[t][0].split('/')[-2]
    mkdir(osp.join(out_path,save_name))
    save_dir = osp.join(out_path,save_name)
    print('<<<<<<<<<<<<<<<<<<<<<<<<------------------>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>{}'.format(t))

    for x in range(100):

        imgs_list = list_testpath[t][x:x + 5]
        imgs_l_torch = read_img_seq(imgs_list)
        with torch.no_grad():
            model.eval()
            model = model.to(device)
            sr = model(imgs_l_torch.unsqueeze(0).to(device))
            out_img = tensor2img(sr.squeeze(0))
            print(osp.join(save_dir,'{}_%.4d.png'.format(save_name) %(x+1)))
            save_img(out_img,osp.join(save_dir,'{}_%.4d.png'.format(save_name) %(x+1)))








# print(list_testpath)

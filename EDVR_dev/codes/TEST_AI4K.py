import os.path as osp
import cv2
import os
import numpy as np
import torch
import models.modules.EDVR_arch as EDVR_arch
import torch.nn as nn
from torchvision.utils import make_grid
import math

testpath = ''
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
# imgs_path = 'dataset/X4/10091373/10091373_0001.png'
# imgdata = read_img(imgs_path)
# print(imgdata[200:300,400:500,:1])
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
def tensor2img(tensor,out_type=np.uint8,min_max=(0,1)):
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0])/(min_max[1] - min_max[0])
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor,nrow=int(math.sqrt(n_img)),normalize=False).numpy()
        img_np = np.transpose(img_np[[2,1,0],:,:],(1,2,0))
    elif n_dim == 3:
        img_np = tensor.numpy()	
        img_np = np.transpose(img_np[[2,1,0],:,:],(1,2,0))
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError('only 4d,3d,2d')
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    return img_np.astype(out_type)
def makedir(path):
    if not osp.exists(path):
        os.makedirs(path)
def save_img(path,img,model='RGB'):
    cv2.imwrite(path,img)
list_testpath = [[] for _ in range(len(os.listdir(testpath)))]

for clip_name in sorted(os.listdir(testpath)):
    # print(clip_name)
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
    # print(list_testpath[index][99:104])
    # for x in range(100):
    #     print(list_testpath[index][x:(x + 5)])
    # for x in range(100):
    #     print(list_testpath[1])
    index += 1
# for x in range(100):
#     print(list_testpath[0][x:x +5])
imgs_list = list_testpath[0][0:5]
imgs_l_torch = read_img_seq(imgs_list)
# print(imgs_l_torch[:,:,200:30400:500])
model_path = '../experiments/AI4K_TEST/models/latest_G.pth'
#device = torch.device('cuda')
#os.environ['CUDA_VISIBLE_DEVICES'] = 0,1
N_in = 5
model = EDVR_arch.EDVR(64, N_in, 4, 2, 10, predeblur=False, HR_in=False,w_TSA=False)
makedir(out_path)
# print(model)
with torch.no_grad():
	model.load_state_dict(torch.load(model_path), strict=True)
	model.eval()
	model = model.cuda()
	sr = model(imgs_l_torch.unsqueeze(0).cuda())
	sr_np = tensor2img(sr)
	save_img(osp.join(out_path,'example.png'),sr_np)

# print(list_testpath)

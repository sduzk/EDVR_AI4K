
import os
import os.path as osp
import sys
import cv2
import lmdb
import pickle
try:
    sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
    import data.util as data_util
    import utils.util as util
except ImportError:
    pass
# print(osp.abspath(__file__))
def AI4K(model = 'gt'):
    model = model
    read_all_imgs = False
    BATCH = 700
    if model == 'gt':
        img_folder = 'dataset/gt'
        lmdb_save_path = 'dataset/train_gt_wval.lmdb'
        H_dst, W_dst = 2160, 3840
    if model == 'X4':
        img_folder = 'dataset/X4'
        lmdb_save_path = 'dataset/train_x4_wval.lmdb'
        H_dst, W_dst = 540, 960
    n_thread = 40
    ########################################################
    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with \'lmdb\'.")
    #### whether the lmdb file exist
    # if osp.exists(lmdb_save_path):
    #     print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
    #     sys.exit(1)
    print('Reading image path list ...')
    all_clips_list = sorted(os.listdir(img_folder))
    all_clips_list_path = []
    for x in all_clips_list:
        all_clips_list_path.append(os.path.join(img_folder,x))
   

    keys = []
    all_imgs_path = []
    index_clip = 0
    for clips_path in all_clips_list_path:
        index_clip += 1
        if model == 'X4':
            for imgs_x4_path in data_util._get_paths_from_images(clips_path):
                all_imgs_path.append(imgs_x4_path)
            for index_imgs_x4 in range(100):
                a = (index_imgs_x4 + 1) // 7 + 1
                b = (index_imgs_x4 + 1) % 7
                if b == 0:
                    b = 7
                c = '%.5d' %(index_clip) + '_' + '%.4d' %(a) + '_' + '%d' %(b)
                keys.append(c)
        else:
            for index,imgs_path in enumerate(data_util._get_paths_from_images(clips_path)):
                if index % 7 == 3:
                    all_imgs_path.append(imgs_path)
            for index_imgs_gt in range(100):
                if index_imgs_gt % 7 == 3:
                    a = (index_imgs_gt + 1) // 7 + 1
                    c = '%.5d' % (index_clip) + '_' + '%.4d' % (a) + '_4'
                    keys.append(c)
  

    data_size_per_img = cv2.imread(all_imgs_path[0], cv2.IMREAD_UNCHANGED).nbytes
    print('data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(all_imgs_path)

    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)
    pbar = util.ProgressBar(len(all_imgs_path))
    txn = env.begin(write=True)
    idx = 1
    for path, key in zip(all_imgs_path, keys):
        idx = idx + 1
        pbar.update('Write {}'.format(key))
        key_byte = key.encode('ascii')
        data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        H, W, C = data.shape  # fixed shape
        assert H == H_dst and W == W_dst and C == 3, 'different shape.'
        txn.put(key_byte, data)
        if not read_all_imgs and idx % BATCH == 1:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()
    print('Finish writing lmdb.')
    #### create meta information
    meta_info = {}
    if model == 'gt':
        meta_info['name'] = 'AI4K_train_GT'
    elif model == 'X4':
        meta_info['name'] = 'AI4K_train_X4'
    meta_info['resolution'] = '{}_{}_{}'.format(3, H_dst, W_dst)
    key_set = set()
    for key in keys:
        a, b, _ = key.split('_')
        key_set.add('{}_{}'.format(a, b))
    meta_info['keys'] = key_set
    pickle.dump(meta_info, open(osp.join(lmdb_save_path, 'meta_info.pkl'), "wb"))
    print('Finish creating lmdb meta info.')


   
if __name__ == "__main__":
    AI4K(model='gt')
    AI4K(model='X4')






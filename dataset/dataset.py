import sys
sys.path.append('.')
from torch.utils.data import Dataset, DataLoader
import json
from config.config_pc2ext import config
import os
import h5py
from util.pc_utils import read_ply_norms,read_ply_norm_cal
import random
import torch
import numpy as np
import torch.nn.functional as F
def pc_normalize(pc):
    """
    对点云数据进行归一化
    :param pc: 需要归一化的点云数据
    :return: 归一化后的点云数据
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc /= m
    return pc

# def pc_normalize(pc):
#     """
#     对点云数据进行归一化
#     :param pc: 需要归一化的点云数据
#     :return: 归一化后的点云数据
#     """
#     centroid = np.mean(pc, axis=0)
#     pc = pc - centroid
#     m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
#     pc /= m
#     return pc

def get_dataloader(config, phase, shuffle=None, ddp=True):

    is_shuffle = phase == "train" if shuffle is None else shuffle
    dataset = PC2ext_dataset(config, phase)
    dataloader = DataLoader(dataset,
                            batch_size=config.batch_size,
                            shuffle=is_shuffle,
                            num_workers=config.num_workers,
                            worker_init_fn=np.random.seed())
    return dataloader

def add_noise(pc_xyz, normal, sigma=0.01):
    '''
    Adds a random gaussian noise for each point in the direction of the normal
    '''
    # print("adding noise")
    num_points, _ = pc_xyz.shape

    sampled_noise = np.random.normal(0.0, sigma, (num_points))
    sampled_noise = np.tile(np.expand_dims(sampled_noise, axis=-1), [1,3])
    noisy_pc = pc_xyz + sampled_noise*normal

    return noisy_pc


class PC2ext_dataset(Dataset):
    def __init__(self, cfg:config, phase) -> None:
        super().__init__()
        self.split_path = cfg.split_path
        with open(self.split_path, "r")as f:
            self.all_data=json.load(f)[phase]
        self.pc_dir = cfg.pc_root
        self.ext_id_dir = cfg.ext_label_root
        self.num_samples = cfg.n_points
        self.add_noise = cfg.add_noise
    def get_data_by_id(self, data_id):
        idx = self.all_data.index(data_id)
        return self.__getitem__(idx)

    def __getitem__(self, index):
        data_id = self.all_data[index]
        # TODO:0 for barrel, 1 for base
        pc_path = os.path.join(self.pc_dir, data_id + '.ply')
        pc_normals = read_ply_norm_cal(pc_path)
        pc, normals = pc_normals[:,:3], pc_normals[:, 3:]
        num = 8196
        pc_sample_idx = random.sample(list(range(pc.shape[0])), self.num_samples)
        pc = pc[pc_sample_idx]
        normals = normals[pc_sample_idx]
        pc = pc_normalize(pc=pc)
        if self.add_noise:
            pc = add_noise(pc, normals)
        pc = torch.tensor(pc, dtype=torch.float32)
        normals = torch.tensor(normals, dtype=torch.float32)
        normals = F.normalize(normals, p=2, dim=1, eps=1e-12)

        h5_path = os.path.join(self.ext_id_dir, data_id + '.h5')
        try:
            with h5py.File(h5_path, 'r') as fp:
                ext_label    = fp["ext_label"][:]
                barrel_label = fp["barrel_label"][:]
                ext_axis     = fp["ext_axis"][:]
                ext_center   = fp["ext_center"][:]
        except Exception as e:
            ext_label = torch.zeros((num, 1), dtype=torch.long)
            barrel_label = torch.zeros((len(pc), 1), dtype=torch.long)
            ext_axis = torch.zeros((8, 3), dtype=torch.long)
            ext_center = torch.zeros((8, 3), dtype=torch.long)
        # labels for extrusion segmentation
        ext_label = ext_label[pc_sample_idx]
        ext_label = torch.tensor(ext_label, dtype=torch.long).squeeze()
        # labels for barrel/base point segmentation
        barrel_label = 1 - barrel_label
        barrel_label = barrel_label[pc_sample_idx]
        barrel_label = torch.tensor(barrel_label, dtype=torch.long).squeeze()
        ext_axis     = torch.tensor(ext_axis, dtype=torch.float32)
        ext_center   = torch.tensor(ext_center, dtype=torch.float32)

        return {
            "id": data_id,
            "pc": pc,
            "ext_label": ext_label,
            "normal_label": normals,
            "barrel_label": barrel_label,
            "gt_ext_axis":  ext_axis,
            "gt_ext_center": ext_center,
        }
    
    def __len__(self):
        return len(self.all_data)



if __name__ == "__main__":
    cfg = config("test")
    dataloader = PC2ext_dataset(cfg=cfg, phase="train")
    print(dataloader[0]['pc'].shape)
    print(dataloader[0]['ext_label'].shape)
    

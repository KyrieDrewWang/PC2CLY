import torch
from models import pointnet_extrusion
from config.config_pc2ext import config
from dataset.dataset import get_dataloader
from losses import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from util.file_utils import ensure_dir, ensure_dirs
import numpy as np
import h5py

def vis_pc_seg(cfg, pc, pred_seg_label, data_id):

    output_dir = os.path.join(cfg.out_vis_dir, "seg")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    title = str(data_id) + '.ply'
    pc_for_vis = [[],[],[]]
    for p in pc:
        pc_for_vis[0].append(p[0])
        pc_for_vis[1].append(p[1])
        pc_for_vis[2].append(p[2])
        
    map_color = {0:'r',1:'g',2:'b',3:'',4:'y',5:'c',6:'m',7:'k'}
    Color = list(map(lambda x: map_color[x], pred_seg_label))
    fig=plt.figure(figsize=(10,10)) 
    ax = plt.subplot(111, projection='3d')
    ax.view_init(elev=30, azim=-60)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_zlim((-1, 1))
    ax.set_title(title) 
    ax.scatter(pc_for_vis[0], pc_for_vis[1], pc_for_vis[2], c = Color, marker = '.', alpha=1, s=20) 
    fig_path = os.path.join(output_dir, data_id + '.png')
    plt.savefig(fig_path, dpi=500, bbox_inches='tight')
    plt.show()

def vis_pc_bb(cfg, pc, pred_bb_label, data_id):

    output_dir = os.path.join(cfg.out_vis_dir, "bb")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    title = str(data_id) + '.ply'
    pc_for_vis = [[],[],[]]
    for p in pc:
        pc_for_vis[0].append(p[0])
        pc_for_vis[1].append(p[1])
        pc_for_vis[2].append(p[2])
        
    map_color = {0:'r',1:'g'}
    Color = list(map(lambda x: map_color[x], pred_bb_label))
    fig=plt.figure(figsize=(10,10)) 
    ax = plt.subplot(111, projection='3d')
    ax.view_init(elev=30, azim=-50)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_zlim((-1, 1))
    ax.set_title(title) 
    ax.scatter(pc_for_vis[0], pc_for_vis[1], pc_for_vis[2], c = Color, marker = '.', alpha=1, s=20) 
    fig_path = os.path.join(output_dir, data_id + '.png')
    plt.savefig(fig_path, dpi=500, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    cfg = config("validation")
    save_h5_dir = os.path.join(cfg.output_dir, "h5")
    ensure_dir(save_h5_dir)

    torch.manual_seed(1234)
    np.random.seed(0)
    device = torch.device('cuda')
    pred_sizes = [3, 16, 1]
    model = pointnet_extrusion.backbone(output_sizes=pred_sizes)

    pc_data_loader = get_dataloader(cfg, "test")
    ckpt = torch.load(cfg.pth_path)['model']
    # print(ckpt)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    
    bar = tqdm(pc_data_loader, desc="starting rendering inference pc")
    
    with torch.no_grad():
        for i, data in enumerate(bar):
            pc = data['pc'].cuda()
            ext_label = data['ext_label'].cuda()
            b,n,_ = pc.shape
            X, W_raw,_ = model(pc)  # normals, geo info M
            
            W_2k = torch.softmax(W_raw, dim=2)
            W_barrel = W_2k[:, :, ::2]
            W_base   = W_2k[:, :, 1::2]
            W        = W_base + W_barrel

            prediction_ext = torch.argmax(W, dim=-1)
            for j in range(len(prediction_ext)):
                file_id = data['id'][j]
                save_h5_path = os.path.join(save_h5_dir, file_id + '.h5')
                ensure_dir(os.path.dirname(save_h5_path))

                prediction_ext_j = prediction_ext[j].detach().cpu().numpy()
                ext_label_j = ext_label[j].detach().cpu().numpy()
                pc_j = pc[j].detach().cpu().numpy()

                with h5py.File(save_h5_path, 'w') as fp:
                    fp.create_dataset('pred_ext', data=prediction_ext_j, dtype=np.float32)
                    fp.create_dataset('label_ext', data=ext_label_j, dtype=np.float32)
                    fp.create_dataset('pc', data=pc_j, dtype=np.float32)

            
    bar.close()
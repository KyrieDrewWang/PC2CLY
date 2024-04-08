import torch
from models import pointnet_extrusion
from config.config import config as cfg
from dataset.pc_loader import get_dataloader
from losses import *
import matplotlib.pyplot as plt
from tqdm import tqdm


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
    torch.manual_seed(1234)
    np.random.seed(0)
    device = torch.device('cuda')
    pred_sizes = [3, 16]
    model = pointnet_extrusion.backbone(output_sizes=pred_sizes)
    pc_data_loader = get_dataloader(cfg.phase, cfg)
    model.load_state_dict(torch.load(cfg.pth_path)['model'])
    model.to(device)
    model.eval()
    
    bar = tqdm(pc_data_loader, desc="starting rendering inference pc")
    
    with torch.no_grad():
        for i, batch in enumerate(bar):
            pc_batch, norm_batch, data_id = batch
            batch_size = pc_batch.size()[0]
            # pc_batch = [pc.to(device) for pc in pc_batch]
            # norm_batch = [n.to(device) for n in norm_batch]
            # pc_batch = torch.stack(pc_batch)
            # norm_batch = torch.stack(norm_batch)
            pc_batch = pc_batch.to(device).to(torch.float)
            X, W_raw = model(pc_batch)  # normals, geo info M
            
            W_2k = torch.softmax(W_raw, dim=2)
            W_barrel = W_2k[:, :, ::2]
            W_base   = W_2k[:, :, 1::2]
            W        = W_base + W_barrel
            '''
            0 for barrel
            1 for base
            ''' 
            BB = torch.zeros(batch_size, cfg.n_points, 2).to(device)
            for j in range(cfg.K):
                BB[:,:,0] += W_2k[:, :, j*2]
                BB[:,:,1] += W_2k[:, :, j*2 + 1]
            
            pc = pc_batch.squeeze().to("cpu").detach().numpy()
            pred_bb_label = torch.argmax(BB, dim=-1).squeeze().to("cpu").detach().numpy()
            pred_seg_label = torch.argmax(W, dim=2).squeeze().to("cpu").detach().numpy()
            vis_pc_bb(cfg=cfg, pc=pc, pred_bb_label=pred_bb_label, data_id=data_id[0])
            vis_pc_seg(cfg=cfg, pc=pc, pred_seg_label=pred_seg_label, data_id=data_id[0])
    bar.close()
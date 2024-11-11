import sys
sys.path.append('.')
from config.config_pc2ext import config
from trainer.trainer_pc2ext import TrainAgentpc2ext, hard_W_encoding, compute_segmentation_iou
from dataset.dataset import get_dataloader
from utils.file_utils import cycle
from tqdm import tqdm
from collections import OrderedDict
import pdb
import os
import torch
from utils.loss_utils import compute_miou_loss, hungarian_matching
from utils.file_utils import ensure_dir, ensure_dirs
import h5py
import numpy as np
def main():

    # pdb.set_trace()

    cfg = config("validation")
    
    tr_agent = TrainAgentpc2ext(cfg=cfg)
    tr_agent.set_optimizer(config=cfg)
    tr_agent.load_ckpt(cfg.pth_path)
    tr_agent.net.eval()
    tr_agent.loss_func.eval()
    test_dataloader = get_dataloader(cfg, 'test')

    save_h5_dir = os.path.join(cfg.output_dir, "h5")
    ensure_dir(save_h5_dir)
    # pdb.set_trace()
    pbar = tqdm(test_dataloader)
    cnt = 0 
    mIoUs = []
    num_pc = 0
    num_true = 0
    for i, data in enumerate(pbar):
        with torch.no_grad():
            pc = data['pc'].cuda()
            b,n,_ = pc.shape
            outputs = tr_agent.net(pc)
            seg_pred = outputs['seg_logits']

            seg_pred_soft = torch.softmax(seg_pred, dim=-1)
            W_barrel = seg_pred_soft[:, :, ::2]
            W_base = seg_pred_soft[:, :, 1::2]

            prediction = W_barrel + W_base
            prediction = torch.softmax(prediction, dim=-1)
            ext_label = data['ext_label'].cuda()

            if cfg.with_label:
                prediction_ = hard_W_encoding(prediction, filter_null_instance=True)
                matching_indices, mask = hungarian_matching(prediction_, ext_label, with_mask=True)
                mask = mask.float()

                mIoU = compute_segmentation_iou(prediction_, ext_label, matching_indices, mask)
                mIoU = mIoU.detach().cpu().numpy()
                mIoUs.append(mIoU.mean())

                prediction_unmasked = torch.gather(prediction_, 2, matching_indices.unsqueeze(1).expand(b, n, cfg.n_max_instances))
                prediction_reordered = torch.where(mask.unsqueeze(1).expand(b, n, cfg.n_max_instances)==1, prediction_unmasked, torch.ones_like(prediction_unmasked)*-1.)
                prediction_ext = torch.argmax(prediction_reordered, dim=-1)
                acc_matrix = (prediction_ext == ext_label).reshape(-1)
                num_pc += len(acc_matrix)
                num_true += int(acc_matrix.sum())

            else:
                prediction_ext = torch.argmax(prediction, dim=-1)

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

        cnt+=len(prediction_ext)
        if cnt > cfg.n_samples_test:
            break
    if cfg.with_label:
        mIoU = np.mean(np.array(mIoUs))
        acc = num_true / num_pc
        save_txt_path = os.path.join(cfg.output_dir, "merits.txt")
        with open(save_txt_path, 'w') as f:
            f.write("mIoU: {} \n".format(str(mIoU)))
            f.write("acc: {} \n".format(str(acc)))




if __name__ == "__main__":

    main()
import os
import shutil
import sys
sys.path.append('.')
from util import ensure_dirs
import json


class config(object):
    
#-----------Frequently Modified Parameters-----------    
    exp_name = "pc_seg" # name of this experiment
    exp_desc = {
        "0":"base",
        "1":"add normal loss",
        "2":"cut operation is omitted",
        "3":"2 with finer settings",
        "4":"extrusion base barrel loss and ext axis loss"
    }
    
#----------Training Parameters----------
    batch_size = 16
    num_workers = 0
    nr_epochs = 400  # total number of epochs to train
    lr = 1e-4
    weight_decay = 1e-5
    lr_reduce_epoch = 100
    # beta1 = 0.5
    grad_clip = None  # initial learning rate
    save_frequency = 5  # save models every x epochs
    val_frequency = 100  # run validation every x steps
    eval_frequency = 5 # evaluation every x epochs
    gamma = 0.7 # step decay rate for lr
    warmup_step = 2000  # step size for learning rate warm up
    resume = False  # continue training from checkpoint
    seg_loss_weight=1.0
    normal_loss_weight=1.0
    barrel_loss_weight=1.0
    ext_axis_loss_weight = 1.0
    pth_path="results/Point2Cyl_DeepCAD/checkpoint.pth"


#-----------Data Settings-----------------
    n_points = 4096  # number of points for encoder
    pc_root    =     "data/deepcad/mesh_test2/cad_pc"  # directory of point clouds data folder
    ext_label_root = "data/deepcad/mesh_test2/seg_label"
    split_path =     "data/deepcad/mesh_test2/train_val_split_extrudenet.json" # path to train-val-test split
    proj_dir   =     "proj_log" # path to project folder where models and logs will be saved
    n_max_instances= 8
    n_samples_test = 2000
    with_label = False
    add_noise = False
    
#---------Model Parameters for Decoder--------
    n_layers = 4                # Number of Encoder blocks
    n_layers_decode = 4         # Number of Decoder blocks
    n_heads = 8                 # Transformer config: number of heads
    dim_feedforward = 512       # Transformer config: FF dimensionality
    d_model = 256               # Transformer config: model dimensionality
    dropout = 0.1                # Dropout rate used in basic layers and Transformers
    dim_z = 256                 # Latent vector dimensionality
    use_group_emb = True
    

    def _state_dict(self):
        a = {k: getattr(self, k) for k, _ in config.__dict__.items() if not k.startswith('_')}
        return a
    
    def __init__(self, phase="train"):
        self.exp_dir = os.path.join(self.proj_dir, self.exp_name)
        self.log_dir = os.path.join(self.exp_dir, 'log')
        self.model_dir = os.path.join(self.exp_dir, 'model')
        self.output_dir = os.path.join(self.exp_dir, 'output') 
        self.is_train = phase == "train"
        ensure_dirs([self.log_dir, self.model_dir, self.output_dir])


if __name__ == "__main__":
    cfg = config()
    print(cfg._state_dict())
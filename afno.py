

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from trainer import main_train

import argparse
import pandas as pd

"""

https://github.com/NVIDIA/modulus/blob/main/modulus/models/afno/afno.py#L406
Adaptive Fourier neural operator (AFNO) model.

    Note
    ----
    AFNO is a model that is designed for 2D images only.

    Parameters
    ----------
    inp_shape : List[int]
        Input image dimensions [height, width]
    in_channels : int
        Number of input channels
    out_channels: int
        Number of output channels
    patch_size : List[int], optional
        Size of image patches, by default [16, 16]
    embed_dim : int, optional
        Embedded channel size, by default 256
    depth : int, optional
        Number of AFNO layers, by default 4
    mlp_ratio : float, optional
        Ratio of layer MLP latent variable size to input feature size, by default 4.0
    drop_rate : float, optional
        Drop out rate in layer MLPs, by default 0.0
    num_blocks : int, optional
        Number of blocks in the block-diag frequency weight matrices, by default 16
    sparsity_threshold : float, optional
        Sparsity threshold (softshrink) of spectral features, by default 0.01
    hard_thresholding_fraction : float, optional
        Threshold for limiting number of modes used [0,1], by default 1

    Example
    -------
    >>> model = modulus.models.afno.AFNO(
    ...     inp_shape=[32, 32],
    ...     in_channels=2,
    ...     out_channels=1,
    ...     patch_size=(8, 8),
    ...     embed_dim=16,
    ...     depth=2,
    ...     num_blocks=2,
    ... )
    >>> input = torch.randn(32, 2, 32, 32) #(N, C, H, W)
    >>> output = model(input)
    >>> output.size()
    torch.Size([32, 1, 32, 32])

    Note
    ----
    Reference: Guibas, John, et al. "Adaptive fourier neural operators:
    Efficient token mixers for transformers." arXiv preprint arXiv:2111.13587 (2021).
    """



if __name__ == "__main__":
    
   
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--id', type=int, help='id of the parameters to use')

    # read from param_list.csv the column with the id
    args = parser.parse_args()
    id = args.id

    
    # read the parameters from the csv file
    #param_list = pd.read_csv("./fno_param_list.csv")
    #param = param_list.iloc[id]

    # # read the parameters id,lr,depth,embed_dim
    # depth =int(param["depth"])
    # embed_dim = int(param["embed_dim"])
    # resolution = int(param["resolution"])
       
 
    # print(f"ID: {id}, depth: {depth}, embed_dim: {embed_dim}")

    # path = "hyper_fno"
    # default_root_dir = f"{path}/{id}"

    
    # while os.path.exists(default_root_dir):
    #     id = id*100 +1

    #     default_root_dir = f"{path}/{id}"
        
    #     truth_value = os.path.exists(default_root_dir)
        
    #print("Training will be saved in",default_root_dir)
    default_root_dir = "afno"
    config_train = {"net":{"type": "afno",
                            "in_channels":2, # input channel dimension
                            "out_channels":2, # output channel dimension
                            "inp_shape":(64,64), # image shape EQUAL TO RESOLUTION
                            "patch_size":(8,8), # patch size
                            "embed_dim":512, # embedded channel size
                            "depth":8, # number of AFNO layers
                            "num_blocks":2,}, # number of blocks in the block-diag frequency weight matrices
                          
                    "max_epochs":10,
                    "patience":5,
                    "optimizer":{'lr':0.001, "cosine_annealing":True},
                    "batch_size":200,
                    "fast_dev_run":False, # True when debugging
                    "resolution":64, # Resolution of the Eucledian mesh (64x64)
                    "fine_tuning":False, # False during pre-training; True during fine-tuning: requires fine_tuning_path with the weights of the model
                   # "fine_tuning_path":"afno/lightning_logs/version_1/checkpoints/epoch=9-step=890.ckpt",
                    "default_root_dir":default_root_dir, # choose where to save your model
                    
                    "num_workers":39 # number of workers for the dataloader. Tune this based on htop command (spawned processes and IO load)

    }

    # clear cuda cache
    torch.cuda.empty_cache()

    main_train(config_train)


import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from trainer import main_train


if __name__ == "__main__":

    default_root_dir = "fno/fine_tuning"
    
    config_train = {"net": {"type":"fno",
                    "in_channels":2,
                    "out_channels":2,
                    "decoder_layers":2,
                    "decoder_layer_size":128,
                    "dimension":2,
                    "latent_channels":512,
                    "num_fno_layers":8,
                    "padding":0 },

                    "max_epochs":10,
                    "patience":5,
                    "optimizer":{'lr':0.0001, "cosine_annealing":True},
                    "batch_size":50,
                    "fast_dev_run":False,
                    "resolution":64,
                    "fine_tuning":False,
                    #"fine_tuning_path":"fno/lightning_logs/version_2/checkpoints/epoch=9-step=14530.ckpt",
                    "default_root_dir":default_root_dir,
                    
                    "num_workers":6

    }

    # clear cuda cache
    torch.cuda.empty_cache()

    main_train(config_train)

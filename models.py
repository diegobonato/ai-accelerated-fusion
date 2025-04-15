import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import lightning.pytorch as pl
import matplotlib.pyplot as plt

from modulus.models.afno.afno import AFNO
from modulus.models.fno.fno import FNO

import numpy as np

########################################################################################
# General LightningModule
########################################################################################


# Define a general LightningModule (nn.Module subclass)
class LitNet(pl.LightningModule):
    
    def __init__(self, config_train):
       
        super().__init__()
        
        print('Network initialized')

        self.save_hyperparameters(config_train)

        if config_train['net']['type'] == 'fno':
            self.net = FNO(in_channels=config_train['net']['in_channels'],
                            out_channels=config_train['net']['out_channels'],
                            decoder_layers=config_train['net']['decoder_layers'],
                            decoder_layer_size=config_train['net']['decoder_layer_size'],
                            dimension=config_train['net']['dimension'],
                            latent_channels=config_train['net']['latent_channels'],
                            num_fno_layers=config_train['net']['num_fno_layers'],
                            padding=config_train['net']['padding'])




        elif config_train['net']['type'] == 'afno':
            self.net = AFNO(in_channels=config_train['net']['in_channels'],
                            out_channels=config_train['net']['out_channels'],
                            inp_shape=config_train['net']['inp_shape'],
                            patch_size=config_train['net']['patch_size'],
                            embed_dim=config_train['net']['embed_dim'],
                            depth=config_train['net']['depth'],
                            #mlp_ratio=config_train['net']['mlp_ratio'],
                            #drop_rate=config_train['net']['drop_rate'],
                            num_blocks=config_train['net']['num_blocks'],
                            #sparsity_threshold=config_train['net']['sparsity_threshold'],
                            #hard_thresholding_fraction=config_train['net']['hard_thresholding_fraction'])
            )
        
        else:              
            raise ValueError("Unknown network type")
        

    def forward(self,x):
        return self.net(x)
    
    def configure_optimizers(self):

        optimizer = Adam(self.net.parameters(), lr=self.hparams.optimizer['lr'])
        # Configure scheduler
        if self.hparams.optimizer['cosine_annealing']:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
            return [optimizer], [scheduler]
        

    def training_step(self, sample, batch_idx=None):
        # training_step defines the train loop. It is independent of forward
        #folders, curnz, forcn, jouln, magne = sample
        folders, magne = sample

        if len(folders) == 1:
            # I don't care about this simulation
            return torch.tensor(0.0, requires_grad=True)

        index_changes = self.find_index_where_sim_changes(folders)

        # I already know that I care about magne only
        #batch = torch.concatenate([curnz, forcn, jouln, magne],dim=1)
        batch = torch.concatenate([magne],dim=1)[:,:2] # shape (batch,2,res,res) (Hx,Hy)
        
        # Create input and target
        input, target = self.create_input_target(batch, index_changes)  

        # if batch was [0,1,2,3,4,5,6,7,8,9]
        # input will be [0,1,2,3,4,5,6,7,8]
        # target will be [1,2,3,4,5,6,7,8,9]
        # this is OK for pre-training
        # for fine-tuning: input will be [0,1,2,3,4,5,6,7], target1 will be [1,2,3,4,5,6,7,8]
        # target2 will be [2,3,4,5,6,7,8,9]

        if not self.hparams.fine_tuning:
            out = self.net(input)
            loss = F.mse_loss(out, target)
            self.log("train_loss", loss.item(), prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
            return loss
        else:
            # fine tuning
            # I need to create two targets
            # target1 will be [1,2,3,4,5,6,7,8]
            # target2 will be [2,3,4,5,6,7,8,9]
            input = input[:-1]
            target1 = target[:-1]
            target2 = target[1:]
            out = self.net(input)
            loss1 = F.mse_loss(out, target1)
            loss2 = F.mse_loss(out, target2)
            loss = loss1 + loss2
            self.log("train_loss", loss.item(), prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
            return loss


    

    def validation_step(self, sample, batch_idx=None):
        #folders, curnz, forcn, jouln, magne = sample
        folders, magne = sample
        if len(folders) == 1:
            # I don't care about this simulation
            return torch.tensor(0.0, requires_grad=True)

        index_changes = self.find_index_where_sim_changes(folders)

        # I already know that I care about magne only
        #batch = torch.concatenate([curnz, forcn, jouln, magne],dim=1)
        batch = torch.concatenate([magne],dim=1)


        # Create input and target
        input, target = self.create_input_target(batch, index_changes)  

        # if batch was [0,1,2,3,4,5,6,7,8,9]
        # input will be [0,1,2,3,4,5,6,7,8]
        # target will be [1,2,3,4,5,6,7,8,9]
        # this is OK for pre-training
        # for fine-tuning: input will be [0,1,2,3,4,5,6,7], target1 will be [1,2,3,4,5,6,7,8]
        # target2 will be [2,3,4,5,6,7,8,9]

        if not self.hparams.fine_tuning:
            out = self.net(input)
            loss = F.mse_loss(out, target)
            self.log("val_loss", loss.item(), prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
            return loss
        else:
            # fine tuning
            # I need to create two targets
            # target1 will be [1,2,3,4,5,6,7,8]
            # target2 will be [2,3,4,5,6,7,8,9]
            input = input[:-1]
            target1 = target[:-1]
            target2 = target[1:]
            out = self.net(input)
            loss1 = F.mse_loss(out, target1)
            loss2 = F.mse_loss(out, target2)
            loss = loss1 + loss2
            self.log("val_loss", loss.item(), prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
            return loss
    
    
    def test_step(self, sample, batch_idx):
           # training_step defines the train loop. It is independent of forward
        folders, curnz, forcn, jouln, magne = sample

        if len(folders) == 1:
            # I don't care about this simulation
            return None

        index_changes = self.find_index_where_sim_changes(folders)

        # I already know that I care about magne only
        #batch = torch.concatenate([curnz, forcn, jouln, magne],dim=1)
        batch = torch.concatenate([magne],dim=1) # shape (batch,3,res,res)
        
        # Create input and target
        input, target = self.create_input_target(batch, index_changes)  

        # if batch was [0,1,2,3,4,5,6,7,8,9]
        # input will be [0,1,2,3,4,5,6,7,8]
        # target will be [1,2,3,4,5,6,7,8,9]
        # this is OK for pre-training
        # for fine-tuning: input will be [0,1,2,3,4,5,6,7], target1 will be [1,2,3,4,5,6,7,8]
        # target2 will be [2,3,4,5,6,7,8,9]

        if not self.hparams.fine_tuning:
            out = self.net(input)
            loss = F.mse_loss(out, target)
            self.log("test_loss", loss.item(), prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
            return loss
        else:
            # fine tuning
            # I need to create two targets
            # target1 will be [1,2,3,4,5,6,7,8]
            # target2 will be [2,3,4,5,6,7,8,9]
            input = input[:-1]
            target1 = target[:-1]
            target2 = target[1:]
            out = self.net(input)
            loss1 = F.mse_loss(out, target1)
            loss2 = F.mse_loss(out, target2)
            loss = loss1 + loss2
            self.log("test_loss", loss.item(), prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
            return loss
   
       
    


    def on_validation_epoch_end(self):

        # Not implemented yet
        return None
        
        print("BATCH INDEX IS ",self.batch_idx)
        
        # for every tstep create a new figure
        # every row will be a channel
        # columns: target, prediction, difference
        fig,axs = plt.subplots(11,3,figsize=(15,15))


        names = ["Hx","Hy","Hz","Jz","phi_x","phi_y","phi_z","F_x","F_y","F_z","Jle"]
        # Looping on channels
        for i in range(11):
            axs[i,0].imshow(self.target[i])
            axs[i,0].set_title(f"Target {names[i]}")
            axs[i,1].imshow(self.out[i])
            axs[i,1].set_title(f"Prediction {names[i]}")
            axs[i,2].imshow(self.target[i]-self.out[i])
            axs[i,2].set_title(f"Difference {names[i]}")
        for ax in axs.flatten():
            ax.axis('off')
            plt.tight_layout()

        # log on tensorboard
        self.logger.experiment.add_figure('Net simulations', fig, self.current_epoch)
        plt.close(fig)
   
    def find_index_where_sim_changes(self,folders):
        
        sim_number = [folder.split("/")[0] for folder in folders]
        index_changes = []
        for i in range(1,len(sim_number)):
            if sim_number[i] != sim_number[i-1]:
                # hai trovato where sim changes
                index_changes.append(i)

        return index_changes
    
    def create_input_target(self,batch, index_changes):
        all_indexes = [0] + index_changes + [batch.shape[0]]

        # Create the input by shifting to the right/left
        all_inputs = []
        all_targets = []
        for i in range(1,len(all_indexes)):
            one_simulation = batch[all_indexes[i-1]:all_indexes[i]]
            
            input_batch = one_simulation[:-1]
            target_batch = one_simulation[1:]
            
            all_inputs.append(input_batch)
            all_targets.append(target_batch)
            
        final_input = torch.concatenate(all_inputs)
        final_target = torch.concatenate(all_targets)
            
        return final_input, final_target

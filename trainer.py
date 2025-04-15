import lightning.pytorch as pl
from models import LitNet
from dataset import WebDataset
import webdataset as wds
from torch.utils.data import DataLoader
import numpy as np
import torch
import os
#Function for main training of the network
def main_train(config_train):
    
    
    # Define the model
    net = config_train['net']
    pl.seed_everything(0)

    # Set the device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        # release all unoccupied cached memory
        torch.cuda.empty_cache()
        # printo GPU info
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        #print('{} {} GPU available'.format(str(device_count), str(device_name)))

    
    print("\n Using number of GPUs: ", device_count)
    # Define the EarlyStopping callback
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',  # Monitor the validation loss
        min_delta=0.01,     # Minimum change in the monitored metric
        patience=config_train['patience'],          # Number of epochs with no improvement after which training will be stopped
        verbose=True,
        mode='min'           # Mode: 'min' if you want to minimize the monitored quantity (e.g., loss)
    )

    # Set the trainer's device to GPU if available
    trainer = pl.Trainer(
    max_epochs=config_train['max_epochs'],
    check_val_every_n_epoch=1,
    log_every_n_steps=50,
    deterministic=True,
    callbacks=[early_stop_callback],
    fast_dev_run=config_train['fast_dev_run'],
    accelerator="gpu",
    devices=device_count,
    default_root_dir=config_train["default_root_dir"]

  )

    # train val test split
    url_train, url_val, url_test = train_test_split()

    # Initialise the datasets

    # add the function that selects only thos who have more than 500 tsteps
    # add functions that takes one tstep every fifty

    train_dataset = wds.DataPipeline(
        wds.SimpleShardList(url_train),
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.select(take_every_50th),
        #wds.to_tuple("__key__", "curnz", "forcn", "jouln", "magne"),
        wds.to_tuple("__key__", "magne"),
        wds.map_tuple(None, WebDataset(config_train).preprocess_data),  # Apply only to text files
       
        #wds.map_tuple(None, WebDataset(config_train).preprocess_data, WebDataset(config_train).preprocess_data, WebDataset(config_train).preprocess_data, WebDataset(config_train).preprocess_data),  # Apply only to text files
        wds.batched(config_train["batch_size"])
    )

    val_dataset = wds.DataPipeline(
        wds.SimpleShardList(url_val),
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.select(take_every_50th),
        #wds.to_tuple("__key__", "curnz", "forcn", "jouln", "magne"),
        wds.to_tuple("__key__", "magne"),
        wds.map_tuple(None, WebDataset(config_train).preprocess_data),  # Apply only to text files
       
        #wds.map_tuple(None, WebDataset(config_train).preprocess_data, WebDataset(config_train).preprocess_data, WebDataset(config_train).preprocess_data, WebDataset(config_train).preprocess_data),  # Apply only to text files
        wds.batched(config_train["batch_size"])
    )


    # test_dataset = (
    #     wds.WebDataset(test_url, shardshuffle=False)
    #     .to_tuple("__key__", "curnz", "forcn", "jouln", "magne")
    #     .map_tuple(None, preprocess_data, preprocess_data, preprocess_data, preprocess_data)  # Apply only to text files
    #     .batched(10)
    # )

    # If fine tuning, load the model; otherwise, train the model
    if config_train['fine_tuning']:
        model = LitNet.load_from_checkpoint(config_train['fine_tuning_path'],config_train)
    else:
        model = LitNet(config_train)

      # Create the dataloaders
    train_dataloader = DataLoader(train_dataset, 
                        batch_size=None, 
                        num_workers=config_train["num_workers"],
                        pin_memory=True)
    
    val_dataloader = DataLoader(val_dataset,
                        batch_size=None,
                        num_workers=config_train["num_workers"],
                        pin_memory=True)
    

    trainer.fit(model, train_dataloader, val_dataloader)
    #trainer.test(model=model,dataloaders=test_dataloader,verbose=True)


    # Add all testing

    return model


def train_test_split():
    # fix seed 
    seed = 42
    np.random.seed(seed)

    # possible indexes
    webdataset_path = "../create_dataset/web_dataset/"
    url = [os.path.join(webdataset_path,tar) for tar in os.listdir(webdataset_path) if tar.endswith("_new.tar")]

    # shuffle indexes
    np.random.shuffle(url)

    # split indexes
    train_url = url[:int(0.8*len(url))]
    val_url = url[int(0.8*len(url)):int(0.9*len(url))]
    test_url = url[int(0.9*len(url)):]
    return train_url, val_url, test_url

sim_with_more_than_500_steps = np.load("sim_with_more_than_500_tsteps.npy")

def take_every_50th(sample):
    if int(sample["__key__"].split("/")[0]) not in sim_with_more_than_500_steps:
        return False
    else:
        # take the first tstep and every 50th
        if (int(sample["__key__"][-6:]) == 1) or (int(sample["__key__"][-6:]) % 50 == 0):
            return True
        else:
            return False
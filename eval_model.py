import numpy as np
import matplotlib.pyplot as plt
import torch
from modulus.models.fno.fno import FNO
from models import LitNet

import ensi_plot as ensi

from scipy.interpolate import griddata
from torch.utils.data import DataLoader
from dataset import WireDeep

"""
Some messy functions to plot the results of the model evolved autoregressively.

USE THE JUPYTER NOTEBOOK INSTEAD (plot_autoreg.ipynb)

"""


def interpolate_on_eucledian_grid(field_to_interpolate,coordinates):
    resolution = 64
    """
    Interpolate the data on a Euclidean grid
    """
    
    """
    Defines a structured grid and interpolates the magnetic field values on it.
    Uses nearest method because I get NaNs otherwise.
    Mask values outside the wire.
    """

    # check that magnetic field is (n_points,)  Give just one components. Treat it as a scalar field
    assert field_to_interpolate.ndim == 1, "Field to interpolate must have shape (n_points,) Give just one component. Treat it as a scalar field"

    # Define the structured grid (Euclidean grid)
    grid_x, grid_y = np.meshgrid(np.linspace(np.min(coordinates[:,0]), np.max(coordinates[:,0]), resolution),
                                np.linspace(np.min(coordinates[:,1]), np.max(coordinates[:,1]), resolution))

    # Interpolate the values from the unstructured grid to the structured grid
    grid_values = griddata(coordinates, field_to_interpolate, (grid_x, grid_y), method='nearest')
    mask = np.sqrt((grid_x-coordinates.mean())**2 + (grid_y-coordinates.mean())**2) > coordinates.max()
    grid_values[mask] = 0.0

    return grid_values


def spatial_plot_only():
    path = "lightning_logs/time_train/checkpoints/epoch=27-step=224.ckpt"

    net = FNO(in_channels=10,
                out_channels=10,
                decoder_layers=2,
                decoder_layer_size=32,
                dimension=2,
                latent_channels=10,
                num_fno_layers=4,
                padding=0,
                )
    
    # Load the model
    net = LitNet(net)

    net.load_state_dict(torch.load(path)["state_dict"])
    net.eval()

    # Load the data
    ensi_path = "./ensi_files/"
    file_to_plot = "MAGNE"

    tsteps_to_plot = np.arange(1,999,100)
    fig,axs = plt.subplots(len(tsteps_to_plot),3,figsize=(15,5*len(tsteps_to_plot)))

    for i,idx in enumerate(tsteps_to_plot):
        file_path = [ensi_path+f'TS-WIRE-TRI03.ensi.{file_to_plot}-{i:06}' for i in [idx,idx+1]]

        _, data_2d_array = ensi.load_multiple_timesteps(file_path,file_to_plot=file_to_plot,skip_lines=4)

        coordinates = ensi.coordinate_reader()
        data_2d_array = (data_2d_array - data_2d_array.min())/(data_2d_array.max() - data_2d_array.min())

        magnetic_field = data_2d_array.reshape(data_2d_array.shape[0],3,coordinates.shape[0]).transpose(0,2,1)

        structured_grid_input  = interpolate_on_eucledian_grid(field_to_interpolate=magnetic_field[0,:,0],coordinates=coordinates) # Working on the first component of the magnetic field for the moment
        structured_grid_output = interpolate_on_eucledian_grid(field_to_interpolate=magnetic_field[1,:,0],coordinates=coordinates) # Working on the first component of the magnetic field for the moment

        # Add a channel dimension   
        structured_grid_input = np.expand_dims(structured_grid_input, axis=0) # Containes the scalar field values in an array
        structured_grid_output = np.expand_dims(structured_grid_output, axis=0) # Containes the scalar field values in an array

        # Create torch tensor and move to device (GPU if available)
        # Output using dataloader is (batch_size,channels,width,height)
        input, _ = torch.tensor(structured_grid_input,device="cpu",dtype=torch.float32), torch.tensor(structured_grid_output,device="cpu",dtype=torch.float32)


        output = net(input.unsqueeze(0))
        vmin = structured_grid_output[0].min()
        vmax = structured_grid_output[0].max()

        # Plotting target, prediction and difference
        axs[i,0].imshow(structured_grid_output[0],vmin=vmin,vmax=vmax)
        axs[i,0].set_title("Target")
        axs[i,1].imshow(output[0,0].detach().numpy(),vmin=vmin,vmax=vmax)
        axs[i,1].set_title("Prediction")
        axs[i,2].imshow(structured_grid_output[0]-output[0,0].detach().numpy()) # Not using vmin and vmax for difference
        axs[i,2].set_title("Difference")

        # Colorbar
        fig.colorbar(axs[i,0].imshow(structured_grid_output[0],vmin=vmin,vmax=vmax), ax=axs[i,0])
        fig.colorbar(axs[i,1].imshow(output[0,0].detach().numpy(),vmin=vmin,vmax=vmax), ax=axs[i,1])
        fig.colorbar(axs[i,2].imshow(structured_grid_output[0]-output[0,0].detach().numpy()), ax=axs[i,2])
    
    # Save figure
    plt.savefig("comparison_100.png")

def autoregressive():
    path = "lightning_logs/autoreg_channels/checkpoints/epoch=59-step=480.ckpt"

    net = FNO(in_channels=10,
                out_channels=10,
                decoder_layers=2,
                decoder_layer_size=32,
                dimension=2,
                latent_channels=32,
                num_fno_layers=4,
                padding=0,
                )

    # Load the model
    net = LitNet(net)

    net.load_state_dict(torch.load(path)["state_dict"])
    net.eval()

    # Load the data
    ensi_path = "./ensi_files/"
    file_to_plot = "MAGNE"

    # Load the data
    file_path = [ensi_path+f'TS-WIRE-TRI03.ensi.{file_to_plot}-{i:06}' for i in range(1,11)]
    _, data_2d_array = ensi.load_multiple_timesteps(file_path,file_to_plot=file_to_plot,skip_lines=4)

    coordinates = ensi.coordinate_reader()
    data_2d_array = (data_2d_array - data_2d_array.min())/(data_2d_array.max() - data_2d_array.min())

    magnetic_field = data_2d_array.reshape(data_2d_array.shape[0],3,coordinates.shape[0]).transpose(0,2,1)
    structured_grid_input  = np.array([interpolate_on_eucledian_grid(field_to_interpolate=magnetic_field[i,:,0],coordinates=coordinates) for i in range(magnetic_field.shape[0])])
    

    # Create torch tensor and move to device (GPU if available)
    # Output using dataloader is (batch_size,channels,width,height)
    input = torch.tensor(structured_grid_input,dtype=torch.float32)
    input = input.unsqueeze(0) # Add batch dimension
    outputs = []
    # Autoregressive prediction
    # Move everything to GPU
    input = input.to("cuda")
    net = net.to("cuda")

    for i in range(1000//10): # 1000 timesteps total, 10 timesteps per prediction
        output = net(input)
        
        outputs.append(output.detach().cpu())
        input = output
    
   # outputs = ([output.detach().cpu() for output in outputs])
    outputs = torch.cat(outputs,dim=1).squeeze(0) # shape is (1,1000,64,64), I squeeze it to (1000,64,64)
   
    return outputs
    

def plot_autoregressive():
    outputs = autoregressive() # shape is (tsteps,1,64,64) where tsteps is 997

    # Load the data
    ensi_path = "./ensi_files/"
    file_to_plot = "MAGNE"

    tsteps_to_plot = np.arange(2,999,100)
    fig,axs = plt.subplots(len(tsteps_to_plot),3,figsize=(15,5*len(tsteps_to_plot)))

        
    file_path = [ensi_path+f'TS-WIRE-TRI03.ensi.{file_to_plot}-{i:06}' for i in tsteps_to_plot]

    _, data_2d_array = ensi.load_multiple_timesteps(file_path,file_to_plot=file_to_plot,skip_lines=4)

    coordinates = ensi.coordinate_reader()
    data_2d_array = (data_2d_array - data_2d_array.min())/(data_2d_array.max() - data_2d_array.min())

    magnetic_field = data_2d_array.reshape(data_2d_array.shape[0],3,coordinates.shape[0]).transpose(0,2,1)
    vmin = magnetic_field.min()
    vmax = magnetic_field.max()

    #outputs = outputs[::len(tsteps_to_plot)]
    for i in range(len(tsteps_to_plot)):

        structured_grid_output = interpolate_on_eucledian_grid(field_to_interpolate=magnetic_field[i,:,0],coordinates=coordinates) # Working on the first component of the magnetic field for the moment


        # Plotting target, prediction and difference
        axs[i,0].imshow(structured_grid_output,vmin=vmin,vmax=vmax)
        axs[i,0].set_title("Target")
        axs[i,1].imshow(outputs[i][0,0].detach().numpy(),vmin=vmin,vmax=vmax)
        axs[i,1].set_title("Prediction")
        axs[i,2].imshow(structured_grid_output-outputs[i][0,0].detach().numpy()) # Not using vmin and vmax for difference
        axs[i,2].set_title("Difference")

        # Colorbar
        fig.colorbar(axs[i,0].imshow(structured_grid_output,vmin=vmin,vmax=vmax), ax=axs[i,0])
        fig.colorbar(axs[i,1].imshow(outputs[i][0,0].detach().numpy(),vmin=vmin,vmax=vmax), ax=axs[i,1])
        fig.colorbar(axs[i,2].imshow(structured_grid_output-outputs[i][0,0].detach().numpy()), ax=axs[i,2])

        
    # Save figure
    plt.savefig("autoregressive_100.png")

def plot_autoregressive_series():
    outputs = autoregressive() # shape is (tsteps,1,64,64) where tsteps is 997

    fig,axs = plt.subplots(20,5,figsize=(50,50))

    # output shape is (1,1000,64,64), where 1 is the batch dimension
    #outputs = outputs[::10]
   
    axs = axs.flatten()
   
    for i in range(100,200):
        axs[i].imshow(outputs[i])
        axs[i].set_title(f"timestep {i}")
        fig.colorbar(axs[i].imshow(outputs[i]), ax=axs[i])
    plt.savefig("autoregressive_series.pdf")

    return None

def plot_time_eval():
    
    path = "lightning_logs/train_with_curr_and_magne/checkpoints/epoch=17-step=4050.ckpt"
    net  = FNO(in_channels=4,
                out_channels=4,
                decoder_layers=2,
                decoder_layer_size=16,
                dimension=2,
                latent_channels=16,
                num_fno_layers=4,
                padding=0,
                )

    # Load the model
    net = LitNet(net)

    net.load_state_dict(torch.load(path)["state_dict"])
    net.eval()

    initial_tstep = 1 # this will be changed to np.random.randint(1,1000) for example
    final_tstep = 1000
    step = 100
    file_to_plot = "MAGNE"
    ensi_path = "./ensi_files/"
    file_path = [ensi_path+f'TS-WIRE-TRI03.ensi.{file_to_plot}-{i:06}' for i in np.arange(start=initial_tstep,stop=final_tstep,step=step)] 

    # Load data across all timesteps
    _, data_2d_array = ensi.load_multiple_timesteps(file_path,file_to_plot=file_to_plot,skip_lines=4)

    # Normalize data
    coordinates = ensi.coordinate_reader()
    data_2d_array = (data_2d_array - data_2d_array.min())/(data_2d_array.max() - data_2d_array.min())


    # Reshape data
    if file_to_plot=="MAGNE":
        magnetic_field = data_2d_array.reshape(data_2d_array.shape[0],3,coordinates.shape[0]).transpose(0,2,1)
        # # self.magnetic_field.shape -> (tsteps, points, components) e.g. (tin+tout steps (the total length of the time window considered),4525,3)

    #structured_grid_input = np.array([interpolate_on_eucledian_grid(magnetic_field[0,:,component],coordinates)  for component in range(magnetic_field.shape[2])])
    #structured_grid_output = np.array([interpolate_on_eucledian_grid(magnetic_field[0,:,component],coordinates) for component in range(magnetic_field.shape[2])])
    
    # Using list in a clever way to create channels and batches
    structured_grid_input = np.concatenate([
        [
            [interpolate_on_eucledian_grid(magnetic_field[tstep,:,component],coordinates)  for component in range(magnetic_field.shape[2])]
        ]for tstep in range(magnetic_field.shape[0])
    ])
    
    structured_grid_output = np.concatenate([
        [
            [interpolate_on_eucledian_grid(magnetic_field[tstep,:,component],coordinates) for component in range(magnetic_field.shape[2])]
        ]for tstep in range(magnetic_field.shape[0])
    ])

    # make tensors 
    input_tensor = torch.tensor(structured_grid_input,dtype=torch.float32)
    print(input_tensor.shape)
    raise ValueError
    # Evaluate
    output = net(input_tensor.unsqueeze(0)).detach().numpy()

    # Plot target prediction and difference for all timesteps
    # Plotting target, prediction and difference
    for component in range(magnetic_field.shape[2]):
        fig,axs = plt.subplots(len(structured_grid_output),3,figsize=(15,5*len(structured_grid_output)))
        vmin, vmax = structured_grid_output.min(), structured_grid_output.max()
        for i in range(len(structured_grid_output)):
            axs[i,0].imshow(structured_grid_output[i],vmin=vmin,vmax=vmax)
            axs[i,0].set_title("Target")
            axs[i,1].imshow(output[i],vmin=vmin,vmax=vmax)
            axs[i,1].set_title("Prediction")
            axs[i,2].imshow(structured_grid_output[i]-output[i])
            axs[i,2].set_title("Difference")

        plt.savefig(f"time_eval_{component}.png")





def use_get_item_to_plot_stuff_easily():

    path = "lightning_logs/all_channels/checkpoints/epoch=17-step=4050.ckpt"
    
    config_train = {"net":FNO(in_channels=11,
                                out_channels=11,
                                decoder_layers=2,
                                decoder_layer_size=16,
                                dimension=2,
                                latent_channels=32,
                                num_fno_layers=4,
                                padding=0,
                                ),
                    "max_epochs":5000,
                    "patience":5,
                    "optimizer":{'lr':0.001},
                    "batch_size":1,
                    "fast_dev_run":False,
                    "resolution":64,
                    "step":100, # Predict t-step in the future
                    "t_input":500, #input ranges from 0 to 500, step 50 -> 10 steps in total
                    "t_output":500, # output ranges from 0 to 500, step 50 -> 10 steps in total
                    "fno_type":"Full Magnet",
                    "files_to_plot": ["MAGNE","CURNZ","FLUXN","FORCN","JOULN"]

    }

    # Load the model
    net = LitNet(config_train["net"])

    net.load_state_dict(torch.load(path)["state_dict"])
    net.eval()

    dataset = WireDeep(step=config_train["step"],fno_type=config_train["fno_type"],t_input=config_train["t_input"], t_output=config_train["t_output"], resolution=config_train["resolution"],file_to_plot=config_train["files_to_plot"],)

    for idx in range(0,900,100):
        # for every tstep create a new figure
        # every row will be a channel
        # columns: target, prediction, difference
        fig,axs = plt.subplots(11,3,figsize=(15,15))

        input, target = dataset.__getitem__(idx)
        # add fictitious batch dimension
        input = input.unsqueeze(0)
        out = net(input).squeeze(0).detach().numpy()
        
        # out shape (4,64,64)
        # target shape (4,64,64)
        names = ["Hx","Hy","Hz","Jz","phi_x","phi_y","phi_z","F_x","F_y","F_z","Jle"]
        for i in range(11):
            axs[i,0].imshow(target[i])
            axs[i,0].set_title(f"Target {names[i]}")
            axs[i,1].imshow(out[i])
            axs[i,1].set_title(f"Prediction {names[i]}")
            axs[i,2].imshow(target[i]-out[i])
            axs[i,2].set_title(f"Difference {names[i]}")
        for ax in axs.flatten():
            ax.axis('off')
            plt.tight_layout()

        plt.savefig(f"get_item_{idx}.png")
        
        





if __name__ == "__main__":
    #plot_autoregressive()
    #plot_time_eval()
    #plot_autoregressive_series()
    use_get_item_to_plot_stuff_easily()
    print("Done")
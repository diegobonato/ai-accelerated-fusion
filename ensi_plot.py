import numpy as np
import matplotlib.pyplot as plt

import matplotlib.tri as tri

import numpy as np
import os
import re
from typing import List
import os

# Function to load data from a single file, skipping initial metadata lines
def load_data(file_path, skip_lines=5):
    # Load data into a numpy array after skipping the metadata lines
    data = np.loadtxt(file_path, skiprows=skip_lines)
    return data


# Function to load data from multiple files and organize by timestep
def load_multiple_timesteps(file_paths: List[str], file_to_plot,skip_lines=4):
    # Dictionary to store data with timestep as the key
    timestep_data = {}

    # Loop through each file, extract timestep, and load data
    for file_path in file_paths:
        # Extract timestep from filename using regex
        match = re.search(rf'{file_to_plot}-(\d+)', file_path)
        if match:
            timestep = int(match.group(1))  # Convert timestep to integer
            # Load data for this timestep and add to dictionary
            timestep_data[timestep] = load_data(file_path, skip_lines=skip_lines)

    # Sort timesteps and create a 2D array with time as the first dimension
    sorted_timesteps = sorted(timestep_data.keys())
    data_array = np.array([timestep_data[t] for t in sorted_timesteps])

    return sorted_timesteps, data_array  # Returning both timesteps and data array for clarity

def coordinate_reader(file_path="TS-WIRE-TRI03/"):
      # Leggi tutte le righe del file
    with open(file_path+"alya_coordinates.dat", "r") as file:
        lines = file.readlines()

    # Read all lines but the first and last one. Skip the first column too (the index)
    coordinates = np.loadtxt(lines[:-1], skiprows=1,usecols=(1,2))
    return coordinates



def triangle_mesh_reader(file_path="TS-WIRE-TRI03/",coordinates=None):

    if coordinates is None:
        raise ValueError("Coordinates must be provided")

    with open(file_path+'alya_elements.dat', 'r') as file:
        lines = file.readlines()

    triangles = np.loadtxt(lines[:-1], skiprows=1,usecols=(1,2,3),dtype=int)

    # Assuming you have:
    # - `points` as an (N, 2) array for the coordinates [x, y] of each point
    # - `triangles` as an (M, 3) array for the triangle definitions (indices of points in each triangle)
    # - `scalar_field` as an (N,) array containing scalar values at each point

    # Ensure triangles use 0-based indexing
    if triangles.min() == 1:
        triangles -= 1  # Convert to 0-based indexing

    triangulation = tri.Triangulation(coordinates[:, 0], coordinates[:, 1],triangles)

    return triangulation

def plot_one_tstep_on_triangular_mesh(triangulation, magnetic_field, tstep_to_plot):
    """
    Plot ONE TSTEP the scalar field on the TRIANGULAR MESH
    
    Scatter plot of coordinates. The color of the points is determined by the value of the first column of data_2d_array
    """
    # check that magnetic_field is 3D
    assert magnetic_field.ndim == 3, f"Expected 3D array, got {magnetic_field.ndim}D"
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    comp_names=['x','y','z']

    # set vmin vmax
    vmin = magnetic_field.min()
    vmax = magnetic_field.max()
    for i in range(3):
        axs[i].tricontourf(triangulation, magnetic_field[tstep_to_plot,:,i], cmap='viridis',vmin=vmin, vmax=vmax)
        axs[i].set_aspect('equal')
        axs[i].set_title(f'Magnetic field component {comp_names[i]}')
        axs[i].set_xlabel('x')
        axs[i].set_ylabel('y')
        # less labels on the x axis
        axs[i].locator_params(axis='x', nbins=4)
        # less labels on the y axis
        axs[i].locator_params(axis='y', nbins=4)
        # colorbar
        fig.colorbar(axs[i].collections[0], ax=axs[i], orientation='horizontal')
    plt.tight_layout()
    plt.savefig(f"magnetic_field_{tstep_to_plot}.pdf")

def plot_multiple_tstep_mesh(triangulation,magnetic_field,sorted_timesteps):
    fig, axs = plt.subplots(10, 3, figsize=(15, 30))
    comp_names=['x','y','z']
    for i in range(10):
        for j in range(3):
            axs[i,j].tricontourf(triangulation, magnetic_field[i,:,j], cmap='viridis')
            #axs[i,j].set_aspect('equal')
            axs[i,j].set_title(f'Magnetic field component {comp_names[j]} at timestep {sorted_timesteps[i]}')
            axs[i,j].set_xlabel('x')
            axs[i,j].set_ylabel('y')
            # less labels on the x axis
            axs[i,j].locator_params(axis='x', nbins=4)
            # less labels on the y axis
            axs[i,j].locator_params(axis='y', nbins=4)
            # colorbar
            fig.colorbar(axs[i,j].collections[0], ax=axs[i,j], orientation='horizontal')

    plt.tight_layout()  
    plt.savefig(f"magnetic_field_multiple_timesteps_MESH.pdf")


def magnetic_field_plot_multiple_tstep(magnetic_field,coordinates,sorted_timesteps,path_to_save):

    # check that coordinates is x,y only
    if coordinates.shape[1]!=2:
        raise ValueError(f"Coordinates must be 2D, got {coordinates.shape}")
    
    # Create a figure with subplots for each component
    fig, axs = plt.subplots(magnetic_field.shape[0], 3, figsize=(15, 30))
    comp_names = ['x', 'y', 'z']

    # Determine the global color range
    vmin = magnetic_field.min()
    vmax = magnetic_field.max()

    # Plot each component in the subplots with the same color range
    for i in range(magnetic_field.shape[0]):
        for j in range(3):
            sc = axs[i, j].scatter(
                coordinates[:, 0], coordinates[:, 1], 
                c=magnetic_field[i, :, j], cmap='viridis', s=10, 
                vmin=vmin, vmax=vmax
            )
            axs[i, j].set_aspect('equal')
            axs[i, j].set_title(f'Component {comp_names[j]}, tstep {sorted_timesteps[i]}')
            axs[i, j].set_xlabel('x')
            axs[i, j].set_ylabel('y')
            axs[i, j].locator_params(axis='x', nbins=4)
            axs[i, j].locator_params(axis='y', nbins=4)
    plt.tight_layout()

    # Add a single colorbar for the entire figure, using the last scatter plot created
    cbar = fig.colorbar(sc, ax=axs, location="top", fraction=0.02, pad=0.04)
    cbar.set_label('Magnetic field intensity')

    plt.savefig(os.path.join(path_to_save, "magnetic_field_multiple_timesteps.jpg"))

def current_plot(data_2d_array,coordinates,sorted_timesteps):

    # Assuming `new_magnetic` and `coordinates` are already defined
    fig, axs = plt.subplots(data_2d_array.shape[0]//3,3, figsize=(15, 30))


    # Determine the global color range
    vmin = data_2d_array.min()
    vmax = data_2d_array.max()


    for i, ax in enumerate(axs.flat):
        # Plot the data for the i-th timestep
        im = ax.tripcolor(coordinates[:, 0], coordinates[:, 1], data_2d_array[i],vmin=vmin, vmax=vmax)
        ax.set_title(f'Timestep {sorted_timesteps[i]}')
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.locator_params(nbins=4)

    plt.tight_layout()
    # Add a colorbar to the last axis
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6, location='top').set_label('Current Intensity')

    plt.savefig(f"current.pdf")

def plot_net_on_mesh(magnetic_field,coordinates,sorted_timesteps,path_to_save="./"):
    num_tsteps = len(sorted_timesteps)
    # plot the magnetic field
    fig, axs = plt.subplots(num_tsteps, magnetic_field.shape[-1], figsize=(15, 3 * num_tsteps))
    comp_names = ['x', 'y', 'z']

    # Determine the global color range
    vmin = magnetic_field.min()
    vmax = magnetic_field.max()

    # Plot each component in the subplots with the same color range
    for i in range(num_tsteps):
        for j in range(magnetic_field.shape[-1]):
            sc = axs[i, j].scatter(
                coordinates[:, 0], coordinates[:, 1], 
                c=magnetic_field[i, :, j], cmap='viridis', s=10, 
                vmin=vmin, vmax=vmax)
            axs[i,j].set_title(f'Magnetic field component {comp_names[j]} at timestep {sorted_timesteps[i]:.4g}')
            axs[i,j].set_xlabel('x')
            axs[i,j].set_ylabel('y')
            # less labels on the x axis
            axs[i,j].locator_params(axis='x', nbins=4)
            # less labels on the y axis
            axs[i,j].locator_params(axis='y', nbins=4)
            # aspect ratio
            axs[i,j].set_aspect('equal')
    plt.tight_layout()
    cbar = fig.colorbar(sc, ax=axs.ravel().tolist(), location="top", fraction=0.02, pad=0.04)
    cbar.set_label('Magnetic field')
    plt.savefig(os.path.join(path_to_save, "net_output_on_mesh.jpg"))


def ensi_plot(ensi_path="./ensi_files/",file_to_plot="MAGNE"):

    """
    MAIN FUNCTION
    """
    if file_to_plot not in ["MAGNE","CURNZ"]:
        raise ValueError("file_to_plot must be either 'MAGNE' or 'CURNZ'")
    
    # Replace `file_paths_example` with a list of paths to your files
    file_paths_example = [ensi_path+f'TS-WIRE-TRI03.ensi.{file_to_plot}-{i:06}' for i in range(1, 999, 100)]
                    
    # Load data across all timesteps
    sorted_timesteps, data_2d_array = load_multiple_timesteps(file_paths_example,file_to_plot=file_to_plot,skip_lines=4)

    # read coordinates
    coordinates = coordinate_reader()

    if file_to_plot=="MAGNE":
        # compute magnetic field
        #magnetic_field = magnetic_field_calculator(data_2d_array,coordinates,sorted_timesteps)
        magnetic_field = data_2d_array.reshape(data_2d_array.shape[0],3,coordinates.shape[0]).transpose(0,2,1)

        # plot multiple timesteps
        #magnetic_field_plot_multiple_tstep(magnetic_field,coordinates,sorted_timesteps)
        plot_net_on_mesh(magnetic_field,coordinates,sorted_timesteps)
       
    elif file_to_plot=="CURNZ":
        current_plot(data_2d_array,coordinates,sorted_timesteps)

if __name__ == "__main__":
    ensi_plot(file_to_plot="CURNZ")
    #ensi_plot(file_to_plot="MAGNE")


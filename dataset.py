from torch.utils.data import Dataset
import numpy as np
import torch

from ensi_plot import coordinate_reader,load_multiple_timesteps
import os
import pandas as pd

from scipy.interpolate import griddata
from numpy import random
from tqdm import tqdm



class WebDataset(Dataset):

    """
    The normalization steps are meant for Hx Hy only.
    """
    def __init__(self, config_train):
        self.config_train = config_train

        self.median, self.iqr = self.load_median_iqr()

    def load_txt(self,sample):
        # Decode binary data to a string
        text = sample.decode("utf-8")

        # Split into lines
        lines = text.splitlines()

        # Skip the first 5 lines (metadata) and convert the rest into numbers
        data = np.loadtxt(lines[4:])  # Skip first 5 lines

        return data

    def interpolate_on_eucledian_grid(self,field_to_interpolate):
        """
        Interpolate the data on a Euclidean grid
        """

        """
        Defines a structured grid and interpolates the magnetic field values on it.
        Uses nearest method because I get NaNs otherwise.
        Mask values outside the wire.
        """
        
        # Load the coordinates
        coordinates = coordinate_reader("/gpfs/scratch/bsc21/bsc580556/create_dataset/dataset/1/")
        
        # check that magnetic field is (n_points,)  Give just one components. Treat it as a scalar field
        assert field_to_interpolate.ndim == 1, "Field to interpolate must be a 1D array, found {}".format(field_to_interpolate.ndim) + "Give just one component. Treat it as a scalar field"

        # Define the structured grid (Euclidean grid)
        grid_x, grid_y = np.meshgrid(np.linspace(np.min(coordinates[:,0]), np.max(coordinates[:,0]), self.config_train["resolution"]),
                                    np.linspace(np.min(coordinates[:,1]), np.max(coordinates[:,1]),  self.config_train["resolution"]))

        # Interpolate the values from the unstructured grid to the structured grid

        grid_values = griddata(coordinates, field_to_interpolate, (grid_x, grid_y), method='nearest')
        mask = np.sqrt((grid_x-coordinates.mean())**2 + (grid_y-coordinates.mean())**2) > (coordinates.max() - coordinates.min())/2
        grid_values[mask] = 0.0

        return grid_values


    def load_median_iqr(self): # Load the normalization data
        median = np.load("median_values.npy") # shape (3,)
        iqr    = np.load("iqr_values.npy") # shape (3,)

        # take only the first two channels
        median = median[:2] # Shape (2,)
        iqr = iqr[:2]

        # Add fictious batch dimension
        median = median[:,np.newaxis,np.newaxis]
        iqr = iqr[:,np.newaxis,np.newaxis]

        return median, iqr

    def preprocess_data(self,sample):
        txt = self.load_txt(sample)
        
        # Interpolate eucledian works for scalar fields only
        if txt.shape[0] == 4525*3:
            txt = txt.reshape(3,4525)
        
        elif txt.shape[0] == 4525:
            txt = txt[np.newaxis,:]
            
        grid = np.array([self.interpolate_on_eucledian_grid(txt[i]) for i in range(txt.shape[0])])

        # Take only the first two channels
        grid = grid[:2]

        # Normalize the data
        grid = (grid - self.median) / self.iqr

        return torch.tensor(grid,dtype=torch.float32)
   
 

class WireDeep(Dataset):

    #
    """
    class used for supervised learining in pure data driven setting
    Idea is to load the ensi file needed using index for tstep (lazy loading)
    Key is field at time t, target is field at time t+1
    """
    def __init__(self, 
                 seed=0,
                 device='cpu', 
                 alya_config_path="TS-WIRE-TRI03/",
                 t_input=100, 
                 t_output=100,
                 file_to_plot="MAGNE",
                 ensi_path="./ensi_files/",
                 resolution=64,
                 fno_type="3D",
                 step=50,
                 path_to_indexes=None):
        """
        
        file_to_plot is either 'MAGNE' or 'CURNZ'. See ensi files for reference
        """
        raise DeprecationWarning("This class is deprecated, use WebDataset instead")

        #assert file_to_plot in ["MAGNE","CURNZ"], f"file_to_plot must be either 'MAGNE' or 'CURNZ', found {file_to_plot}"
        assert fno_type in ["3D","2D","Autoregressive Channels","Full Magnet"], "fno_type must be either '3D' or '2D' or 'Autoregressive Channels' or 'Full Magnet'"
        self.seed = seed
        self.device = device
       
        # Load coordinates
        # Every data will be concatenataed to coordinates mesh ("Euclidean grid needed for FNO" for example)
        #self.coordinates = coordinate_reader(alya_config_path)
               
        #self.ensi_path = ensi_path
        self.ensi_path = None
       
        self.files_to_plot = file_to_plot
        
        # t input is the number of tsteps fed as input (3D tensor) (t=10 in the paper)
        # t output is the number of tsteps fed as output (3D tensor) (big T in the paper)
        self.t_input = t_input
        self.t_output = t_output

        # Initialize data_2d_array and sorted_timesteps
        self.sorted_timesteps = None
        self.data_2d_array = None
        self.magnetic_field = None

        self.resolution = resolution
        self.fno_type = fno_type
        self.step = step

        # I need to save the tstep for the target
        self.tstep = None



        # Create indexes for data
        #self.data_indexing = self.dataset_indexing()

        self.path_to_indexes = path_to_indexes

        self.dataframe= pd.read_csv(path_to_indexes)

        # Remove problematic folders
        self.remove_problematic_folders()

        self.folder_indexes = self.dataframe["folder"].unique()


        # Setting dataset length variable, it gets updated when call dataset_length
        # I call the function manually before the training, to avoid weird behavior in the dataloader
        self.len_data = None
    

    def remove_problematic_folders(self):
        """
        Remove problematic folders from the dataset
        """
        problematic_folders = problematic_folders_list()
        #self.folder_indexes = self.folder_indexes[~self.folder_indexes.isin(problematic_folders)]
        self.dataframe = self.dataframe[~self.dataframe["folder"].isin(problematic_folders)]

    def normalize_data(self):
        """
        Normalize everything between 0 and 1
        I would have liked between -1 and 1 but it'd be weird to have a time between -1 and 1

        Consider using sklearn.preprocessing.MinMaxScaler or similar.

        HERE I AM NORMALIZING THE BATCH, INSTEAD I SHOULD DO IT ON THE WHOLE DATASET
        """
        self.data_2d_array = np.array(self.data_2d_array)
        self.sorted_timesteps = np.array(self.sorted_timesteps)

        self.coordinates = (self.coordinates - self.coordinates.min())/(self.coordinates.max() - self.coordinates.min())
        self.sorted_timesteps = (self.sorted_timesteps - self.sorted_timesteps.min())/(self.sorted_timesteps.max() - self.sorted_timesteps.min() + 1e-6)
        
        self.data_2d_array = (self.data_2d_array - self.data_2d_array.min())/(self.data_2d_array.max() - self.data_2d_array.min() + 1e-6)

        
    
    def reshape_data(self):
        """
        Reshape the data to the correct shape
        """
        return self.data_2d_array.reshape(self.data_2d_array.shape[0],3,self.coordinates.shape[0]).transpose(0,2,1)
    
    def __len__(self):

        if self.fno_type == "3D":
            raise NotImplementedError("This function is not implemented yet")
            return 1000 # I should put 1 but to simulate batch sizes I put 1000 # Here I'll put the len of the dataset. The number of differnet configurations. For the moment, I only have one
        elif self.fno_type == "2D":
            raise NotImplementedError("This function is not implemented yet")
            return 1000 -100 # Last tstep of simulation - t_input - t_output (time window size)
        elif self.fno_type == "Autoregressive Channels":
            raise NotImplementedError("This function is not implemented yet")
            return 1000 - 20
        elif self.fno_type == "Full Magnet":
            return self.len_data
            
    def __getitem__(self,idx):

        if self.fno_type == "2D":
            return self.get_item_autoregressive(idx)        
        elif self.fno_type == "3D":
            return self.get_item_time(idx)
        elif self.fno_type == "Autoregressive Channels":
            return self.get_item_autoregressive_CHANNELS(idx)
        elif self.fno_type == "Full Magnet":
            try:
                return self.get_item_pandas(idx)
            except:
                return self.get_item_pandas(idx+1)
           
        else:
            raise ValueError("fno_type must be either '3D' or '2D' or 'Autoregressive Channels' or 'Full Magnet'")

    def interpolate_on_eucledian_grid(self,field_to_interpolate):
        """
        Interpolate the data on a Euclidean grid
        """
        
        """
        Defines a structured grid and interpolates the magnetic field values on it.
        Uses nearest method because I get NaNs otherwise.
        Mask values outside the wire.
        """

        # check that magnetic field is (n_points,)  Give just one components. Treat it as a scalar field
        assert field_to_interpolate.ndim == 1, "Field to interpolate must be a 1D array, found {}".format(field_to_interpolate.ndim) + "Give just one component. Treat it as a scalar field"

        # Define the structured grid (Euclidean grid)
        grid_x, grid_y = np.meshgrid(np.linspace(np.min(self.coordinates[:,0]), np.max(self.coordinates[:,0]), self.resolution),
                                    np.linspace(np.min(self.coordinates[:,1]), np.max(self.coordinates[:,1]), self.resolution))

        # Interpolate the values from the unstructured grid to the structured grid
        
        grid_values = griddata(self.coordinates, field_to_interpolate, (grid_x, grid_y), method='nearest')
        mask = np.sqrt((grid_x-self.coordinates.mean())**2 + (grid_y-self.coordinates.mean())**2) > (self.coordinates.max() - self.coordinates.min())/2
        grid_values[mask] = 0.0

        return grid_values
    

    def get_item_autoregressive(self,idx):
        raise NotImplementedError("This function is not implemented yet")

        """
        Takes one 2D image as input and targets the next 2D image after a certain number of timesteps
        """


        # idx should start from 1, instead of 0
        idx += 1

        """
        The idea is that dataset will load one single fensi file at a time.
        It'll do so based on "file to plot" which can be MAGNE or CURZ or other (see ensi files) and 
        based on the index of the timestep.

        Files are loaded from the same directory. It'llbe necessary to change ensi path when I'll have the full dataset.
        """
        # If you want, this can be extended to be a list of files that the function load multiple tsteps can load at once
        # Loading timestep i and i+1, which will be the input and the target
        file_path = [self.ensi_path+f'TS-WIRE-TRI03.ensi.{self.file_to_plot}-{i:06}' for i in [idx,idx+100]] #PREDICT 100 tsteps ahead
                        
        # Load data across all timesteps
        # Sorted timesteps is useful only when you have multiple timesteps
        # For the moment, I have only one tstep at a time
        self.sorted_timesteps, self.data_2d_array = load_multiple_timesteps(file_path,file_to_plot=self.file_to_plot,skip_lines=4)

        # Normalize data
        # This can be substituted with sklearn.preprocessing.MinMaxScaler or similar
        self.normalize_data()

        # Reshape data
        if self.file_to_plot=="MAGNE":
            # Magnetic field data requires careful handling (need to reshape)
            self.magnetic_field = self.data_2d_array.reshape(self.data_2d_array.shape[0],3,self.coordinates.shape[0]).transpose(0,2,1)
            # # self.magnetic_field.shape -> (tsteps, points, components) e.g. (2,4525,3)
        # Concatanate the mesh coordinates to the data
        # This is needed for FNO
        # This can be substituted with a different mesh

        # I want (2,channels(mesh+data),width,height) 
        # FNO requires structured mesh 
        # I will interpolate the data on a structured grid # # shape (resolution,resolution)
        
        structured_grid_input = self.interpolate_on_eucledian_grid(self.magnetic_field[0,:,0]) # Working on the first component of the magnetic field for the moment
        structured_grid_output = self.interpolate_on_eucledian_grid(self.magnetic_field[1,:,0]) # Working on the first component of the magnetic field for the moment
        
        # Add a channel dimension   
        structured_grid_input = np.expand_dims(structured_grid_input, axis=0) # Containes the scalar field values in an array
        structured_grid_output = np.expand_dims(structured_grid_output, axis=0) # Containes the scalar field values in an array

        
        # Concatenate the structured grid to the coordinates
        # NOTE: FOR THE MOMENT I AVOID CONCATENATING THE STRUCTURED GRID TO THE COORDINATES
        # SEE WHAT HAPPENS USING DATA ONLY
        #structured_grid = np.concatenate([structured_grid, self.coordinates], axis=0)


        # Create torch tensor and move to device (GPU if available)
        # Output using dataloader is (batch_size,channels,width,height)
        return torch.tensor(structured_grid_input,device=self.device,dtype=torch.float32), torch.tensor(structured_grid_output,device=self.device,dtype=torch.float32)


    def get_item_autoregressive_CHANNELS(self,idx):
        raise NotImplementedError("This function is not implemented yet")

        """
        Takes one 2D image as input and targets the next 2D image after a certain number of timesteps
        """


        # idx should start from 1, instead of 0
        idx += 1 # Taking random timesteps or fix it? You decide

        """
        The idea is that dataset will load one single fensi file at a time.
        It'll do so based on "file to plot" which can be MAGNE or CURZ or other (see ensi files) and 
        based on the index of the timestep.

        Files are loaded from the same directory. It'llbe necessary to change ensi path when I'll have the full dataset.
        """
        # If you want, this can be extended to be a list of files that the function load multiple tsteps can load at once
        # Loading timestep i and i+1, which will be the input and the target
        file_path = [self.ensi_path+f'TS-WIRE-TRI03.ensi.{self.file_to_plot}-{i:06}' for i in range(idx,idx+20)]
        # Load data across all timesteps
        # Sorted timesteps is useful only when you have multiple timesteps
        # For the moment, I have only one tstep at a time
        self.sorted_timesteps, self.data_2d_array = load_multiple_timesteps(file_path,file_to_plot=self.file_to_plot,skip_lines=4)

        # Normalize data
        # This can be substituted with sklearn.preprocessing.MinMaxScaler or similar
        self.normalize_data()

        # Reshape data
        if self.file_to_plot=="MAGNE":
            # Magnetic field data requires careful handling (need to reshape)
            self.magnetic_field = self.data_2d_array.reshape(self.data_2d_array.shape[0],3,self.coordinates.shape[0]).transpose(0,2,1)
            # # self.magnetic_field.shape -> (tsteps, points, components) e.g. (2,4525,3)
        # Concatanate the mesh coordinates to the data
        # This is needed for FNO
        # This can be substituted with a different mesh

        # I want (2,channels(mesh+data),width,height) 
        # FNO requires structured mesh 
        # I will interpolate the data on a structured grid # # shape (resolution,resolution)

        ## Now taking the first half timesteps as input and the second half as output
        structured_grid_input = np.array([self.interpolate_on_eucledian_grid(self.magnetic_field[i,:,0]) for i in range(self.magnetic_field.shape[0]//2)]) # Working on the first component of the magnetic field for the moment
        structured_grid_output = np.array([self.interpolate_on_eucledian_grid(self.magnetic_field[i,:,0]) for i in range(self.magnetic_field.shape[0]//2,self.magnetic_field.shape[0])]) # Working on the first component of the magnetic field for the moment
       
        #print("magnetic field shape",self.magnetic_field.shape)
        #print("structured grid input shape",structured_grid_input.shape)
        #print("structured grid output shape",structured_grid_output.shape)
        #raise ValueError


        # Concatenate the structured grid to the coordinates
        # NOTE: FOR THE MOMENT I AVOID CONCATENATING THE STRUCTURED GRID TO THE COORDINATES
        # SEE WHAT HAPPENS USING DATA ONLY
        #structured_grid = np.concatenate([structured_grid, self.coordinates], axis=0)
        
        # Create torch tensor and move to device (GPU if available)
        # Output using dataloader is (batch_size,channels,width,height)
        return torch.tensor(structured_grid_input,device=self.device,dtype=torch.float32), torch.tensor(structured_grid_output,device=self.device,dtype=torch.float32)


    def get_item_time(self,idx):
        raise NotImplementedError("This function is not implemented yet")
        """
        Takes the first t_input timesteps as input and targets the next t_output timesteps
        inputs and outputs are 3D tensors (2D with a channel dimension)
        
        Inputs are always the first t_input timesteps
        Outputs are always the next t_output timesteps

        For the moment, always feed the net from tstep 0 to tstep t_input-1
        Next, I can try training with random t_input windows
        """
        # Qui andrò a prendere input e output dalla cartella 
        # Devo prendere i primi t_input+t_output timesteps
        # Divideró in input e output successivamente

        # avrò qualcosa del tipo ensi path = dataset/idx/TS-WIRE-TRI03.ensi.MAGNE-*
        # quindi idx qui serve per andare a prendere la cartella, non per prender i tstep da analizzare 
        # La randomizzazione del tstep da prendere sará fatta pescando random da una distribuzione con numpy 
        initial_tstep = 1 # this will be changed to np.random.randint(1,1000) for example
        file_path = [self.ensi_path+f'TS-WIRE-TRI03.ensi.{self.file_to_plot}-{i:06}' for i in np.arange(start=initial_tstep,stop=self.t_input+self.t_output+1,step=self.step)] 

        # Load data across all timesteps
        self.sorted_timesteps, self.data_2d_array = load_multiple_timesteps(file_path,file_to_plot=self.file_to_plot,skip_lines=4)

        # Normalize data
        self.normalize_data()

        # Reshape data
        if self.file_to_plot=="MAGNE":
            self.magnetic_field = self.data_2d_array.reshape(self.data_2d_array.shape[0],3,self.coordinates.shape[0]).transpose(0,2,1)
            # # self.magnetic_field.shape -> (tsteps, points, components) e.g. (tin+tout steps (the total length of the time window considered),4525,3)

        # Interpolate on structured grid
        # That function accepts only one timestep at a time
        # I will loop over all timesteps and concatenate the results on the channel dimensionç
        # Here I'm taking just the x component
       
        structured_grid_input = np.array([self.interpolate_on_eucledian_grid(self.magnetic_field[i,:,0]) for i in range(self.t_input//self.step)]) 
        structured_grid_output = np.array([self.interpolate_on_eucledian_grid(self.magnetic_field[i,:,0]) for i in range(self.t_input//self.step,(self.t_input+self.t_output)//self.step)])

        # Add a channel dimension by reshaping
        #print(structured_grid_input.shape)
        #print(structured_grid_output.shape)
        
        
        # Create torch tensor and move to device (GPU if available)
        # Output using dataloader is (batch_size,channels,width,height)
        return torch.tensor(structured_grid_input,device=self.device,dtype=torch.float32), torch.tensor(structured_grid_output,device=self.device,dtype=torch.float32)

    def folder_definition(self,idx):
        raise DeprecationWarning("This method is deprecated and should not be used.")

        """
        Define the folder where the ensi files are stored
        idx will select a folder in dataset/idx/ where the ensi files and alya_config files are stored
        """
        if idx >= len(self.folder_indexes):
            print(f"idx {idx} is out of range for dataset len={len(self.folder_indexes)}, setting it to 0")
            idx = 0
        # idx reads from folder list and selects the name of the folder to read.
        # let me call it idx still for clarity  
        folder_number = self.folder_indexes[idx]

        # Define the folder where the ensi files are stored
        # idx starts from 1 
        #idx = idx + 1

        # Skip folders that raise problems
        problematic_folders = problematic_folders_list()
        # avoiding problematic folders
        count= 0
        while (folder_number in problematic_folders) or (folder_number+self.step in problematic_folders):
            folder_number += 1
            count += 1
            if count > len(self.folder_indexes):
                raise ValueError("All folders are problematic")

       
        folder = f"/gpfs/scratch/bsc21/bsc580556/create_dataset/dataset/{folder_number}/"

        # Define self.ensi_path
        self.ensi_path = folder

        # Load the coordinates
        self.coordinates = coordinate_reader(folder)

        # I may need to save idx, only to deal with problems in load dataset function
        self.idx = idx




    def get_item_full_magnet(self,idx):
        raise DeprecationWarning("This method is deprecated, use get_item_pandas instead")

        """
        New implementation with full dataset.
        idx will select a folder number from the list given in input. Data will be in
        dataset/folder_indexes[idx]/ where the ensi files and alya_config files are stored
        Then a random index will select the specific timestep.
        """


        """
        Takes different outputs from MAGNET.
        One item has a fixed tstep.
        Channels will have many channels:
        3 components of the magnetic field + current + force + the rest that magnet outputs
        """

        self.folder_definition(idx)

       

        # Loading data one index at a time to avoid confusion.
        # I think it's easier to read
        # Reset self.tstep
        self.tstep = None
        data = self.load_data() # contains a list with magne, curnz, fluxn, joul, forcn

        concatenate_me = []
        for field in data:
            # The way data is interpolated on eucledian grid depends on wheather it's a scalar field or a vector field
            if field.ndim == 2: # dimensions are time_steps, n_points
                # Scalar field eg CURNZ, JOULN
                structured_grid = np.array([self.interpolate_on_eucledian_grid(field[t,:]) for t in range(field.shape[0])])
            elif field.ndim == 3: # dimensions are time_steps, n_points, components
                # Vector field eg MAGNE, FORCN
                structured_grid = np.array([self.interpolate_on_eucledian_grid(field[t,:,i]) for t in range(field.shape[0]) for i in range(field.shape[2])])
            else:
                raise ValueError("Field to interpolate must be either 1D or 2D, found {}".format(field.ndim))
            
            concatenate_me.append(structured_grid)

        # Concatenate the structured grid to the coordinates
        input_grid = np.concatenate(concatenate_me, axis=0)

        # Interpolate on structured grid
        #magnetic_grid = np.array([self.interpolate_on_eucledian_grid(magnetic_field[t,:,i]) for t in range(magnetic_field.shape[0]) for i in range(magnetic_field.shape[2])])
        #current_grid = np.array([self.interpolate_on_eucledian_grid(current[t,:]) for t in range(current.shape[0])])

        # Concatenate the structured grid to the coordinates
        #input_grid = np.concatenate([magnetic_grid,current_grid], axis=0)

        # Same thing for target
        # tstep will not be None, and will be used to load the target with tstep + step
        self.tstep = self.tstep + self.step
       
        data = self.load_data()
        concatenate_me = []
        for field in data:
            if field.ndim == 2: 
                structured_grid = np.array([self.interpolate_on_eucledian_grid(field[t,:]) for t in range(field.shape[0])])
            elif field.ndim == 3: 
                structured_grid = np.array([self.interpolate_on_eucledian_grid(field[t,:,i]) for t in range(field.shape[0]) for i in range(field.shape[2])])
            else:
                raise ValueError("Field to interpolate must be either 1D or 2D, found {}".format(field.ndim))
            concatenate_me.append(structured_grid)

        target_grid = np.concatenate(concatenate_me, axis=0)


        # Create torch tensor and move to device (GPU if available)
        return torch.tensor(input_grid,device=self.device,dtype=torch.float32), torch.tensor(target_grid,device=self.device,dtype=torch.float32)

    def get_item_pandas(self,idx):
        """
        Indexes are stored in a csv with two columns:
        folder and tstep
        """
        

        # Get folder and tstep
        folder = self.dataframe.iloc[idx]["folder"]
        self.tstep = self.dataframe.iloc[idx]["tstep"]

        
        self.check_folder_and_tstep_for_pandas_access(folder,idx)

                

        # Define the folder where the ensi files are stored
        self.ensi_path = f"/gpfs/scratch/bsc21/bsc580556/create_dataset/dataset/{folder}/"

        

        # Load the coordinates
        self.coordinates = coordinate_reader(self.ensi_path)

        # I may need to save idx, only to deal with problems in load dataset function
        self.idx = idx

        data = self.load_data()

        concatenate_me = []
        for field in data:
            # The way data is interpolated on eucledian grid depends on wheather it's a scalar field or a vector field
            if field.ndim == 2: # dimensions are time_steps, n_points
                # Scalar field eg CURNZ, JOULN
                structured_grid = np.array([self.interpolate_on_eucledian_grid(field[t,:]) for t in range(field.shape[0])])
            elif field.ndim == 3: # dimensions are time_steps, n_points, components
                # Vector field eg MAGNE, FORCN
                structured_grid = np.array([self.interpolate_on_eucledian_grid(field[t,:,i]) for t in range(field.shape[0]) for i in range(field.shape[2])])
            else:
                raise ValueError("Field to interpolate must be either 1D or 2D, found {}".format(field.ndim))
            
            concatenate_me.append(structured_grid)

        # Concatenate the structured grid to the coordinates
        input_grid = np.concatenate(concatenate_me, axis=0)

        # Same thing for target
        # tstep will not be None, and will be used to load the target with tstep + step
        self.tstep = self.tstep + self.step
        data = self.load_data() # now with different tstep  
        concatenate_me = []
        for field in data:
            if field.ndim == 2: 
                structured_grid = np.array([self.interpolate_on_eucledian_grid(field[t,:]) for t in range(field.shape[0])])
            elif field.ndim == 3: 
                structured_grid = np.array([self.interpolate_on_eucledian_grid(field[t,:,i]) for t in range(field.shape[0]) for i in range(field.shape[2])])
            else:
                raise ValueError("Field to interpolate must be either 1D or 2D, found {}".format(field.ndim))
            concatenate_me.append(structured_grid)

        target_grid = np.concatenate(concatenate_me, axis=0)


        # Create torch tensor and move to device (GPU if available)
        return torch.tensor(input_grid,device=self.device,dtype=torch.float32), torch.tensor(target_grid,device=self.device,dtype=torch.float32)


    def check_folder_and_tstep_for_pandas_access(self,folder,idx):
        # I have to be sure that there exists a tstep + step
        if self.dataframe[(self.dataframe["folder"]==folder) & (self.dataframe["tstep"]==self.tstep+self.step)].shape[0] == 0:
            # If it doesn't exist, it means that that is the target. Pick the related input
            try:
                # check that the previous tstep exists
                assert self.dataframe[(self.dataframe["folder"]==folder) & (self.dataframe["tstep"]==self.tstep-self.step)].shape[0] > 0, f"Folder {folder} has no tstep {self.tstep}"
                self.tstep -= self.step

            except:
                # If you're here, it means that the chosen tstep has neither the previous nor the next tstep (with this value of step)
                # Check that the folder contains at least n_steps, if not change folder
                try:
                    assert self.dataframe[self.dataframe["folder"]==folder].max()["tstep"] >= self.step
                    # If it contains at least n_steps, change tstep
                    self.tstep = self.dataframe[self.dataframe["folder"]==folder].min()["tstep"]

                except:
                    # If it doesn't contain at least n_steps, change folder
                    print(f"Folder {folder} has not enough tsteps, skipping")
                   # save_problematic(folder)
                    self.dataframe = self.dataframe[self.dataframe["folder"]!=folder]
                    return self.get_item_pandas(idx+1)
        

    def check_folder_and_tstep_for_random_access(self):

        raise DeprecationWarning()
        """
        Function that was used to check if the folders in dataset were OK
        and to see if there existed the tstep + step.
        I used it because I once accessed data randomly within the folder, 
        which is something that doesn't make much sense.
        
        """
        max_tstep = self.find_min_max_tstep()

        if max_tstep - self.step < 1:
            while max_tstep - self.step < 1:
                
                print(f"Folder {self.ensi_path} has not enough tsteps, skipping")
                save_problematic(self.folder_indexes[self.idx])

                # remove problematic folder from the list
                self.folder_indexes = self.folder_indexes[self.folder_indexes != self.folder_indexes[self.idx]]
                self.folder_definition(self.idx)
                max_tstep = self.find_min_max_tstep()



        # Randomly sample a tstep
        # Since not all simulations have the same number of tsteps, I need to handle errors properly

        # If self.tstep is None, it means that I'm loading the first tstep, target will be sampled with tstep +step
        if self.tstep is None:
            # count how many iterations in while loop
            count = 0
            i=0
            while i < len(self.files_to_plot):
                file_to_plot = self.files_to_plot[i]
                try:
                    tstep = np.random.randint(1, max_tstep - self.step)  # I need to be sure that I can sample the target
                    file_paths = [self.ensi_path + f'TS-WIRE-TRI03.ensi.{file_to_plot}-{i:06}' for i in [tstep,tstep + self.step]]
                    self.sorted_timesteps, self.data_2d_array = load_multiple_timesteps(file_paths, file_to_plot=file_to_plot, skip_lines=4)
                    self.tstep = tstep
                    i+=1
                    
                except: 
                    # go back to the first file to plot
                    i = 0
                    count += 1
                    #if count%1000 == 0:
                       # print(f"Count: {count}")
                    if count > 10_000:
                        # reset count
                        count=0
                        #print(f"Fodler {self.ensi_path} has problems, skipping")
                        save_problematic(self.folder_indexes[self.idx])

                        # remove problematic folder from the list
                        self.folder_indexes = self.folder_indexes[self.folder_indexes != self.folder_indexes[self.idx]]
                        
                        self.folder_definition(self.idx)
                        


        

    def load_data(self):
        
        """
        Load data from ensi files
        files to plot should be a list of strings
        """
                    
        data = []

        for file_to_plot in self.files_to_plot: # ["MAGNE","CURNZ","FLUXN","JOULN","FORCN"]:
           
           
           # raise ValueError("I'm not sure this is working")
            file_paths = [self.ensi_path+f'TS-WIRE-TRI03.ensi.{file_to_plot}-{i:06}' for i in [self.tstep]]
            try:
                self.sorted_timesteps, self.data_2d_array = load_multiple_timesteps(file_paths,file_to_plot=file_to_plot,skip_lines=4)
            except:
                # Save the log in a file
                log_errors(f"load_data error for folder {self.ensi_path} and tstep {self.tstep}")
                # remove the item
                self.dataframe = self.dataframe.drop(index=self.idx)
                self.get_item_pandas(self.idx)

            # To do: extend for different kinds of normalizations
            self.normalize_data()
            
           # if file_to_plot == "MAGNE":
           #     magnetic_field = self.data_2d_array.reshape(self.data_2d_array.shape[0],3,self.coordinates.shape[0]).transpose(0,2,1)
           # elif file_to_plot == "CURNZ":
           #     current = self.data_2d_array
           # else:
            #    raise ValueError("File to plot must be either MAGNE or CURNZ")

            try :
                self.data_2d_array = self.data_2d_array.reshape(self.data_2d_array.shape[0],3,self.coordinates.shape[0]).transpose(0,2,1)
            except:
                pass # It means data is 1 D

            data.append(self.data_2d_array)
        return data

    

    def find_min_max_tstep(self):
        """
        Not all files to plot have the same number of timesteps.
        It could be that MAGNE has 5000 tsteps, while CURNZ has only 540.
        To avoid useless random search, I find the minimum maximum tstep among all files to plot
        
        """
        max_tstep = 10_000
        for file_to_plot in self.files_to_plot:
            # All files end with the tstep number (6 digits)
            max_tstep_file_to_plot = max([int(f[-6:]) for f in os.listdir(self.ensi_path) if f"ensi.{file_to_plot}" in f])
            
            if max_tstep_file_to_plot < max_tstep:
                max_tstep = max_tstep_file_to_plot
        return max_tstep
    

    def dataset_length(self): # optimisable with numba
        
        """
        Return the length of the dataset.
        The length is measuered as the total number of timesteps that can be used as input
        In this context, a timestep is a set of the channels for a fixed tstep and simulation
        """
        # I have to iterate find_min_max_tstep for each folder in the dataset
        dataset_length= 0
        for folder in tqdm(self.folder_indexes,desc="Finding dataset length"):
            # define ensipath
            self.ensi_path= f"/gpfs/scratch/bsc21/bsc580556/create_dataset/dataset/{folder}/"
            max_tstep = self.find_min_max_tstep()
            dataset_length += max_tstep - self.step
        print("Dataset length:",dataset_length) # Dataset length:Dataset length: 702_268
        self.len_data = dataset_length
        return dataset_length
    
    def dataset_indexing(self):
        raise DeprecationWarning("Use slurm script to assign indexes to data")
        """
        Return the indexing of the dataset
        For every folder, assign a index to every input file
        """

        print("start dataset indexing")

        if os.path.exists("dataset_indexing.csv"):
            df = pd.read_csv("dataset_indexing.csv")
            return df

        # Define pandas dataframe
        df = pd.DataFrame(columns=["folder","tstep"])
        
        for folder in tqdm(self.folder_indexes, desc="Indexing dataset"):
            # First, find the available tsteps that can be actually used with this tstep
            #self.folder_definition(folder)
            self.ensi_path= f"/gpfs/scratch/bsc21/bsc580556/create_dataset/dataset/{folder}/"
            max_tstep = self.find_min_max_tstep()
            # For every folder, assign a index to every input file
            # This will be used to load the data
            
            # using self.files_to_plot[0] as a proxy, just to get the list of files. 
            # Get item will use the final tstep number in the file 
            file_paths = [f for f in os.listdir(self.ensi_path) if f"ensi.{self.files_to_plot[0]}" in f]
            # save the timestep (last 6 digit of file name)
            tsteps = [int(f[-6:]) for f in file_paths]

            # remove tsteps larger than max_tstep 
            tsteps = [tstep for tstep in tsteps if tstep <= max_tstep]

            # save in pandas
            for tstep in tsteps:
                df = pd.concat([df, pd.DataFrame({"folder": [folder], "tstep": [tstep]})], ignore_index=True)

            
        df.to_csv("dataset_indexing.csv",index=False)

        return df

def save_problematic(idx):
    """
    Save problematic folders
    """
    with open("problematic_folders.txt","a") as f:
        f.write(str(idx) + "\n")

def log_errors(error):
    """
    Log errors in a txt file
    """
    with open("log_train_errors.txt","a") as f:
        f.write(f"{error}\n")

def problematic_folders_list(path= "./problematic_folders.txt"):
    """
    List of problematic folders is saved automatically in a txt file 
    """
            
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [int(line.strip()) for line in lines]

    return lines


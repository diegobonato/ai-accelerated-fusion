

# AI-ACCELERATED MAGNET

This repository contains the scripts needed to train and evaluate two AI models to generate autoregressively the solution of Maxwell's equations for a infinitely long current-carrying HTS wire. The data is produced by MAGNET. 

There are two architectures: Fourier Neural Operator (FNO) and Adaptive Fourier Neural Operator (AFNO). 

The models are trained using Lightning Pytorch version 2.4 (https://lightning.ai/docs/pytorch/2.4.0/). The libraries are contained in a Singularity container. 

1. The top-level scripts are afno.py and fno.py. These are the scripts that you should launch to start a training. You can launch them directly using python (for debugging) or launch them using slurm (for the complete training). To use Slurm, launch the job arrays using afno_train.sh and fno_train.sh. All the parameters should be changed in the config_train dictionary. 

2. trainer.py contains the functions to prepare the Lighting Pytorch trainer. This script is called by afno.py and fno.py

3. dataset.py contains the pre-processing funtions to load the results from MAGNET and interpolate them to a Eucledian mesh. It uses webdataset. The actual input preparation (concatenation of the different physical quantities in channels) is done in the training loop in models.py. 

4. models.py contains the AI architectures and training functions. It follows the rules of Lightning Pytorch to wrap classical pytroch scripts. 

5. iqr_valiues.npy and median_values.npy contain the numpy arrays imported by models.py to normalize the data.

6. ensi_plot.py contains useful functions to plot the .ensi. files produced by MAGNET.

7. batch_job_arrays.sh is needed to launch automatically many Slurm arrays, when they cannot be submitted all in once (eg. when you have thousands of jobs to schedule)

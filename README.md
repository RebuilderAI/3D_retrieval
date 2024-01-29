# 3D_retrieval

This project focuses on retrieval of 3D models of objects that resemble the most to the given 2D image.
The selected algorithm for this task is Uni3D.

## Before proceeding with the task make sure you have installed all the required packages within a new conda virtual environment for this respository:

* conda create --n uni3d python=3.8 
* cd uni3d
* conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
* pip install -r requirements.txt 

## install pointnet2 extensions from https://github.com/erikwijmans/Pointnet2_PyTorch
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

## Retrieval

Run the following to execute the retrieval system 

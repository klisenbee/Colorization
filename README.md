# Colorization
### Towards More Vibrant Colorful Image Colorization
Final project for CSE 203B Convex Optimization Course

## Group Members 
  * Ahan Mukhopadhyay
  * Kolin Guo
  * Kyle Lisenbee
  * Ulyana Tkachenko

## Prerequisites
  * Ubuntu 18.04
  * NVIDIA GPU with CUDA version >= 10.1
  * [Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) version >= 19.03, API >= 1.40
  * [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker#ubuntu-16041804-debian-jessiestretchbuster) (previously known as nvidia-docker)  
  
Command to test if all prerequisites are met:  
  `sudo docker run -it --rm --gpus all ubuntu nvidia-smi`
  
## Setup Instructions
  `bash ./setup.sh`  
You should be greeted by the Docker container **colorization** when this script finishes. The working directory is */* and the repo is mounted at */Colorization*.  

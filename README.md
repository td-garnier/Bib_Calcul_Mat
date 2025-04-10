# Bib_Calcul_Mat
Bibliotheque de calcul matricielle


Sur terminal - créer son env:

conda create -n Cult_Info python=3.12.3
conda activate Cult_Info
conda install -c conda-forge cuda-nvcc cuda-nvrtc "cuda-version>=12.0"
conda install -c conda-forge cuda-python
sudo apt-get install nvidia-cuda-toolkit
conda install numpy time numba matplotlib scipy cupy
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda-repo-ubuntu2404-12-8-local_12.8.1-570.124.06-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-8-local_12.8.1-570.124.06-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-12-8-local/cuda-B2775641-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
sudo apt-get install -y cuda-drivers

Obs.: Après l'installation complète, redémarrer l'ordinateur

SUR VSCODE: cliquer sur installer ipykernell (voir image)
[alt text](<Screenshot from 2025-04-10 16-36-01.png>)

Resources: 
Pour CUDA Toolkit : https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_local


Attention : 

En avril 2025, il y avait un problème sur la libération de la mémoire avec Nvidia. En cas de saturation de la mémoire : ne pas pleurer et redémarrer le kernel ou redemarrer la machine. 

!
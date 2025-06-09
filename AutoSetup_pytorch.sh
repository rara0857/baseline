yes | pip install matplotlib --user
yes | pip install tfrecord
yes | pip install tensorboard
yes | pip install pillow --user
yes | pip install opencv-python-headless
yes | sudo apt update
yes | sudo apt install libgl1-mesa-glx
yes | sudo apt update && sudo apt install -y libsm6 libxext6
yes | sudo apt-get install libsm6
yes | sudo apt-get install libxrender1

#sudo dpkg -i ./ASAP_package/ASAP-1.7-Linux-python35.deb
yes | sudo add-apt-repository universe
yes | sudo apt-get update
yes | sudo apt-get install libboost-all-dev
yes | sudo apt --fix-broken install
yes | sudo apt-get install libboost-all-dev
yes | sudo apt-get install libjpeg-dev libpng-dev libtiff-dev
yes | sudo apt --fix-broken install
#yes | sudo dpkg -i ./ASSETS/ASAP_package/ASAP-1.9-Linux-Ubuntu1804.deb
yes | sudo apt-get install -f

yes | sudo pip install awscli --force-reinstall --upgrade
yes | pip install --upgrade pip
yes | sudo apt-get install openslide-tools
yes | sudo apt-get install python-openslide
yes | sudo apt install libvips-tools
yes | pip install openslide-python
yes | pip install torchvision
yes | pip install wandb
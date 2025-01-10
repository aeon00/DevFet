# DevFet
These are the steps to get the devfet environment up and running on mesocentre. It also includes the way to clone and pull code from this repo to mesocentre.

# Install miniconda on mesocentre
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

# Activate miniconda
source ~/miniconda3/bin/activate

# install needed packages
first create a virtual env

conda create --name devfet python=3.8.2

pip install brain-slam

pip install matplotlib==3.4.3

pip install pandas==1.4.4

pip install seaborn==0.11.2

pip install plotly==5.5.0

pip install -U kaleido

# Clone on Mesocentre
git clone https://github.com/aeon00/DevFet.git #perform command while on the head node

# Pull request
git pull



#!/bin/bash

set -e # exit script if an error occurs

pip install gdown

echo ""
echo "########################################"
echo "Download data, pretrained model weight"
echo "########################################"
echo ""

mkdir -p ./saved_models/3DM
# gdown https://drive.google.com/uc?id=1kUtX1977ly9Kj5sdpcVl_lMPmwQTXlpN
gdown --fuzzy https://drive.google.com/file/d/1jfhmuKFFQdGYd2WlOTYBQ_Z5sXHlrGkm/view?usp=sharing

cp ./3DM_UNet_pretrained.pth ./saved_models/3DM/G_BA_26000.pth
echo "Downloaded model weight"
# gdown https://drive.google.com/uc?id=1gmi_bjMFJ-0dfAHURKrhI15W7VQmXpmZ
gdown --fuzzy https://drive.google.com/file/d/1ExNXnJ-tgypJcbUfRg3jsd_dClkH1YU4/view?usp=sharing

echo "Downloaded data"

python -m jupyter notebook demo.ipynb

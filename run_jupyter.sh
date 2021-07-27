#!/bin/bash

set -e # exit script if an error occurs

pip install gdown

echo ""
echo "########################################"
echo "Download data, pretrained model weight"
echo "########################################"
echo ""

gdown https://drive.google.com/uc?id=1kUtX1977ly9Kj5sdpcVl_lMPmwQTXlpN
echo "Downloaded model weight"
gdown https://drive.google.com/uc?id=1gmi_bjMFJ-0dfAHURKrhI15W7VQmXpmZ
echo "Downloaded data"

python -m jupyter notebook demo.ipynb

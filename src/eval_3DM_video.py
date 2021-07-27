import argparse
import os
import sys
sys.path.append(".")
import warnings
warnings.filterwarnings('ignore')

import skimage.io as skio

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from src.unet3d.unet_model import UNet

parser = argparse.ArgumentParser()
parser.add_argument("--video_name", type=str, default="4p2VPS_S")
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--exp_name", type=str, default="3DM", help="name of the dataset")
opt = parser.parse_args()
print(opt)


def load(path):
    return torch.from_numpy(skio.imread(path).astype(float)).type(torch.FloatTensor).squeeze().unsqueeze(0).unsqueeze(0).cuda()

def path2result(path, model):
    wf = load(path).cuda()
    tmp = wf.max().data
    wf  = wf / tmp

    result = model(wf)
    result = torch.nn.functional.relu(result * tmp)[0, 0, ...]

    return result

cuda = torch.cuda.is_available()

# Initialize generator and discriminator
G_BA = UNet(n_channels=1, n_classes=1, depth=4, channel=32,
            normalization=None, activation='lrelu', bias=True)

if cuda:
    G_BA = G_BA.cuda()

if opt.epoch != 0:
    # Load pretrained models
    G_BA.load_state_dict(torch.load("saved_models/%s/G_BA_%d.pth" % (opt.exp_name, opt.epoch)))
else:
    raise RuntimeError


if opt.video_name == "4p2VPS_S":
    os.makedirs(f"images/{opt.exp_name}/video_{opt.epoch}_4p2VPS_S", exist_ok=True)
    tlength = 1260
if opt.video_name == "4p2VPS_Y":
    os.makedirs(f"images/{opt.exp_name}/video_{opt.epoch}_4p2VPS_Y", exist_ok=True)
    tlength = 1260
elif opt.video_name == "1VPS_S":
    os.makedirs(f"images/{opt.exp_name}/video_{opt.epoch}_1VPS_S", exist_ok=True)
    tlength = 960
elif opt.video_name == "1VPS_Y":
    os.makedirs(f"images/{opt.exp_name}/video_{opt.epoch}_1VPS_Y", exist_ok=True)
    tlength = 960

with torch.no_grad():
    for i in range(tlength):
        if opt.video_name == "4p2VPS_S":
            real_B_path = f"./Dataset_3DM/real_3DM/BEAR_4p2VPS/{str(i+1).zfill(4)}.tif"
        elif opt.video_name == "4p2VPS_Y":
            real_B_path = f"./data/3DM_widefield/raw_210227_3DM_casperGCaMP7a_4dpf_S2_R1_16XWI_4p2VPS_48planex1260vol/{i}.tif"
        elif opt.video_name == "1VPS_S":
            real_B_path = f"./Dataset_3DM/real_3DM/BEAR_1VPS/{str(i+1).zfill(3)}.tif"
        elif opt.video_name == "1VPS_Y":
            real_B_path = f"./data/3DM_widefield/raw_210227_3DM_casperGCaMP7a_4dpf_S2_R2_16XWI_1VPS_48planex1200vol/{i}.tif"

        fake_A = path2result(real_B_path, G_BA)

        if opt.video_name == "4p2VPS_S":
            skio.imsave(f"images/{opt.exp_name}/video_{opt.epoch}_4p2VPS_S/{i+1}.tif", fake_A.cpu().numpy())
        elif opt.video_name == "4p2VPS_Y":
            skio.imsave(f"images/{opt.exp_name}/video_{opt.epoch}_4p2VPS_Y/{i+1}.tif", fake_A.cpu().numpy())
        elif opt.video_name == "1VPS_S":
            skio.imsave(f"images/{opt.exp_name}/video_{opt.epoch}_1VPS_S/{i+1}.tif", fake_A.cpu().numpy())
        elif opt.video_name == "1VPS_Y":
            skio.imsave(f"images/{opt.exp_name}/video_{opt.epoch}_1VPS_Y/{i+1}.tif", fake_A.cpu().numpy())

        if i % 100 == 0:
            print(i, end=' ')
    
print('\n')

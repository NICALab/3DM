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


os.makedirs(f"images/{opt.exp_name}/eval_210418_S5_NN_{opt.exp_name}_{opt.epoch}", exist_ok=True)
os.makedirs(f"images/{opt.exp_name}/eval_210418_S5_BEAR_NN_{opt.exp_name}_{opt.epoch}", exist_ok=True)
os.makedirs(f"images/{opt.exp_name}/eval_210502_S1_NN_{opt.exp_name}_{opt.epoch}", exist_ok=True)
os.makedirs(f"images/{opt.exp_name}/eval_210502_S1_BEAR_NN_{opt.exp_name}_{opt.epoch}", exist_ok=True)


with torch.no_grad():
    for i in range(60):
        # 210418 S5

        # Raw wide-field
        B_path = f"./Dataset_3DM/evaluation/Confocal_timeLapse_4D/210418_Casper_GCaMP7a_4dpf_S5/resampled_WF/resampled_WF_rawVideo_210418_Casper_GCaMP7a_4dpf_sample5_t{i + 1}.tif"
        fake_A = path2result(B_path, G_BA)

        skio.imsave(f"images/{opt.exp_name}/eval_210418_S5_NN_{opt.exp_name}_{opt.epoch}/{i + 1}.tif", fake_A.cpu().numpy())

        # BEAR wide-field
        B_path = f"./Dataset_3DM/evaluation/Confocal_timeLapse_4D/210418_Casper_GCaMP7a_4dpf_S5/BEAR_WF/resampled_WF_BEAR_rawVideo_210418_Confocal_sample5_t{i + 1}.tif"
        fake_A = path2result(B_path, G_BA)

        skio.imsave(f"images/{opt.exp_name}/eval_210418_S5_BEAR_NN_{opt.exp_name}_{opt.epoch}/{i + 1}.tif", fake_A.cpu().numpy())

        # 210502 S1
    
        # Raw wide-field
        B_path = f"./Dataset_3DM/evaluation/Confocal_timeLapse_4D/210502_Casper_GCaMP7a_4dpf_S1/resampled_WF/resampled_WF_rawVideo_210502_Casper_GCaMP7a_4dpf_sample1_t{i + 1}.tif"
        fake_A = path2result(B_path, G_BA)

        skio.imsave(f"images/{opt.exp_name}/eval_210502_S1_NN_{opt.exp_name}_{opt.epoch}/{i + 1}.tif", fake_A.cpu().numpy())

        # BEAR wide-field
        B_path = f"./Dataset_3DM/evaluation/Confocal_timeLapse_4D/210502_Casper_GCaMP7a_4dpf_S1/BEAR_WF/resampled_WF_BEAR_rawVideo_210502_Confocal_sample1_t{i + 1}.tif"
        fake_A = path2result(B_path, G_BA)

        skio.imsave(f"images/{opt.exp_name}/eval_210502_S1_BEAR_NN_{opt.exp_name}_{opt.epoch}/{i + 1}.tif", fake_A.cpu().numpy())

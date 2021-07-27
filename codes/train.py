import argparse
import os
import numpy as np
import itertools
import sys

sys.path.append(".")
import time
import warnings
warnings.filterwarnings('ignore')

import skimage.io as skio

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from datasets import confocalpairDataset, syntheticbeadDataset
from util import plotter
from codes.unet3d.unet_model import UNet

import matplotlib.pyplot as plt

# random.seed(1234)
# torch.manual_seed(1234)
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=50000, help="number of epochs of training")
parser.add_argument("--exp_name", type=str, default="3DM", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--lr", type=float, default=4e-5, help="adam: learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=100, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)


import matplotlib
matplotlib.use('Agg')
fig = plotter(opt)
fig.clear()

# Create sample and checkpoint directories
os.makedirs("images/%s" % opt.exp_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.exp_name, exist_ok=True)

def load(path):
    return torch.from_numpy(skio.imread(path).astype(float)).type(torch.FloatTensor).squeeze().unsqueeze(0).unsqueeze(0).cuda()

class weighted_l1_loss(torch.nn.L1Loss):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    def forward(self, inp, target):
        return self.weight * super(weighted_l1_loss, self).forward(inp, target)

pixelwise_loss = weighted_l1_loss(weight=1)

cuda = torch.cuda.is_available()


# Initialize generator and discriminator
G_BA = UNet(n_channels=1, n_classes=1, depth=4, channel=32,
            normalization=None, activation='lrelu', bias=True)

# print(G_BA)

if cuda:
    G_BA = G_BA.cuda()

if opt.epoch != 0:
    # Load pretrained models
    G_BA.load_state_dict(torch.load("saved_models/%s/G_BA_%d.pth" % (opt.exp_name, opt.epoch)))
else:
    pass

# Optimizers
optimizer_G_BA = torch.optim.Adam(
    itertools.chain(G_BA.parameters()), lr=opt.lr
)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor


# ----------
# Dataset
# ----------
dataset_1 = confocalpairDataset(path='./Dataset_3DM',
                                patchsize=[48, 160, 160],
                                noise=True,
                                batch_size=opt.batch_size)
print("Loaded dataset 1, confocalpairDataset.")
dataset_2 = syntheticbeadDataset(path='./Dataset_3DM/synthetic_bead_3DM',
                                 patchsize=[48, 64, 64],
                                 noise=True,
                                 batch_size=opt.batch_size)
print("Loaded dataset 2, syntheticbeadDataset.\n")

dataloader_1 = DataLoader(dataset_1, batch_size=opt.batch_size, shuffle=True, num_workers=opt.batch_size, pin_memory=True)
dataloader_2 = DataLoader(dataset_2, batch_size=opt.batch_size, shuffle=True, num_workers=opt.batch_size, pin_memory=True)


# ----------
# Loss array
# is composed as '2' dataset, '1' losses' and 'number of updates'
# ----------

loss_array = np.zeros((2, 1, opt.n_epochs))

# ----------
# Load evaluation data
# ----------

# real_B_path = "./data/3DM_widefield/Decomp_raw_210227_3DM_casperGCaMP7a_4dpf_S2_R1_16XWI_4p2VPS_48planex1260vol/13.tif"
real_B_path = "./Dataset_3DM/real_3DM/BEAR_4p2VPS/0002.tif"
real_B_wf = load(real_B_path).cuda()
real_B_wf /= real_B_wf.max()

real_B_path = "./Dataset_3DM/evaluation/Confocal_timeLapse_4D/210418_Casper_GCaMP7a_4dpf_S5/resampled_WF/resampled_WF_rawVideo_210418_Casper_GCaMP7a_4dpf_sample5_t1.tif"
real_B_eval1 = load(real_B_path).cuda()
real_B_eval1 /= real_B_eval1.max()

real_B_path = "./Dataset_3DM/evaluation/Confocal_timeLapse_4D/210418_Casper_GCaMP7a_4dpf_S5/BEAR_resampled/BEAR_resampled_rawVideo_210418_Confocal_sample5_t1.tif"
real_B_eval2 = load(real_B_path).cuda()
real_B_eval2 /= real_B_eval2.max()
print("Loaded evaluation data.\n")

# ----------
# Training
# ----------

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader_1):
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))
        real_B_noise = Variable(batch["B_n"].type(Tensor))

        optimizer_G_BA.zero_grad()
        fake_A = G_BA(real_B_noise)
        loss_pixelwise = pixelwise_loss(fake_A, real_A)

        loss_pixelwise.backward()
        optimizer_G_BA.step()

        # Print log
        if epoch % (opt.sample_interval // 10) == 0:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"[{ts}] D1 E [{epoch}/{opt.n_epochs}] B [{i}/{len(dataloader_1)}] Pixel : {loss_pixelwise.item():.4f}")

        loss_array[0, :, epoch] = [loss_pixelwise.item()]

        fig.clear()
        fig.plot(loss_array, epoch)
        
        with torch.no_grad():
            if epoch % opt.sample_interval == 0:
                skio.imsave(f"images/{opt.exp_name}/{epoch}_D1_A.tif", real_A[0:2, ...].data.cpu().numpy())
                skio.imsave(f"images/{opt.exp_name}/{epoch}_D1_B.tif", real_B[0:2, ...].data.cpu().numpy())
                skio.imsave(f"images/{opt.exp_name}/{epoch}_D1_B_n.tif", real_B_noise[0:2, ...].data.cpu().numpy())
                skio.imsave(f"images/{opt.exp_name}/{epoch}_D1_BA.tif", fake_A[0:2, ...].data.cpu().numpy())

    for i, batch in enumerate(dataloader_2):
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))
        real_B_noise = Variable(batch["B_n"].type(Tensor))

        optimizer_G_BA.zero_grad()
        fake_A = G_BA(real_B_noise)
        loss_pixelwise = pixelwise_loss(fake_A, real_A)

        loss_pixelwise.backward()
        optimizer_G_BA.step()

        # Print log
        if epoch % (opt.sample_interval // 10) == 0:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"[{ts}] D2 E [{epoch}/{opt.n_epochs}] B [{i}/{len(dataloader_2)}] Pixel : {loss_pixelwise.item():.4f}\n")

        loss_array[1, :, epoch] = [loss_pixelwise.item()]

        fig.clear()
        fig.plot(loss_array, epoch)
        
        with torch.no_grad():
            if epoch % opt.sample_interval == 0:
                skio.imsave(f"images/{opt.exp_name}/{epoch}_D2_A.tif", real_A[0:2, ...].data.cpu().numpy())
                skio.imsave(f"images/{opt.exp_name}/{epoch}_D2_B.tif", real_B[0:2, ...].data.cpu().numpy())
                skio.imsave(f"images/{opt.exp_name}/{epoch}_D2_B_n.tif", real_B_noise[0:2, ...].data.cpu().numpy())
                skio.imsave(f"images/{opt.exp_name}/{epoch}_D2_BA.tif", fake_A[0:2, ...].data.cpu().numpy())

    # Evaluation
    with torch.no_grad():
        if epoch % opt.sample_interval == 0:
            fake_A_wf = G_BA(real_B_wf)
            fake_A_eval1 = G_BA(real_B_eval1)
            fake_A_eval2 = G_BA(real_B_eval2)

            # skio.imsave(f"images/{opt.exp_name}/{epoch}_ref_B.tif", real_B_wf.cpu().numpy())
            skio.imsave(f"images/{opt.exp_name}/{epoch}_ref_BA_wf.tif", fake_A_wf.cpu().numpy())
            skio.imsave(f"images/{opt.exp_name}/{epoch}_ref_BA_eval1.tif", fake_A_eval1.cpu().numpy())
            skio.imsave(f"images/{opt.exp_name}/{epoch}_ref_BA_eval2.tif", fake_A_eval2.cpu().numpy())


    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G_BA.state_dict(), "saved_models/%s/G_BA_%d.pth" % (opt.exp_name, epoch))
        plt.savefig("images/%s/%s_figure.pdf" % (opt.exp_name, epoch))
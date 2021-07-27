import time

import torch
from torch.utils.data import Dataset

import skimage.io as skio

from os import listdir
from os.path import splitext

import random

from util import preprocess


def load(path):
    return torch.from_numpy(skio.imread(path).astype(float)).type(torch.FloatTensor).squeeze().unsqueeze(0)


class confocalpairDataset(Dataset):
    def __init__(self, path='./Dataset_3DM', patchsize=[48, 160, 160], noise=True, batch_size=None):
        self.ids = list(range(1, batch_size+1))
        self.raw_Ss, self.blur_Ss = [], []
        self.batch_size = batch_size
        self.patchsize = patchsize
        self.noise = noise

        # 210220 Confocal zStack
        for _ in range(1):
            raw_path = f"{path}/training/Confocal_zStack/210220_Casper_GCaMP7a_4dpf/resampled/resampled_210220_Confocal_16X0p8NA_water_Casper_GCaMP7a_4dpf.tif"
            blur_path = f"{path}/training/Confocal_zStack/210220_Casper_GCaMP7a_4dpf/resampled_WF/resampled_WF_210220_Confocal_16X0p8NA_water_Casper_GCaMP7a_4dpf.tif"
            self.raw_Ss.append(load(raw_path))
            self.blur_Ss.append(load(blur_path))
        
        for i in range(10):
            print(f"    [in datasets.py - confocalpairDataset - __init__] [{i}/10] loaded")
            # 210410 data S
            raw_path = f"{path}/training/Confocal_timeLapse_4D/210410_Casper_GCaMP7a_4dpf_S1/BEAR_resampled/BEAR_resampled_rawVideo_210410_Confocal_sample1_t{i * 6 + 1}.tif"
            blur_path = f"{path}/training/Confocal_timeLapse_4D/210410_Casper_GCaMP7a_4dpf_S1/BEAR_WF/resampled_WF_BEAR_rawVideo_210410_Confocal_sample1_t{i * 6 + 1}.tif"
            self.raw_Ss.append(load(raw_path))
            self.blur_Ss.append(load(blur_path))

            # Y
            raw_path = f"{path}/training/Confocal_timeLapse_4D/210410_Casper_GCaMP7a_4dpf_S1/resampled/Resample_rawVideo_210410_Confocal_sample1_5um_{i * 6 + 1}.tif"
            blur_path = f"{path}/training/Confocal_timeLapse_4D/210410_Casper_GCaMP7a_4dpf_S1/resampled_WF/resampled_WF_rawVideo_210410_Confocal_sample1_t{i * 6 + 1}.tif"
            self.raw_Ss.append(load(raw_path))
            self.blur_Ss.append(load(blur_path))

            # 210418 data
            for j in range(1, 4):
                # S
                raw_path = f"{path}/training/Confocal_timeLapse_4D/210418_Casper_GCaMP7a_4dpf_S{j}/BEAR_resampled/BEAR_resampled_rawVideo_210418_Confocal_sample{j}_t{i * 6 + 1}.tif"
                blur_path = f"{path}/training/Confocal_timeLapse_4D/210418_Casper_GCaMP7a_4dpf_S{j}/BEAR_WF/resampled_WF_BEAR_rawVideo_210418_Confocal_sample{j}_t{i * 6 + 1}.tif"
                self.raw_Ss.append(load(raw_path))
                self.blur_Ss.append(load(blur_path))

                # Y
                raw_path = f"{path}/training/Confocal_timeLapse_4D/210418_Casper_GCaMP7a_4dpf_S{j}/resampled/Resample_rawVideo_210418_Casper_GCaMP7a_4dpf_sample{j}_{i * 6 + 1}.tif"
                blur_path = f"{path}/training/Confocal_timeLapse_4D/210418_Casper_GCaMP7a_4dpf_S{j}/resampled_WF/resampled_WF_rawVideo_210418_Casper_GCaMP7a_4dpf_sample{j}_t{i * 6 + 1}.tif"
                self.raw_Ss.append(load(raw_path))
                self.blur_Ss.append(load(blur_path))
        print("\n")

        # Sanity check
        for (r, b) in zip(self.raw_Ss, self.blur_Ss):
            assert r.min() >= 0
            assert b.min() >= 0
            assert r.isnan().float().sum() == 0
            assert b.isnan().float().sum() == 0

    def __len__(self):
        return len(list(range(1, self.batch_size+1)))

    def __getitem__(self, i):
        idx = random.randint(0, len(self.raw_Ss)-1)
        raw_S = self.raw_Ss[idx]
        blur_S = self.blur_Ss[idx]
        if self.noise:
            raw, blur, blur_noise = preprocess([raw_S, blur_S], self.patchsize, self.noise)

            return {'A' : raw,
                    'B' : blur,
                    'B_n' : blur_noise
            }
        else:
            raw, blur = preprocess([raw_S, blur_S], self.patchsize, self.noise)

            return {'A' : raw,
                    'B' : blur
            }


class syntheticbeadDataset(Dataset):
    def __init__(self, path, patchsize=[48, 64, 64], noise=False, batch_size=None):
        self.raw_dir = f'{path}/raw/'
        self.blur_dir = f'{path}/blur/'
        self.batch_size = batch_size

        self.ids = [splitext(file)[0] for file in listdir(self.blur_dir)
                    if not file.startswith('.')]
        self.raws, self.blurs = [], []
        for _id in self.ids:
            blur = torch.from_numpy(skio.imread(f'{self.blur_dir}{_id}.tif').astype(float)).type(torch.FloatTensor).squeeze().unsqueeze(0)
            raw = torch.from_numpy(skio.imread(f'{self.raw_dir}{_id}.tif').astype(float)).type(torch.FloatTensor).squeeze().unsqueeze(0)
            self.blurs.append(blur)
            self.raws.append(raw)

        self.patchsize = patchsize
        self.noise = noise

    def __len__(self):
        return len(list(range(1, self.batch_size+1)))

    def __getitem__(self, i):
        i = random.randint(0, len(self.raws)-1)
        raw, blur = self.raws[i], self.blurs[i]

        if self.noise:
            raw, blur, blur_noise = preprocess([raw, blur], self.patchsize, self.noise)

            return {'A' : raw,
                    'B' : blur,
                    'B_n' : blur_noise
            }
        else:
            raw, blur = preprocess([raw, blur], self.patchsize, self.noise)

            return {'A' : raw,
                    'B' : blur
            }

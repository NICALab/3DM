import torch
import random
import numpy as np
import matplotlib.pyplot as plt


class plotter():
    def __init__(self, opt):
        self.fig, self.ax = plt.subplots(2, 1, figsize=(5, 4))
        self.fig.canvas.set_window_title(opt.exp_name)
        self.fig.canvas.draw()
        plt.tight_layout()
        self.losslist = ['G pix loss']

    def clear(self):
        for i in range(2):
            self.ax[i].clear()
            text =  f"D{i+1} pixel loss"
            self.ax[i].set_title(text)

    def plot(self, loss_array, epoch):
        for i in range(2):
            self.ax[i].plot(loss_array[i, 0, :epoch+1], color='#1f77b4')
        self.fig.canvas.draw()
        plt.pause(1)


def preprocess(images, patchsize, noise=False, noise_range=[100, 3000]):
    def rand_flag():
        return torch.rand(1).item() > 0.5

    def patchslice(images, patchsize, pad=0):  # pad=20, 30, 40
        _, z, y, x = images[-1].size()
        if patchsize[1] + pad * 2 >= y or patchsize[2] + pad * 2 >= x:
            pad = 0
        zt = random.randint(0, z-patchsize[0])
        yt = random.randint(0+pad, y-pad-patchsize[1])
        xt = random.randint(0+pad, x-pad-patchsize[2])
        images = [img[:, zt:zt+patchsize[0], yt:yt+patchsize[1], xt:xt+patchsize[2]] for img in images]
        return images
    
    def normalize(images):
        norm_fac = (images[-1].max() + 1e-9) # / 1000
        return [img / norm_fac for img in images]
    
    def flip(images):
        if rand_flag():
            images = [torch.flip(img, [2]) for img in images]
        if rand_flag():
            images = [torch.flip(img, [3]) for img in images]
        if rand_flag():
            images = [torch.rot90(img, 1, (2, 3)) for img in images]
        return images

    images = patchslice(images, patchsize, pad=0)
    images = normalize(images)
    images = flip(images)

    if noise: # Should apply only for synthetic WF data!
        noise_level = random.randint(*noise_range)
        images.append(torch.from_numpy(np.random.poisson(images[-1].numpy() * noise_level) / noise_level).type(torch.FloatTensor))

    return images

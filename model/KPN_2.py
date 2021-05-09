import torch
import torch.nn as nn
import torch.nn.functional as F
from model.KPN import KPN


# Simple Two Stage Model
class ImageInpainter(nn.Module):
    def __init__(self):
        super(ImageInpainter, self).__init__()
        self.generator = KPN(burst_length=1, blind_est=True, kernel_size=[5])
        self.denoiser = KPN(burst_length=1, blind_est=True, kernel_size=[3])

    def forward(self, img):
        img_gen = self.generator(img)
        img_denoised = self.denoiser(img_gen)
        return img_gen, img_denoised

import torch
import torch.nn as nn
import numpy as np


class incremental_Loss_txt(nn.Module):
    """
    Loss function of incremental.

    Args:
        code_length(int): Hashing code length.
        gamma, theta(float): Hyper-parameter.
    """
    def __init__(self, code_length, gamma, eta):
        super(incremental_Loss_txt, self).__init__()
        self.code_length = code_length
        self.gamma = gamma
        self.eta = eta

    def forward(self, F, B, omega):
        omega = np.array(omega.cpu()) #win10√ linux×
        quantization_loss = ((F - B[omega, :]) ** 2).sum()
        correlation_loss = ((F.t() @ torch.ones(F.shape[0], 1, device=F.device)) ** 2).sum()
        loss = (self.gamma * quantization_loss + self.eta * correlation_loss) / (F.shape[0])

        return loss


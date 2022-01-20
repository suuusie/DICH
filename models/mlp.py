# 定义MLP模型

import torch.nn as nn
import torch.nn.functional as F
LAYER1_NODE = 8192
def load_model(y_dim,code_length):
    """
    Load MLP model.

    Args
        code_length (int): Hashing code length.
        y_dim: dimension of tags

    Returns
        model (torch.nn.Module): MLP model.
    """
    model = Net(y_dim, code_length)
    return  model
def weights_init(m):
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight.data, 0.0, 0.01)
        nn.init.normal_(m.bias.data, 0.0, 0.01)
class Net(nn.Module):
    def __init__(self, y_dim, code_length):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(y_dim, LAYER1_NODE)
        self.fc2 = nn.Linear(LAYER1_NODE,code_length)

    def forward(self, x):

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.tanh(x)
        x = x.squeeze()

        return x
# model = Net(500,32)
# # 打印出来看是否正确
# print(model)

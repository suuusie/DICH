import torch.nn as nn
import torch
import math


from torch.hub import load_state_dict_from_url


def load_model(code_length,pretrain_model):
    """
    Load CNN model.

    Args
        code_length (int): Hashing code length.

    Returns
        model (torch.nn.Module): CNN model.
    """
    model = ImgNet(code_length,pretrain_model)

    return  model

class ImgNet(nn.Module):
    def __init__(self, code_len, pretrain_model=None):
        super(ImgNet, self).__init__()
        self.features = nn.Sequential(
            # 0 conv1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4),
            # 1 relu1
            nn.ReLU(inplace=True),
            # 2 norm1
            nn.LocalResponseNorm(size=2, k=2),
            # 3 pool1
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # 4 conv2
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5, stride=1, padding=2),
            # 5 relu2
            nn.ReLU(inplace=True),
            # 6 norm2
            nn.LocalResponseNorm(size=2, k=2),
            # 7 pool2
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # 8 conv3
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 9 relu3
            nn.ReLU(inplace=True),
            # 10 conv4
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 11 relu4
            nn.ReLU(inplace=True),
            # 12 conv5
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 13 relu5
            nn.ReLU(inplace=True),
            # 14 pool5
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            # 15 full_conv6
            nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=6),
            # 16 relu6
            nn.ReLU(inplace=True),
            # 17 full_conv7
            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1),
            # 18 relu7
            nn.ReLU(inplace=True),
        )
        # fc8

        self.classifier = nn.Linear(in_features=4096, out_features=code_len)
        self.classifier.weight.data = torch.randn(code_len, 4096) * 0.01
        # self.classifier.bias.data = torch.randn(code_len) * 0.01
        # self.mean = torch.zeros(3, 224, 224)
        self.alpha = 1.0
        if pretrain_model:
            self._init(pretrain_model)

    def _init(self, data):
        weights = data['layers'][0]
        # self.mean = torch.from_numpy(data['normalization'][0][0][0].transpose()).type(torch.float)
        for k, v in self.features.named_children():
            k = int(k)
            if isinstance(v, nn.Conv2d):
                if k > 1:
                    k -= 1
                v.weight.data = torch.from_numpy(weights[k][0][0][0][0][0].transpose())
                # v.bias.data = torch.from_numpy(weights[k][0][0][0][0][1].reshape(-1))

    def forward(self, x):
        # if x.is_cuda:
        #     x = x - self.mean.cuda()
        # else:
        #     x = x - self.mean
        x = self.features(x)
        feat = x.squeeze()
        hid = self.classifier(feat)
        code = torch.tanh(self.alpha * hid)
        # return feat, hid, code
        return code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)
# model = AlexNet(32)
# # 打印出来看是否正确
# print(model)

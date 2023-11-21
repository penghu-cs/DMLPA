import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import Parameter
from torchvision import models

from util import *

class FCNN(nn.Module):
    """Network to learn image representations"""

    def __init__(self, input_dim=4096, out_dim=1024, mid=2048, num_fc=2, bn=False, dropout=0.5):
        super(FCNN, self).__init__()
        # self.bn = nn.BatchNorm1d(input_dim)
        # self.denseL1 = nn.Linear(input_dim, out_dim)
        if num_fc < 1 and not isinstance(num_fc, int):
            raise ValueError("'num_fc' should be a positive integer!")
        
        layers, bn_dim = [], mid
        for i in range(num_fc):
            mid = mid if i != num_fc - 1 else out_dim
            if i == 0:
                layers += [nn.Linear(input_dim, mid)]
            # elif i == num_fc - 1:
            #     layers += [nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(mid, out_dim)]
            elif bn:
                layers += [nn.BatchNorm1d(bn_dim), nn.ReLU(inplace=True), nn.Linear(bn_dim, mid)]
            else:
                layers += [nn.ReLU(inplace=True), nn.Linear(bn_dim, mid)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x = self.bn(x)
        # out = F.relu(self.denseL1(x))
        out = self.net(x)
        return out


class Text_CNN(nn.Module):
    def __init__(self, mode, word_dim, out_dim, filters, filter_num, dropout_prob, wv_matrix, in_channel=1, mid=1024, num_fc=2, bn=False, dropout=0.5):
        super(Text_CNN, self).__init__()
        self.mode = mode
        self.word_dim = word_dim
        self.vocab_size = wv_matrix.shape[0]
        self.out_dim = out_dim
        self.filters = filters
        self.filter_num = filter_num
        self.dropout_prob = dropout_prob
        self.in_channel = in_channel

        assert (len(self.filters) == len(self.filter_num))
        self.embedding = nn.Embedding(self.vocab_size, self.word_dim, padding_idx=self.vocab_size - 1)
        if self.mode == "static" or self.mode == 'non-static' or self.mode == 'multichannel':
            self.wv_matrix = wv_matrix
            self.embedding.weight.data.copy_(torch.from_numpy(self.wv_matrix))
            if self.mode == 'static':
                self.embedding.weight.requires_grad = False
            elif self.mode == 'multichannel':
                self.embedding2 = nn.Embedding(self.vocab_size, self.word_dim, padding_idx=self.vocab_size - 1)
                self.embedding2.weight.data.copy_(torch.from_numpy(self.wv_matrix))
                self.embedding2.weight.requires_grad = False
                self.in_channel = 2

        self.convs1 = nn.ModuleList([nn.Conv2d(self.in_channel, out_channel, (K, self.word_dim)) for out_channel, K in zip(self.filter_num, self.filters)])
        layers, bn_dim = [], mid
        for i in range(num_fc):
            mid = mid if i != num_fc - 1 else out_dim
            if i == 0:
                # layers += ([nn.BatchNorm1d(sum(self.filter_num)), nn.Linear(sum(self.filter_num), mid)] if bn else [nn.Linear(sum(self.filter_num), mid)]) if num_fc > 1 else [nn.Dropout(dropout), nn.Linear(sum(self.filter_num), out_dim)]
                layers += [nn.Linear(sum(self.filter_num), mid)]
            # elif i == num_fc - 1:
            #     layers += [nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(mid, out_dim)]
            elif bn:
                layers += [nn.BatchNorm1d(bn_dim), nn.ReLU(inplace=True), nn.Linear(bn_dim, mid)]
            else:
                layers += [nn.ReLU(inplace=True), nn.Linear(bn_dim, mid)]
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        out = self.embedding(x).unsqueeze(1)

        if self.mode == 'multichannel':
            out2 = self.embedding2(x).unsqueeze(1)
            out = torch.cat((out, out2), 1)

        out = [F.relu(_conv(out)).squeeze(3) for _conv in self.convs1]
        out = [F.max_pool1d(o, o.size(2)).squeeze(2) for o in out]
        out1 = torch.cat(out, 1)
        # out = self.dropout(out)
        out2 = self.fc(out1)
        # out2 = out2 / out2.norm(dim=1, keepdim=True)
        return out2


class Half_MNIST_CNN_Net(nn.Module):
    def __init__(self, out_dim, in_channel=1, mid=1024, one_layer=False):
        super(Half_MNIST_CNN_Net, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Linear(64 * 5 * 1, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, out_dim)
        )
        

    def forward(self, x):
        # # in_size = 64
        # in_size = x.size(0) # one batch
        # # x: 64*10*12*12
        # out1 = F.relu(self.conv1(x))
        # # x: 64*20*4*4
        # out2 = F.relu(self.mp(self.conv2(out1)))
        # # x: 64*320
        # out3 = F.relu(self.conv3(out2))
        # out4 = F.relu(self.mp(self.conv4(out3)))

        # out4 = out4.view(in_size, -1) # flatten the tensor
        # # x: 64*10
        # out5 = F.relu(self.fc1(out4))
        # out6 = self.fc2(out5)
        return self.net(x)

class MNIST_CNN_Net(nn.Module):
    def __init__(self, out_dim, in_channel=1, mid=1024, one_layer=False):
        super(MNIST_CNN_Net, self).__init__()
        self.one_layer = one_layer
        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=3)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        # self.bn2 = nn.BatchNorm2d(64)
        self.mp = nn.MaxPool2d(2)
        # fully connect
        # self.fc1 = nn.Linear(64 * 5, 128)
        self.fc1 = nn.Linear(64 * 4 * 4, mid)
        # self.bn3 = nn.BatchNorm1d(128)
        if not self.one_layer:
            self.fc2 = nn.Linear(mid, out_dim)

    def forward(self, x):
        # in_size = 64
        in_size = x.size(0) # one batch
        # x: 64*10*12*12
        out1 = F.relu(self.conv1(x))
        # x: 64*20*4*4
        out2 = F.relu(self.mp(self.conv2(out1)))
        # x: 64*320
        out3 = F.relu(self.conv3(out2))
        out4 = F.relu(self.mp(self.conv4(out3)))

        out4 = out4.view(in_size, -1) # flatten the tensor
        # x: 64*10
        out5 = F.relu(self.fc1(out4))
        out6 = self.fc2(out5)
        return out6


class SAD_CNN_Net(nn.Module):
    def __init__(self, out_dim, in_channel=1, mid=1024, one_layer=False):
        super(SAD_CNN_Net, self).__init__()
        self.one_layer = one_layer
        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=3)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(64)
        self.mp = nn.MaxPool2d(2)
        # fully connect
        # self.fc1 = nn.Linear(64 * 5, 128)
        self.fc1 = nn.Linear(64 * 4 * 1, mid)
        # self.bn3 = nn.BatchNorm1d(128)
        if not one_layer:
            self.fc2 = nn.Linear(mid, out_dim)

    def forward(self, x):
        # in_size = 64
        in_size = x.size(0) # one batch
        # x: 64*10*12*12
        out1 = F.relu(self.conv1(x))
        # x: 64*20*4*4
        out2 = F.relu(self.mp(self.conv2(out1)))
        # x: 64*320
        out3 = F.relu(self.conv3(out2))
        out4 = F.relu(self.mp(self.conv4(out3)))

        out4 = out4.view(in_size, -1) # flatten the tensor
        # x: 64*10
        out5 = F.relu(self.fc1(out4))
        out6 = self.fc2(out5)
        return out6

class DMNet(nn.Module):
    def __init__(self, in_dims, out_dim, wv_matrix=None, num_fc_img=2, num_fc_txt=1, bn=False, dropout=0.5):
        super(DMNet, self).__init__()
        self.nets = nn.ModuleList()
        for dim in in_dims:
            if wv_matrix is not None and dim[-1] != 4096:
                self.nets.append(Text_CNN('multichannel', wv_matrix.shape[-1], out_dim=out_dim, filters=[3, 4, 5], filter_num=[100, 100, 100], dropout_prob=0.5, wv_matrix=wv_matrix, num_fc=num_fc_txt, bn=bn, dropout=dropout))
            elif len(dim) == 3 and dim[-1] == 14:
                self.nets.append(Half_MNIST_CNN_Net(in_channel=dim[0], out_dim=out_dim))
            elif len(dim) == 3 and dim[-1] == 28:
                self.nets.append(MNIST_CNN_Net(in_channel=dim[0], out_dim=out_dim))
            elif len(dim) == 3 and dim[-1] == 13:
                self.nets.append(SAD_CNN_Net(in_channel=dim[0], out_dim=out_dim))
            else:
                self.nets.append(FCNN(input_dim=dim[-1], out_dim=out_dim, num_fc=num_fc_img, bn=bn, dropout=dropout))

    def forward(self, in_data):
        outs, batch_size = [], 5000
        for v in range(len(in_data)):
            numcases = in_data[v].shape[0]
            if numcases > 5000:
                tmp = []
                for i in range(int(np.ceil(float(numcases) / batch_size))):
                    tmp.append(self.nets[v](in_data[v][i * batch_size: (i + 1) * batch_size]))
                outs.append(torch.cat(tmp))
            else:
                outs = [self.nets[v](in_data[v]) for v in range(len(in_data))]
        return outs

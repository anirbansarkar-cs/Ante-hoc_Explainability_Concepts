# -*- coding: utf-8 -*-
""" Code for training and evaluating Self-Explaining Neural Networks.
Copyright (C) 2018 David Alvarez-Melis <dalvmel@mit.edu>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License,
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import pdb
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function
# from torch.legacy import nn as nn_legacy
from torch.autograd import Variable

import torch.utils.model_zoo as model_zoo
# from custom_modules import SequentialWithArgs, FakeReLU

#===============================================================================
#==========================      REGULARIZERS        ===========================
#===============================================================================

# From https://discuss.pytorch.org/t/how-to-create-a-sparse-autoencoder-neural-network-with-pytorch/3703
class L1Penalty(Function):
    @staticmethod
    def forward(ctx, input, l1weight):
        ctx.save_for_backward(input)
        ctx.l1weight = l1weight
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_variables
        grad_input = input.clone().sign().mul(self.l1weight)
        grad_input += grad_output
        return grad_input


#===============================================================================
#=======================       MODELS FOR IMAGES       =========================
#===============================================================================

class input_conceptizer(nn.Module):
    """ Dummy conceptizer for images: each input feature (e.g. pixel) is a concept.

        Args:
            indim (int): input concept dimension
            outdim (int): output dimension (num classes)

        Inputs:
            x: Image (b x c x d x d) or Generic tensor (b x dim)

        Output:
            - H:  H(x) matrix of concepts (b x dim  x 1) (for images, dim = x**2)
                  or (b x dim +1 x 1) if add_bias = True
    """

    def __init__(self, add_bias = True):
        super(input_conceptizer, self).__init__()
        self.add_bias = add_bias
        self.learnable = False

    def forward(self, x):
        if len(list(x.size())) == 4:
            # This is an images
            out = x.view(x.size(0), x.size(-1)**2, 1)
        else:
            out = x.view(x.size(0), x.size(1), 1)
        if self.add_bias:
            pad = (0,0,0,1) # Means pad to next to last dim, 0 at beginning, 1 at end
            out = F.pad(out, pad, mode = 'constant', value = 1)
        return out


class AutoEncoder(nn.Module):
    """
        A general autoencoder meta-class with various penalty choices.

        Takes care of regularization, etc. Children of the AutoEncoder class
        should implement encode() and decode() functions.
        Encode's output should be same size/dim as decode input and viceversa.
        Ideally, AutoEncoder should not need to do any resizing (TODO).

    """
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # self.sparsity = sparsity is not None
        # self.l1weight = sparsity if (sparsity) else 0.1

    def forward(self, x):
        encoded, logits, out = self.encode(x)
        # if self.sparsity:
        #     #encoded = L1Penalty.apply(encoded, self.l1weight)    # Didn't work
        decoded = self.decode(encoded)
        return encoded, logits, decoded.view_as(x), out

class image_fcc_conceptizer(AutoEncoder):
    """ MLP-based conceptizer for concept basis learning.

        Args:
            din (int): input size
            nconcept (int): number of concepts
            cdim (int): concept dimension

        Inputs:
            x: Image (b x c x d x d)

        Output:
            - Th: Tensor of encoded concepts (b x nconcept x cdim)
    """

    def __init__(self, din, nconcept, cdim): #, sparsity = None):
        super(image_fcc_conceptizer, self).__init__()
        self.din      = din        # Input dimension
        self.nconcept = nconcept   # Number of "atoms"/concepts
        self.cdim     = cdim       # Dimension of each concept
        self.learnable = True

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128), nn.ReLU(True),
            nn.Linear(128, 64), nn.ReLU(True),
            nn.Linear(64, 12), nn.ReLU(True),
            nn.Linear(12, nconcept*cdim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(nconcept*cdim, 12), nn.ReLU(True),
            nn.Linear(12, 64), nn.ReLU(True),
            nn.Linear(64, 128), nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Tanh()
        )
    def encode(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x).view(-1, self.nconcept, self.cdim)
        return encoded

    def decode(self, z):
        decoded = self.decoder(z.view(-1, self.cdim*self.nconcept))
        return decoded

    # def forward(self, x):
    #     x_dims = x.size()
    #     x = x.view(x.size(0), -1)
    #     encoded = self.encoder(x).view(-1, self.nconcept, self.cdim)
    #     decoded = self.decoder(encoded.view(-1, self.cdim*self.nconcept)).view(x_dims)
    #     return encoded, decoded
    #


class image_cnn_conceptizer(AutoEncoder):
    """ CNN-based conceptizer for concept basis learning.

        Args:
            din (int): input size
            nconcept (int): number of concepts
            cdim (int): concept dimension

        Inputs:
            x: Image (b x c x d x d)

        Output:
            - Th: Tensor of encoded concepts (b x nconcept x cdim)
    """

    def __init__(self, din, nconcept, cdim=None, nchannel =1): #, sparsity = None):
        super(image_cnn_conceptizer, self).__init__()
        self.din      = din        # Input dimension
        self.nconcept = nconcept   # Number of "atoms"/concepts
        self.cdim     = cdim       # Dimension of each concept
        self.nchannel = nchannel
        self.learnable = True
        self.add_bias = False
        self.dout     = int(np.sqrt(din)//4 - 3*(5-1)//4) # For kernel = 5 in both, and maxppol stride = 2 in both

        # Encoding
        self.conv1  = nn.Conv2d(nchannel,10, kernel_size=5)    # b, 10, din - (k -1),din - (k -1)
        # after pool layer (functional)                        # b, 10,  (din - (k -1))/2, idem
        self.conv2  = nn.Conv2d(10, nconcept, kernel_size=5)   # b, 10, (din - (k -1))/2 - (k-1), idem
        # after pool layer (functional)                        # b, 10,  din/4 - 3(k-1))/4, idem
        self.linear1 = nn.Linear(self.dout**2, self.cdim)       # b, nconcepts, cdim
        # self.linear2 = nn.Linear(self.dout**2, self.cdim)
        # self.linear3 = nn.Linear(self.nconcept, 10)
        self.linear2 = nn.Linear(self.nconcept * (self.dout**2), 128)
        self.linear3 = nn.Linear(128,64)
        self.linear4 = nn.Linear(64,20)
        self.linear5 = nn.Linear(20,10)

        # Decoding
        self.unlinear = nn.Linear(self.cdim,self.dout**2)                # b, nconcepts, dout*2
        self.deconv3  = nn.ConvTranspose2d(nconcept, 16, 5, stride = 2)  # b, 16, (dout-1)*2 + 5, 5
        self.deconv2  = nn.ConvTranspose2d(16, 8, 5)                     # b, 8, (dout -1)*2 + 9
        self.deconv1  = nn.ConvTranspose2d(8, nchannel, 2, stride=2, padding=1) # b, nchannel, din, din


    def encode(self, x):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # x.to(device)
        # x = x.to('cpu')
        # print (x.type())
        p = self.conv1(x)
        # print (p.type())
        p       = F.relu(F.max_pool2d(p, 2))
        # p       = F.relu(F.max_pool2d(self.conv1(x), 2))
        p       = F.relu(F.max_pool2d(self.conv2(p), 2))
        # print (p.shape)
        encoded = self.linear1(p.view(-1, self.nconcept, self.dout**2))
        # intermid = self.linear2(p.view(-1, self.nconcept, self.dout**2))
        # print (intermid.shape)
        # logits = self.linear3(intermid.view(-1, self.nconcept))
        intermid = self.linear2(p.view(-1, self.nconcept*(self.dout**2)))
        intermid1 = F.relu(self.linear3(intermid.view(-1,128)))
        intermid2 = F.relu(self.linear4(intermid1.view(-1,64)))
        logits = self.linear5(intermid2.view(-1,20))


        # print (self.din)
        # print (self.nconcept)
        # print (self.cdim)
        # print (self.dout)
        # print (p.shape)
        # print (encoded.shape)
        # print (intermid.shape)
        # print (logits.shape)
        return encoded, logits

    def decode(self, z):
        q       = self.unlinear(z).view(-1, self.nconcept, self.dout, self.dout)
        q       = F.relu(self.deconv3(q))
        q       = F.relu(self.deconv2(q))
        decoded = F.tanh(self.deconv1(q))
        return decoded
    #
    #
    # def forward(self, x):
    #     
    #
    #     # Encoding
    #     p       = F.relu(F.max_pool2d(self.conv1(x), 2))
    #     p       = F.relu(F.max_pool2d(self.conv2(p), 2))
    #     encoded = self.linear(p.view(-1, self.nconcept, 4 * 4))
    #
    #     # Decoding
    #     q       = self.unlinear(encoded).view(-1, self.nconcept, 4, 4)
    #     q       = F.relu(self.deconv3(q))
    #     q       = F.relu(self.deconv2(q))
    #     decoded = F.tanh(self.deconv1(q))
    #     # decoded =
    #     #
    #     # encoded  = self.Linear(conv_out.view())
    #     # decoded = self.decoder(encoded)
    #     # print(encoded.size())
    #     # encoded = encoded.view(x.size(0), self.nconcept, self.cdim)
    #     return encoded, decoded


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class image_resnet_conceptizer(AutoEncoder):
    """ CNN-based conceptizer for concept basis learning.

        Args:
            din (int): input size
            nconcept (int): number of concepts
            cdim (int): concept dimension

        Inputs:
            x: Image (b x c x d x d)

        Output:
            - Th: Tensor of encoded concepts (b x nconcept x cdim)
    """

    def __init__(self, din, nconcept, nclass, cdim=None, nchannel =1, block=Bottleneck, num_blocks=[3,4,23,3]): #, sparsity = None):
        super(image_resnet_conceptizer, self).__init__()
        self.din      = din        # Input dimension
        self.nconcept = nconcept   # Number of "atoms"/concepts
        self.nclass = nclass       # Number of classes
        self.cdim     = cdim       # Dimension of each concept
        self.nchannel = nchannel
        self.learnable = True
        self.add_bias = False
        self.dout     = int(np.sqrt(din)//4 - 3*(5-1)//4) # For kernel = 5 in both, and maxppol stride = 2 in both
        self.concept_in_planes = 64

        # Encoding
        # self.conv1  = nn.Conv2d(nchannel,10, kernel_size=5)    # b, 10, din - (k -1),din - (k -1)
        # # after pool layer (functional)                        # b, 10,  (din - (k -1))/2, idem
        # self.conv2  = nn.Conv2d(10, nconcept, kernel_size=5)   # b, 10, (din - (k -1))/2 - (k-1), idem
        # # after pool layer (functional)                        # b, 10,  din/4 - 3(k-1))/4, idem
        # self.linear1 = nn.Linear(self.dout**2, self.cdim)       # b, nconcepts, cdim
        # # self.linear2 = nn.Linear(self.dout**2, self.cdim)
        # # self.linear3 = nn.Linear(self.nconcept, 10)
        # self.linear2 = nn.Linear(self.nconcept * (self.dout**2), 128)
        # self.linear3 = nn.Linear(128,64)
        # self.linear4 = nn.Linear(64,20)
        # self.linear5 = nn.Linear(20,10)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conc_layer1 = self._make_layer_concept(block, 64, num_blocks[0], stride=1)
        self.conc_layer2 = self._make_layer_concept(block, 128, num_blocks[1], stride=2)
        self.conc_layer3 = self._make_layer_concept(block, 256, num_blocks[2], stride=2)
        self.conc_layer4 = self._make_layer_concept(block, 512, num_blocks[3], stride=2)
        # self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conc_linear1 = nn.Linear(512*block.expansion, self.nconcept)
        self.conc_linear2 = nn.Linear(512*block.expansion, self.nclass)
        # self.conc_linear3 = nn.Linear(256, num_concepts)

        # Decoding
        self.unlinear = nn.Linear(self.cdim,self.dout**2)                # b, nconcepts, dout*2
        self.deconv3  = nn.ConvTranspose2d(nconcept, 16, 5, stride = 2)  # b, 16, (dout-1)*2 + 5, 5
        self.deconv2  = nn.ConvTranspose2d(16, 8, 5)                     # b, 8, (dout -1)*2 + 9
        self.deconv1  = nn.ConvTranspose2d(8, nchannel, 2, stride=2, padding=1) # b, nchannel, din, din

        # self.unlinear = nn.Linear(self.nconcept, 1024)
        # self.deconv_6 = nn.ConvTranspose2d(1024, 512, kernel_size = 5, stride = 2, padding = 1, bias = True) # 8, 7,7
        # # self.deconv_6 = nn.ConvTranspose2d(512, 512, kernel_size = 7, stride = 2, padding = 0, bias = True) # 8, 7,7
        # self.deconv_5 = nn.ConvTranspose2d(512, 256, kernel_size = 7, stride = 2, padding = 0, bias = True) # 8, 7,7
        # self.deconv_4 = nn.ConvTranspose2d(256, 128, kernel_size = 5, stride = 2, padding = 0, bias = True) # 8, 7,7
        # self.deconv_3 = nn.ConvTranspose2d(128, 64, kernel_size = 7, stride = 2, padding = 0, bias = True) # 16, 9, 9
        # self.deconv_2 = nn.ConvTranspose2d(64, 32, kernel_size = 5, stride = 2, padding = 1, bias = True) # 3, 17, 17
        # self.deconv_1 = nn.ConvTranspose2d(32, 3, kernel_size = 4, stride = 2, padding = 0, bias = True) # 8, 32, 32


    def _make_layer_concept(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        # print (strides)
        for stride in strides:
            layers.append(block(self.concept_in_planes, planes, stride))
            self.concept_in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def encode(self, x):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # x.to(device)
        # x = x.to('cpu')
        # print (x.type())

        # p = self.conv1(x)
        # # print (p.type())
        # p       = F.relu(F.max_pool2d(p, 2))
        # # p       = F.relu(F.max_pool2d(self.conv1(x), 2))
        # p       = F.relu(F.max_pool2d(self.conv2(p), 2))
        # # print (p.shape)
        # encoded = self.linear1(p.view(-1, self.nconcept, self.dout**2))
        # # intermid = self.linear2(p.view(-1, self.nconcept, self.dout**2))
        # # print (intermid.shape)
        # # logits = self.linear3(intermid.view(-1, self.nconcept))
        # intermid = self.linear2(p.view(-1, self.nconcept*(self.dout**2)))
        # intermid1 = F.relu(self.linear3(intermid.view(-1,128)))
        # intermid2 = F.relu(self.linear4(intermid1.view(-1,64)))
        # logits = self.linear5(intermid2.view(-1,20))

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.conc_layer1(out)
        out = self.conc_layer2(out)
        out = self.conc_layer3(out)
        out = self.conc_layer4(out)
        # out = F.relu(self.conv2(out))
        out = F.avg_pool2d(out, 4)
        # out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        # print (out.shape)
        encoded = F.relu(self.conc_linear1(out)).view(-1, self.nconcept, self.cdim)
        logits = self.conc_linear2(out)

        # print (self.din)
        # print (self.nconcept)
        # print (self.cdim)
        # print (self.dout)
        # print (p.shape)
        # print (encoded.shape)
        # print (intermid.shape)
        # print (logits.shape)
        return encoded, logits, out

    def decode(self, z):
        q       = self.unlinear(z).view(-1, self.nconcept, self.dout, self.dout)
        q       = F.relu(self.deconv3(q))
        q       = F.relu(self.deconv2(q))
        decoded = F.tanh(self.deconv1(q))
        return decoded
        # print (z.shape)
        # recon_out = self.unlinear(z.squeeze()).view(-1, 1024, 1, 1)
        # recon_out = F.relu(self.deconv_6(recon_out))
        # recon_out = F.relu(self.deconv_5(recon_out))
        # recon_out = F.relu(self.deconv_4(recon_out))
        # recon_out = F.relu(self.deconv_3(recon_out))
        # recon_out = F.relu(self.deconv_2(recon_out))
        # recon_out = F.relu(self.deconv_1(recon_out))
        # return recon_out
    #
    #
    # def forward(self, x):
    #     
    #
    #     # Encoding
    #     p       = F.relu(F.max_pool2d(self.conv1(x), 2))
    #     p       = F.relu(F.max_pool2d(self.conv2(p), 2))
    #     encoded = self.linear(p.view(-1, self.nconcept, 4 * 4))
    #
    #     # Decoding
    #     q       = self.unlinear(encoded).view(-1, self.nconcept, 4, 4)
    #     q       = F.relu(self.deconv3(q))
    #     q       = F.relu(self.deconv2(q))
    #     decoded = F.tanh(self.deconv1(q))
    #     # decoded =
    #     #
    #     # encoded  = self.Linear(conv_out.view())
    #     # decoded = self.decoder(encoded)
    #     # print(encoded.size())
    #     # encoded = encoded.view(x.size(0), self.nconcept, self.cdim)
    #     return encoded, decoded

def ResNet18(din, nconcept, cdim=None, nchannel =1):
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


#EfficientNet

def swish(x):
    return x * x.sigmoid()


def drop_connect(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


class SE(nn.Module):
    '''Squeeze-and-Excitation block with Swish.'''

    def __init__(self, in_channels, se_channels):
        super(SE, self).__init__()
        self.se1 = nn.Conv2d(in_channels, se_channels,
                             kernel_size=1, bias=True)
        self.se2 = nn.Conv2d(se_channels, in_channels,
                             kernel_size=1, bias=True)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = swish(self.se1(out))
        out = self.se2(out).sigmoid()
        out = x * out
        return out


class Block(nn.Module):
    '''expansion + depthwise + pointwise + squeeze-excitation'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 expand_ratio=1,
                 se_ratio=0.,
                 drop_rate=0.):
        super(Block, self).__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        self.expand_ratio = expand_ratio

        # Expansion
        channels = expand_ratio * in_channels
        self.conv1 = nn.Conv2d(in_channels,
                               channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        # Depthwise conv
        self.conv2 = nn.Conv2d(channels,
                               channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=(1 if kernel_size == 3 else 2),
                               groups=channels,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        # SE layers
        se_channels = int(in_channels * se_ratio)
        self.se = SE(channels, se_channels)

        # Output
        self.conv3 = nn.Conv2d(channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Skip connection if in and out shapes are the same (MV-V2 style)
        self.has_skip = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        out = x if self.expand_ratio == 1 else swish(self.bn1(self.conv1(x)))
        out = swish(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        if self.has_skip:
            if self.training and self.drop_rate > 0:
                out = drop_connect(out, self.drop_rate)
            out = out + x
        return out


class EfficientNet(AutoEncoder):
    def __init__(self, din, nconcept, num_classes=10, cdim=None, nchannel=3):
        super(EfficientNet, self).__init__()
        self.din = din
        self.num_concept = nconcept
        self.cdim     = cdim
        self.learnable = True
        self.dout     = int(np.sqrt(din)//4 - 3*(5-1)//4) # For kernel = 5 in both, and maxppol stride = 2 in both
        self.cfg = {
                    'num_blocks': [1, 2, 2, 3, 3, 4, 1],
                    'expansion': [1, 6, 6, 6, 6, 6, 6],
                    'out_channels': [16, 24, 40, 80, 112, 192, 320],
                    'kernel_size': [3, 3, 5, 3, 5, 5, 3],
                    'stride': [1, 2, 2, 2, 1, 2, 1],
                    'dropout_rate': 0.2,
                    'drop_connect_rate': 0.2,
                }
        self.conv1 = nn.Conv2d(3,
                               32,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_channels=32)
        self.linear = nn.Linear(self.cfg['out_channels'][-1], num_classes)
        self.linear1 = nn.Linear(self.cfg['out_channels'][-1], self.num_concept)

        # Decoding
        self.unlinear = nn.Linear(self.cdim,self.dout**2)                # b, nconcepts, dout*2
        self.deconv3  = nn.ConvTranspose2d(nconcept, 16, 5, stride = 2)  # b, 16, (dout-1)*2 + 5, 5
        self.deconv2  = nn.ConvTranspose2d(16, 8, 5)                     # b, 8, (dout -1)*2 + 9
        self.deconv1  = nn.ConvTranspose2d(8, nchannel, 2, stride=2, padding=1) # b, nchannel, din, din

    def _make_layers(self, in_channels):
        layers = []
        cfg = [self.cfg[k] for k in ['expansion', 'out_channels', 'num_blocks', 'kernel_size',
                                     'stride']]
        b = 0
        blocks = sum(self.cfg['num_blocks'])
        for expansion, out_channels, num_blocks, kernel_size, stride in zip(*cfg):
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                drop_rate = self.cfg['drop_connect_rate'] * b / blocks
                layers.append(
                    Block(in_channels,
                          out_channels,
                          kernel_size,
                          stride,
                          expansion,
                          se_ratio=0.25,
                          drop_rate=drop_rate))
                in_channels = out_channels
        return nn.Sequential(*layers)

    def encode(self, x):
        out = swish(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        dropout_rate = self.cfg['dropout_rate']
        if self.training and dropout_rate > 0:
            out = F.dropout(out, p=dropout_rate)
        # print (out.shape)
        logits = self.linear(out)
        encoded = self.linear1(out).view(-1, self.num_concept, 1)
        return encoded, logits

    def decode(self, z):
        q       = self.unlinear(z).view(-1, self.num_concept, self.dout, self.dout)
        q       = F.relu(self.deconv3(q))
        q       = F.relu(self.deconv2(q))
        decoded = F.tanh(self.deconv1(q))
        return decoded


# def EfficientNetB0():
#     cfg = {
#         'num_blocks': [1, 2, 2, 3, 3, 4, 1],
#         'expansion': [1, 6, 6, 6, 6, 6, 6],
#         'out_channels': [16, 24, 40, 80, 112, 192, 320],
#         'kernel_size': [3, 3, 5, 3, 5, 5, 3],
#         'stride': [1, 2, 2, 2, 1, 2, 1],
#         'dropout_rate': 0.2,
#         'drop_connect_rate': 0.2,
#     }
#     return EfficientNet(cfg)


class Bottleneck_D(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck_D, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(AutoEncoder):
    def __init__(self, din, nconcept, num_classes=10, cdim=None, nchannel=3, block=Bottleneck_D, nblocks=[6,12,24,16], growth_rate=12, reduction=0.5):
        super(DenseNet, self).__init__()
        self.din = din
        self.num_concept = nconcept
        self.cdim     = cdim
        self.learnable = True
        self.dout     = int(np.sqrt(din)//4 - 3*(5-1)//4) # For kernel = 5 in both, and maxppol stride = 2 in both
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)
        self.linear1 = nn.Linear(num_planes, self.num_concept)

        # Decoding
        self.unlinear = nn.Linear(self.cdim,self.dout**2)                # b, nconcepts, dout*2
        self.deconv3  = nn.ConvTranspose2d(nconcept, 16, 5, stride = 2)  # b, 16, (dout-1)*2 + 5, 5
        self.deconv2  = nn.ConvTranspose2d(16, 8, 5)                     # b, 8, (dout -1)*2 + 9
        self.deconv1  = nn.ConvTranspose2d(8, nchannel, 2, stride=2, padding=1) # b, nchannel, din, din

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def encode(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        encoded = self.linear1(out).view(-1, self.num_concept, 1)
        return encoded, logits

    def decode(self, z):
        q       = self.unlinear(z).view(-1, self.num_concept, self.dout, self.dout)
        q       = F.relu(self.deconv3(q))
        q       = F.relu(self.deconv2(q))
        decoded = F.tanh(self.deconv1(q))
        return decoded

def DenseNet121():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

def DenseNet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def DenseNet201():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def DenseNet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

def densenet_cifar():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12)


#===============================================================================
#=======================       MODELS FOR TEXT       ===========================
#===============================================================================


class text_input_conceptizer(nn.Module):
    """ Dummy conceptizer for images: each token is a concept.

        Args:

        Inputs:
            x: Text tensor (one hot) (b x 1 x L)

        Output:
            - H:  H(x) matrix of concepts (b x L x 1) [TODO: generalize to set maybe?]
    """

    def __init__(self):
        super(text_input_conceptizer, self).__init__()
        # self.din    = din
        # self.nconcept = nconcept
        # self.cdim   = cdim

    def forward(self, x):
        #return x.view(x.size(0), x.size(-1)**2, 1)
        #return x.transpose(1,2)._fill(1)
        return Variable(torch.ones(x.size())).transpose(1,2)


class text_embedding_conceptizer(nn.Module):
    """ H(x): word embedding of word x.

        Can be used in a non-learnt way (e.g. if embeddings are already trained)
        TODO: Should we pass this to parametrizer?

        Args:
            embeddings (optional): pretrained embeddings to initialize method

        Inputs:
            x: array of word indices (L X B X 1)

        Output:
            enc: encoded representation (L x B x D)
    """

    def __init__(self, embeddings = None, train_embeddings = False):
        super(text_embedding_conceptizer, self).__init__()
        vocab_size, hidden_dim = embeddings.shape
        self.hidden_dim = hidden_dim
        self.embedding_layer = nn.Embedding(vocab_size,hidden_dim)
        print(type(embeddings))
        if embeddings is not None:
            self.embedding_layer.weight.data = torch.from_numpy( embeddings )
            print('Text conceptizer: initializing embeddings')
        self.embedding_layer.weight.requires_grad = train_embeddings
        if embeddings is None and not train_embeddings:
            print('Warning: embeddings not initialized from pre-trained and train = False')


    def forward(self, x):
        encoded = self.embedding_layer(x.squeeze(1))
        #encoded = encoded.transpose(0,1) # To have Batch dim again in 0
        return encoded

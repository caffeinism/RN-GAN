import torch
import torch.nn as nn 
import math
from torch.autograd import Variable
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

def nl():
    return nn.LeakyReLU(0.2, inplace=True)

def conv(ic, oc, k, s, p, bn=True):
    model = []

    model.append(nn.Conv2d(ic, oc, k, s, p))
    if bn:
        model.append(nn.BatchNorm2d(oc))
    model.append(nl())

    return nn.Sequential(*model)

class FeatureExtractor(nn.Module):
    def __init__(self, filters):
        super(FeatureExtractor, self).__init__()
        
        layers = []
        
        for i, (in_channels, out_channels, kernel_size, stride, padding) in enumerate(filters):
            layers.append(conv(in_channels, out_channels, kernel_size, stride, padding, bn=True if i > 0 else False))
            self.out_channels = out_channels            
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
    
    
class ModelG(nn.Module):
    def __init__(self, channels):
        super(ModelG, self).__init__()
        
        channels[0] = (channels[0] + 2) * 2

        layers = []
        for in_plane, out_plane in zip(channels[:-1], channels[1:]):
            layers.append(nn.Linear(in_plane, out_plane))
            layers.append(nl())
                     
        self.out_channels = channels[-1]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    

class ModelF(nn.Module):
    def __init__(self, channels):
        super(ModelF, self).__init__()

        layers = []
        for in_plane, out_plane in zip(channels[:-1], channels[1:]):
            layers.append(nn.Linear(in_plane, out_plane))
            layers.append(nl())
        layers.append(nn.Dropout(p=0.5))
        layers.append(nn.Linear(channels[-1], 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
        
        
class RNDiscriminator(nn.Module):
    def __init__(self, feature_extractor, g, f):
        super(RNDiscriminator, self).__init__()
        self.feature_extractor = feature_extractor
        self.g = g
        self.f = f

        self.featuremap_size = None
        
    def forward(self, image):
        x = self.feature_extractor(image)
        if not self.featuremap_size:
            self.featuremap_size = x.size(2)

        assert self.featuremap_size == x.size(2)

        batch_size = x.size(0)
        k = x.size(1)
        d = x.size(2)
        
        # tag arbitrary coordinate
        coordinate = torch.arange(-1, 1 + 0.00001, 2 / (d-1)).cuda()
        coordinate_x = coordinate.expand(batch_size, 1, d, d)
        coordinate_y = coordinate.view(d, 1).expand(batch_size, 1, d, d)
        x = torch.cat([x, coordinate_x, coordinate_y], 1)
        k += 2
        
        x = x.view(batch_size, k, d ** 2).permute(0, 2, 1)
        
        # concatnate o_i, o_j and q
        x_left = x.unsqueeze(1).repeat(1, d ** 2, 1, 1).view(batch_size, d ** 4, k)
        x_right = x.unsqueeze(2).repeat(1, 1, d ** 2, 1).view(batch_size, d ** 4, k)
        x = torch.cat([x_left, x_right], 2)        
        
        x = x.view(batch_size * (d ** 4), k * 2)
        
        # g(o_i, o_j, q)
        x = self.g(x)
        x = x.view(batch_size, d ** 4, x.size(1))
        # Σg(o_i, o_j, q)
        x = torch.sum(x, dim=1)
        # f(Σg(o_i, o_j, q))
        x = self.f(x)
        
        return x
    
class NormalDiscriminator(nn.Module):
    def __init__(self, ndf, nc, n_layer):
        super(NormalDiscriminator, self).__init__()

        layer = []
        
        layer.append(nn.Conv2d(nc, ndf, 4, 2, 1))
        layer.append(nl())
        
        layer.append(nn.Conv2d(ndf, ndf * 2, 4, 2, 1))
        layer.append(nn.BatchNorm2d(ndf * 2))
        layer.append(nl())
        
        layer.append(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1))
        layer.append(nn.BatchNorm2d(ndf * 4))
        layer.append(nl())
        
        layer.append(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1))
        layer.append(nn.BatchNorm2d(ndf * 8))
        layer.append(nl())

        for _ in range(n_layer):
            layer.append(nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1))
            layer.append(nn.BatchNorm2d(ndf * 8))
            layer.append(nl())

        layer.append(nn.Conv2d(ndf * 8, 1, 4, 1, 0))

        self.model = nn.Sequential(*layer)

    def forward(self, x):
        return self.model(x).view(-1, 1)
    
def define_D(model_type, image_size, ndf, nc):
    n_layer = int(math.log2(image_size)) - 6
    assert n_layer >= 0
    if model_type == 'dcgan':
        return NormalDiscriminator(64, 3, n_layer)
    elif model_type == 'rngan':        
        feature_extractor = FeatureExtractor([
            (nc, ndf * 1, 4, 2, 1),
            (ndf * 1, ndf * 2, 4, 2, 1),
            (ndf * 2, ndf * 4, 4, 2, 1),
            (ndf * 4, ndf * 8, 4, 2, 1),
        ] + [(ndf * 8, ndf * 8, 4, 2, 1)] * n_layer)

        prev_out_channels = feature_extractor.out_channels
        g = ModelG([prev_out_channels, 512, 512, 512, 512])
        
        prev_out_channels = g.out_channels
        f = ModelF([prev_out_channels, 512, 512])
        
        return RNDiscriminator(feature_extractor, g, f)
    else:
        raise NotImplementedError

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc, image_size):
        super(Generator, self).__init__()

        n_layer = int(math.log2(image_size)) - 6
        assert n_layer >= 0
        
        layer = []
        
        layer.append(nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0))
        layer.append(nn.BatchNorm2d(ngf * 8))
        layer.append(nl())
        
        for _ in range(n_layer):
            layer.append(nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1))
            layer.append(nn.BatchNorm2d(ngf * 8))
            layer.append(nl())

        layer.append(nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1))
        layer.append(nn.BatchNorm2d(ngf * 4))
        layer.append(nl())
        layer.append(nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1))
        layer.append(nn.BatchNorm2d(ngf * 2))
        layer.append(nl())
        layer.append(nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1))
        layer.append(nn.BatchNorm2d(ngf))
        layer.append(nl())
        layer.append(nn.ConvTranspose2d(ngf, nc, 4, 2, 1))
        layer.append(nn.Tanh())
        
        self.model = nn.Sequential(*layer)

    def forward(self, z):
        return self.model(z)


def define_G(model_type, image_size, nz, ngf, nc):
    if model_type == 'dcgan':
        return Generator(nz, ngf, nc, image_size)
    else:
        raise NotImplementedError
import pickle
import random
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.datasets as dset
import numpy as np
import random
import os
import argparse
from networks import define_D, define_G
from tqdm import tqdm
from utils import init_weights 
import tf_recorder as tensorboard

parser = argparse.ArgumentParser("train")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--lr_d', type=float, default=2e-4)
parser.add_argument('--lr_g', type=float, default=2e-4)
parser.add_argument('--dataset', type=str, default='./celebA-HQ')
parser.add_argument('--init', type=str, default='kaiming')
parser.add_argument('--nz', type=int, default=128)
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--result_dir', type=str, default='result')
parser.add_argument('--model_dir', type=str, default='models')
parser.add_argument('--gan_type', type=str, required=True)
parser.add_argument('--g_type', type=str, default='dcgan')
parser.add_argument('--d_type', type=str, default='rngan')
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--n_cpu', type=int, default=4)
config, _ = parser.parse_known_args()

if __name__ == '__main__':
    os.system('mkdir ' + config.result_dir)
    os.system('mkdir ' + config.model_dir)
    torch.manual_seed(config.seed)

    discriminator = define_D(config.d_type, config.image_size, config.ndf, 3)
    generator = define_G(config.g_type, config.image_size, config.nz, config.ngf, 3)
    if config.gan_type == 'vanila':
        criterion = nn.BCEWithLogitsLoss()
    elif config.gan_type == 'lsgan':
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError

    dataset = dset.ImageFolder(config.dataset, transform=transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
    dataloader = torch.utils.data.DataLoader(dataset, config.batch_size, shuffle=True, 
                                             num_workers=config.n_cpu, pin_memory=True)

    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.lr_d, betas=(0, 0.9))
    optimizer_G = optim.Adam(generator.parameters(), lr=config.lr_g, betas=(0, 0.9))
    
    discriminator.cuda()
    generator.cuda()

    init_weights(discriminator, config.init)
    init_weights(generator, config.init)

    tb = tensorboard.tf_recorder('RNGAN')

    print(generator)
    print(discriminator)
    print(config)
    
    for epoch in range(config.n_epoch):
        generator.train()
        for data, _ in tqdm(dataloader):
            batch_size = data.size(0)
            label_real = torch.full((batch_size, 1), 1.0).cuda()
            label_fake = torch.full((batch_size, 1), 0.0).cuda()
            
            real = data.cuda()
            z = torch.randn(batch_size, config.nz, 1, 1).cuda()
            fake = generator(z)
            
            # train discriminator
            optimizer_D.zero_grad()

            d_real = discriminator(real)
            loss_d_real = criterion(d_real, label_real)
            loss_d_real.backward()

            d_fake = discriminator(fake.detach())
            loss_d_fake = criterion(d_fake, label_fake)
            loss_d_fake.backward()

            loss_d = loss_d_real + loss_d_fake

            optimizer_D.step()

            # train generator
            optimizer_G.zero_grad()
            d_fake = discriminator(fake)
            loss_g = criterion(d_fake, label_real)
            loss_g.backward()

            optimizer_G.step()

            tb.add_scalar('loss_g', loss_g.item())
            tb.add_scalar('loss_d', loss_d.item())
            tb.iter()

        generator.eval()
        with torch.no_grad():
            z = torch.randn(config.batch_size, config.nz, 1, 1).cuda()
            fake = generator(z)

        fake = (fake + 1.0) / 2.0
        vutils.save_image(fake, '{}/epoch_{:04d}.png'.format(config.result_dir, epoch))
        torch.save({
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'optimizer_g': optimizer_G.state_dict(),
            'optimizer_d': optimizer_D.state_dict(),
        }, '{}/{:04d}.pth'.format(config.model_dir, epoch))

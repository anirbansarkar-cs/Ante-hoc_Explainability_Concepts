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

# Standard Imports
import sys, os
import numpy as np
import pdb
import pickle
import argparse
import operator
import matplotlib
# import matplotlib.pyplot as plt
from PIL import Image

# Torch Imports
import torch
from torch.utils.data import TensorDataset
from torch.autograd import Variable
import torchvision
from torchvision import transforms
import torchvision.datasets as dset
from torchvision.datasets import CIFAR10
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data.dataloader as dataloader

# # Imports from my other repos
# from robust_interpret.explainers import gsenn_wrapper
# from robust_interpret.utils import lipschitz_boxplot, lipschitz_argmax_plot

# Local imports
from SENN.arglist import get_senn_parser #parse_args as parse_senn_args
from SENN.models import GSENN
from SENN.conceptizers import image_fcc_conceptizer, image_cnn_conceptizer, input_conceptizer, image_resnet_conceptizer, EfficientNet, DenseNet
# from SENN.conceptizers import *

from SENN.parametrizers import image_parametrizer, torchvision_parametrizer, vgg_parametrizer
from SENN.aggregators import linear_scalar_aggregator, additive_scalar_aggregator
from SENN.trainers import HLearningClassTrainer, VanillaClassTrainer, GradPenaltyTrainer
from SENN.utils import plot_theta_stability, generate_dir_names, noise_stability_plots, concept_grid
from SENN.eval_utils import estimate_dataset_lipschitz

# from prettytable import PrettyTable

os.environ['CUDA_VISIBLE_DEVICES']='6,7'

def load_cifar_data(valid_size=0.1, shuffle=True, resize = None, random_seed=2008, batch_size = 64,
                    num_workers = 1):
    """
        We return train and test for plots and post-training experiments
    """

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Normalize the test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # normalized according to pytorch torchvision guidelines https://chsasank.github.io/vision/models.html
    train = CIFAR10('/dataset/CIFAR-10/', train=True, download=True, transform=transform_train)
    test  = CIFAR10('/dataset/CIFAR-10/', train=False, download=True, transform=transform_test)

    num_train = len(train)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)


    # Create DataLoader
    dataloader_args = dict(batch_size=batch_size,num_workers=num_workers)
    train_loader = dataloader.DataLoader(train, sampler=train_sampler, **dataloader_args)
    valid_loader = dataloader.DataLoader(train, sampler=valid_sampler, **dataloader_args)
    dataloader_args['shuffle'] = False
    test_loader = dataloader.DataLoader(test, **dataloader_args)
    # print (test_loader.shape)

    return train_loader, valid_loader, test_loader, train, test

def parse_args():
    senn_parser = get_senn_parser()

    ### Local ones
    parser = argparse.ArgumentParser(parents =[senn_parser],add_help=False,
        description='Interpteratbility robustness evaluation on MNIST')

    # #setup
    parser.add_argument('-d','--datasets', nargs='+',
                        default = ['heart', 'ionosphere', 'breast-cancer','wine','heart',
                        'glass','diabetes','yeast','leukemia','abalone'], help='<Required> Set flag')
    parser.add_argument('--lip_calls', type=int, default=10,
                        help='ncalls for bayes opt gp method in Lipschitz estimation')
    parser.add_argument('--lip_eps', type=float, default=0.01,
                        help='eps for Lipschitz estimation')
    parser.add_argument('--lip_points', type=int, default=100,
                        help='sample size for dataset Lipschitz estimation')
    parser.add_argument('--optim', type=str, default='gp',
                        help='black-box optimization method')

    #####

    args = parser.parse_args()
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    return args

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print("Total Trainable Params: {}".format(total_params))
    return total_params

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.nclasses = 10
    args.theta_dim = args.nclasses
    if (args.theta_arch == 'simple') or ('vgg' in args.theta_arch):
        H, W = 32, 32
    else:
        # Need to resize to have access to torchvision's models
        H, W = 224, 224
    args.input_dim = H*W

    model_path, log_path, results_path = generate_dir_names('cifar', args)

    train_loader, valid_loader, test_loader, train_tds, test_tds = load_cifar_data(
                        batch_size=args.batch_size,num_workers=args.num_workers,
                        resize=(H,W)
                        )

    if args.h_type == 'input':
        conceptizer  = input_conceptizer()
        args.nconcepts = args.input_dim + int(not args.nobias)
    elif args.h_type == 'cnn':
        
        # biase. They treat it like any other concept.
        #args.nconcepts +=     int(not args.nobias)
        # conceptizer  = image_cnn_conceptizer(args.input_dim, args.nconcepts, args.concept_dim, nchannel = 3) #, sparsity = sparsity_l)
        conceptizer = image_resnet_conceptizer(args.input_dim, args.nconcepts, args.nclasses, args.concept_dim, nchannel = 3)
        # conceptizer = EfficientNet(args.input_dim, args.nconcepts, args.nclasses, args.concept_dim, nchannel = 3)
        # conceptizer = DenseNet(args.input_dim, args.nconcepts, args.nclasses, args.concept_dim, nchannel = 3)
    else:
        #args.nconcepts +=     int(not args.nobias)
        conceptizer  = image_fcc_conceptizer(args.input_dim, args.nconcepts, args.concept_dim, nchannel = 3) #, sparsity = sparsity_l)


    if args.theta_arch == 'simple':
        parametrizer = image_parametrizer(args.input_dim, args.nconcepts, args.theta_dim, nchannel = 3, only_positive = args.positive_theta)
    elif 'vgg' in args.theta_arch:
        parametrizer = vgg_parametrizer(args.input_dim, args.nconcepts, args.theta_dim, arch = args.theta_arch, nchannel = 3, only_positive = args.positive_theta) #torchvision.models.alexnet(num_classes = args.nconcepts*args.theta_dim)
    else:
        parametrizer = torchvision_parametrizer(args.input_dim, args.nconcepts, args.theta_dim, arch = args.theta_arch, nchannel = 3, only_positive = args.positive_theta) #torchvision.models.alexnet(num_classes = args.nconcepts*args.theta_dim)


    aggregator   = additive_scalar_aggregator(args.nconcepts, args.concept_dim, args.nclasses)

    # model        = GSENN(conceptizer, parametrizer, aggregator) #, learn_h = args.train_h)
    model        = GSENN(conceptizer, aggregator)


    if args.theta_reg_type in ['unreg','none', None]:
        trainer = VanillaClassTrainer(model, args)
    elif args.theta_reg_type == 'grad1':
        trainer = GradPenaltyTrainer(model, args, typ = 1)
    elif args.theta_reg_type == 'grad2':
        trainer = GradPenaltyTrainer(model, args, typ = 2)
    elif args.theta_reg_type == 'grad3':
        trainer = GradPenaltyTrainer(model, args, typ = 3)
    elif args.theta_reg_type == 'crosslip':
        trainer = CLPenaltyTrainer(model, args)
    else:
        raise ValueError('Unrecoginzed theta_reg_type')

    if args.train or not args.load_model or (not os.path.isfile(os.path.join(model_path,'model_best.pth.tar'))):
        trainer.train(train_loader, valid_loader, epochs = args.epochs, save_path = model_path)
        trainer.plot_losses(save_path=results_path)
    else:
        checkpoint = torch.load(os.path.join(model_path,'model_best.pth.tar'), map_location=lambda storage, loc: storage)
        checkpoint.keys()
        model = checkpoint['model']
        trainer =  VanillaClassTrainer(model, args) # arbtrary trained, only need to compuyte val acc

    model.eval()

     #Check accuracy with the best model
    checkpoint = torch.load(os.path.join(model_path,'model_best.pth.tar'), map_location=lambda storage, loc: storage)
    checkpoint.keys()
    model = checkpoint['model']
    trainer =  VanillaClassTrainer(model, args)
    trainer.evaluate(test_loader, fold = 'test')

    # # pytorch_total_params = sum(p.numel() for p in model.parameters())
    # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print (pytorch_total_params)
    # count_parameters(model)

    # All_Results = {}

    # 0. Concept Grid for Visualization
    concept_grid(model, test_loader, top_k = 10, cuda = args.cuda, save_path = results_path + '/concept_grid.pdf')

if __name__ == '__main__':
    main()

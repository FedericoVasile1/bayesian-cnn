from __future__ import print_function

import os
import argparse
import random

import torch
import numpy as np
from torch.optim import Adam, lr_scheduler
from torch.nn import functional as F

import data
import utils
import metrics
from models.BayesianModels.Bayesian3Conv3FC import BBB3Conv3FC
from models.BayesianModels.BayesianAlexNet import BBBAlexNet
from models.BayesianModels.BayesianLeNet import BBBLeNet

# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def getModel(net_type, inputs, outputs, priors, layer_type, activation_type):
    if (net_type == 'lenet'):
        return BBBLeNet(outputs, inputs, priors, layer_type, activation_type)
    elif (net_type == 'alexnet'):
        return BBBAlexNet(outputs, inputs, priors, layer_type, activation_type)
    elif (net_type == '3CONV3FC'):
        return BBB3Conv3FC(outputs, inputs, priors, layer_type, activation_type)
    else:
        raise ValueError('Network should be either [LeNet / AlexNet / 3Conv3FC')


def train_model(net, optimizer, criterion, trainloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    net.train()
    training_loss = 0.0
    accs = []
    kl_list = []
    for i, (inputs, labels) in enumerate(trainloader, 1):

        optimizer.zero_grad()

        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)

        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1)
        
        kl = kl / num_ens
        kl_list.append(kl.item())
        log_outputs = utils.logmeanexp(outputs, dim=2)

        beta = metrics.get_beta(i-1, len(trainloader), beta_type, epoch, num_epochs)
        loss = criterion(log_outputs, labels, kl, beta)
        loss.backward()
        optimizer.step()

        accs.append(metrics.acc(log_outputs.data, labels))
        training_loss += loss.cpu().data.numpy()
    return training_loss/len(trainloader), np.mean(accs), np.mean(kl_list)


def validate_model(net, criterion, validloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    """Calculate ensemble accuracy and NLL Loss"""
    net.train()
    valid_loss = 0.0
    accs = []

    for i, (inputs, labels) in enumerate(validloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)
        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1).data

        log_outputs = utils.logmeanexp(outputs, dim=2)

        beta = metrics.get_beta(i-1, len(validloader), beta_type, epoch, num_epochs)
        valid_loss += criterion(log_outputs, labels, kl, beta).item()
        accs.append(metrics.acc(log_outputs, labels))

    return valid_loss/len(validloader), np.mean(accs)

def run(args):
    dataset = args.dataset
    net_type = args.net_type

    # Hyper Parameter settings
    layer_type = args.layer_type
    activation_type = args.activation_type
    priors = args.priors

    train_ens = args.train_ens
    valid_ens = args.valid_ens
    n_epochs = args.n_epochs
    lr_start = args.lr_start
    num_workers = args.num_workers
    valid_size = args.valid_size
    batch_size = args.batch_size
    beta_type = args.beta_type

    trainset, testset, inputs, outputs = data.getDataset(dataset)
    train_loader, valid_loader, test_loader = data.getDataloader(
        trainset, testset, valid_size, batch_size, num_workers)
    net = getModel(net_type, inputs, outputs, priors, layer_type, activation_type).to(device)

    ckpt_dir = f'checkpoints/{dataset}/bayesian'
    ckpt_name = f'checkpoints/{dataset}/bayesian/model_{net_type}_{layer_type}_{activation_type}.pt'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    criterion = metrics.ELBO(len(trainset)).to(device)
    optimizer = Adam(net.parameters(), lr=lr_start)
    lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
    best_valid_acc = -1
    best_valid_acc_epoch = -1
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        train_loss, train_acc, train_kl = train_model(net, optimizer, criterion, train_loader, num_ens=train_ens, beta_type=beta_type, epoch=epoch, num_epochs=n_epochs)
        valid_loss, valid_acc = validate_model(net, criterion, valid_loader, num_ens=valid_ens, beta_type=beta_type, epoch=epoch, num_epochs=n_epochs)
        lr_sched.step(valid_loss)

        print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f} \ttrain_kl_div: {:.4f}'.format(
            epoch, train_loss, train_acc, valid_loss, valid_acc, train_kl))

        # save model if validation accuracy has increased
        if valid_loss > best_valid_acc:
            #print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            #    valid_loss_max, valid_loss))
            torch.save(net.state_dict(), ckpt_name)
            best_valid_acc = valid_acc
            best_valid_acc_epoch = epoch

    log = '--- Best validation accuracy is {:.1f}% obtained at epoch {} ---'.format(best_valid_acc, best_valid_acc_epoch)
    print(log)

    # TODO: add testing phase
    # TODO: add tensorboard

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "PyTorch Bayesian Model Training")
    parser.add_argument('--net_type', default='lenet', type=str, help='model = [lenet/alexnet/3CONV3FC]')
    parser.add_argument('--dataset', default='MNIST', type=str, help='dataset = [MNIST/CIFAR10/CIFAR100/CRC]')

    parser.add_argument('--layer_type', default='lrt', type=str)        # 'bbb' or 'lrt'
    parser.add_argument('--activation_type', default='softplus', type=str)  # 'softplus' or 'relu' or 'tanh'
    parser.add_argument('--prior_mu', default=0, type=float)            #
    parser.add_argument('--prior_sigma', default=0.1, type=float)
    parser.add_argument('--posterior_mu_initial', default='0,0.1', type=str)    # (mean, std) normal_
    parser.add_argument('--posterior_rho_initial', default='-5,0.1', type=str)  # (mean, std) normal_

    parser.add_argument('--n_epochs', default=200, type=int)
    parser.add_argument('--lr_start', default=0.001, type=float)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--valid_size', default=0.2, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--train_ens', default=1, type=int)     # the number of samples
    parser.add_argument('--valid_ens', default=1, type=int)
    parser.add_argument('--beta_type', default=0.1, type=float)     # 'Blundell', 'Standard', etc. Use float for const value

    parser.add_argument('--seed', default=25, type=int)

    args = parser.parse_args()

    run(args, args.dataset, args.net_type)

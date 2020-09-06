import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets

import argparse
import time
import os

import _init_paths
from src.datasets.BiologicalDataset import BiologicalDataset
from src.models.Dropout3Conv3FC import Dropout3Conv3FC

def load_dataset(dataset_name, val_split, batch_size, mode='train'):
    if mode == 'train':
        train = True
        dataset = {phase: None for phase in ['train', 'val']}
    elif mode == 'test':
        train = False
        dataset = {'test': None}
    else:
        raise Exception('Wrong mode parameter; ' + mode + ' is not supported.'
                        ' Supported modes: [train|test]')

    if dataset_name == 'MNIST':
        input_channels = 1
        num_classes = 10
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        dataset[mode] = datasets.MNIST(root=os.path.join(os.getcwd(), 'data'),
                                       train=train,
                                       download=True,
                                       transform=transform)
    elif dataset_name == 'CIFAR10':
        input_channels = 3
        num_classes = 10
        transform = transforms.Compose([
            transforms.ToTensor(),
            # values taken from: https://github.com/kuangliu/pytorch-cifar/issues/19
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
        ])
        dataset[mode] = datasets.CIFAR10(root=os.path.join(os.getcwd(), 'data'),
                                         train=train,
                                         download=True,
                                         transform=transform)
    elif dataset_name == 'CRC':
        input_channels = 3
        num_classes = 3
        dataset[mode] = BiologicalDataset(train)
    else:
        raise Exception('Wrong --dataset option; ' + dataset_name + ' is not supported.'
                        ' Supported datasets: [CRC|MNIST|CIFAR10]')

    dataloader = {}
    if train:
        n_val_samples = int(len(dataset['train']) * val_split)
        n_train_samples = len(dataset['train']) - n_val_samples
        dataset['train'], dataset['val'] = torch.utils.data.random_split(dataset['train'],
                                                                         [n_train_samples, n_val_samples])
        dataloader['train'] = torch.utils.data.DataLoader(dataset['train'],
                                                          batch_size,
                                                          False)
        dataloader['val'] = torch.utils.data.DataLoader(dataset['val'],
                                                        batch_size,
                                                        False)
    else:
        dataloader['test'] = torch.utils.data.DataLoader(dataset['test'],
                                                         batch_size,
                                                         False)

    return dataloader, input_channels, num_classes

def load_model(model, input_channels, num_classes, dropout):
    if model == '3CONV3FC':
        model = Dropout3Conv3FC(num_classes, input_channels, dropout)
    else:
        raise Exception('Wrong --model option; ' + model + ' is not supported.'
                        ' Supported models: [3CONV3FC]')

    return model

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader, input_channels, num_classes = load_dataset(args.dataset, args.val_split, args.batch_size)
    model = load_model(args.model, input_channels, num_classes, args.dropout)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
    criterion = nn.CrossEntropyLoss().to(device)

    model.train(True)       # always true since we are using variational dropout

    phases = ['train', 'val']
    model = model.to(device)
    for epoch in range(1, args.epochs+1, 1):
        start = time.time()
        loss_epoch = {phase: 0 for phase in phases}
        accuracy_epoch = {phase: 0 for phase in phases}

        for phase in phases:
            training = True if phase == 'train' else False
            with torch.set_grad_enabled(training):
                for batch_idx, (images, targets) in enumerate(dataloader[phase], start=1):
                    if batch_idx > 2:
                        break
                    batch_size = images.shape[0]

                    if training:
                        optimizer.zero_grad()

                    images = images.to(device)
                    scores = model(images)

                    targets = targets.to(device)
                    loss = criterion(scores, targets)

                    if training:
                        loss.backward()
                        optimizer.step()

                    loss_epoch[phase] += loss.item() * batch_size
                    accuracy_epoch[phase] += (scores.argmax(dim=1) == targets).sum().item()

                    if args.verbose:
                        print('[{:5}] Epoch: {}/{}  Iteration: {}  Loss: {:.4f}'.
                              format(phase, epoch, args.epochs, batch_idx, loss.item()))

        lr_sched.step(loss_epoch['val']/len(dataloader['val'].dataset))

        end = time.time()
        print('[{}]  Loss: {:.4f}  Accuracy: {:.1f}%  |[{}]  Loss: {:.4f}  Accuracy: {:.1f}%'.
              format('train',
                     loss_epoch['train']/len(dataloader['train'].dataset),
                     accuracy_epoch['train']/len(dataloader['train'].dataset) * 100,
                     'val',
                     loss_epoch['val']/len(dataloader['val'].dataset),
                     accuracy_epoch['val']/len(dataloader['val'].dataset) * 100))
        print('Running time: {:.1f}s'.format(end - start))

if __name__ == '__main__':
    base_dir = os.getcwd()
    base_dir = base_dir.split('/')[-1]
    if base_dir != 'Project8':
        raise Exception('Wrong base dir, this file must be run from Project8/ directory.')

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='3CONV3FC', type=str)
    parser.add_argument('--dataset', default='MNIST', type=str)
    parser.add_argument('--dropout', default=0.5, type=float)

    # These hyperparameter default values are equal to the ones of bayesian models.
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--val_split', default=0.2, type=float)

    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    main(args)
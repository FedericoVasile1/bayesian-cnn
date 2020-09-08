import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter

import argparse
import time
import os
import sys

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
            transforms.Resize(32),  # in order to have the same size as MNIST and CRC
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
                                                          True)
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
    this_dir = os.path.join(os.path.dirname(__file__), '.')
    save_dir = os.path.join(this_dir, 'checkpoints')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    checkpoint_file = 'model-{}-dataset-{}-dropout-{}.pth'.format(args.model,
                                                                  args.dataset,
                                                                  args.dropout)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader, input_channels, num_classes = load_dataset(args.dataset, args.val_split, args.batch_size)
    model = load_model(args.model, input_channels, num_classes, args.dropout)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
    criterion = nn.CrossEntropyLoss().to(device)
    model.train(True)       # always true since we are using variational dropout

    writer = SummaryWriter()
    command = 'python ' + ' '.join(sys.argv)
    f = open(writer.log_dir + '/run_command.txt', 'w+')
    f.write(command)
    f.close()

    best_val_accuracy = -1
    best_epoch = -1
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
                    writer.add_scalar('Loss_iter/'+phase, loss.item(), batch_idx)

        lr_sched.step(loss_epoch['val'] / len(dataloader['val'].dataset))

        end = time.time()
        print('Epoch: {:3} |[{}]  Loss: {:.4f}  Accuracy: {:.1f}%  |[{}]  Loss: {:.4f}  Accuracy: {:.1f}%  |Running time: {:.1f}s'.
              format(epoch,
                     'train',
                     loss_epoch['train'] / len(dataloader['train'].dataset),
                     accuracy_epoch['train'] / len(dataloader['train'].dataset) * 100,
                     'val',
                     loss_epoch['val'] / len(dataloader['val'].dataset),
                     accuracy_epoch['val'] / len(dataloader['val'].dataset) * 100,
                     end - start))
        writer.add_scalar('Loss_epoch/train', loss_epoch['train'] / len(dataloader['train'].dataset), epoch)
        writer.add_scalar('Loss_epoch/val', loss_epoch['val'] / len(dataloader['val'].dataset), epoch)
        writer.add_scalar('Accuracy_epoch/train', accuracy_epoch['train'] / len(dataloader['train'].dataset) * 100, epoch)
        writer.add_scalar('Accuracy_epoch/val', accuracy_epoch['val'] / len(dataloader['val'].dataset) * 100, epoch)

        if best_val_accuracy < (accuracy_epoch['val'] / len(dataloader['val'].dataset) * 100):
            best_val_accuracy  = accuracy_epoch['val'] / len(dataloader['val'].dataset) * 100
            best_epoch = -1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, checkpoint_file))
    print('--- Best validation accuracy is {:.1f}% obtained at epoch {} ---'.format(best_val_accuracy, best_epoch))

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
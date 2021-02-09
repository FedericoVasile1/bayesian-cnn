import argparse
import time
import os
import sys
import json
import shutil

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'lib', 'PyTorchBayesianCNN'))
from src.utils.pipeline import set_seed, load_dataset, load_model

def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_name = 'model-{}-dataset-{}-dropout-{}-actfunc-{}-batchsize-{}'.\
        format(args.model, args.dataset, args.dropout, args.activation_function, args.batch_size)
    if args.suppress_epoch_print:
        print(base_name)

    writer_dir = os.path.join('results', 'mc_dropout', base_name)
    if os.path.isdir(writer_dir):
        shutil.rmtree(writer_dir)
    writer = SummaryWriter(log_dir=writer_dir)
    command = 'python ' + ' '.join(sys.argv)
    f = open(writer.log_dir + '/log.txt', 'w+')
    f.write(command + '\n')
    f.close()

    dataloader = load_dataset(args.dataset, args.val_split, args.batch_size, mode='train')
    model = load_model(args.model, args.input_channels, args.num_classes, args.activation_function, args.dropout, None, None)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
    criterion = nn.CrossEntropyLoss().to(device)
    model.train(True)

    best_val_accuracy = -1
    best_epoch = -1
    phases = ['train', 'val']
    model = model.to(device)
    for epoch in range(1, args.epochs+1, 1):
        start = time.time()
        loss_epoch = {phase: 0 for phase in phases}
        accuracy_epoch = {phase: 0 for phase in phases}

        for phase in phases:
            training = True if phase=='train' else False
            with torch.set_grad_enabled(training):
                for batch_idx, (images, targets) in enumerate(dataloader[phase], start=1):
                    batch_size = images.shape[0]
                    images = images.to(device)
                    targets = targets.to(device)

                    if training:
                        writer.add_histogram('conv1.weight', model.conv1.weight, global_step=epoch)
                        writer.add_histogram('conv1.bias', model.conv1.bias, global_step=epoch)
                        writer.close()

                        optimizer.zero_grad()

                    scores = model(images)

                    loss = criterion(scores, targets)

                    if training:
                        loss.backward()
                        optimizer.step()

                    loss_epoch[phase] += loss.item() * batch_size
                    accuracy_epoch[phase] += (scores.argmax(dim=1) == targets).sum().item()

                    if args.verbose:
                        print('[{:5}] Epoch: {}/{}  Iteration: {}  Loss: {:.4f}'.
                              format(phase, epoch, args.epochs, batch_idx, loss.item()))

        lr_sched.step(loss_epoch['val'] / len(dataloader['val'].dataset))

        end = time.time()
        log = 'Epoch: {:3} |'
        log += '[{}]  Loss: {:.4f}  Accuracy: {:.1f}%  |'
        log += '[{}]  Loss: {:.4f}  Accuracy: {:.1f}%  |'
        log += 'Running time: {:.1f}s'
        log = str(log).format(epoch,
                              'train',
                              loss_epoch['train'] / len(dataloader['train'].dataset),
                              accuracy_epoch['train'] / len(dataloader['train'].dataset) * 100,
                              'val',
                              loss_epoch['val'] / len(dataloader['val'].dataset),
                              accuracy_epoch['val'] / len(dataloader['val'].dataset) * 100,
                              end - start)
        if not args.suppress_epoch_print:
            print(log)
        f = open(writer.log_dir + '/log.txt', 'a+')
        f.write(log + '\n')
        f.close()

        writer.add_scalars('Loss_epoch/train_val',
                           {phase: loss_epoch[phase] / len(dataloader[phase].dataset) for phase in phases},
                           epoch)
        writer.add_scalars('Accuracy_epoch/train_val',
                           {phase: accuracy_epoch[phase] / len(dataloader[phase].dataset) * 100 for phase in phases},
                           epoch)
        writer.close()

        if best_val_accuracy < (accuracy_epoch['val'] / len(dataloader['val'].dataset) * 100):
            best_val_accuracy  = accuracy_epoch['val'] / len(dataloader['val'].dataset) * 100
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'accuracy': best_val_accuracy,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(writer.log_dir, 'best_model.pth'))
    log = '--- Best validation accuracy is {:.1f}% obtained at epoch {} ---\n'.format(best_val_accuracy, best_epoch)
    print(log)
    f = open(writer.log_dir + '/log.txt', 'a+')
    f.write(log + '\n')
    f.close()

if __name__ == '__main__':
    base_dir = os.getcwd()
    base_dir = base_dir.split('/')[-1]
    if base_dir != 'Project8':
        raise Exception('Wrong base dir, this file must be run from Project8/ directory.')

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='MNIST', type=str)
    parser.add_argument('--model', default='MCDROPALEXNET', type=str)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--activation_function', default='relu', type=str)
    parser.add_argument('--batch_size', default=256, type=int)

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--val_split', default=0.2, type=float)

    parser.add_argument('--seed', default=25, type=int)

    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--suppress_epoch_print', default=False, action='store_true')
    parser.add_argument('--data_info',  default='data/data_info.json' ,type=str)

    args = parser.parse_args()

    if args.model not in ('MCDROP3CONV3FC', 'MCDROPALEXNET'):
        raise ValueError('Wrong --model argument. '
                         'This training pipeline supports only [MCDROP3CONV3FC|MCDROPALEXNET] models')

    with open(args.data_info, 'r') as f:
        data_info = json.load(f)[args.dataset]
    args.class_index = data_info['class_index']
    args.input_channels = data_info['input_channels']
    args.num_classes = len(args.class_index)

    main(args)
import argparse
import time
import os
import sys
import json

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'lib', 'PyTorchBayesianCNN'))
from src.utils.pipeline import set_seed, load_dataset, load_model
from lib.PyTorchBayesianCNN import metrics, utils

def main(args):
    # settings
    prior_mu = args.prior_mu
    prior_sigma = args.prior_sigma
    posterior_mu_initial = (int(args.posterior_mu_initial.split(',')[0]), float(args.posterior_mu_initial.split(',')[1]))
    posterior_rho_initial = (int(args.posterior_rho_initial.split(',')[0]), float(args.posterior_rho_initial.split(',')[1]))
    priors = {'prior_mu': prior_mu,
              'prior_sigma': prior_sigma,
              'posterior_mu_initial': posterior_mu_initial,
              'posterior_rho_initial': posterior_rho_initial}

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_name = 'model-{}-dataset-{}-actfunc-{}-batchsize-{}-pm-{}-ps-{}-pmi-{}-pri-{}-te-{}-ve-{}'.\
        format(args.model, args.dataset, args.activation_function, args.batch_size,
               prior_mu, prior_sigma, posterior_mu_initial, posterior_rho_initial, args.train_ens, args.valid_ens)
    writer = SummaryWriter(log_dir='results/varinf/'+base_name)
    command = 'python ' + ' '.join(sys.argv)
    if args.suppress_epoch_print:
        print(command)
    f = open(writer.log_dir + '/log.txt', 'w+')
    f.write(command)
    f.close()

    dataloader = load_dataset(args.dataset, args.val_split, args.batch_size, num_workers=args.num_workers, mode='train')
    model = load_model(args.model, args.input_channels, args.num_classes, args.activation_function, None, priors, args.layer_type)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
    # TODO: better check the loss provided, is the implementation correct?
    criterion = metrics.ELBO(len(dataloader['train'].dataset)+len(dataloader['val'].dataset)).to(device)
    #model.train(True)

    best_val_accuracy = -1
    best_epoch = -1
    phases = ['train', 'val']
    model = model.to(device)
    for epoch in range(1, args.epochs+1, 1):
        start = time.time()
        loss_epoch = {phase: 0.0 for phase in phases}
        accuracy_epoch = {phase: 0.0 for phase in phases}
        kl_epoch = {phase: 0.0 for phase in phases}

        for phase in phases:
            training = True if phase=='train' else False
            ens = args.train_ens if training else args.valid_ens
            with torch.set_grad_enabled(training):
                model.train()

                for batch_idx, (images, targets) in enumerate(dataloader[phase], start=1):
                    batch_size = images.shape[0]
                    images = images.to(device)
                    targets = targets.to(device)

                    if training:
                        optimizer.zero_grad()

                    outputs = torch.zeros(batch_size, args.num_classes, ens).to(device)
                    kl = 0.0

                    for j in range(ens):
                        scores, _kl = model(images)
                        kl += _kl
                        outputs[:, :, j] = F.log_softmax(scores, dim=1)
                    # TODO: better check what is kl. is it one per batch? the criterion should receive
                    #        the sum or the mean of kls?
                    kl /= ens
                    log_outputs = utils.logmeanexp(outputs, dim=2)

                    beta = metrics.get_beta(batch_idx-1, len(dataloader[phase]), args.beta_type, epoch, args.epochs)
                    loss = criterion(log_outputs, targets, kl, beta)

                    if training:
                        loss.backward()
                        optimizer.step()

                    loss_epoch[phase] += loss.item() * batch_size
                    accuracy_epoch[phase] += (log_outputs.argmax(dim=1) == targets).sum().item()
                    kl_epoch[phase] += kl.item()

                    if args.verbose:
                        print('[{:5}] Epoch: {}/{}  Iteration: {}  Loss: {:.4f}  KL_div: {:.4f}'.
                              format(phase, epoch, args.epochs, batch_idx, loss.item(), kl))
                    writer.add_scalar('Loss_iter/'+phase, loss.item(), batch_idx)

        lr_sched.step(loss_epoch['val'] / len(dataloader['val'].dataset))

        end = time.time()
        log = 'Epoch: {:3} |'
        log += '[{}]  Loss: {:.4f}  Accuracy: {:.1f}%  KL_div: {:.4f}  |'
        log += '[{}]  Loss: {:.4f}  Accuracy: {:.1f}%  KL_div: {:.4f}  |'
        log += 'Running time: {:.1f}s'
        log = str(log).format(epoch,
                              'train',
                              loss_epoch['train'] / len(dataloader['train'].dataset),
                              accuracy_epoch['train'] / len(dataloader['train'].dataset) * 100,
                              kl_epoch['train'] / len(dataloader['train']),
                              'val',
                              loss_epoch['val'] / len(dataloader['val'].dataset),
                              accuracy_epoch['val'] / len(dataloader['val'].dataset) * 100,
                              kl_epoch['val'] / len(dataloader['val']),
                              end - start)
        if not args.suppress_epoch_print:
            print(log)
        f = open(writer.log_dir + '/log.txt', 'a+')
        f.write(log)
        f.close()

        writer.add_scalar('Loss_epoch/train', loss_epoch['train'] / len(dataloader['train'].dataset), epoch)
        writer.add_scalar('Loss_epoch/val', loss_epoch['val'] / len(dataloader['val'].dataset), epoch)
        writer.add_scalar('Accuracy_epoch/train', accuracy_epoch['train'] / len(dataloader['train'].dataset) * 100, epoch)
        writer.add_scalar('Accuracy_epoch/val', accuracy_epoch['val'] / len(dataloader['val'].dataset) * 100, epoch)

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
    f.write(log)
    f.close()

if __name__ == '__main__':
    base_dir = os.getcwd()
    base_dir = base_dir.split('/')[-1]
    if base_dir != 'Project8':
        raise Exception('Wrong base dir, this file must be run from Project8/ directory.')

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='MNIST', type=str)
    parser.add_argument('--model', default='VARINF3CONV3FC', type=str)
    parser.add_argument('--activation_function', default='softplus', type=str)
    parser.add_argument('--batch_size', default=256, type=int)

    parser.add_argument('--layer_type', default='lrt', type=str)        # 'bbb' or 'lrt'
    parser.add_argument('--prior_mu', default=0, type=int)
    parser.add_argument('--prior_sigma', default=0.1, type=float)
    parser.add_argument('--posterior_mu_initial', default='0,0.1', type=str)    # (mean, std) normal_
    parser.add_argument('--posterior_rho_initial', default='-5,0.1', type=str)  # (mean, std) normal_

    parser.add_argument('--train_ens', default=1, type=int)     # the number of samples
    parser.add_argument('--valid_ens', default=1, type=int)
    parser.add_argument('--beta_type', default=0.1, type=float)     # 'Blundell', 'Standard', etc. Use float for const value

    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--val_split', default=0.2, type=float)
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--seed', default=25, type=int)

    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--suppress_epoch_print', default=False, action='store_true')
    parser.add_argument('--data_info',  default='data/data_info.json' ,type=str)

    args = parser.parse_args()

    if args.model != 'VARINF3CONV3FC':
        raise ValueError('Wrong --model argument. This training pipeline supports only VARINF3CONV3FC model')

    with open(args.data_info, 'r') as f:
        data_info = json.load(f)[args.dataset]
    args.class_index = data_info['class_index']
    args.input_channels = data_info['input_channels']
    args.num_classes = len(args.class_index)

    main(args)
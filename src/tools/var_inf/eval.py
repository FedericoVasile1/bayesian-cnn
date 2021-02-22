import shutil
import os
import argparse
import sys
import time
import json

import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report
import numpy as np

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'lib', 'PyTorchBayesianCNN'))
from src.utils.uncertainty import compute_uncertainties
from src.utils.visualize import plot_histogram, plot_histogram_classes, plot_confusion_matrix, save_fig_to_tensorboard
from src.utils.pipeline import set_seed, load_dataset, load_model

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
    base_name = 'model-{}-dataset-{}-actfunc-{}-batchsize-{}-pm-{}-ps-{}-pmi-{}-pri-{}-lrt-{}-te-{}-ve-{}'.\
        format(args.model, args.dataset, args.activation_function, args.batch_size,
               prior_mu, prior_sigma, posterior_mu_initial, posterior_rho_initial,
               args.layer_type, args.train_ens, args.valid_ens)
    checkpoint_path = os.path.join(os.getcwd(), 'results', 'var_inf', base_name, 'best_model.pth')
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
    else:
        raise(RuntimeError('Cannot find the checkpoint {}'.format(checkpoint_path)))

    eval_dir = os.path.join(os.getcwd(), 'results', 'var_inf', base_name, 'eval_K='+str(args.K))
    if os.path.isdir(eval_dir):
        shutil.rmtree(eval_dir)
    os.mkdir(eval_dir)
    writer = SummaryWriter(log_dir=eval_dir)
    command = 'python ' + ' '.join(sys.argv)
    f = open(writer.log_dir + '/log.txt', 'w+')
    f.write(command + '\n')
    f.close()

    dataloader = load_dataset(args.dataset, None, args.batch_size, num_workers=args.num_workers, mode='test')

    model = load_model(args.model, args.input_channels, args.num_classes, args.activation_function, None, priors, args.layer_type)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train(True)           # always true since we are using variational dropout

    model = model.to(device)
    with torch.set_grad_enabled(False):

        if not args.skip_K1:
            log = 'Inference on test set without computing uncertainties(i.e. K=1) ...'
            print(log)
            f = open(writer.log_dir + '/log.txt', 'a+')
            f.write(log + '\n')
            f.close()

            accuracy = 0
            scores_all = []
            targets_all = []
            start = time.time()
            for batch_idx, (images, targets) in enumerate(dataloader['test'], start=1):
                images = images.to(device)
                scores, _ = model(images)

                scores = scores.argmax(dim=1)
                targets = targets.to(device)
                accuracy += (scores == targets).sum().item()

                scores_all.append(scores)
                targets_all.append(targets)
            end = time.time()

            log = '... Accuracy on test set: {:.1f}%  |Running time: {:.1f}s\n'.\
                format(accuracy / len(dataloader['test'].dataset) * 100, end - start)
            print(log)
            f = open(writer.log_dir + '/log.txt', 'a+')
            f.write(log + '\n')
            f.close()

            scores_all = torch.cat(scores_all).cpu().detach().numpy()
            targets_all = torch.cat(targets_all).cpu().detach().numpy()
            title = '[{}|K=1]/ConfusionMatrix'.format(args.dataset)
            fig = plot_confusion_matrix(targets_all,
                                        scores_all,
                                        args.class_index,
                                        title)
            save_fig_to_tensorboard(fig, writer, title)

        log = 'Inference on test set computing uncertainties with K={} ...'.format(args.K)
        print(log)
        f = open(writer.log_dir + '/log.txt', 'a+')
        f.write(log + '\n')
        f.close()

        predictions_uncertainty_all = []
        predicted_class_variance_all = []
        predicted_class_variance_al_all = []
        predicted_class_variance_ep_all = []
        targets_all = []
        accuracy = 0
        start = time.time()
        for batch_idx, (images, targets) in enumerate(dataloader['test'], start=1):
            images = images.to(device)
            predictions_uncertainty, predicted_class_variance, predicted_class_variance_al, predicted_class_variance_ep =\
                compute_uncertainties(model, images, K=args.K)

            targets = targets.to(device)
            accuracy += (predictions_uncertainty == targets).sum().item()

            predictions_uncertainty_all.append(predictions_uncertainty)
            predicted_class_variance_all.append(predicted_class_variance)
            predicted_class_variance_al_all.append(predicted_class_variance_al)
            predicted_class_variance_ep_all.append(predicted_class_variance_ep)
            targets_all.append(targets)
        end = time.time()

        avg_unc = torch.cat(predicted_class_variance_all).sum() / len(dataloader['test'].dataset)
        al = torch.cat(predicted_class_variance_al_all).sum() / len(dataloader['test'].dataset)
        ep = torch.cat(predicted_class_variance_ep_all).sum() / len(dataloader['test'].dataset)
        log = '... Accuracy on test set: {:.1f}%  with average uncertainty={:.3f} (al={:.3f}, ep={:.3f})|Running time: {:.1f}s'.\
            format(accuracy / len(dataloader['test'].dataset) * 100, avg_unc, al, ep, end - start)
        print(log)
        f = open(writer.log_dir + '/log.txt', 'a+')
        f.write(log + '\n')
        f.close()

        targets_all = torch.cat(targets_all).cpu().detach().numpy()
        predictions_uncertainty_all = torch.cat(predictions_uncertainty_all).cpu().detach().numpy()
        predicted_class_variance_all = torch.cat(predicted_class_variance_all).cpu().detach().numpy()

        clas_rep = classification_report(targets_all,
                                         predictions_uncertainty_all,
                                         np.arange(len(args.class_index)).tolist(),
                                         args.class_index)
        f = open(writer.log_dir + '/log.txt', 'a+')
        f.write(clas_rep + '\n')
        f.close()

        title = '[{}|K={}]/ConfusionMatrix'.format(args.dataset, args.K)
        fig = plot_confusion_matrix(targets_all,
                                    predictions_uncertainty_all,
                                    args.class_index,
                                    title)
        save_fig_to_tensorboard(fig, writer, title)

        max = -1
        for t in predicted_class_variance:
            if max < t.max():
                max = t.max()
        range = (0, max)

        title = '[{}|K={}]/AllPredictions'.format(args.dataset, args.K)
        fig = plot_histogram(predicted_class_variance_all, color='b', title=title, range=range)
        save_fig_to_tensorboard(fig, writer, title)

        title = '[{}|K={}]/CorrectPredictions'.format(args.dataset, args.K)
        fig = plot_histogram(predicted_class_variance_all[predictions_uncertainty_all == targets_all],
                             color='green',
                             title=title,
                             range=range)
        save_fig_to_tensorboard(fig, writer, title)

        title = '[{}|K={}]/WrongPredictions'.format(args.dataset, args.K)
        fig = plot_histogram(predicted_class_variance_all[predictions_uncertainty_all != targets_all],
                             color='red',
                             title=title,
                             range=range)
        save_fig_to_tensorboard(fig, writer, title)

        title = '[{}|K={}]/ClassesAllPredictions'.format(args.dataset, args.K)
        fig = plot_histogram_classes(predicted_class_variance_all,
                                     targets_all,
                                     args.class_index,
                                     title=title,
                                     range=range)
        save_fig_to_tensorboard(fig, writer, title)

        title = '[{}|K={}]/ClassesCorrectPredictions'.format(args.dataset, args.K)
        mask = predictions_uncertainty_all == targets_all
        fig = plot_histogram_classes(predicted_class_variance_all[mask],
                                     targets_all[mask],
                                     args.class_index,
                                     title=title,
                                     range=range)
        save_fig_to_tensorboard(fig, writer, title)

        title = '[{}|K={}]/ClassesWrongPredictions'.format(args.dataset, args.K)
        mask = predictions_uncertainty_all != targets_all
        fig = plot_histogram_classes(predicted_class_variance_all[mask],
                                     targets_all[mask],
                                     args.class_index,
                                     title=title,
                                     range=range)
        save_fig_to_tensorboard(fig, writer, title)


if __name__ == '__main__':
    base_dir = os.getcwd()
    base_dir = base_dir.split('/')[-1]
    if base_dir != 'Project8':
        raise Exception('Wrong base dir, this file must be run from Project8/ directory.')

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='MNIST', type=str)
    parser.add_argument('--model', default='VARINFALEXNET', type=str)
    parser.add_argument('--activation_function', default='softplus', type=str)
    parser.add_argument('--batch_size', default=256, type=int)

    parser.add_argument('--layer_type', default='lrt', type=str)        # 'bbb' or 'lrt'
    parser.add_argument('--prior_mu', default=0.0, type=float)
    parser.add_argument('--prior_sigma', default=0.1, type=float)
    parser.add_argument('--posterior_mu_initial', default='0,0.1', type=str)    # (mean, std) normal_
    parser.add_argument('--posterior_rho_initial', default='-5,0.1', type=str)  # (mean, std) normal_

    parser.add_argument('--train_ens', default=1, type=int)     # the number of samples
    parser.add_argument('--valid_ens', default=1, type=int)

    parser.add_argument('--K', default=100, type=int)
    parser.add_argument('--skip_K1', default=False, action='store_true')

    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--seed', default=25, type=int)

    parser.add_argument('--data_info',  default='data/data_info.json' ,type=str)

    args = parser.parse_args()

    if args.model not in ('VARINF3CONV3FC', 'VARINFALEXNET'):
        raise ValueError('Wrong --model argument. '
                         'This training pipeline supports only [VARINF3CONV3FC|VARINFALEXNET] model')

    with open(args.data_info, 'r') as f:
        data_info = json.load(f)[args.dataset]
    args.class_index = data_info['class_index']
    args.input_channels = data_info['input_channels']
    args.num_classes = len(args.class_index)

    main(args)




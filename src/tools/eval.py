import torch
from torch.utils.tensorboard import SummaryWriter

import os
import argparse
import sys
import time
import json

import _init_paths
from src.tools.train import load_dataset, load_model
from src.utils.uncertainty import compute_uncertainties
from src.utils.visualize import plot_histogram, plot_confusion_matrix, save_fig_to_tensorboard

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader = load_dataset(args.dataset, None, args.batch_size, 'test')

    checkpoint_file = 'model-{}-dataset-{}-dropout-{}.pth'.format(args.model,
                                                                  args.dataset,
                                                                  args.dropout)
    checkpoint_path = os.path.join(os.getcwd(), 'src', 'tools', 'checkpoints', checkpoint_file)
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
    else:
        raise(RuntimeError('Cannot find the checkpoint {}'.format(checkpoint_path)))
    model = load_model(args.model, args.input_channels, args.num_classes, args.dropout)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train(True)           # always true since we are using variational dropout

    writer = SummaryWriter()
    command = 'python ' + ' '.join(sys.argv)
    f = open(writer.log_dir + '/run_command.txt', 'w+')
    f.write(command)
    f.close()

    model = model.to(device)
    with torch.set_grad_enabled(False):
        print('Inference on test set without computing uncertainties(i.e. K=1) ...')
        accuracy = 0
        scores_all = []
        targets_all = []
        start = time.time()
        for batch_idx, (images, targets) in enumerate(dataloader['test'], start=1):
            images = images.to(device)
            scores = model(images).argmax(dim=1)

            targets = targets.to(device)
            accuracy += (scores == targets).sum().item()

            scores_all.append(scores)
            targets_all.append(targets)
        end = time.time()
        print('... Accuracy on test set: {:.1f}%  |Running time: {:.1f}s\n'.
              format(accuracy / len(dataloader['test'].dataset) * 100, end - start))

        scores_all = torch.cat(scores_all)
        targets_all = torch.cat(targets_all)
        title = 'Confusion matrix - {}/K=1'.format(args.dataset)
        fig = plot_confusion_matrix(targets_all.numpy(), scores_all.numpy(), args.class_index, title)
        save_fig_to_tensorboard(fig, writer, title)

        print('Inference on test set computing uncertainties with K={} ...'.format(args.K))
        predictions_uncertainty_all = []
        predicted_class_variance_all = []
        scores_all = []
        targets_all = []
        accuracy = 0
        start = time.time()
        for batch_idx, (images, targets) in enumerate(dataloader['test'], start=1):
            images = images.to(device)
            predictions_uncertainty, predicted_class_variance = compute_uncertainties(model, images, K=args.K)
            predictions_uncertainty_all.append(predictions_uncertainty)
            predicted_class_variance_all.append(predicted_class_variance)

            targets = targets.to(device)
            accuracy += (predictions_uncertainty == targets).sum().item()

            scores_all.append(scores)
            targets_all.append(targets)
        end = time.time()
        print('... Accuracy on test set: {:.1f}%  |Running time: {:.1f}s'.
              format(accuracy / len(dataloader['test'].dataset) * 100, end - start))

        scores_all = torch.cat(scores_all)
        targets_all = torch.cat(targets_all)
        predictions_uncertainty_all = torch.cat(predictions_uncertainty_all)
        predicted_class_variance_all = torch.cat(predicted_class_variance_all)

        title = 'Confusion matrix - {}/K={}'.format(args.dataset, args.K)
        fig = plot_confusion_matrix(targets_all.numpy(), scores_all.numpy(), args.class_index, title)
        save_fig_to_tensorboard(fig, writer, title)

        title = '{} test/All predictions'.format(args.dataset)
        fig = plot_histogram(predicted_class_variance_all.numpy(), color='b', title=title)
        save_fig_to_tensorboard(fig, writer, title)

        title = '{} test/Correct predictions'.format(args.dataset)
        fig = plot_histogram(predicted_class_variance_all[predictions_uncertainty_all == targets_all].numpy(),
                             color='green',
                             title=title)
        save_fig_to_tensorboard(fig, writer, title)

        title = '{} test/Wrong predictions'.format(args.dataset)
        fig = plot_histogram(predicted_class_variance_all[predictions_uncertainty_all != targets_all].numpy(),
                             color='red',
                             title=title)
        save_fig_to_tensorboard(fig, writer, title)

if __name__ == '__main__':
    base_dir = os.getcwd()
    base_dir = base_dir.split('/')[-1]
    if base_dir != 'Project8':
        raise Exception('Wrong base dir, this file must be run from Project8/ directory.')

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='3CONV3FC', type=str)
    parser.add_argument('--dataset', default='MNIST', type=str)
    parser.add_argument('--dropout', default=0.5, type=float)

    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--K', default=100, type=int)
    parser.add_argument('--data_info',  default='data/data_info.json' ,type=str)

    args = parser.parse_args()

    with open(args.data_info, 'r') as f:
        data_info = json.load(f)[args.dataset]
    args.class_index = data_info['class_index']
    args.input_channels = data_info['input_channels']
    args.num_classes = len(args.class_index)

    main(args)




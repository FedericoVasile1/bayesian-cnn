import torch
from torch.utils.tensorboard import SummaryWriter

import os
import argparse
import sys
import time
import numpy as np

import _init_paths
from src.tools.train import load_dataset, load_model
from src.utils.uncertainty import compute_uncertainties
from src.utils.visualize import my_histogram, plot_to_image

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader, input_channels, num_classes = load_dataset(args.dataset, None, args.batch_size, 'test')

    checkpoint_file = 'model-{}-dataset-{}-dropout-{}.pth'.format(args.model,
                                                                  args.dataset,
                                                                  args.dropout)
    checkpoint_path = os.path.join(os.getcwd(), 'src', 'tools', 'checkpoints', checkpoint_file)
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
    else:
        raise(RuntimeError('Cannot find the checkpoint {}'.format(checkpoint_path)))
    model = load_model(args.model, input_channels, num_classes, args.dropout)
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
        start = time.time()
        accuracy = 0
        for batch_idx, (images, targets) in enumerate(dataloader['test'], start=1):
            images = images.to(device)
            scores = model(images)

            targets = targets.to(device)
            accuracy += (scores.argmax(dim=1) == targets).sum().item()
        end = time.time()
        print('... Accuracy on test set: {:.1f}%  |Running time: {:.1f}s\n\n'.
              format(accuracy / len(dataloader['test'].dataset) * 100, end - start))
        # TODO: PLOT CONFUSION MATRIX

        print('Inference on test set computing uncertainties with K={} ...'.format(args.K))
        predictions_uncertainty_all = []
        predicted_class_variance_all = []
        start = time.time()
        accuracy = 0
        for batch_idx, (images, targets) in enumerate(dataloader['test'], start=1):
            images = images.to(device)
            predictions_uncertainty, predicted_class_variance = compute_uncertainties(model, images, K=args.K)
            predictions_uncertainty_all.append(predictions_uncertainty)
            predicted_class_variance_all.append(predicted_class_variance)

            targets = targets.to(device)
            accuracy += (predictions_uncertainty == targets).sum().item()
        end = time.time()
        print('... Accuracy on test set: {:.1f}%  |Running time: {:.1f}s'.
              format(accuracy / len(dataloader['test'].dataset) * 100, end - start))
        # TODO: PLOT CONFUSION MATRIX

        predictions_uncertainty_all = torch.cat(predictions_uncertainty_all)
        predicted_class_variance_all = torch.cat(predicted_class_variance_all)

        title = '{} test - All predictions'.format(args.dataset)
        fig = my_histogram(predicted_class_variance_all, color='b', title=title)
        fig = plot_to_image(fig)
        writer.add_image(title, np.transpose(fig, (2, 0, 1)), 0)
        writer.close()

        title = '{} test - Correct predictions'.format(args.dataset)

        fig = my_histogram(predicted_class_variance_all[predictions_uncertainty_all == targets],
                           color='green',
                           title=title)
        fig = plot_to_image(fig)
        writer.add_image(title, np.transpose(fig, (2, 0, 1)), 0)
        writer.close()

        title = '{} test - Wrong predictions'.format(args.dataset)
        fig = my_histogram(predicted_class_variance_all[predictions_uncertainty_all != targets],
                           color='red',
                           title=title)
        fig = plot_to_image(fig)
        writer.add_image(title, np.transpose(fig, (2, 0, 1)), 0)
        writer.close()




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

    args = parser.parse_args()
    main(args)




import torch
from torch.utils.tensorboard import SummaryWriter

import os
import argparse
import sys
import time

import _init_paths
from src.tools.train import load_dataset, load_model

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
    start = time.time()
    accuracy = 0
    with torch.set_grad_enabled(False):
        for batch_idx, (images, targets) in enumerate(dataloader['test'], start=1):
            images = images.to(device)
            scores = model(images)

            targets = targets.to(device)
            accuracy += (scores.argmax(dim=1) == targets).sum().item()
    end = time.time()
    print('Accuracy on test set: {:.1f}%  |Running time: {:.1f}s'.
          format(accuracy / len(dataloader['test'].dataset) * 100, end - start))

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

    args = parser.parse_args()
    main(args)




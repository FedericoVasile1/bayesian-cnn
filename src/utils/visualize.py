import torch
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter

from PIL import Image
import cv2

import io
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def plot_random_images(images, labels=None, idxlabel_to_namelabel=None, examples=16, fig_suptitle=None, figsize=(8, 8), fpath=None):
    '''
    Given a 'images' dataset a random subset is selected and showed
    @param images: Numpy array of shape (N, H, W, C) with pixels in range [0, 1] float or [0, 255] int
    @param labels: Numpy array of shape (N,) containing the labels of the images
    @param idxlabel_to_namelabel: Python list containing at position i the name of class i
    @param examples: int number of samples to randomly draw from images
    @param fig_suptitle: String containing the figure title
    @param figsize: tuple of int containing height and width of the figure
    @param fpath: string containing the path where to save the figure, if neeeded, None otherwise
    '''
    imgs_index = np.random.choice(np.arange(len(images)), examples, replace=False)

    fig, axes = plt.subplots(int(examples / np.sqrt(examples)),
                             int(examples / np.sqrt(examples)),
                             figsize=figsize)

    axes = axes.ravel()
    image_shape = images[0].shape
    for idx, _ in enumerate(axes):
        X = images[imgs_index[idx]]
        if len(image_shape) == 2:
            axes[idx].imshow(X=X, cmap="gray")
        else:
            axes[idx].imshow(X=X)
        axes[idx].axis('off')

        if labels is not None:
            idx_y = labels[imgs_index[idx]]
            y = idxlabel_to_namelabel[idx_y]
            axes[idx].set_title(str(imgs_index[idx]) + ': ' + y)

    fig.suptitle(fig_suptitle, fontsize=12)

    if fpath:
        fig.savefig(fpath)

def plot_confusion_matrix(y_true,
                          y_pred,
                          idx_to_nameclass,
                          normalize=True,
                          figsize=(7, 7),
                          title='Confusion matrix',
                          cmap=plt.cm.Greys,
                          show_image=False):
    '''
    It constructs and plot a confusion matrix.
    @param y_true: Numpy array of shape (N,) containing the true labels
    @param y_pred: Numpy array of shape (N,) containing the predicted labels
    @param idx_to_nameclass: list containing at index i the name of i-th class
    @param normalize: boolean indicating if normalize confusion matrix or not
    @param figsize: int tuple containing the size of the figure
    @param title: string, the title of the confusion matrix
    @param cmap: color of the confusion matrix
    '''
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        print('Confusion matrix without normalization')
    figure = plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(idx_to_nameclass))
    plt.xticks(tick_marks, idx_to_nameclass, rotation=45)
    plt.yticks(tick_marks, idx_to_nameclass)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid(False)
    if show_image:
        plt.show()
    return figure

def my_histogram(data, figsize=(8, 5), color='b', title=None, show_image=False):
    figure = plt.figure(figsize=figsize)
    plt.hist(data, bins=25, color=color)
    plt.title(title, color='black')
    plt.xlabel('F(std)')
    plt.ylabel('# of images')
    if show_image:
        plt.show()
    return figure

def matplotlib_imshow(img):
    """
    Helper function to show an image
    @param img: Numpy array of shape (C, H, W) representing an image. Pay attention to the fact that the image
                here must be unnormalized, i.e. elements in range [0, 255] and dtype == uint8
    """
    C, _, _ = img.shape
    img = img.numpy()
    if C == 1:
        plt.imshow(img.squeeze(axis=0), cmap="Greys")
    elif C == 3:
        plt.imshow(img.transpose((1, 2, 0)))
    else:
        raise Exception('Wrong number of input image channels. Expected 1 or 3, ' + C + ' provided.')

def plot_to_image(figure):
      """Converts the matplotlib plot specified by 'figure' to a PNG image and
      returns it. The supplied figure is closed and inaccessible after this call."""
      # Save the plot to a PNG in memory.
      buf = io.BytesIO()
      plt.savefig(buf, format='png')
      # Closing the figure prevents it from being displayed directly inside
      # the notebook.
      plt.close(figure)
      buf.seek(0)
      # Convert PNG buffer to numpy array
      image = np.frombuffer(buf.getvalue(), dtype=np.uint8)
      buf.close()

      image = cv2.imdecode(image, 1)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      return image

def stats_classes(dataset_name, train=True, figsize=(5, 5), show_image=False):
    if dataset_name == 'CRC':
        y = np.load(os.path.join(os.getcwd(), 'data', 'crc_3_noisy', 'Y_train.npy' if train else 'Y_test.npy'))
        y = torch.from_numpy(y)
    elif dataset_name == 'MNIST':
        dataset = datasets.MNIST(root=os.path.join(os.getcwd(), 'data'), train=train, download=True)
        y = dataset.targets
    elif dataset_name == 'CIFAR10':
        dataset = datasets.MNIST(root=os.path.join(os.getcwd(), 'data'), train=train, download=True)
        y = dataset.targets
    else:
        raise Exception('Wrong --dataset option; ' + dataset_name + ' is not supported.'
                        ' Supported datasets: [CRC|MNIST|CIFAR10]')

    assert len(y.shape) == 1
    figure = plt.figure(figsize=figsize)

    classes_and_counts = torch.unique(y, sorted=True, return_counts=True)
    idx_classes = classes_and_counts[0].numpy()
    num_classes = len(idx_classes)
    colors = plt.get_cmap('Blues')(np.linspace(0, 1, num_classes))[::-1]

    # for each class generate a color: the more elements a class have and the more darker the color associated will be
    class_to_count = {classes_and_counts[0][idx].item(): classes_and_counts[1][idx].item()
                      for idx in range(num_classes)}
    class_to_count = sorted(class_to_count.items(), key=lambda x: x[1], reverse=True)
    ordered_classes = [idx_and_count[0] for idx_and_count in class_to_count]
    new_seq_colors = np.zeros_like(colors)
    for idx, idx_class in enumerate(ordered_classes):
        new_seq_colors[idx_class] = colors[idx]

    plt.bar(classes_and_counts[0], classes_and_counts[1], color=new_seq_colors)
    plt.xticks(classes_and_counts[0])
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    if show_image:
        plt.show()
    return figure

if __name__ == '__main__':
    # WARNING: this module is not intended to be runnable, so this main is only for testing purposes
    X_train = np.load(os.path.join(os.getcwd(), 'data', 'crc_3_noisy', 'X_train.npy'))
    Y_train = np.load(os.path.join(os.getcwd(), 'data', 'crc_3_noisy', 'Y_train.npy'))
    mean_image = np.load(os.path.join(os.getcwd(), 'data', 'crc_3_noisy', 'mean_x_train.npy'))

    X_train *= 255.
    X_train[:, :, :, 0] += mean_image[0]
    X_train[:, :, :, 1] += mean_image[1]
    X_train[:, :, :, 2] += mean_image[2]
    X_train = X_train.astype('uint8')

    if True:
        dataset = 'MNIST'
        writer = SummaryWriter()
        a = stats_classes(dataset)
        b = plot_to_image(a)
        writer.add_image(dataset+'_classes', np.transpose(b, (2, 0, 1)), 0)
        writer.close()

    if False:
        plot_random_images(X_train, Y_train, examples=16, fig_suptitle='X_train_samples', figsize=(6, 6))
        plt.show()

    if False:
        SAMPLES = 32
        Y_pred = np.random.randint(0, 3, SAMPLES)
        idx_samples = np.random.choice(np.arange(len(Y_train)), SAMPLES, replace=False)
        Y_true = Y_train[idx_samples]
        plot_confusion_matrix(Y_true, Y_pred)
        plt.show()


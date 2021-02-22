import torch
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter

import cv2

import io
import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def plot_random_images(images,
                       labels=None,
                       class_index=None,
                       examples=16,
                       fig_suptitle=None,
                       figsize=(8, 8),
                       fpath=None):
    '''
    Given a 'images' dataset a random subset is selected and showed
    @param images: Numpy array of shape (N, H, W, C) with pixels in range [0, 1] float or [0, 255] int
    @param labels: Numpy array of shape (N,) containing the labels of the images
    @param class_index: Python list containing at position i the name of class i
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
            y = class_index[idx_y]
            axes[idx].set_title(str(imgs_index[idx]) + ': ' + y)

    fig.suptitle(fig_suptitle, fontsize=12)

    if fpath:
        fig.savefig(fpath)

def plot_confusion_matrix(y_true,
                          y_pred,
                          class_index,
                          normalize=True,
                          figsize=(7, 7),
                          title='Confusion matrix',
                          cmap=plt.cm.Greys,
                          show_image=False):
    '''
    It constructs and plot a confusion matrix.
    @param y_true: Numpy array of shape (N,) containing the true labels
    @param y_pred: Numpy array of shape (N,) containing the predicted labels
    @param class_index: list containing at index i the name of i-th class
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
    tick_marks = np.arange(len(class_index))
    plt.xticks(tick_marks, class_index, rotation=45)
    plt.yticks(tick_marks, class_index)

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

def plot_histogram_classes(data,
                           target_all,
                           class_index,
                           figsize=(14, 8),
                           title=None,
                           show_image=False,
                           xlabel='F(std)',
                           ylabel='# of images'):
    # hard-coded
    LEGEND = class_index
    STACKED = True
    COLORS = cm.rainbow(np.linspace(0, 1, len(class_index)))

    # convert data from a tensor containing all values to a list
    #  in which the element at position i is a tensor containing all
    #  values of class i
    new_data = []
    for idx, cls in enumerate(class_index):
        cls_values = data[target_all == idx]    # cls_values is a torch tensor
        new_data.append(cls_values)

    plot_histogram(new_data, figsize, COLORS, title, show_image, xlabel, ylabel, legend=LEGEND, stacked=STACKED)

def plot_histogram(data,
                   figsize=(8, 5),
                   color='b',
                   title=None,
                   show_image=False,
                   xlabel='F(std)',
                   ylabel='# of images',
                   legend=None,
                   stacked=False):
    figure = plt.figure(figsize=figsize)
    plt.hist(data, bins=25, color=color, linewidth=1.2, edgecolor='black', stacked=stacked)
    plt.title(title, color='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend is not None:
        plt.legend(legend)
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

def save_fig_to_tensorboard(fig, writer, title):
    fig = plot_to_image(fig)
    writer.add_image(title, np.transpose(fig, (2, 0, 1)), 0)
    writer.close()

def stats_classes(dataset_name, class_index, train=True, figsize=(5, 5), show_image=False):
    if dataset_name == 'CRC':
        y = np.load(os.path.join(os.getcwd(), 'data', 'crc_3_noisy', 'real_classes_train_int.npy' if train else 'real_classes_test_int.npy'))
        y = torch.as_tensor(y, dtype=torch.int64)
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
    plt.xticks(classes_and_counts[0], class_index, rotation=45)
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    if show_image:
        plt.show()
    return figure
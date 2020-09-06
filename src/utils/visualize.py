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

def plot_confusion_matrix(y_true, y_pred, normalize=True, figsize=(7, 7), title='Confusion matrix', cmap=plt.cm.Greys):
    '''
    It constructs and plot a confusion matrix.
    @param y_true: Numpy array of shape (N,) containing the true labels
    @param y_pred: Numpy array of shape (N,) containing the predicted labels
    @param normalize: boolean indicating if normalize confusion matrix or not
    @param figsize: int tuple containing the size of the figure
    @param title: string, the title of the confusion matrix
    @param cmap: color of the confusion matrix
    '''
    cm = confusion_matrix(y_true, y_pred)
    classes = ['H', 'AC', 'AD']
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        print('Confusion matrix without normalization')
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

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

if __name__ == '__main__':
    # WARNING: this module is not intended to be runnable, so this main is only for testing purposes
    X_train = np.load(os.path.join('..', '..', 'data', 'X_train.npy'))
    Y_train = np.load(os.path.join('..', '..', 'data', 'Y_train.npy'))
    mean_image = np.load(os.path.join('..', '..', 'data', 'mean_x_train.npy'))

    X_train *= 255.
    X_train[:, :, :, 0] += mean_image[0]
    X_train[:, :, :, 1] += mean_image[1]
    X_train[:, :, :, 2] += mean_image[2]
    X_train = X_train.astype('uint8')

    if True:
        plot_random_images(X_train, Y_train, examples=16, fig_suptitle='X_train_samples', figsize=(6, 6))
        plt.show()

    if False:
        SAMPLES = 32
        Y_pred = np.random.randint(0, 3, SAMPLES)
        idx_samples = np.random.choice(np.arange(len(Y_train)), SAMPLES, replace=False)
        Y_true = Y_train[idx_samples]
        plot_confusion_matrix(Y_true, Y_pred)
        plt.show()


from typing import Union, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn

from torchvision.utils import make_grid


def visualize_batch(img_batch):
    n_rows = int(np.sqrt(img_batch.size()[0])) + 1
    grid = make_grid(img_batch, nrow=n_rows).cpu().numpy()
    plt.imshow(np.transpose(grid, (1, 2, 0)), interpolation='bilinear')
    plt.show()


def plot_confusion_matrix(
        cm: np.ndarray,
        categories: Union[str, list] = 'auto',
        count: bool = True,
        percent: bool = True,
        cbar: bool = True,
        xyticks: bool = True,
        xyplotlabels: bool = True,
        sum_stats: bool = True,
        cmap: Optional[str] = 'Blues',
        sort: bool = True,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True
):
    """
    Plot Confusion Matrix cm using a Seaborn heatmap visualization

    :param cm: 2D numpy array with confusion matrix
    :param categories: List of strings containing the categories to be displayed on the x,y axis
    :param count: If True, show counts in the confusion matrix cells
    :param percent: If True, show relative percentage in the confusion matrix cells
    :param cbar: If True, show the color bar. The cbar values are based off the values in the
        confusion matrix
    :param xyticks: If True, show x and y ticks
    :param xyplotlabels: If True, show 'True Labels' and 'Predicted Labels' on the figure
    :param sum_stats: If True, display summary statistics below the figure
    :param cmap: Colormap of the values displayed from matplotlib.pyplot.cm
    :param sort: If True, Confusion Matrix will be sorted based on the diagonal values
    :param title: Title for confusion matrix plot
    :param save_path: Path where to save consufion matrix plot
    :param show: Whether to show the confusion matrix plot
    """
    epsilon = 1e-7

    # Sort Confusion matrix if needed
    sorted_indices = None
    if sort:
        sorted_indices = np.argsort(np.diag(cm))[::-1]
        cm = cm[sorted_indices, :][:, sorted_indices]

    if categories != 'auto':
        # Append Not predicted class label if categories do not include it
        if cm.shape[0] - len(categories) == 1:
            categories.append('Not predicted')

        if sort:
            categories = np.array(categories)[sorted_indices]

    # Code to generate text inside each square
    blanks = ['' for i in range(cm.size)]

    if count:
        group_counts = ['{0:0.0f}\n'.format(value) for value in cm.flatten()]
    else:
        group_counts = blanks

    if percent:
        values = cm / (np.sum(cm, axis=1)[:, np.newaxis] + epsilon)
        values = values.flatten()
        group_percentages = ['{0:.2%}'.format(value) for value in values]
    else:
        group_percentages = blanks

    box_labels = [f'{v1}{v2}'.strip()
                  for v1, v2 in zip(group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cm.shape[0], cm.shape[1])

    # Code to generate summary statistics and text for summary stats
    if sum_stats:
        average_precision = np.array(
            [cm[i, i] / cm[:, i].sum() for i in range(cm.shape[0])]).mean()
        stats_text = f'\nAverage precision={average_precision:0.2f}'
    else:
        stats_text = ''

    if not xyticks:
        # Do not show categories if xyticks is False
        categories = False

    figsize = (cm.shape[0], cm.shape[0])
    if figsize[0] == 0:
        figsize = (10, 10)

    # Make the heatmap visualization
    plt.figure(figsize=figsize)
    seaborn.heatmap(
        cm,
        annot=box_labels,
        fmt='',
        cmap=cmap,
        cbar=cbar,
        xticklabels=categories,
        yticklabels=categories,
    )

    if xyplotlabels:
        plt.ylabel('True labels')
        plt.xlabel(f'Predicted labels {stats_text}')
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()

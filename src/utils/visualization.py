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
        cm,
        categories='auto',
        count=True,
        percent=True,
        cbar=True,
        xyticks=True,
        xyplotlabels=True,
        sum_stats=True,
        cmap='Blues',
        title=None,
        sort=True,
        path=None
):
    '''
    Plot Confusion Matrix cm using a Seaborn heatmap visualization

    Args:
        cm (np.ndarray): Confusion matrix

        categories (str, optional): List of strings containing the
        categories to be displayed on the x,y axis

        count (bool, optional): If True, show counts in the
        confusion matrix cells

        percent (bool, optional): If True, show relative percentage in
        the confusion matrix cells

        cbar (bool, optional): If True, show the color bar.
        The cbar values are based off the values in the confusion matrix

        xyticks (bool, optional): If True, show x and y ticks

        xyplotlabels (bool, optional): If True, show 'True Labels' and
        'Predicted Labels' on the figure

        sum_stats (bool, optional): If True, display summary statistics
        below the figure

        cmap (str, optional): Colormap of the values displayed
        from matplotlib.pyplot.cm

        sort(bool, optional): If True, Confusion Matrix will be sorted
        based on the diagonal values

        path (None, optional): Path to save the plot to. Shows the plot
        on the screen if path is None
    '''
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

    if path is not None:
        plt.savefig(path)
    plt.show()

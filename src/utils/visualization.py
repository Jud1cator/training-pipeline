from typing import Union, Optional, List

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn

from matplotlib import patches

from torchvision.utils import make_grid, draw_bounding_boxes


def visualize_batch(img_batch: Union[torch.Tensor, List[torch.Tensor]]):
    n_rows = int(np.sqrt(len(img_batch))) + 1
    grid = make_grid(img_batch, nrow=n_rows).cpu().numpy()
    plt.imshow(np.transpose(grid, (1, 2, 0)), interpolation='bilinear')
    plt.show()


def visualize_with_boxes(img_batch: torch.Tensor, boxes: torch.Tensor):
    imgs = []
    for i in range(img_batch.size()[0]):
        img = (img_batch[i] * 255).byte()
        colors = ['red'] * boxes[i].size()[0]
        img = draw_bounding_boxes(
            img.cpu(), boxes[i][:, [1, 0, 3, 2]].cpu(), colors=colors, width=2)
        imgs.append(img)
    n_rows = int(np.sqrt(len(img_batch))) + 1
    grid = make_grid(imgs, nrow=n_rows).numpy()
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


def get_rectangle_edges_from_pascal_bbox(bbox):
    xmin_top_left, ymin_top_left, xmax_bottom_right, ymax_bottom_right = bbox

    top_left = (xmin_top_left, ymin_top_left)
    width = xmax_bottom_right - xmin_top_left
    height = ymin_top_left - ymax_bottom_right

    return top_left, width, height


def get_rectangle_edges_from_coco_bbox(bbox):
    xmin_top_left, ymin_top_left, width, height = bbox

    top_left = (xmin_top_left, ymin_top_left)

    return top_left, width, height


def draw_bboxes(plot_ax, bboxes, get_rectangle_corners_fn):
    for bbox in bboxes:
        top_left, width, height = get_rectangle_corners_fn(bbox)

        rect_1 = patches.Rectangle(
            top_left,
            width,
            height,
            linewidth=4,
            edgecolor="black",
            fill=False,
        )
        rect_2 = patches.Rectangle(
            top_left,
            width,
            height,
            linewidth=2,
            edgecolor="white",
            fill=False,
        )

        # Add the patch to the Axes
        plot_ax.add_patch(rect_1)
        plot_ax.add_patch(rect_2)


def draw_pascal_voc_bboxes(plot_ax, bboxes):
    draw_bboxes(plot_ax, bboxes, get_rectangle_edges_from_pascal_bbox)


def draw_coco_bboxes(plot_ax, bboxes):
    draw_bboxes(plot_ax, bboxes, get_rectangle_edges_from_coco_bbox)


def show_image(image, draw_bboxes_fn, bboxes=None, figsize=(10, 10)):
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)

    if bboxes is not None:
        draw_bboxes_fn(ax, bboxes)

    plt.show()


def show_image_pascal_voc(image, bboxes=None, figsize=(10, 10)):
    show_image(image, draw_bboxes_fn=draw_pascal_voc_bboxes, bboxes=bboxes, figsize=figsize)


def show_image_coco(image, bboxes=None, figsize=(10, 10)):
    show_image(image, draw_bboxes_fn=draw_coco_bboxes, bboxes=bboxes, figsize=figsize)

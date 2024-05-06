from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import Counter
from torch.utils.data import DataLoader


__all__ = ['plot_distribution']


def plot_distribution(
    dataloader: DataLoader,
    *,
    title: str,
    labels_map: Optional[Dict[int, str]] = None,
    fig_show: bool = True
) -> Tuple[Counter, plt.Figure, plt.Axes]:
    """
    Plots the distribution of labels in the dataset represented by a DataLoader.

    This function counts occurrences of each label in the dataset and produces a bar
    chart displaying the distribution. The labels can optionally be mapped from integers
    to human-readable strings using the `labels_map` dictionary.

    Args:
        dataloader (DataLoader): The DataLoader for which to plot label distribution.
        title (str): The title for the plot.
        labels_map (Optional[Dict[int, str]]): An optional dictionary mapping label indices
                                                to human-readable labels. If provided, these
                                                labels are used on the x-axis.
        fig_show (bool): If True, the figure will be displayed using `plt.show()`.
                         Set to False to prevent the figure from being shown, useful
                         for saving the figure to a file instead.

    Returns:
        Tuple[Counter, plt.Figure, plt.Axes]: A tuple containing a Counter object with label
                                              counts, the Matplotlib Figure object, and the
                                              Matplotlib Axes object used for the plot.
    """
    label_counts = Counter()

    # Invert labels_map for easy lookup if provided
    inverted_labels_map = {v: k for k, v in labels_map.items()} if labels_map else None

    for _, labels in dataloader:
        labels = labels.numpy()  # Ensure labels are NumPy array
        if inverted_labels_map:
            labels = [inverted_labels_map[label] for label in labels]
        label_counts.update(labels)

    # Plotting the distribution
    fig, ax = plt.subplots()
    fig.suptitle(title)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_ylabel('Number of Occurrences')
    ax.set_xlabel('Labels')
    bars = ax.bar(label_counts.keys(), label_counts.values())

    # Enhance plot aesthetics
    ax.spines['top'].set_visible(False)
    ax.set_xticks(range(len(label_counts)))
    ax.set_xticklabels(label_counts.keys(), rotation=45, ha="right")

    # Annotate bars with count value
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    if fig_show:
        plt.show()

    return label_counts, fig, ax

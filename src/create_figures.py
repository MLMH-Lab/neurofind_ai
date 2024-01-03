"""Figures to visualise results for AAE and brain age.

These figures are created on the test datasets, each incl. a patient group and healthy controls.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def get_box_plot(df, metric='deviations', grouping_var='scanner', output_path=None, filename='box_plot'):
    """ Box-and-whisker plot for the chosen deviation metric.

    Parameters
    ----------
    df: NDFrame
        Dataframe containing the metric to plot.
    metric: str, default='deviations'
        Name of the column with the chosen metric to plot. Default is the autoencoder output.
    grouping_var: str, default='scanner'
        Name of variable that determines how the plot is displayed. If 'scanner' the overall metric is displayed
        separately for each scanner; if 'group_scanner' the curves for each scanner*diagnostic group are shown
        separately instead.
    output_path: str, default=output_path
        Name of directory where figures will be saved.

    Returns
    ---------
    g: Matplotlib figure
        Box-and-whisker plot for the chosen deviation metric.
    """
    names = df[grouping_var].unique()
    names = np.sort(names)
    n_cols = len(names)

    # box_fig = plt.figure(figsize=(20, 20))
    box_fig, axes = plt.subplots(n_cols, 1, figsize=(10, 8), sharex='col')

    if grouping_var != 'Diagn':
        for name, ax in zip(names, axes.flatten()):
            sns.boxplot(data=df[df[grouping_var] == name], x=metric, y='Diagn', orient='h', ax=ax).set_title(name)
            # fig.set_title(name)
    else:
        sns.boxplot(data=df, x=metric, y='Diagn', orient='h').set_title(grouping_var)

    plt.tight_layout()
    # plt.show()

    image_path = f'{output_path}/{filename}.png'
    if output_path is not None:
        plt.savefig(image_path)

    # box_fig.savefig(Path(output_path / 'box_plot.png'))

    return image_path

""" Functions for plotting behavioral variables directly from metadata or similar outputs"""
import seaborn as sns
import matplotlib.pyplot as plt
from .. import utils, load, paths
import os


def plt_behavior_from_meta(meta, staging='parsed_10stage', save_folder=None):
    """
    Function for plotting

    :param staging: str, the way you want to set stages of learning
    :return: saves plot of behavioral variables
    """

    # get mouse from metadata index
    mouse = meta.reset_index()['mouse'].unique()[0]

    # add your learning stage column if it doesn't exist
    if 'parsed_stage' not in meta.columns and 'parsed_stage' in staging:
        meta = utils.add_5stages_to_meta(meta)
    elif 'parsed_10stage' not in meta.columns and 'parsed_10stage' in staging:
        meta = utils.add_10stages_to_meta(meta)

    # columns to plot
    col_to_plot = ['neuropil', 'pre_speed', 'delta_speed', 'pre_licks', 'delta_lickrate', 'test_fac']

    # plot all behavior variables as a single long plot with subplot
    fig, axes = plt.subplots(1, len(col_to_plot), figsize=(7 *len(col_to_plot) ,6))

    # loop over and plot each sublot column of behavior data
    for c, var in enumerate(col_to_plot):
        g = sns.barplot(data=meta,
                        y=var,
                        x=staging,
                        hue='condition',
                        palette=cas.lookups.color_dict, ax=axes[c])
        axes[c].set_xticklabels(axes[c].get_xticklabels(), rotation=45, ha='right');
        axes[c].set_title(f'{var}\n', size=18)

        if c == len(col_to_plot) - 1:
            axes[c].set_title(f'{mouse}: rank 9: component 8\n', size=18)

    plt.savefig(os.path.join(save_folder, '{mouse} behavior {staging}.png'), bbox_inches='tight')


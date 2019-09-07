"""
Functions for plotting trial history effects using modeling from the
Pillow lab and tensortools TCA results.
"""
import flow
import pool
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from .. import paths, load
from .. import trialhistory
from scipy.stats import pearsonr


def groupmouse_index_heatmap(
        mice=['OA27', 'OA26', 'OA67', 'VF226', 'CC175', 'OA32', 'OA34', 'OA36'],
        words=None,
        group_by='all',
        rank_num=18,
        verbose=True):

    # get all dataframes
    all_dfs = trialhistory.groupmouse_th_index_dataframe(
                mice=mice,
                words=words,
                group_by=group_by,
                rank_num=rank_num,
                verbose=verbose)

    # create colormap
    cmap = sns.diverging_palette(220, 10, sep=30, as_cmap=True)

    # create x labels
    xlab = ['sensory_history', 'reward_history',
            'reward_history - sensory_history', 'learning_index']

    # sort according to degree of modulation by behavioral performance
    # a.k.a., learning
    sorter = np.argsort(all_dfs['learning_index'].values)
    plt_df = all_dfs.reset_index(['component']).values

    cs_to_check = ['plus', 'minus', 'neutral']
    for cs in cs_to_check:
        cs_bool = all_dfs.reset_index()['condition'].values == cs
        sort_bool = cs_bool[sorter]
        cs_plt_df = plt_df[sorter, 1:][sort_bool]
        cs_y_label = plt_df[sorter, 0][sort_bool]

        plt.figure()
        sns.heatmap(cs_plt_df, center=0, vmax=1, vmin=-1, cmap=cmap,
                    yticklabels=cs_y_label, xticklabels=xlab)
        plt.title('Condition: ' + cs)

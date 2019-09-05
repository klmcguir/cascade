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

    # sort according to degree of modulation by behavioral performance
    # a.k.a., learning
    sorter = np.argsort(all_dfs['learning_index'].values)

    # sorter = np.arange(0,rank)
    xlab = ['sensory_history', 'reward_history', 'reward_history - sensory_history', 'learning_idx']
    plt.figure()
    sns.heatmap(trial_hist_mod[sorter, :], center=0, vmax=1, vmin=-1, cmap=cmap, yticklabels=np.arange(1, rank+1)[sorter], xticklabels=xlab)
    plt.title('Orientation ' + str(ori))

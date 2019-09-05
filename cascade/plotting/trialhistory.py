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
from scipy.stats import pearsonr


def groupmouse_index_heatmap(
    mice=['OA27', 'OA26'],
    words=['tray', 'obligations']
    group_by='all'):

            
    cmap = sns.diverging_palette(220, 10, sep=30, as_cmap=True)
    sorter = np.argsort(trial_hist_mod[:, -1])
    # sorter = np.arange(0,rank)
    xlab = ['sensory_history', 'reward_history', 'reward_history - sensory_history', 'learning_idx']
    plt.figure()
    sns.heatmap(trial_hist_mod[sorter, :], center=0, vmax=1, vmin=-1, cmap=cmap, yticklabels=np.arange(1, rank+1)[sorter], xticklabels=xlab)
    plt.title('Orientation ' + str(ori))
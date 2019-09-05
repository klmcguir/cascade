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




# ori = 'all'
# save_pls = True
iteration = 0
ori_to_check = [0, 135, 270]
for ori in ori_to_check:
# for rank in tensor.results:
    for rank in [18]:
        data = {}
        for i in range(rank):
            fac = tensor.results[rank][iteration].factors[2][:,i]
            data['factor_' + str(i+1)] = fac
        fac_df = pd.DataFrame(data=data, index=meta1.index)

        # loop over single oris
        psy_fac = pd.concat([psy1, fac_df], axis=1).drop(columns='orientation')
        ori_bool = (meta1['orientation'] == ori)  & (meta1['learning_state'] == 'learning')  # only look during initial learning
        single_ori = psy_fac.loc[ori_bool]
#         single_meta = meta1.loc[ori_bool]

        # get means for each factor for each type of trial history
        trial_history = {}
        trial_hist_mod = np.zeros((rank, 4))
        for i in range(rank):
            single_factor = single_ori['factor_' + str(i+1)].values
            bool_curr = single_ori['ori_' + str(ori)] == 1
            bool_prev = single_ori['ori_' + str(ori) + '_th'] == 1
            
            prev_same = np.nanmean(single_factor[single_ori['ori_' + str(ori) + '_th_prev'] == 1]) 
            prev_diff = np.nanmean(single_factor[single_ori['ori_' + str(ori) + '_th_prev'] == 0])
            sensory_history = (prev_diff - prev_same)/np.nanmean(single_factor)
            
            prev_same = np.nanmean(single_factor[single_ori['prev_reward_th'] == 1]) 
            prev_diff = np.nanmean(single_factor[single_ori['prev_reward_th'] == 0])
            reward_history = (prev_diff - prev_same)/np.nanmean(single_factor)
            
            high_dp = np.nanmean(single_factor[single_ori['dprime'] >= 2]) 
            low_dp = np.nanmean(single_factor[single_ori['dprime'] < 2])
            learning_idx = (high_dp - low_dp)/np.nanmean(single_factor)
            
            trial_hist_mod[i, 0] = sensory_history
            trial_hist_mod[i, 1] = reward_history
            trial_hist_mod[i, 2] = reward_history - sensory_history
            trial_hist_mod[i, 3] = learning_idx
            
            print('rank', str(i+1), sensory_history, reward_history, learning_idx)
            
    cmap = sns.diverging_palette(220, 10, sep=30, as_cmap=True)
    sorter = np.argsort(trial_hist_mod[:, -1])
    # sorter = np.arange(0,rank)
    xlab = ['sensory_history', 'reward_history', 'reward_history - sensory_history', 'learning_idx']
    plt.figure()
    sns.heatmap(trial_hist_mod[sorter, :], center=0, vmax=1, vmin=-1, cmap=cmap, yticklabels=np.arange(1, rank+1)[sorter], xticklabels=xlab)
    plt.title('Orientation ' + str(ori))
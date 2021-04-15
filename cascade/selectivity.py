import numpy as np
import pandas as pd
from . import drive, utils, paths, lookups, load, categorize
from sklearn.metrics import roc_auc_score
from copy import deepcopy

import seaborn as sns
import matplotlib.pyplot as plt
import warnings

"""
TODO: function to get preferred tuning from component category vector 
"""

def calculate_index_core_reversal(match_to='onsets', index_type='ab_ab', on='go', staging='parsed_4stage'):

    # calculate drive 
    # TODO save and auto load this npy if it exists
    drive_mat_list = drive.drive_day_mat_from_core_reversal_data(match_to='onsets')

    # load all of your raw reversal n=7 data
    load_dict = load.core_reversal_data(limit_to=None, match_to=match_to)

    # load models for cell categorization
    ensemble = np.load(paths.analysis_file('tca_ensemble_v4i10_noT0_20210307.npy', 'tca_dfs'), allow_pickle=True).item()
    tca_data_dict = np.load(paths.analysis_file('input_data_v4i10_noT0_20210307.npy', 'tca_dfs'), allow_pickle=True).item()
    mod = 'v4i10_norm_on_noT0' if match_to == 'onsets' else 'v4i10_norm_off_noT0'
    mouse_mod = 'v4i10_on_mouse_noT0' if match_to == 'onsets' else 'v4i10_off_mouse_noT0'
    rank = 9 if match_to == 'onsets' else 8
    factors = ensemble[mod].results[rank][0].factors
    cell_cats = categorize.best_comp_cats(factors)
    mouse_vec = tca_data_dict[mouse_mod]

    # calculate drivenness on each cell looping over mice
    si_list = []
    for meta, ids, tensor, dmat in zip(load_dict['meta_list'], load_dict['id_list'], load_dict['tensor_list'], drive_mat_list):

        # make sure binning method is in meta
        meta = utils.add_stages_to_meta(meta, staging)

        # get cetegory vector for individual mice
        mouse_boo = mouse_vec == utils.meta_mouse(meta)
        mouse_cats = cell_cats[mouse_boo]

        # calculate index of choice
        si_mat = selectivity_index(meta, tensor, drive_mat=dmat, index_type=index_type, on=on,
                          staging=staging, cats=mouse_cats, plot_please=False)
        si_list.append(si_mat)
    si_mat = np.vstack(si_list)

    return si_mat


def selectivity_index(meta,
                      tensor,
                      drive_mat=None,
                      index_type='ab_ab',
                      on='go',
                      staging='parsed_4stage',
                      cats=None,
                      plot_please=False):

    # for plotting cell tuning
    # trial_means = np.nanmean(tensor[:, 17:, :], axis=1)
    # cues = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']
    # cue_means = np.zeros((trial_means.shape[0], len(cues))) + np.nan
    # for cc, cue in enumerate(cues):
    #     cue_boo = meta.mismatch_condition.isin([cue])
    #     cue_means[:, cc] = np.nanmean(trial_means[:, cue_boo], axis=1)
    # pref_cue = np.argmax(cue_means, axis=1)
    # pref_list = [cues[s] for s in pref_cue]

    # add reversal condition to metadata dataframe. 
    if 'mismatch_condition' not in meta.columns:
        meta = utils.add_reversal_mismatch_condition_to_meta(meta)

    # nan trials for undriven days/cues 
    cues = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']
    if drive_mat is not None:
        tensor = deepcopy(tensor)
        for ci, cue in enumerate(cues):
            cue_boo = meta.mismatch_condition.isin([cue]).values
            trial_boo = drive_mat[:, :, ci] == 0
            for celli in range(tensor.shape[0]):
                tensor[celli, :, trial_boo[celli, :] & cue_boo] = np.nan

    stages = lookups.staging[staging]
    m_si_index = np.zeros((tensor.shape[0], len(stages))) + np.nan
    for c, di in enumerate(stages):
        stage_boo = meta[staging].isin([di]).values
        day_stage_boo = stage_boo
        # stage_days = meta.loc[stage_boo, :].reset_index()['date'].unique()
        # day_means = np.zeros((tensor.shape[0], tensor.shape[1], len(stage_days)))
        # day_means[:] = np.nan
        #         for c2, di2 in tqdm(enumerate(stage_days),
        #                             desc=f'{utils.meta_mouse(meta)}: {di}, cue SVM',
        #                             total=len(stage_days)):
        #             day_boo = meta.reset_index()['date'].isin([di2]).values

        # define trials you calc index on
        if on == 'go':
            hit_boo = meta.trialerror.isin([0, 3, 5]) #3,5
            cr_boo = meta.trialerror.isin([1, 2, 4]) #2, 4
        elif on == 'th':
            prev_same = (meta.prev_same_plus | meta.prev_same_neutral | meta.prev_same_minus)
            hit_boo = ~prev_same
            cr_boo = prev_same
        elif on == 'speed':
            hit_boo = meta.pre_speed.ge(10) #3,5
            cr_boo = meta.pre_speed.le(5) #2, 4
        elif on == 'rh' or on == 'reward':
            # reward history, focusing only on FC-FC trials 
            hit_boo = meta.prev_reward & (meta.prev_same_plus | meta.prev_same_neutral | meta.prev_same_minus)
            cr_boo = ~meta.prev_reward & (meta.prev_same_plus | meta.prev_same_neutral | meta.prev_same_minus)
        elif on == 'ph' or on == 'punishment':
            # punishment history, focusing only on QC-QC trials 
            hit_boo = meta.prev_punish & (meta.prev_same_plus | meta.prev_same_neutral | meta.prev_same_minus)
            cr_boo = ~meta.prev_punish & (meta.prev_same_plus | meta.prev_same_neutral | meta.prev_same_minus)

        cr_tensor = tensor[:, 17:, day_stage_boo & cr_boo]
        hit_tensor = tensor[:, 17:, day_stage_boo & hit_boo]
        # plt_cr_tensor =  tensor[:, :, day_stage_boo & cr_boo]
        # plt_hit_tensor = tensor[:, :, day_stage_boo & hit_boo]

        # nan the unpreferred cue types for each cell
        if cats is not None:
            hit_meta = meta.loc[day_stage_boo & hit_boo]
            cr_meta = meta.loc[day_stage_boo & cr_boo]
            cue_groups = [[ 0,  5,  6], [4,  8], [1,  2,  3], [7], [-1]]
            cue_groups_labels =[
                ['becomes_unrewarded'],
                ['remains_unrewarded'],
                ['becomes_rewarded'],
                ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded'],
                ['supressed']  # will nan out
            ]
            # nan unpreffered cue for cells based on TCA preferred tuning
            for cg, lbl in zip(cue_groups, cue_groups_labels):
                cell_cg_boo = np.isin(cats, cg)
                hit_lbl_boo = ~hit_meta.mismatch_condition.isin(lbl)
                cr_lbl_boo = ~cr_meta.mismatch_condition.isin(lbl)
                for celli in np.where(cell_cg_boo)[0]:
                    cr_tensor[celli, :, cr_lbl_boo] = np.nan
                    hit_tensor[celli, :, hit_lbl_boo] = np.nan
                    # plt_cr_tensor[celli, :, cr_lbl_boo] = np.nan
                    # plt_hit_tensor[celli, :, hit_lbl_boo] = np.nan
        cr_mean = np.nanmean(cr_tensor, axis=2)
        hit_mean = np.nanmean(hit_tensor, axis=2)

        # calculate discriminability variable for hit and CR trials
        if index_type == 'selectivity':
            # SI = 2*(AUC([prevdiff_discrim_dot, prevsame_discrim_dot])-0.5)
            # Value-guided remapping of sensory cortex by lateral orbitofrontal cortex, 2020
            # Abhishek Banerjee1,3, Giuseppe Parente1,4, Jasper Teutsch1,3,4, Christopher Lewis1,
            # Fabian F. Voigt1,2 & Fritjof Helmchen1,2

            # calculate descrimination score
            dot_mat_cr = np.zeros((cr_tensor.shape[0], cr_tensor.shape[2]))
            for i in range(cr_tensor.shape[0]):
                dot_mat_cr[i, :] = cr_mean[i,:] @ cr_tensor[i, :, :] - hit_mean[i,:] @ cr_tensor[i, :, :]
            dot_mat_hit = np.zeros((hit_tensor.shape[0], hit_tensor.shape[2]))
            for i in range(hit_tensor.shape[0]):
                dot_mat_hit[i, :] = hit_mean[i,:] @ hit_tensor[i, :, :] - cr_mean[i,:] @ hit_tensor[i, :, :]

                # calculate si index for stage, skip if all trials are nan
            y = np.array(['hit'] * np.sum(day_stage_boo & hit_boo) + ['CR'] * np.sum(day_stage_boo & cr_boo))
            y_hat = np.concatenate([dot_mat_hit, dot_mat_cr], axis=1)
            si_vec = np.zeros(y_hat.shape[0]) + np.nan
            for i in range(y_hat.shape[0]):
                if np.isnan(y_hat[i, :]).all():
                    continue
                notna = ~np.isnan(y_hat[i, :])
                gcount = np.sum(np.isin(y[notna], 'hit'))
                ngcount = np.sum(np.isin(y[notna], 'CR'))
                if gcount < 10 and ngcount < 10:
                    print(f'Go: {gcount} NoGO: {ngcount}')
                    continue
                try:
                    si = 2*(roc_auc_score(y[notna], y_hat[i, notna]) - 0.5)
                except:
                    si = np.nan
                si_vec[i] = si

            m_si_index[:, c] = si_vec

        elif index_type == 'ab_ab':
            
            # a - b / a + b
            dot_mat_hit = np.nanmean(np.nanmean(hit_tensor, axis=1), axis=1)
            dot_mat_cr = np.nanmean(np.nanmean(cr_tensor, axis=1), axis=1)
            dot_mat_hit[dot_mat_hit < 0] = 0.0000001
            dot_mat_cr[dot_mat_cr < 0] = 0.0000001
            m_si_index[:, c] = (dot_mat_hit - dot_mat_cr) / (dot_mat_hit + dot_mat_cr)

        elif index_type == 'ab_mean':

            # a - b / mean(all a and b trials)
            all_trial_tensor = np.nanmean(np.nanmean(np.concatenate([hit_tensor, cr_tensor], axis=2), axis=1), axis=1)
            dot_mat_hit = np.nanmean(np.nanmean(hit_tensor, axis=1), axis=1)
            dot_mat_cr = np.nanmean(np.nanmean(cr_tensor, axis=1), axis=1)
            dot_mat_hit[dot_mat_hit < 0] = np.nan
            dot_mat_cr[dot_mat_cr < 0] = np.nan
            all_trial_tensor[all_trial_tensor < 0] = 0.0000001
            m_si_index[:, c] = (dot_mat_hit - dot_mat_cr) / all_trial_tensor

        elif index_type == 'auc':
            
            dot_mat_hit = np.nanmean(hit_tensor, axis=1)
            dot_mat_cr = np.nanmean(cr_tensor, axis=1)
            y = np.array(['hit'] * np.sum(day_stage_boo & hit_boo) + ['CR'] * np.sum(day_stage_boo & cr_boo))
            y_hat = np.concatenate([dot_mat_hit, dot_mat_cr], axis=1)
            si_vec = np.zeros(y_hat.shape[0]) + np.nan
            for i in range(y_hat.shape[0]):
                if np.isnan(y_hat[i, :]).all():
                    continue
                notna = ~np.isnan(y_hat[i, :])
                gcount = np.sum(np.isin(y[notna], 'hit'))
                ngcount = np.sum(np.isin(y[notna], 'CR'))
                if gcount < 10 and ngcount < 10:
                    print(f'Go: {gcount} NoGO: {ngcount}')
                    continue
                try:
                    si = 2*(roc_auc_score(y[notna], y_hat[i, notna]) - 0.5)
                except:
                    si = np.nan
                si_vec[i] = si

            m_si_index[:, c] = si_vec

#         # save for plotting
#         if c == 1:
#             plt_cr_tensor1 =  np.nanmean(plt_cr_tensor, axis=2)
#             plt_hit_tensor1 = np.nanmean(plt_hit_tensor, axis=2)
#             yhat1 = deepcopy(y_hat)
#             y1 = deepcopy(y)
#         elif c == 3:
#             plt_cr_tensor3 =  np.nanmean(plt_cr_tensor, axis=2)
#             plt_hit_tensor3 = np.nanmean(plt_hit_tensor, axis=2)
#             yhat3 = deepcopy(y_hat)
#             y3 = deepcopy(y)

#     if plot_please:
#         counter = 0
#         for i in range(m_si_index.shape[0]):

#             cell_si = m_si_index[i, :]
#             tuning = pref_list[i]

#             if not np.isnan(cell_si[1]) and not np.isnan(cell_si[3]):

#                 if counter > 10:
#                     continue

#                 # looking at upper left quadrant
#                 if cell_si[1] > -0.2 or  cell_si[3] < 0.2:
#                     continue
#                 fig, ax = plt.subplots(2, 2, sharex='col', sharey='col')
#                 ax[0, 0].plot(plt_cr_tensor1[i,:], label='nogo', color='black')
#                 ax[0, 0].plot(plt_hit_tensor1[i,:], label='go', color=lookups.color_dict[tuning])
#                 ax[0, 0].legend()
#                 ax[0, 0].axvline(15.5, color='red', linestyle='--')
#                 ax[1, 0].plot(plt_cr_tensor3[i,:], label='nogo', color='black')
#                 ax[1, 0].plot(plt_hit_tensor3[i,:], label='go', color=lookups.color_dict[tuning])
#                 ax[1, 0].legend()
#                 ax[1, 0].axvline(15.5, color='red', linestyle='--')
#                 c1 = [lookups.color_dict[tuning] if s == 'hit' else 'black' for s in y1]
#                 ax[0, 1].scatter(np.arange(yhat1.shape[1]), yhat1[i, :], c=c1)
#                 ax[0, 1].legend(labels=['go', 'nogo'])
#                 c3 = [lookups.color_dict[tuning] if s == 'hit' else 'black' for s in y3]
#                 ax[1, 1].scatter(np.arange(yhat3.shape[1]), yhat3[i, :], c=c3)
#                 ax[1, 1].legend(labels=['go', 'nogo'])
#                 ax[0, 0].set_ylabel(f'late_learning responses\nSI={round(cell_si[1], 3)}', ha='right', rotation=0)
#                 ax[1, 0].set_ylabel(f'late_reversal responses\nSI={round(cell_si[3], 3)}', ha='right', rotation=0)
#                 ax[1, 0].set_xlabel('time from onset', ha='right', rotation=0)
#                 ax[1, 1].set_xlabel('trials', ha='right', rotation=0)
#                 ax[0, 0].set_title('average responses')
#                 ax[0, 1].set_title('trial dot product')

#                 plt.suptitle(f'{utils.meta_mouse(meta)} celln{i} {tuning}')
#                 save_file = paths.analysis_file(f'{utils.meta_mouse(meta)}_celln{i}_{tuning}.png', 'selectivity_testing/')
#                 plt.savefig(save_file, bbox_inches='tight', facecolor='white')
#                 counter += 1
#         plt.close('all')

    return m_si_index



def onset_histogram(index, cell_cats=None, cell_sorter=None, rr=9):

    # get param to define caegories in usual sort order (by TCA comp)
    cues = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']
    cats_in_order = pd.unique(cell_cats[cell_sorter])
    new_order = cats_in_order[:-1]  # last cat is -1 is cells with no cattegory (0 weight)

    _, ax = plt.subplots(rr, 4, figsize=(15,25), sharex=True, sharey='row')
    for stage_n in range(4):
        for ni, cati in enumerate(new_order):
            cvboo = cell_cats == cati
            for ci, cue in enumerate(cues):
                if ci != 0 and ni <= 2:
                    continue
                if ci != 1 and (ni > 2 and ni <= 4):
                    continue
                if ci != 2 and (ni > 4 and ni <= 7):
                    continue

                r_index = index[cvboo, stage_n]
                sns.histplot(r_index, bins=np.arange(-1, 1.1, 0.1),
                            ax=ax[ni, stage_n], color=lookups.color_dict[cue])
                ax[ni, stage_n].axvline(0, linestyle='--', color='black')
                        # ignore python and numpy divide by zero warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    with np.errstate(invalid='ignore', divide='ignore'):
                        r_mean = np.nanmean(r_index)
                ax[ni, stage_n].axvline(r_mean, linestyle='--', color='red')
                
                if ni == rr-1:
                    ax[ni, stage_n].set_xlabel('RH index')
                
    # plt.suptitle(f'ONSET running modulation index', position=(0.5, 0.9), size=20)
    # plt.savefig(os.path.join(save_folder, f'on_RUNmodindex_balancestage.png'))
            
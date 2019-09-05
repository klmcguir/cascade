"""
Functions for plotting trial history effects using modeling from the
Pillow lab and tensortools TCA results.
"""
import flow
import pool
import pandas as pd
import numpy as np
import os
from . import load


def groupmouse_th_index_dataframe(
        mice=['OA27', 'OA26', 'OA67', 'VF226', 'CC175', 'OA32', 'OA34', 'OA36'],
        words=None,
        group_by='all',
        rank_num=18,
        verbose=True):
    """
    Create a pandas dataframe of trial history modulation indices across all
    mice.
    """

    # ensure that 'words=None' allows defaults to run in th_index_dataframe
    if not words:
        words = [words]*len(mice)

    # get all single mouse dataframes
    df_list = []
    for m, w in zip(mice, words):
        th_df = th_index_dataframe(
                    m,
                    word=w,
                    rank_num=rank_num,
                    group_by=group_by,
                    verbose=verbose)
        df_list.append(th_df)
    all_dfs = pd.concat(df_list, axis=0)

    return all_dfs


def th_index_dataframe(
        mouse,
        word=None,
        rank_num=18,
        group_by='all',
        verbose=True):
    """
    Create a pandas dataframe of trial history modulation indices for one
    mouse. Only looks at initial learning stage.
    """

    # default TCA params to use
    if not word:
        if mouse == 'OA27':
            word = 'tray'
        else:
            word = 'already'  # should be updated to 'obligations'
        if verbose:
            print('Creating dataframe for ' + mouse + '-' + word)

    ms = flow.Mouse(mouse)
    psy = ms.psytracker(verbose=True)
    dateRuns = psy.data['dateRuns']
    trialRuns = psy.data['runLength']

    # create your trial indices per day and run
    trial_idx = []
    for i in trialRuns:
        trial_idx.extend(range(i))

    # get date and run vectors
    date_vec = []
    run_vec = []
    for c, i in enumerate(dateRuns):
        date_vec.extend([i[0]]*trialRuns[c])
        run_vec.extend([i[1]]*trialRuns[c])

    # create your data dict, transform from log odds to odds ratio
    data = {}
    for c, i in enumerate(psy.weight_labels):
        # adding multiplication step here with binary vector !!!!!!
        data[i] = np.exp(psy.fits[c, :])*psy.inputs[:, c].T
    ori_0_in = [i[0] for i in psy.data['inputs']['ori_0']]
    ori_135_in = [i[0] for i in psy.data['inputs']['ori_135']]
    ori_270_in = [i[0] for i in psy.data['inputs']['ori_270']]
    blank_in = [
        0 if i == 1 else 1 for i in
        np.sum((ori_0_in, ori_135_in, ori_270_in), axis=0)]

    # loop through psy data create a binary vectors for trial history
    binary_cat = ['ori_0', 'ori_135', 'ori_270', 'prev_reward']
    for cat in binary_cat:
        data[cat + '_th'] = [i[0] for i in psy.data['inputs'][cat]]
        data[cat + '_th_prev'] = [i[1] for i in psy.data['inputs'][cat]]

    # create a single list of orientations to match format of meta
    ori_order = [0, 135, 270, -1]
    data['orientation'] = [
        ori_order[np.where(np.isin(i, 1))[0][0]]
        for i in zip(ori_0_in, ori_135_in, ori_270_in, blank_in)]

    # create your index out of relevant variables
    index = pd.MultiIndex.from_arrays([
                [mouse]*len(trial_idx),
                date_vec,
                run_vec,
                trial_idx
                ],
                names=['mouse', 'date', 'run', 'trial_idx'])

    # make master dataframe
    dfr = pd.DataFrame(data, index=index)

    # load TCA data
    tensor, ids, clus, meta = load.groupday_tca(
                                mouse,
                                word=word,
                                group_by=group_by)

    # add in continuous dprime
    dp = pool.calc.psytrack.dprime(flow.Mouse(mouse))
    dfr['dprime'] = dp

    # filter out blank trials
    psy_df = dfr.loc[(dfr['orientation'] >= 0), :]

    # check that all runs have matched trial orienations
    new_psy_df_list = []
    new_meta_df_list = []
    dates = meta.reset_index()['date'].unique()
    for d in dates:
        psy_day_bool = psy_df.reset_index()['date'].isin([d]).values
        meta_day_bool = meta.reset_index()['date'].isin([d]).values
        psy_day_df = psy_df.iloc[psy_day_bool, :]
        meta_day_df = meta.iloc[meta_day_bool, :]
        runs = meta_day_df.reset_index()['run'].unique()
        for r in runs:
            psy_run_bool = psy_day_df.reset_index()['run'].isin([r]).values
            meta_run_bool = meta_day_df.reset_index()['run'].isin([r]).values
            psy_run_df = psy_day_df.iloc[psy_run_bool, :]
            meta_run_df = meta_day_df.iloc[meta_run_bool, :]
            psy_run_idx = psy_run_df.reset_index()['trial_idx'].values
            meta_run_idx = meta_run_df.reset_index()['trial_idx'].values

            # drop extra trials from trace2P that don't have associated imaging
            max_trials = np.min([len(psy_run_idx), len(meta_run_idx)])

            # get just your orientations for checking that trials are matched
            meta_ori = meta_run_df['orientation'].iloc[:max_trials]
            psy_ori = psy_run_df['orientation'].iloc[:max_trials]

            # make sure all oris match between vectors of the same length each day
            assert np.all(psy_ori.values == meta_ori.values)

            # if everything looks good, copy meta index into psy
            meta_new = meta_run_df.iloc[:max_trials]
            psy_new = psy_run_df.iloc[:max_trials]
            data = {}
            for i in psy_new.columns:
                data[i] = psy_new[i].values
            new_psy_df_list.append(pd.DataFrame(data=data, index=meta_new.index))
            new_meta_df_list.append(meta_new)

    meta1 = pd.concat(new_meta_df_list, axis=0)
    psy1 = pd.concat(new_psy_df_list, axis=0)

    iteration = 0
    ori_to_check = [0, 135, 270]
    ori_vec, cond_vec = [], []
    for ori in ori_to_check:
    # for rank in tensor.results:
        for rank in [rank_num]:
            data = {}
            for i in range(rank):
                fac = tensor.results[rank][iteration].factors[2][:,i]
                data['factor_' + str(i+1)] = fac
            fac_df = pd.DataFrame(data=data, index=meta1.index)

            # loop over single oris
            psy_fac = pd.concat([psy1, fac_df], axis=1).drop(columns='orientation')
            ori_bool = (meta1['orientation'] == ori)  & (meta1['learning_state'] == 'learning')  # only look during initial learning
            single_ori = psy_fac.loc[ori_bool]

            # check the condition for this ori
            single_meta = meta1.loc[ori_bool]
            cond = single_meta['condition'].unique()
            assert len(cond) == 1

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

            trial_history['sensory_history'] = trial_hist_mod[:, 0]
            trial_history['reward_history'] = trial_hist_mod[:, 1]
            trial_history['diff_reward_sensory'] = trial_hist_mod[:, 2]
            trial_history['learning_index'] = trial_hist_mod[:, 3]

        ori_vec.extend([ori]*rank_num)
        cond_vec.extand([cond]*rank_num)
    # create your index out of relevant variables
    index = pd.MultiIndex.from_arrays([
                [mouse]*rank_num,
                ori_vec,
                cond_vec,
                list(np.arange(1, rank_num) + 1)
                ],
                names=['mouse', 'orientation', 'condition', 'component'])

    # make master dataframe
    th_df = pd.DataFrame(trial_history, index=index)

    return th_df

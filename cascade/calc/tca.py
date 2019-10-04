"""
Functions for making simple calculations on tensortools TCA results.
Saves into MongoDB database for quick retrieval.
"""
from pool.database import memoize
import flow
import pool
import pandas as pd
import numpy as np
import os
from copy import deepcopy
from .. import load, utils


@memoize(across='mouse', updated=191002, returns='other', large_output=False)
def trial_factor_tuning(
        mouse,
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        word=None,
        group_by='all',
        nan_thresh=0.85,
        score_threshold=0.8,
        rank_num=18,
        verbose=True):
    """
    Create a pandas dataframe of trial factor tuning for one
    mouse. Only looks at initial learning stage. I repeat, this is
    only calculated on initial learning!!!
    """

    # default TCA params to use
    if not word:
        if mouse.mouse == 'OA27':
            word = 'restaurant'
        else:
            word = 'whale'
        if verbose:
            print('Creating dataframe for ' + mouse.mouse + '-' + word)

    ms = deepcopy(mouse)
    mouse = mouse.mouse
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
    load_kwargs = {'mouse': mouse,
                   'method': method,
                   'cs': cs,
                   'warp': warp,
                   'word': word,
                   'group_by': group_by,
                   'nan_thresh': nan_thresh,
                   'score_threshold': score_threshold,
                   'rank': rank_num}
    tensor, _, _ = load.groupday_tca_model(**load_kwargs)
    meta = load.groupday_tca_meta(**load_kwargs)

    # add in continuous dprime
    dp = pool.calc.psytrack.dprime(ms)
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
    cs_to_check = ['plus', 'minus', 'neutral']
    ori_vec, cond_vec, comp_vec = [], [], []
    df_data = {}
    mean_response_mat = np.zeros((rank_num, 3))
    for c, ori in enumerate(ori_to_check):
        for rank in [rank_num]:
            data = {}
            for i in range(rank):
                fac = tensor.results[rank][iteration].factors[2][:,i]
                data['factor_' + str(i+1)] = fac
            fac_df = pd.DataFrame(data=data, index=meta1.index)

            # loop over single oris
            psy_fac = pd.concat([psy1, fac_df], axis=1).drop(columns='orientation')
            ori_bool = (meta1['orientation'] == ori)  & (meta1['learning_state'] == 'learning')  # only look during initial learning
            single_psy = psy_fac.loc[ori_bool]

            # check the condition for this ori
            single_meta = meta1.loc[ori_bool]
            cond = single_meta['condition'].unique()
            if len(cond) > 1:
                multi_ori = []
                for ics in cs_to_check:
                    multi_ori.append(ics in cond)
                assert np.sum(multi_ori) == 1
                cond = cs_to_check[np.where(multi_ori)[0][0]]
            else:
                # assert len(cond) == 1
                cond = cond[0]

            # get means for each factor for each type of trial history
            for i in range(rank):
                single_factor = single_psy['factor_' + str(i+1)].values
                mean_response = np.nanmean(single_factor)
                mean_response_mat[i, c] = mean_response

        # save mean per ori
        df_data['mean_' + str(ori) + '_response'] = mean_response_mat[:, c]

    # get the mean response of the whole trial factor for across learning
    total_bool = (meta1['learning_state'] == 'learning')
    total_df = psy_fac.loc[total_bool]
    total_mean_list = []
    data = {}
    for i in range(rank):
        single_factor = total_df['factor_' + str(i+1)].values
        total_mean_list.append(np.nanmean(single_factor))
    df_data['mean_total_response'] = total_mean_list

    # bootstrap to check for significant tuning compared to mean response
    boot_num = 1000
    boot_means_per_comp = np.zeros((rank_num, boot_num))
    for bi in range(boot_num):
        for ri in range(rank):
            single_factor = total_df['factor_' + str(i+1)].values
            rand_samp = np.random.choice(
                single_factor, size=int(np.round(len(single_factor)/3)),
                replace=False)
            boot_means_per_comp[ri, bi] = np.nanmean(rand_samp)

    # test tuning
    bonferonni_correction = len(cs_to_check)
    tuning = []
    for i in range(rank_num):
        a = np.sum(boot_means_per_comp[i, :] >= mean_response_mat[i, 0])/1000 < 0.05/bonferonni_correction
        b = np.sum(boot_means_per_comp[i, :] >= mean_response_mat[i, 1])/1000 < 0.05/bonferonni_correction
        c = np.sum(boot_means_per_comp[i, :] >= mean_response_mat[i, 2])/1000 < 0.05/bonferonni_correction
        d = np.where([a, b, c])[0]

        if len(d) > 1 or len(d) == 0:
            tuning.append('broad')
        else:
            tuning.append(str(ori_to_check[d[0]]))

    # get tuning in terms of CS
    tuning_cs = []
    learning_meta = meta1.loc[total_bool]
    for ti in tuning:
        if ti == 'broad':
            tuning_cs.append('broad')
        else:
            tuning_cs.append(
                    learning_meta['condition']
                    .loc[learning_meta['orientation'].isin([int(ti)]), :]
                    .unique()[0])

    # save tuning into dict
    df_data['preferred_tuning'] = tuning
    df_data['preferred_tuning_cs'] = tuning_cs

    # make final df
    index = pd.MultiIndex.from_arrays([
                [mouse]*rank_num,
                list(np.arange(1, rank_num + 1))
                ],
                names=['mouse', 'component'])

    # make master dataframe
    tuning_df = pd.DataFrame(df_data, index=index)

    return tuning_df

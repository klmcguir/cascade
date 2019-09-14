"""
Functions for plotting trial history effects using modeling from the
Pillow lab and tensortools TCA results.
"""
import flow
import pool
import pandas as pd
import numpy as np
import os
from copy import deepcopy
from . import load, utils


def groupmouse_th_index_byday(
        mice=['OA27', 'OA26', 'OA67', 'VF226', 'CC175', 'OA32', 'OA34', 'OA36'],
        words=None,
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        group_by='all',
        nan_thresh=0.85,
        score_threshold=0.8,
        category_thresh=8,
        rank_num=18,
        cont_dprime=False,
        daymean=True,
        stage=None,
        verbose=True):
    """
    Create a pandas dataframe of trial history modulation indices across all
    mice. Calculated as a mean per day. Only use days above a reasonable
    number of trials in each category.
    """

    # ensure that 'words=None' allows defaults to run in th_index_dataframe
    if not words:
        words = [words]*len(mice)

    # get all single mouse dataframes
    df_list = []
    for m, w in zip(mice, words):
        th_df = th_index_dataframe_byday(
                    m,
                    word=w,
                    trace_type=trace_type,
                    method=method,
                    cs=cs,
                    warp=warp,
                    nan_thresh=nan_thresh,
                    score_threshold=score_threshold,
                    category_thresh=category_thresh,
                    rank_num=rank_num,
                    # stage=stage,
                    group_by=group_by,
                    # cont_dprime=cont_dprime,
                    verbose=verbose)

        df_list.append(th_df)
    all_dfs = pd.concat(df_list, axis=0)

    # optionally take the means of indices across days
    if daymean:
        all_dfs = _get_means_across_days(all_dfs)

    return all_dfs


def groupmouse_th_index_dataframe(
        mice=['OA27', 'OA26', 'OA67', 'VF226', 'CC175', 'OA32', 'OA34', 'OA36'],
        words=None,
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        group_by='all',
        nan_thresh=0.85,
        score_threshold=0.8,
        rank_num=18,
        cont_dprime=False,
        stage=None,
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
                    trace_type=trace_type,
                    method=method,
                    cs=cs,
                    warp=warp,
                    nan_thresh=nan_thresh,
                    score_threshold=score_threshold,
                    rank_num=rank_num,
                    stage=stage,
                    group_by=group_by,
                    cont_dprime=cont_dprime,
                    verbose=verbose)
        df_list.append(th_df)
    all_dfs = pd.concat(df_list, axis=0)

    return all_dfs


def groupmouse_th_index_log2_dataframe(
        mice=['OA27', 'OA26', 'OA67', 'VF226', 'CC175', 'OA32', 'OA34', 'OA36'],
        words=None,
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        group_by='all',
        nan_thresh=0.85,
        score_threshold=0.8,
        rank_num=18,
        verbose=True):
    """
    Create a pandas dataframe of trial history modulation indices across all
    mice. Use a ramp index --> log2(mean(late trials)/mean(early trials))
    """

    # ensure that 'words=None' allows defaults to run in th_index_dataframe
    if not words:
        words = [words]*len(mice)

    # get all single mouse dataframes
    df_list = []
    for m, w in zip(mice, words):
        th_df = th_index_log2_dataframe(
                    m,
                    word=w,
                    trace_type=trace_type,
                    method=method,
                    cs=cs,
                    warp=warp,
                    nan_thresh=nan_thresh,
                    score_threshold=score_threshold,
                    rank_num=rank_num,
                    group_by=group_by,
                    verbose=verbose)
        df_list.append(th_df)
    all_dfs = pd.concat(df_list, axis=0)

    return all_dfs


def groupmouse_th_tuning_dataframe(
        mice=['OA27', 'OA26', 'OA67', 'VF226', 'CC175', 'OA32', 'OA34', 'OA36'],
        words=None,
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        group_by='all',
        nan_thresh=0.85,
        score_threshold=0.8,
        rank_num=18,
        verbose=True):
    """
    Create a pandas dataframe of component tuning across all mice.
    """

    # ensure that 'words=None' allows defaults to run in th_index_dataframe
    if not words:
        words = [words]*len(mice)

    # get all single mouse dataframes
    df_list = []
    for m, w in zip(mice, words):
        th_df = th_tuning_dataframe(
                    m,
                    word=w,
                    trace_type=trace_type,
                    method=method,
                    cs=cs,
                    warp=warp,
                    nan_thresh=nan_thresh,
                    score_threshold=score_threshold,
                    rank_num=rank_num,
                    group_by=group_by,
                    verbose=verbose)
        df_list.append(th_df)
    all_dfs = pd.concat(df_list, axis=0)

    return all_dfs


def th_index_dataframe(
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
        cont_dprime=False,
        stage=None,
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
            word = 'obligations'  # should be updated to 'obligations'
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
    binary_cat = ['ori_0', 'ori_135', 'ori_270', 'prev_reward', 'prev_punish']
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

    # add in continuous dprime so psytracker data frame
    dp = pool.calc.psytrack.dprime(flow.Mouse(mouse))
    dfr['dprime'] = dp

    # add in non continuous dprime to meta dataframe
    meta = utils.add_dprime_to_meta(meta)

    # filter out blank trials
    psy_df = dfr.loc[(dfr['orientation'] >= 0), :]

    # check that all runs have matched trial orientations
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

    # check which stim
    plus_ori = meta1.iloc[(meta1['condition'].values == 'plus') &
                          (meta1['learning_state'].values == 'learning'), :]
    plus_ori = plus_ori['orientation'].unique()[0]
    if verbose:
        print(mouse + ': Plus orientation is: ' + str(plus_ori))

    # filter on stage if you are going to
    if stage:
        if stage == 'high_dp':
            if cont_dprime:
                stage_bool = psy1['dprime'] >= 2
            else:
                stage_bool = meta1['dprime'] >= 2
        elif stage == 'low_dp':
            if cont_dprime:
                stage_bool = psy1['dprime'] < 2
            else:
                stage_bool = meta1['dprime'] < 2
        # meta1 = meta1.loc[stage_bool, :]
        # psy1 = psy1.loc[stage_bool, :]
        stage_tag = stage
        if verbose:
            print(mouse + ': Stage is set: ' + stage)

    iteration = 0
    ori_to_check = [0, 135, 270]
    ori_vec, cond_vec, comp_vec = [], [], []
    trial_history = {}
    trial_hist_mod = np.zeros((rank_num*len(ori_to_check), 5))
    for c, ori in enumerate(ori_to_check):
    # for rank in tensor.results:
        for rank in [rank_num]:
            data = {}
            for i in range(rank):
                fac = tensor.results[rank][iteration].factors[2][:,i]
                # if stage:
                #     fac = fac[stage_bool.values]
                data['factor_' + str(i+1)] = fac
            fac_df = pd.DataFrame(data=data, index=meta1.index)

            # loop over single oris
            psy_fac = pd.concat([psy1, fac_df], axis=1).drop(columns='orientation')
            # only look during initial learning
            ori_bool = ((meta1['orientation'] == ori) &
                        (meta1['learning_state'] == 'learning'))
            single_psy = psy_fac.loc[ori_bool]

            # check the condition for this ori
            single_meta = meta1.loc[ori_bool]
            cond = single_meta['condition'].unique()
            if len(cond) > 1:
                cs_to_check = ['plus', 'minus', 'neutral']
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

                # save an unindexed unfiltered copy if you are using staging
                if stage and i == range(rank)[0]:
                    save_psy = deepcopy(single_psy)
                    save_meta = deepcopy(single_meta)
                elif stage:
                    single_psy = deepcopy(save_psy)
                    single_meta = deepcopy(save_meta)

                single_factor = single_psy['factor_' + str(i+1)].values

                # learning index
                if cont_dprime:
                    # continuous smooth dprime from pillow
                    high_dp = np.nanmean(
                        single_factor[single_psy['dprime'] >= 2])
                    low_dp = np.nanmean(
                        single_factor[single_psy['dprime'] < 2])
                else:
                    # daily dprime
                    high_dp = np.nanmean(
                        single_factor[single_meta['dprime'] >= 2])
                    low_dp = np.nanmean(
                        single_factor[single_meta['dprime'] < 2])
                learning_idx = (high_dp - low_dp)/np.nanmean(single_factor)

                # if you are filtering on a stage calculate learning index first
                if stage:
                    stag_ori_bool = stage_bool[ori_bool]
                    single_factor = single_factor[stag_ori_bool.values]
                    single_psy = single_psy.loc[stag_ori_bool]
                    single_meta = single_meta.loc[stag_ori_bool]

                # Trial history calculated as in Ramesh and Burgess 2018
                prev_same = np.nanmean(
                    single_factor[
                        single_psy['ori_' + str(ori) + '_th_prev'] == 1])
                prev_diff = np.nanmean(
                    single_factor[
                        single_psy['ori_' + str(ori) + '_th_prev'] == 0])
                sensory_history_2018 = (prev_diff - prev_same)/np.nanmean(single_factor)

                # ori_X_th_prev is the one-back set of orientations. They
                # define trials that were preceded by a given stimulus X.
                # Avoid trials that were preceded by reward or punishment.
                prev_same = np.nanmean(
                    single_factor[
                        (single_psy['ori_' + str(ori) + '_th_prev'] == 1) &
                        (single_psy['prev_reward_th'] == 0) &
                        (single_psy['prev_punish_th'] == 0)
                        ])
                prev_diff = np.nanmean(
                    single_factor[
                        (single_psy['ori_' + str(ori) + '_th_prev'] == 0) &
                        (single_psy['prev_reward_th'] == 0) &
                        (single_psy['prev_punish_th'] == 0)
                        ])
                sensory_history = (prev_diff - prev_same)/(prev_diff + prev_same)

                # previously rewarded trials
                # only make the comparison between trials preceded by FC trials
                prev_rew = np.nanmean(
                    single_factor[
                        (single_psy['prev_reward_th'] == 1) &
                        (single_psy['ori_' + str(plus_ori) + '_th_prev'] == 1)
                        ])
                prev_unrew = np.nanmean(
                    single_factor[
                        (single_psy['prev_reward_th'] == 0) &
                        (single_psy['prev_punish_th'] == 0) &
                        (single_psy['ori_' + str(plus_ori) + '_th_prev'] == 1)
                        ])
                # reward_history = (prev_unrew - prev_rew)/np.nanmean(single_factor)
                reward_history = (prev_unrew - prev_rew)/(prev_unrew + prev_rew)

                trial_hist_mod[i + (rank*c), 0] = sensory_history_2018
                trial_hist_mod[i + (rank*c), 1] = sensory_history
                trial_hist_mod[i + (rank*c), 2] = reward_history
                trial_hist_mod[i + (rank*c), 3] = reward_history - sensory_history
                trial_hist_mod[i + (rank*c), 4] = learning_idx

        ori_vec.extend([ori]*rank_num)
        cond_vec.extend([cond]*rank_num)
        comp_vec.extend(list(np.arange(1, rank_num + 1)))

    trial_history['sensory_history_2018'] = trial_hist_mod[:, 0]
    trial_history['sensory_history'] = trial_hist_mod[:, 1]
    trial_history['reward_history'] = trial_hist_mod[:, 2]
    trial_history['diff_reward_sensory'] = trial_hist_mod[:, 3]
    trial_history['learning_index'] = trial_hist_mod[:, 4]

    # create your index out of relevant variables
    index = pd.MultiIndex.from_arrays([
                [mouse]*(rank_num*len(ori_to_check)),
                ori_vec,
                cond_vec,
                comp_vec
                ],
                names=['mouse', 'orientation', 'condition', 'component'])

    # make master dataframe
    th_df = pd.DataFrame(trial_history, index=index)

    return th_df


def th_index_log2_dataframe(
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
        cont_dprime=False,
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
            word = 'obligations'  # should be updated to 'obligations'
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
    binary_cat = ['ori_0', 'ori_135', 'ori_270', 'prev_reward', 'prev_punish']
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

    # add in continuous dprime so psytracker data frame
    dp = pool.calc.psytrack.dprime(flow.Mouse(mouse))
    dfr['dprime'] = dp

    # add in non continuous dprime to meta dataframe
    meta = utils.add_dprime_to_meta(meta)

    # filter out blank trials
    psy_df = dfr.loc[(dfr['orientation'] >= 0), :]

    # check that all runs have matched trial orientations
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

    # check which stim
    plus_ori = meta1.iloc[(meta1['condition'].values == 'plus') &
                          (meta1['learning_state'].values == 'learning'), :]
    plus_ori = plus_ori['orientation'].unique()[0]
    if verbose:
        print(mouse + ': Plus orientation is: ' + str(plus_ori))

    iteration = 0
    ori_to_check = [0, 135, 270]
    ori_vec, cond_vec, comp_vec = [], [], []
    trial_history = {}
    trial_hist_mod = np.zeros((rank_num*len(ori_to_check), 5))
    for c, ori in enumerate(ori_to_check):
    # for rank in tensor.results:
        for rank in [rank_num]:
            data = {}
            for i in range(rank):
                fac = tensor.results[rank][iteration].factors[2][:,i]
                data['factor_' + str(i+1)] = fac
            fac_df = pd.DataFrame(data=data, index=meta1.index)

            # loop over single oris
            psy_fac = pd.concat([psy1, fac_df], axis=1).drop(columns='orientation')
            # only look during initial learning
            ori_bool = ((meta1['orientation'] == ori) &
                        (meta1['learning_state'] == 'learning'))
            single_psy = psy_fac.loc[ori_bool]

            # check the condition for this ori
            single_meta = meta1.loc[ori_bool]
            cond = single_meta['condition'].unique()
            if len(cond) > 1:
                cs_to_check = ['plus', 'minus', 'neutral']
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

                # Trial history calculated as in Ramesh and Burgess 2018
                prev_same = np.nanmean(
                    single_factor[
                        single_psy['ori_' + str(ori) + '_th_prev'] == 1])
                prev_diff = np.nanmean(
                    single_factor[
                        single_psy['ori_' + str(ori) + '_th_prev'] == 0])
                sensory_history_2018 = np.log2(prev_diff/prev_same)

                # ori_X_th_prev is the one-back set of orientations. They
                # define trials that were preceded by a given stimulus X.
                # Avoid trials that were preceded by reward or punishment.
                prev_same = np.nanmean(
                    single_factor[
                        (single_psy['ori_' + str(ori) + '_th_prev'] == 1) &
                        (single_psy['prev_reward_th'] == 0) &
                        (single_psy['prev_punish_th'] == 0)
                        ])
                prev_diff = np.nanmean(
                    single_factor[
                        (single_psy['ori_' + str(ori) + '_th_prev'] == 0) &
                        (single_psy['prev_reward_th'] == 0) &
                        (single_psy['prev_punish_th'] == 0)
                        ])
                sensory_history = np.log2(prev_diff/prev_same)

                # previously rewarded trials
                # only make the comparison between trials preceded by FC trials
                prev_rew = np.nanmean(
                    single_factor[
                        (single_psy['prev_reward_th'] == 1) &
                        (single_psy['ori_' + str(plus_ori) + '_th_prev'] == 1)
                        ])
                prev_unrew = np.nanmean(
                    single_factor[
                        (single_psy['prev_reward_th'] == 0) &
                        (single_psy['prev_punish_th'] == 0) &
                        (single_psy['ori_' + str(plus_ori) + '_th_prev'] == 1)
                        ])
                # reward_history = (prev_unrew - prev_rew)/np.nanmean(single_factor)
                reward_history = np.log2(prev_unrew/prev_rew)

                if cont_dprime:
                    # continuous smooth dprime from pillow
                    high_dp = np.nanmean(
                        single_factor[single_psy['dprime'] >= 2])
                    low_dp = np.nanmean(
                        single_factor[single_psy['dprime'] < 2])
                else:
                    # daily dprime
                    high_dp = np.nanmean(
                        single_factor[single_meta['dprime'] >= 2])
                    low_dp = np.nanmean(
                        single_factor[single_meta['dprime'] < 2])
                learning_idx = np.log2(high_dp/low_dp)

                trial_hist_mod[i + (rank*c), 1] = sensory_history_2018
                trial_hist_mod[i + (rank*c), 1] = sensory_history
                trial_hist_mod[i + (rank*c), 2] = reward_history
                trial_hist_mod[i + (rank*c), 3] = reward_history - sensory_history
                trial_hist_mod[i + (rank*c), 4] = learning_idx

        ori_vec.extend([ori]*rank_num)
        cond_vec.extend([cond]*rank_num)
        comp_vec.extend(list(np.arange(1, rank_num + 1)))

    trial_history['sensory_history_2018_log2'] = trial_hist_mod[:, 0]
    trial_history['sensory_history_log2'] = trial_hist_mod[:, 1]
    trial_history['reward_history_log2'] = trial_hist_mod[:, 2]
    trial_history['diff_reward_sensory_log2'] = trial_hist_mod[:, 3]
    trial_history['learning_index_log2'] = trial_hist_mod[:, 4]

    # create your index out of relevant variables
    index = pd.MultiIndex.from_arrays([
                [mouse]*(rank_num*len(ori_to_check)),
                ori_vec,
                cond_vec,
                comp_vec
                ],
                names=['mouse', 'orientation', 'condition', 'component'])

    # make master dataframe
    th_df = pd.DataFrame(trial_history, index=index)

    return th_df


def th_tuning_dataframe(
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
    Create a pandas dataframe of trial history modulation indices for one
    mouse. Only looks at initial learning stage.
    """

    # default TCA params to use
    if not word:
        if mouse == 'OA27':
            word = 'tray'
        else:
            word = 'obligations'  # should be updated to 'obligations'
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
                cs_to_check = ['plus', 'minus', 'neutral']
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
    tuning = []
    for i in range(rank_num):
        a = np.sum(boot_means_per_comp[i, :] >= mean_response_mat[i, 0])/1000 < 0.05
        b = np.sum(boot_means_per_comp[i, :] >= mean_response_mat[i, 1])/1000 < 0.05
        c = np.sum(boot_means_per_comp[i, :] >= mean_response_mat[i, 2])/1000 < 0.05
        d = np.where([a, b, c])[0]

        if len(d) > 1 or len(d) == 0:
            tuning.append('broad')
        else:
            tuning.append(str(ori_to_check[d[0]]))

    # save tuning into dict
    df_data['preferred_tuning'] = tuning

    # make final df
    index = pd.MultiIndex.from_arrays([
                [mouse]*rank_num,
                list(np.arange(1, rank_num + 1))
                ],
                names=['mouse', 'component'])

    # make master dataframe
    tuning_df = pd.DataFrame(df_data, index=index)

    return tuning_df


def th_index_dataframe_byday(
        mouse,
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        word=None,
        group_by='all',
        nan_thresh=0.85,
        score_threshold=0.8,
        category_thresh=8,
        rank_num=18,
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
            word = 'obligations'  # should be updated to 'obligations'
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
    binary_cat = ['ori_0', 'ori_135', 'ori_270', 'prev_reward', 'prev_punish']
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

    # check which stim
    plus_ori = meta1.iloc[(meta1['condition'].values == 'plus') &
                          (meta1['learning_state'].values == 'learning'), :]
    plus_ori = plus_ori['orientation'].unique()[0]
    if verbose:
        print(mouse + ': Plus orientation is: ' + str(plus_ori))

    # preallocate
    ori_to_check = [0, 135, 270]
    day_rank_df_list = []

    for rank in [rank_num]:
    # for rank in tensor.results:

        # get tensor data for a given rank
        data = {}
        for i in range(rank):
            fac = tensor.results[rank][0].factors[2][:,i]
            data['factor_' + str(i+1)] = fac
        fac_df = pd.DataFrame(data=data, index=meta1.index)

        for cc, d in enumerate(dates):

            # filter on days
            ls_bool = (meta1['learning_state'] == 'learning').values
            psy1_day_bool = psy1.reset_index()['date'].isin([d]).values
            combined_bool = ls_bool & psy1_day_bool
            psy1_day_df = psy1.iloc[combined_bool, :]
            meta1_day_df = meta1.iloc[combined_bool, :]
            fac1_day_df = fac_df.iloc[combined_bool, :]
            psy_fac = pd.concat(
                [psy1_day_df, fac1_day_df], axis=1).drop(columns='orientation')

            # make sure you have trials left!
            if len(meta1_day_df) == 0:
                if verbose:
                    print('Skipping day: ' + str(d) + ', no trials passed ' +
                          'filtering on learning_state & day.')
                continue

            # make sure you have cues of all types left!
            if len(meta1_day_df['orientation'].unique()) < 3:
                if verbose:
                    print('Skipping day: ' + str(d) + ', all cues were not ' +
                          'presented.')
                continue

            # preallocate
            ori_vec, cond_vec, comp_vec = [], [], []
            trial_history = {}
            trial_hist_mod = np.zeros((rank_num*len(ori_to_check), 5))

            # loop over single oris
            for c, ori in enumerate(ori_to_check):

                # only look during initial learning
                ori_bool = meta1_day_df['orientation'] == ori

                # filter down to a single ori
                single_psy = psy_fac.loc[ori_bool]

                # check the condition for this ori
                single_meta = meta1_day_df.loc[ori_bool]
                cond = single_meta['condition'].unique()
                if len(cond) > 1:
                    cs_to_check = ['plus', 'minus', 'neutral']
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
                    bool_curr = single_psy['ori_' + str(ori)] == 1
                    bool_prev = single_psy['ori_' + str(ori) + '_th'] == 1

                    # Trial history calculated as in Ramesh and Burgess 2018
                    prev_same = np.nanmean(
                        single_factor[
                            single_psy['ori_' + str(ori) + '_th_prev'] == 1])
                    prev_diff = np.nanmean(
                        single_factor[
                            single_psy['ori_' + str(ori) + '_th_prev'] == 0])
                    sensory_history_2018 = (prev_diff - prev_same)/(prev_diff + prev_same)
                    if not _check_ntrial(
                        single_psy['ori_' + str(ori) + '_th_prev'] == 1,
                            category_thresh):
                        sensory_history_2018 = np.nan
                    if not _check_ntrial(
                        single_psy['ori_' + str(ori) + '_th_prev'] == 0,
                            category_thresh):
                        sensory_history_2018 = np.nan


                    # ori_X_th_prev is the one-back set of orientations. They
                    # define trials that were preceded by a given stimulus X.
                    # Avoid trials that were preceded by reward or punishment.
                    prev_same = np.nanmean(
                        single_factor[
                            (single_psy['ori_' + str(ori) + '_th_prev'] == 1) &
                            (single_psy['prev_reward_th'] == 0) &
                            (single_psy['prev_punish_th'] == 0)
                            ])
                    prev_diff = np.nanmean(
                        single_factor[
                            (single_psy['ori_' + str(ori) + '_th_prev'] == 0) &
                            (single_psy['prev_reward_th'] == 0) &
                            (single_psy['prev_punish_th'] == 0)
                            ])
                    sensory_history = (prev_diff - prev_same)/(prev_diff + prev_same)
                    if not _check_ntrial(
                            (single_psy['ori_' + str(ori) + '_th_prev'] == 1) &
                            (single_psy['prev_reward_th'] == 0) &
                            (single_psy['prev_punish_th'] == 0),
                            category_thresh):
                        sensory_history = np.nan
                    if not _check_ntrial(
                            (single_psy['ori_' + str(ori) + '_th_prev'] == 0) &
                            (single_psy['prev_reward_th'] == 0) &
                            (single_psy['prev_punish_th'] == 0),
                            category_thresh):
                        sensory_history = np.nan

                    # previously rewarded trials
                    # only make the comparison between trials preceded by FC trials
                    prev_rew = np.nanmean(
                        single_factor[
                            (single_psy['prev_reward_th'] == 1) &
                            (single_psy['ori_' + str(plus_ori) + '_th_prev'] == 1)
                            ])
                    prev_unrew = np.nanmean(
                        single_factor[
                            (single_psy['prev_reward_th'] == 0) &
                            (single_psy['prev_punish_th'] == 0) &
                            (single_psy['ori_' + str(plus_ori) + '_th_prev'] == 1)
                            ])
                    reward_history = (prev_unrew - prev_rew)/(prev_unrew + prev_rew)
                    if not _check_ntrial(
                            (single_psy['prev_reward_th'] == 1) &
                            (single_psy['ori_' + str(plus_ori) + '_th_prev'] == 1),
                            category_thresh):
                        reward_history = np.nan
                    if not _check_ntrial(
                            (single_psy['prev_reward_th'] == 0) &
                            (single_psy['prev_punish_th'] == 0) &
                            (single_psy['ori_' + str(plus_ori) + '_th_prev'] == 1),
                            category_thresh):
                        reward_history = np.nan

                    trial_hist_mod[i + (rank*c), 0] = sensory_history_2018
                    trial_hist_mod[i + (rank*c), 1] = sensory_history
                    trial_hist_mod[i + (rank*c), 2] = reward_history
                    trial_hist_mod[i + (rank*c), 3] = reward_history - sensory_history
                    # trial_hist_mod[i + (rank*c), 4] = learning_idx

                ori_vec.extend([ori]*rank_num)
                cond_vec.extend([cond]*rank_num)
                comp_vec.extend(list(np.arange(1, rank_num + 1)))

            trial_history['sensory_history_2018'] = trial_hist_mod[:, 0]
            trial_history['sensory_history'] = trial_hist_mod[:, 1]
            trial_history['reward_history'] = trial_hist_mod[:, 2]
            trial_history['diff_reward_sensory'] = trial_hist_mod[:, 3]
            # trial_history['learning_index'] = trial_hist_mod[:, 4]

            # create your index out of relevant variables
            index = pd.MultiIndex.from_arrays([
                        [mouse]*(rank_num*len(ori_to_check)),
                        [d]*(rank_num*len(ori_to_check)),
                        [cc+1]*(rank_num*len(ori_to_check)),
                        ori_vec,
                        cond_vec,
                        comp_vec
                        ],
                        names=['mouse', 'date', 'day_number', 'orientation',
                               'condition', 'component'])

            # make master dataframe
            th_df = pd.DataFrame(trial_history, index=index)

            # stick all days and ranks together in a list
            day_rank_df_list.append(th_df)

    # make them all into one large dataframe
    final_df = pd.concat(day_rank_df_list, axis=0)

    return final_df


def _check_ntrial(trial_bool, category_thresh=8):
    """
    Helper function to check the number of trials being used in a calculation
    is above a threshold. Returns True or False as to whether threshold was
    passed.
    """
    return np.sum(trial_bool) >= category_thresh


def _get_means_across_days(df_byday):
    """
    Helper function to take the mean of each type of index across a day.
    """

    # loop over each day, component, and orientation/condition
    df_byday = (df_byday
                .groupby(['mouse', 'orientation', 'condition', 'component'])
                .mean())

    return df_byday

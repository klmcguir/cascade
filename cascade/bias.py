"""Functions for calculating FC bias."""
import pandas as pd
import numpy as np
import flow
import pool
import os
import matplotlib.pyplot as plt
import seaborn as sns
from . import paths
from . import tca
from . import utils


def get_bias(
        mouse,
        trace_type='zscore_day',
        drive_threshold=20,
        drive_type='visual'):
    """
    Returns:
    --------
    FC_bias : ndarray
        bias per cell per day
    """

    # get tensor, metadata, and ids to get things rolling
    ten, met, id = build_tensor(
        mouse, drive_threshold=drive_threshold, trace_type=trace_type)

    # get boolean indexer for period stim is on screen
    stim_window = np.arange(-1, 7, 1/15.5)[0:108]
    stim_window = (stim_window > 0) & (stim_window < 3)

    # get vector and count of dates for the loop
    date_vec = met.reset_index()['date']
    date_num = len(np.unique(date_vec))

    # preallocate tensors
    FC_ten = np.zeros((np.shape(ten)[0], np.shape(ten)[1], date_num))
    QC_ten = np.zeros((np.shape(ten)[0], np.shape(ten)[1], date_num))
    NC_ten = np.zeros((np.shape(ten)[0], np.shape(ten)[1], date_num))

    # preallocate lists
    ls_list = []
    dprime_list = []

    # boolean vecs for each CS
    FC_bool = met['condition'].isin(['plus']).values
    QC_bool = met['condition'].isin(['minus']).values
    NC_bool = met['condition'].isin(['neutral']).values

    # loop through and get mean response of each cell per day for three CSs
    for c, day in enumerate(np.unique(date_vec)):

        # indexing for the day
        day_bool = date_vec.isin([day]).values

        # mean responses
        day_FC = ten[:, :, day_bool & FC_bool]
        day_QC = ten[:, :, day_bool & QC_bool]
        day_NC = ten[:, :, day_bool & NC_bool]
        FC_ten[:, :, c] = np.nanmean(day_FC, axis=2)
        QC_ten[:, :, c] = np.nanmean(day_QC, axis=2)
        NC_ten[:, :, c] = np.nanmean(day_NC, axis=2)

        # learning state
        ls = np.unique(met['learning_state'].values[day_bool])
        ls_list.append(ls)

        # dprime
        dp = pool.calc.performance.dprime(flow.Date(mouse, date=day))
        dprime_list.append(dp)

    FC_mean = np.nanmean(FC_ten[:, stim_window, :], axis=1)
    QC_mean = np.nanmean(QC_ten[:, stim_window, :], axis=1)
    NC_mean = np.nanmean(NC_ten[:, stim_window, :], axis=1)

    # do not consider cells that are negative to all three cues
    neg_bool = (FC_mean < 0) & (QC_mean < 0) & (NC_mean < 0)
    FC_mean[FC_mean < 0] = 0
    QC_mean[QC_mean < 0] = 0
    NC_mean[NC_mean < 0] = 0
    FC_mean[neg_bool] = np.nan
    QC_mean[neg_bool] = np.nan
    NC_mean[neg_bool] = np.nan

    # calculate bias
    FC_bias = FC_mean/(FC_mean + QC_mean + NC_mean)

    return FC_bias, dprime_list, ls_list


def get_mean_response(
        mouse,
        trace_type='zscore_day',
        drive_threshold=20,
        drive_type='visual'):

    # get tensor, metadata, and ids to get things rolling
    ten, met, id = build_tensor(
        mouse, drive_threshold=drive_threshold, trace_type=trace_type)

    # get boolean indexer for period stim is on screen
    stim_window = np.arange(-1, 7, 1/15.5)[0:108]
    stim_window = (stim_window > 0) & (stim_window < 3)

    # get vector and count of dates for the loop
    date_vec = met.reset_index()['date']
    date_num = len(np.unique(date_vec))

    # preallocate tensors
    FC_ten = np.zeros((np.shape(ten)[0], np.shape(ten)[1], date_num))
    QC_ten = np.zeros((np.shape(ten)[0], np.shape(ten)[1], date_num))
    NC_ten = np.zeros((np.shape(ten)[0], np.shape(ten)[1], date_num))

    # preallocate lists
    ls_list = []
    dprime_list = []

    # boolean vecs for each CS
    FC_bool = met['condition'].isin(['plus']).values
    QC_bool = met['condition'].isin(['minus']).values
    NC_bool = met['condition'].isin(['neutral']).values

    # loop through and get mean response of each cell per day for three CSs
    for c, day in enumerate(np.unique(date_vec)):

        # indexing for the day
        day_bool = date_vec.isin([day]).values

        # mean responses
        day_FC = ten[:, :, day_bool & FC_bool]
        day_QC = ten[:, :, day_bool & QC_bool]
        day_NC = ten[:, :, day_bool & NC_bool]
        FC_ten[:, :, c] = np.nanmean(day_FC, axis=2)
        QC_ten[:, :, c] = np.nanmean(day_QC, axis=2)
        NC_ten[:, :, c] = np.nanmean(day_NC, axis=2)

        # learning state
        ls = np.unique(met['learning_state'].values[day_bool])
        ls_list.append(ls)

        # dprime
        dp = pool.calc.performance.dprime(flow.Date(mouse, date=day))
        dprime_list.append(dp)

    FC_mean = np.nanmean(FC_ten[:, stim_window, :], axis=1)
    QC_mean = np.nanmean(QC_ten[:, stim_window, :], axis=1)
    NC_mean = np.nanmean(NC_ten[:, stim_window, :], axis=1)

    # do not consider cells that are negative to all three cues
    neg_bool = (FC_mean < 0) & (QC_mean < 0) & (NC_mean < 0)
    FC_mean[neg_bool] = np.nan
    QC_mean[neg_bool] = np.nan
    NC_mean[neg_bool] = np.nan

    return FC_mean, QC_mean, NC_mean, dprime_list, ls_list


def get_stage_average(FC_bias, dprime_list, ls_list, dprime_thresh=2):
    '''
    Helper function that calculates average bias/response using stages of
    learning and dprime.

    Returns:
    --------
    RNCB_mean1 : list
        mean considering all cells per day independently,
        matches Ramesh & Burgess
    aligned_mean2 : list
        mean considering all cells per day using alignment to first get mean
        bias per cell across a learning stage
    '''

    dprime_list = np.array(dprime_list)
    stage_mean1 = []
    stage_mean2 = []
    for stage in ['naive', 'learning', 'reversal1']:
        if stage == 'naive':
            naive_list = [
                'naive' if 'naive' in s else 'nope' for s in ls_list]
            naive_bool = np.isin(naive_list, stage).flatten()
            naive_bias = FC_bias[:, naive_bool]
            stage_mean1.append(np.nanmean(naive_bias[:]))
            stage_mean2.append(np.nanmean(np.nanmean(naive_bias, axis=1), axis=0))
        elif stage == 'learning':
            learn_list = [
                'learning' if 'learning' in s else 'nope' for s in ls_list]
            low_learn_bool = (np.isin(learn_list, stage).flatten() &
                              (dprime_list < dprime_thresh))
            high_learn_bool = (np.isin(learn_list, stage).flatten() &
                               (dprime_list >= dprime_thresh))
            low_learn_bias = FC_bias[:, low_learn_bool]
            high_learn_bias = FC_bias[:, high_learn_bool]
            stage_mean1.append(np.nanmean(low_learn_bias[:]))
            stage_mean2.append(
                np.nanmean(np.nanmean(low_learn_bias, axis=1), axis=0))
            stage_mean1.append(np.nanmean(high_learn_bias[:]))
            stage_mean2.append(
                np.nanmean(np.nanmean(high_learn_bias, axis=1), axis=0))
            # changing the inner mean to 'np.mean' would force only cells
            # fully aligned across stage to be considered
        elif stage == 'reversal1':
            rev1_list = [
                'reversal1' if 'reversal1' in s else 'nope' for s in ls_list]
            low_rev1_bool = (np.isin(rev1_list, stage).flatten() &
                             (dprime_list < dprime_thresh))
            high_rev1_bool = (np.isin(rev1_list, stage).flatten() &
                              (dprime_list >= dprime_thresh))
            low_rev1_bias = FC_bias[:, low_rev1_bool]
            high_rev1_bias = FC_bias[:, high_rev1_bool]
            stage_mean1.append(np.nanmean(low_rev1_bias[:]))
            stage_mean2.append(
                np.nanmean(np.nanmean(low_rev1_bias, axis=1), axis=0))
            stage_mean1.append(np.nanmean(high_rev1_bias[:]))
            stage_mean2.append(
                np.nanmean(np.nanmean(high_rev1_bias, axis=1), axis=0))

    return stage_mean1, stage_mean2


def build_tensor(
        mouse,
        tags=None,

        # grouping params
        group_by='all',
        up_or_down='up',
        use_dprime=False,
        dprime_threshold=2,

        # tensor params
        trace_type='zscore_day',
        cs='',
        downsample=True,
        start_time=-1,
        end_time=6,
        clean_artifacts=None,
        thresh=20,
        warp=False,
        smooth=True,
        smooth_win=5,
        nan_trial_threshold=None,
        verbose=True,

        # filtering params
        exclude_tags=('disengaged', 'orientation_mapping', 'contrast', 'retinotopy', 'sated'),
        exclude_conds=('blank', 'blank_reward', 'pavlovian'),
        driven=True,
        drive_css=('0', '135', '270'),
        drive_threshold=15,
        drive_type='visual'):

    """
    Perform tensor component analysis (TCA) on data aligned
    across a group of days. Builds one large tensor.

    Algorithms from https://github.com/ahwillia/tensortools.

    Parameters
    -------
    methods, tuple of str
        'cp_als', fits CP Decomposition using Alternating
            Least Squares (ALS).
        'ncp_bcd', fits nonnegative CP Decomposition using
            the Block Coordinate Descent (BCD) Method.
        'ncp_hals', fits nonnegative CP Decomposition using
            the Hierarchical Alternating Least Squares
            (HALS) Method.
        'mcp_als', fits CP Decomposition with missing data using
            Alternating Least Squares (ALS).

    rank, int
        number of components you wish to fit

    replicates, int
        number of initializations/iterations fitting for each rank

    Returns
    -------

    """

    # set grouping parameters
    if group_by.lower() == 'naive':
        tags = 'naive'
        use_dprime = False
        exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                        'retinotopy', 'sated', 'learning_start')

    elif group_by.lower() == 'high_dprime_learning':
        use_dprime = True
        up_or_down = 'up'
        tags = 'learning'
        exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                        'retinotopy', 'sated', 'learning_start',
                        'reversal1_start')

    elif group_by.lower() == 'low_dprime_leanrning':
        use_dprime = True
        up_or_down = 'down'
        tags = 'learning'
        exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                        'retinotopy', 'sated', 'learning_start')

    elif group_by.lower() == 'high_dprime_reversal1':
        use_dprime = True
        up_or_down = 'up'
        tags = 'reversal1'
        exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                        'retinotopy', 'sated', 'reversal2_start')

    elif group_by.lower() == 'low_dprime_reversal1':
        use_dprime = True
        up_or_down = 'down'
        tags = 'reversal1'
        exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                        'retinotopy', 'sated')

    elif group_by.lower() == 'high_dprime_reversal2':
        use_dprime = True
        up_or_down = 'up'
        tags = 'reversal2'
        exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                        'retinotopy', 'sated')

    elif group_by.lower() == 'low_dprime_reversal2':
        use_dprime = True
        up_or_down = 'down'
        tags = 'reversal2'
        exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                        'retinotopy', 'sated')

    elif group_by.lower() == 'naive_vs_high_dprime':
        use_dprime = True
        up_or_down = 'up'
        tags = None
        days = flow.DateSorter.frommeta(mice=[mouse], tags='naive')
        days.extend(flow.DateSorter.frommeta(mice=[mouse], tags='learning'))
        dates = set(days)
        exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                        'retinotopy', 'sated', 'learning_start',
                        'reversal1_start')

    elif group_by.lower() == 'l_vs_r1':  # high dprime
        use_dprime = True
        up_or_down = 'up'
        tags = None
        days = flow.DateSorter.frommeta(mice=[mouse], tags='learning')
        days.extend(flow.DateSorter.frommeta(mice=[mouse], tags='reversal1'))
        dates = set(days)
        exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                        'retinotopy', 'sated', 'learning_start',
                        'reversal1_start')

    elif group_by.lower() == 'all':
        tags = None
        use_dprime = False
        if mouse == 'OA27':
            exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                            'retinotopy', 'sated', 'learning_start',
                            'reversal1_start', 'reversal2_start')
        else:
            exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                            'retinotopy', 'sated')


    else:
        print('Using input parameters without modification by group_by=...')

    # create folder structure and save dir
    pars = {'tags': tags,
            'trace_type': trace_type, 'cs': cs, 'downsample': downsample,
            'start_time': start_time, 'end_time': end_time,
            'clean_artifacts': clean_artifacts, 'thresh': thresh,
            'warp': warp, 'smooth': smooth, 'smooth_win': smooth_win,
            'exclude_tags': exclude_tags, 'exclude_conds': exclude_conds,
            'driven': driven, 'drive_css': drive_css,
            'drive_threshold': drive_threshold}
    group_pars = {'group_by': group_by, 'up_or_down': up_or_down,
                  'use_dprime': use_dprime,
                  'dprime_threshold': dprime_threshold}
    # save_dir = paths.tca_path(mouse, 'group', pars=pars, group_pars=group_pars)

    # get DateSorter object
    if np.isin(group_by.lower(), ['naive_vs_high_dprime', 'l_vs_r1']):
        days = flow.DateSorter(dates=dates)
    else:
        days = flow.DateSorter.frommeta(mice=[mouse], tags=tags)

    # filter DateSorter object if you are filtering on dprime
    if use_dprime:
        dprime = []
        for day1 in days:
            # for comparison with naive make sure dprime keeps naive days
            if np.isin('naive', day1.tags):
                if up_or_down.lower() == 'up':
                    dprime.append(np.inf)
                else:
                    dprime.append(-np.inf)
            else:
                dprime.append(pool.calc.performance.dprime(day1))
        if up_or_down.lower() == 'up':
            days = [d for c, d in enumerate(days) if dprime[c]
                    > dprime_threshold]
        elif up_or_down.lower() == 'down':
            days = [d for c, d in enumerate(days) if dprime[c]
                    <= dprime_threshold]

    # preallocate for looping over a group of days/runs
    meta_list = []
    tensor_list = []
    id_list = []
    for c, day1 in enumerate(days, 0):

        # get cell_ids
        d1_ids = flow.xday._read_crossday_ids(day1.mouse, day1.date)
        d1_ids = np.array([int(s) for s in d1_ids])

        # filter cells based on visual/trial drive across all cs, prevent
        # breaking when only pavs are shown
        if driven:
            good_ids = tca._group_drive_ids(
                days, drive_css, drive_threshold, drive_type=drive_type)
            d1_ids_bool = np.isin(d1_ids, good_ids)
            d1_sorter = np.argsort(d1_ids[d1_ids_bool])
        else:
            d1_ids_bool = np.ones(np.shape(d1_ids)) > 0
            d1_sorter = np.argsort(d1_ids[d1_ids_bool])
        ids = d1_ids[d1_ids_bool][d1_sorter]

        # TODO add in additional filter for being able to check for quality of xday alignment

        # get all runs for both days
        d1_runs = day1.runs()

        # filter for only runs without certain tags
        d1_runs = [run for run in d1_runs if not
                   any(np.isin(run.tags, exclude_tags))]

        # build tensors for all correct runs and trials after filtering
        if d1_runs:
            d1_tensor_list = []
            d1_meta = []
            for run in d1_runs:
                t2p = run.trace2p()
                # trigger all trials around stimulus onsets
                run_traces = utils.getcstraces(
                    run, cs=cs, trace_type=trace_type,
                    start_time=start_time, end_time=end_time,
                    downsample=True, clean_artifacts=clean_artifacts,
                    thresh=thresh, warp=warp, smooth=smooth,
                    smooth_win=smooth_win)
                # filter and sort
                run_traces = run_traces[d1_ids_bool, :, :][d1_sorter, :, :]
                # get matched trial metadata/variables
                dfr = tca._trialmetafromrun(run)
                # subselect metadata if you are only running certain cs
                if cs != '':
                    if cs == 'plus' or cs == 'minus' or cs == 'neutral':
                        dfr = dfr.loc[(dfr['condition'].isin([cs])), :]
                    elif cs == '0' or cs == '135' or cs == '270':
                        dfr = dfr.loc[(dfr['orientation'].isin([cs])), :]
                    else:
                        print('ERROR: cs called - "' + cs + '" - is not\
                              a valid option.')

                # subselect metadata to remove certain conditions
                if len(exclude_conds) > 0:
                    run_traces = run_traces[:, :, (~dfr['condition'].isin(exclude_conds))]
                    dfr = dfr.loc[(~dfr['condition'].isin(exclude_conds)), :]

                # drop trials with nans and add to lists
                keep = np.sum(np.sum(np.isnan(run_traces), axis=0,
                              keepdims=True),
                              axis=1, keepdims=True).flatten() == 0
                dfr = dfr.iloc[keep, :]
                d1_tensor_list.append(run_traces[:, :, keep])
                d1_meta.append(dfr)

            # concatenate matched cells across trials 3rd dim (aka, 2)
            tensor = np.concatenate(d1_tensor_list, axis=2)

            # concatenate all trial metadata in pd dataframe
            meta = pd.concat(d1_meta, axis=0)

            meta_list.append(meta)
            tensor_list.append(tensor)
            id_list.append(ids)

    # get total trial number across all days/runs
    meta = pd.concat(meta_list, axis=0)
    trial_num = len(meta.reset_index()['trial_idx'])

    # get union of ids. Use these for indexing and splicing tensors together
    id_union = np.unique(np.concatenate(id_list, axis=0))
    cell_num = len(id_union)

    # build a single large tensor leaving zeros where cell is not found
    trial_start = 0
    trial_end = 0
    group_tensor = np.zeros((cell_num, np.shape(tensor_list[0])[1], trial_num))
    group_tensor[:] = np.nan
    for i in range(len(tensor_list)):
        trial_end += np.shape(tensor_list[i])[2]
        for c, k in enumerate(id_list[i]):
            celln_all_trials = tensor_list[i][c, :, :]
            group_tensor[(id_union == k), :, trial_start:trial_end] = celln_all_trials
        trial_start += np.shape(tensor_list[i])[2]

    # allow for cells with low number of trials to be dropped
    if nan_trial_threshold:
        # update saving tag
        nt_tag = '_nantrial' + str(nan_trial_threshold)
        # remove cells with too many nan trials
        ntrials = np.shape(group_tensor)[2]
        nbadtrials = np.sum(np.isnan(group_tensor[:, 0, :]), 1)
        badtrialratio = nbadtrials/ntrials
        badcell_indexer = badtrialratio < nan_trial_threshold
        group_tensor = group_tensor[badcell_indexer, :, :]
        if verbose:
            print('Removed ' + str(np.sum(~badcell_indexer)) +
                  ' cells from tensor:' + ' badtrialratio < ' +
                  str(nan_trial_threshold))
            print('Kept ' + str(np.sum(badcell_indexer)) +
                  ' cells from tensor:' + ' badtrialratio < ' +
                  str(nan_trial_threshold))
    else:
        nt_tag = ''

    # just so you have a clue how big the tensor is
    if verbose:
        print('Tensor built: tensor shape = ' + str(np.shape(group_tensor)))

    return group_tensor, meta, id_union

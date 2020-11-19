"""Functions for loading tensors filtering on different tags, etc."""

import numpy as np
import flow
import pool
import pandas as pd
import os
from . import utils
from . import paths
from .tca import _trialmetafromrun, _group_ids_score
from .tca import _group_drive_ids, _get_speed_pupil_npil_traces
from .tca import _remove_stimulus_corr, _three_point_temporal_trace


def groupday_tensor(
        mouse,
        tags=None,

        # TCA params
        rank=20,
        method=('ncp_hals',),
        replicates=3,
        fit_options=None,
        skip_modes=[],
        negative_modes=[],
        tensor_init='rand',

        # grouping params
        group_by='all3',
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
        smooth_win=6,
        verbose=True,

        # filtering params
        exclude_tags=('disengaged', 'orientation_mapping', 'contrast', 'retinotopy', 'sated'),
        exclude_conds=('blank', 'blank_reward', 'pavlovian'),
        driven=True,
        drive_css=('0', '135', '270'),
        drive_threshold=15,
        nan_trial_threshold=0.85,
        score_threshold=0.8,

        # other params
        update_meta=False,
        three_pt_tf=False,
        remove_stim_corr=False):
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

    elif group_by.lower() == 'learning':
        use_dprime = False
        tags = 'learning'
        exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                        'retinotopy', 'sated', 'learning_start',
                        'reversal1_start')

    elif group_by.lower() == 'naive_and_learning':
        use_dprime = False
        tags = ['naive', 'learning']
        days = flow.DateSorter.frommeta(
            mice=[mouse], tags='naive', exclude_tags=['bad'])
        days.extend(
            flow.DateSorter.frommeta(
                mice=[mouse], tags='learning', exclude_tags=['bad']))
        dates = [s.date for s in days]
        exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                        'retinotopy', 'sated', 'learning_start',
                        'reversal1_start')

    elif group_by.lower() == 'high_dprime_learning':
        use_dprime = True
        up_or_down = 'up'
        tags = 'learning'
        exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                        'retinotopy', 'sated', 'learning_start',
                        'reversal1_start')

    elif group_by.lower() == 'low_dprime_learning':
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
        days = flow.DateSorter.frommeta(
            mice=[mouse], tags='naive', exclude_tags=['bad'])
        days.extend(
            flow.DateSorter.frommeta(
                mice=[mouse], tags='learning', exclude_tags=['bad']))
        dates = [s.date for s in days]
        exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                        'retinotopy', 'sated', 'learning_start',
                        'reversal1_start')

    elif group_by.lower() == 'l_vs_r1':  # high dprime
        use_dprime = True
        up_or_down = 'up'
        tags = None
        days = flow.DateSorter.frommeta(
            mice=[mouse], tags='learning', exclude_tags=['bad'])
        days.extend(
            flow.DateSorter.frommeta(
                mice=[mouse], tags='reversal1', exclude_tags=['bad']))
        dates = [s.date for s in days]
        exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                        'retinotopy', 'sated', 'learning_start',
                        'reversal1_start')

    elif group_by.lower() == 'all3':
        tags = None
        use_dprime = False
        if mouse == 'OA27':
            exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                            'retinotopy', 'sated', 'learning_start',
                            'reversal1_start', 'reversal2_start')
        else:
            exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                            'retinotopy', 'sated')

    elif group_by.lower() == 'all3':
        tags = None
        use_dprime = False
        if mouse == 'OA27':
            exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                            'retinotopy', 'sated', 'learning_start',
                            'reversal1_start', 'reversal2_start', 'reversal2')
        else:
            exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                            'retinotopy', 'sated', 'reversal2_start', 'reversal2')


    elif group_by.lower() == 'all100':  # first 100 trials of every day
        tags = None
        use_dprime = False
        if mouse == 'OA27':
            exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                            'retinotopy', 'sated', 'learning_start',
                            'reversal2_start', 'reversal2')
        else:
            exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                            'retinotopy', 'sated', 'reversal2_start', 'reversal2')

    elif group_by.lower() == 'all3':
        tags = None
        use_dprime = False
        if mouse == 'OA27':
            exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                            'retinotopy', 'sated', 'learning_start',
                            'reversal2_start', 'reversal2')
        else:
            exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                            'retinotopy', 'sated', 'reversal2_start', 'reversal2')


    else:
        print('Using input parameters without modification by group_by=...')

    # create folder structure and save dir
    if fit_options is None:
        fit_options = {'tol': 0.0001, 'max_iter': 500, 'verbose': False}
    pars = {'tags': tags, 'rank': rank, 'method': method,
            'replicates': replicates, 'fit_options': fit_options,
            'trace_type': trace_type, 'cs': cs, 'downsample': downsample,
            'start_time': start_time, 'end_time': end_time,
            'clean_artifacts': clean_artifacts, 'thresh': thresh,
            'warp': warp, 'smooth': smooth, 'smooth_win': smooth_win,
            'exclude_tags': exclude_tags, 'exclude_conds': exclude_conds,
            'driven': driven, 'drive_css': drive_css,
            'drive_threshold': drive_threshold}
    if three_pt_tf:
        pars['three_pt_trace'] = True
    if remove_stim_corr:
        pars['removed_stim_corr'] = True
    if len(negative_modes) > 0:
        pars['negative_modes'] = negative_modes,
    group_pars = {'group_by': group_by, 'up_or_down': up_or_down,
                  'use_dprime': use_dprime,
                  'dprime_threshold': dprime_threshold}
    save_dir = paths.tca_path(mouse, 'group', pars=pars, group_pars=group_pars)

    # get DateSorter object
    if np.isin(group_by.lower(),
               ['naive_vs_high_dprime', 'l_vs_r1', 'naive_and_learning']):
        days = flow.DateSorter.frommeta(
            mice=[mouse], dates=dates, exclude_tags=['bad'])
    else:
        days = flow.DateSorter.frommeta(
            mice=[mouse], tags=tags, exclude_tags=['bad'])

    # only include days with xday alignment
    days = [s for s in days if 'xday' in s.tags]

    # add monitor condition to exclusions
    exclude_conds += ('monitor',)

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
    bhv_list = []
    id_list = []
    for c, day1 in enumerate(days, 0):

        # get cell_ids
        d1_ids = flow.xday._read_crossday_ids(day1.mouse, day1.date)
        # skip empty if there is no crossday alignment file
        if len(d1_ids) == 0:
            continue
        d1_ids = np.array([int(s) for s in d1_ids])

        # filter cells based on visual/trial drive across all cs, prevent
        # breaking when only pavs are shown
        if driven:
            good_ids = _group_drive_ids(days, drive_css, drive_threshold)
            # filter for being able to check for quality of xday alignment
            if score_threshold > 0:
                orig_num_ids = len(good_ids)
                highscore_ids = _group_ids_score(days, score_threshold)
                good_ids = np.intersect1d(good_ids, highscore_ids)
                if verbose and c == 0:
                    print('Cell score threshold ' + str(score_threshold) + ':'
                          + ' ' + str(len(highscore_ids)) + ' above threshold:'
                          + ' good_ids updated to ' + str(len(good_ids)) + '/'
                          + str(orig_num_ids) + ' cells.')
                # update saving tag
                score_tag = '_score0pt' + str(int(score_threshold * 10))
            else:
                score_tag = ''
            d1_ids_bool = np.isin(d1_ids, good_ids)
            d1_sorter = np.argsort(d1_ids[d1_ids_bool])
        else:
            good_ids = d1_ids
            # filter for being able to check for quality of xday alignment
            if score_threshold > 0:
                orig_num_ids = len(good_ids)
                highscore_ids = _group_ids_score(days, score_threshold)
                good_ids = np.intersect1d(good_ids, highscore_ids)
                if verbose and c == 0:
                    print('Cell score thresh ' + str(score_threshold) + ':'
                          + ' ' + str(len(highscore_ids)) + ' above thresh:'
                          + ' good_ids updated to ' + str(len(good_ids)) + '/'
                          + str(orig_num_ids) + ' cells.')
                # update saving tag
                score_tag = '_score0pt' + str(int(score_threshold * 10))
            else:
                score_tag = ''
            d1_ids_bool = np.isin(d1_ids, good_ids)
            d1_sorter = np.argsort(d1_ids[d1_ids_bool])
        ids = d1_ids[d1_ids_bool][d1_sorter]

        # get all runs for both days
        d1_runs = day1.runs(exclude_tags=['bad'], run_types='training')

        # filter for only runs without certain tags
        d1_runs = [run for run in d1_runs if not
        any(np.isin(run.tags, exclude_tags))]

        # build tensors for all correct runs and trials after filtering
        if d1_runs:
            d1_tensor_list = []
            d1_bhv_list = []
            d1_meta = []
            for run in d1_runs:
                t2p = run.trace2p()
                # trigger all trials around stimulus onsets
                run_traces = utils.getcstraces(
                    run, cs=cs, trace_type=trace_type,
                    start_time=start_time, end_time=end_time,
                    downsample=downsample, clean_artifacts=clean_artifacts,
                    thresh=thresh, warp=warp, smooth=smooth,
                    smooth_win=smooth_win, exclude_tags=exclude_tags)
                bhv_traces = _get_speed_pupil_npil_traces(
                    run,
                    cs=cs,
                    start_time=start_time,
                    end_time=end_time,
                    downsample=downsample,
                    cutoff_before_lick_ms=-1)

                # filter and sort
                run_traces = run_traces[d1_ids_bool, :, :][d1_sorter, :, :]
                # get matched trial metadata/variables
                dfr = _trialmetafromrun(run)
                # skip runs with no stimulus presentations
                if len(dfr) == 0:
                    continue
                # skip runs with only one type of stimulus presentation
                ori_to_match = np.unique(dfr['orientation'].values)
                ori_wo_blanks = len(ori_to_match) - np.sum(ori_to_match == -1)
                if cs == '' and ori_wo_blanks <= 2:
                    if verbose:
                        print('Skipping, only {} ori presented: '.format(ori_wo_blanks), run)
                    continue
                # subselect metadata if you are only running certain cs
                if cs != '':
                    if cs == 'plus' or cs == 'minus' or cs == 'neutral':
                        run_traces = run_traces[:, :, (~dfr['condition'].isin([cs]))]
                        bhv_traces = bhv_traces[:, :, (~dfr['condition'].isin([cs]))]
                        dfr = dfr.loc[(dfr['condition'].isin([cs])), :]
                    elif cs == '0' or cs == '135' or cs == '270':
                        run_traces = run_traces[:, :, (~dfr['orientation'].isin([cs]))]
                        bhv_traces = bhv_traces[:, :, (~dfr['orientation'].isin([cs]))]
                        dfr = dfr.loc[(dfr['orientation'].isin([cs])), :]
                    else:
                        print('ERROR: cs called - "' + cs + '" - is not\
                              a valid option.')

                # subselect metadata to remove certain conditions
                if len(exclude_conds) > 0:
                    run_traces = run_traces[:, :, (~dfr['condition'].isin(exclude_conds))]
                    bhv_traces = bhv_traces[:, :, (~dfr['condition'].isin(exclude_conds))]
                    dfr = dfr.loc[(~dfr['condition'].isin(exclude_conds)), :]

                # drop trials with nans and add to lists
                keep = np.sum(np.sum(np.isnan(run_traces), axis=0,
                                     keepdims=True),
                              axis=1, keepdims=True).flatten() == 0
                dfr = dfr.iloc[keep, :]
                d1_tensor_list.append(run_traces[:, :, keep])
                d1_bhv_list.append(bhv_traces[:, :, keep])
                d1_meta.append(dfr)

            # if you did not add any runs for the day, continue
            if len(d1_tensor_list) == 0:
                continue

            # concatenate matched cells across trials 3rd dim (aka, 2)
            tensor = np.concatenate(d1_tensor_list, axis=2)

            # concatenate matched cells across trials 3rd dim (aka, 2)
            bhv_tensor = np.concatenate(d1_bhv_list, axis=2)

            # concatenate all trial metadata in pd dataframe
            meta = pd.concat(d1_meta, axis=0)

            meta_list.append(meta)
            tensor_list.append(tensor)
            bhv_list.append(bhv_tensor)
            id_list.append(ids)

    # get total trial number across all days/runs
    meta = pd.concat(meta_list, axis=0)
    trial_num = len(meta.reset_index()['trial_idx'])

    # get union of ids. Use these for indexing and splicing tensors together
    id_union = np.unique(np.concatenate(id_list, axis=0))
    cell_num = len(id_union)

    # build final behavior trace tensor
    group_bhv_tensor = np.concatenate(bhv_list, axis=2)

    # build a single large tensor leaving nans where cell is not found
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
        badtrialratio = nbadtrials / ntrials
        badcell_indexer = badtrialratio < nan_trial_threshold
        group_tensor = group_tensor[badcell_indexer, :, :]
        id_union = id_union[badcell_indexer]
        if verbose:
            print('Removed ' + str(np.sum(~badcell_indexer)) +
                  ' cells from tensor:' + ' badtrialratio < ' +
                  str(nan_trial_threshold))
            print('Kept ' + str(np.sum(badcell_indexer)) +
                  ' cells from tensor:' + ' badtrialratio < ' +
                  str(nan_trial_threshold))
    else:
        nt_tag = ''

    # optionally remove stimulus correlations for each cell
    if remove_stim_corr:
        group_tensor = _remove_stimulus_corr(group_tensor, meta)

    # optionally use the average across the baseline, stim, and post stim avg
    # to define a three pt temporal trace
    if three_pt_tf:
        group_tensor = _three_point_temporal_trace(group_tensor, meta)

    # just so you have a clue how big the tensor is
    if verbose:
        print('Tensor building finishing: tensor shape = '
              + str(np.shape(group_tensor)))

    # print output so you don't go crazy waiting
    if verbose:
        print(str(day1.mouse) + ': group_by=' + str(group_by) + ': loaded.')

    return group_tensor, id_union, group_bhv_tensor, meta


def load_all_groupday(
        mouse='OA27',
        trace_type='zscore_day',
        method='ncp_hals',
        cs='',
        warp=False,
        word='prints',
        rank=15,
        group_by='all3',
        nan_thresh=0.85,
        score_threshold=0.8,
        full_output=False,
        unsorted=True,
        with_model=True,
        verbose=False):
    """
    Load all existing data from fitting a TCA model.
    """

    # load TCA model
    if with_model:
        model, sorts = groupday_tca_model(
            mouse=mouse,
            trace_type=trace_type,
            method=method,
            cs=cs,
            warp=warp,
            word=word,
            rank=rank,
            group_by=group_by,
            nan_thresh=nan_thresh,
            score_threshold=score_threshold,
            full_output=full_output,
            unsorted=unsorted,
            verbose=verbose)
    else:
        model, sorts = [], []

    # load cell ids
    ids = groupday_tca_ids(
        mouse=mouse,
        trace_type=trace_type,
        method=method,
        cs=cs,
        warp=warp,
        word=word,
        group_by=group_by,
        nan_thresh=nan_thresh,
        score_threshold=score_threshold)

    # load input tensor
    tensor = groupday_tca_input_tensor(
        mouse=mouse,
        trace_type=trace_type,
        method=method,
        cs=cs,
        warp=warp,
        word=word,
        group_by=group_by,
        nan_thresh=nan_thresh,
        score_threshold=score_threshold)

    # load metadata
    meta = groupday_tca_meta(
        mouse=mouse,
        trace_type=trace_type,
        method=method,
        cs=cs,
        warp=warp,
        word=word,
        group_by=group_by,
        nan_thresh=nan_thresh,
        score_threshold=score_threshold)

    # load behavioral traces (i.e., pupil and running)
    bhv = groupday_tca_bhv(
        mouse=mouse,
        trace_type=trace_type,
        method=method,
        cs=cs,
        warp=warp,
        word=word,
        group_by=group_by,
        nan_thresh=nan_thresh,
        score_threshold=score_threshold)

    return model, ids, tensor, meta, bhv, sorts


def singleday_tensor(
        mouse,
        date,
        tags=None,

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
        smooth_win=6,
        verbose=True,

        # filtering params
        exclude_tags=('orientation_mapping', 'contrast', 'retinotopy'),
        exclude_conds=('blank', 'blank_reward', 'pavlovian'),
        driven=True,
        drive_css=('0', '135', '270'),
        drive_threshold=15):
    """
    Build tensor component analysis (TCA) on for a single day.

    Parameters
    ----------

    Returns
    -------

    """

    # create folder structure and save dir
    pars = {'tags': tags,
            'trace_type': trace_type, 'cs': cs, 'downsample': downsample,
            'start_time': start_time, 'end_time': end_time,
            'clean_artifacts': clean_artifacts, 'thresh': thresh,
            'warp': warp, 'smooth': smooth, 'smooth_win': smooth_win,
            'exclude_tags': exclude_tags, 'exclude_conds': exclude_conds,
            'driven': driven, 'drive_css': drive_css,
            'drive_threshold': drive_threshold}
    save_dir = paths.tca_path(mouse, 'single', pars=pars)

    day1 = flow.DateSorter.frommeta(
        mice=[mouse], dates=[date], tags=tags, exclude_tags=['bad'])[0]

    # get cell_ids
    d1_ids = flow.xday._read_crossday_ids(day1.mouse, day1.date)
    d1_ids = np.array([int(s) for s in d1_ids])

    # filter cells based on visual/trial drive across all cs, prevent
    # breaking when only pavs are shown
    if driven:
        d1_drive = []
        for dcs in drive_css:
            try:
                d1_drive.append(pool.calc.driven.trial(day1, dcs))
            except KeyError:
                print(str(day1) + ' requested ' + dcs +
                      ': no match to what was shown (probably pav only).')
        d1_drive = np.max(d1_drive, axis=0)
        # account for rare cases where lost xday ids are final id (making _ids
        # 1 shorter than _drive). Add a fake id to the end and force drive to
        # be false for that id
        if len(d1_drive) > len(d1_ids):
            print('Warning: ' + str(day1) + ': _ids was ' +
                  str(len(d1_drive) - len(d1_ids)) +
                  ' shorter than _drive: added pseudo-id.')
            d1_drive[-1] = 0
            d1_ids = np.concatenate((d1_ids, np.array([-1])))
        d1_ids_bool = np.array(d1_drive) > drive_threshold
        d1_drive_ids = d1_ids[np.array(d1_drive) > drive_threshold]
        d1_sorter = np.argsort(d1_ids[d1_ids_bool])
    else:
        d1_ids_bool = np.ones(np.shape(d1_ids)) > 0
        d1_sorter = np.argsort(d1_ids[d1_ids_bool])
    ids = d1_ids[d1_ids_bool][d1_sorter]

    # get all runs for both days
    d1_runs = day1.runs(exclude_tags=['bad'])

    # filter for only runs without certain tags
    d1_runs = [run for run in d1_runs if not any(np.isin(run.tags, exclude_tags))]

    # build tensors for all correct runs and trials after filtering
    if d1_runs:
        d1_tensor_list = []
        d1_meta = []
        for run in d1_runs:
            t2p = run.trace2p()
            # trigger all trials around stimulus onsets
            run_traces = utils.getcstraces(run, cs=cs, trace_type=trace_type,
                                           start_time=start_time, end_time=end_time,
                                           downsample=True, clean_artifacts=clean_artifacts,
                                           thresh=thresh, warp=warp, smooth=smooth,
                                           smooth_win=smooth_win)
            # filter and sort
            run_traces = run_traces[d1_ids_bool, :, :][d1_sorter, :, :]
            # get matched trial metadata/variables
            dfr = _trialmetafromrun(run)
            # subselect metadata if you are only running certain cs
            if cs != '':
                if cs == 'plus' or cs == 'minus' or cs == 'neutral':
                    dfr = dfr.loc[(dfr['condition'].isin([cs])), :]
                elif cs == '0' or cs == '135' or cs == '270':
                    dfr = dfr.loc[(dfr['orientation'].isin([cs])), :]
                else:
                    print('ERROR: cs called - "' + cs + '" - is not\
                          a valid option.')

            # subselect metadata to remove certain condtions
            if len(exclude_conds) > 0:
                run_traces = run_traces[:, :, (~dfr['condition'].isin(exclude_conds))]
                dfr = dfr.loc[(~dfr['condition'].isin(exclude_conds)), :]

            # drop trials with nans and add to lists
            keep = np.sum(np.sum(np.isnan(run_traces), axis=0, keepdims=True),
                          axis=1, keepdims=True).flatten() == 0
            dfr = dfr.iloc[keep, :]
            d1_tensor_list.append(run_traces[:, :, keep])
            d1_meta.append(dfr)

        # concatenate matched cells across trials 3rd dim (aka, 2)
        tensor = np.concatenate(d1_tensor_list, axis=2)

        # concatenate all trial metadata in pd dataframe
        meta = pd.concat(d1_meta, axis=0)

        # concatenate and save df for the day
        meta_path = os.path.join(save_dir, str(day1.mouse) + '_'
                                 + str(day1.date) + '_df_single_meta.pkl')
        input_tensor_path = os.path.join(save_dir, str(day1.mouse) + '_'
                                         + str(day1.date) + '_single_tensor_'
                                         + str(trace_type) + '.npy')
        input_ids_path = os.path.join(save_dir, str(day1.mouse) + '_'
                                      + str(day1.date) + '_single_ids_'
                                      + str(trace_type) + '.npy')

        # meta.to_pickle(meta_path)
        # np.save(input_tensor_path, tensor)
        # np.save(input_ids_path, ids)

        # print output so you don't go crazy waiting
        if verbose:
            print('Day: ' + str(day1.date) + ': ' + str(day1.mouse) + ': done.')

        return tensor, meta, ids


def groupday_tca_ids(
        mouse='OA27',
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        word='tray',
        group_by='all3',
        nan_thresh=0.85,
        score_threshold=None):
    """
    Load existing TCA ids (absolute cell ids for all aligned cells in tensor).

    Parameters
    ----------

    Returns
    -------
    ids array

    """

    mouse = mouse
    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}
    group_pars = {'group_by': group_by}

    # if cells were removed with too many nan trials
    if nan_thresh:
        load_tag = '_nantrial' + str(nan_thresh)
    else:
        load_tag = ''

    # update saving tag if you used a cell score threshold
    if score_threshold:
        load_tag = '_score0pt' + str(int(score_threshold * 10)) + load_tag

    # load dir
    load_dir = paths.tca_path(
        mouse, 'group', pars=pars, word=word, group_pars=group_pars)
    ids_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + load_tag
                  + '_group_ids_' + str(trace_type) + '.npy')

    # load your data
    ids = np.load(ids_path)

    return ids


def groupday_tca_model(
        mouse='OA27',
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        rank=18,
        word='tray',
        group_by='all3',
        nan_thresh=0.85,
        score_threshold=None,
        train_test_split=0.8,
        cv=False,
        full_output=False,
        unsorted=False,
        verbose=False):
    """
    Load existing tensor component analysis (TCA) model and ids.

    Parameters
    ----------

    Returns
    -------

    """

    mouse = mouse
    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}
    group_pars = {'group_by': group_by}

    # if cells were removed with too many nan trials
    if nan_thresh:
        load_tag = '_nantrial' + str(nan_thresh)
    else:
        load_tag = ''

    # update saving tag if you used a cell score threshold
    if score_threshold:
        load_tag = '_score0pt' + str(int(score_threshold * 10)) + load_tag

    # if train-test split was made
    load_tag_ids = load_tag
    if cv:
        load_tag = load_tag + '_cv' + str(train_test_split)

    # load dir
    load_dir = paths.tca_path(
        mouse, 'group', pars=pars, word=word, group_pars=group_pars)
    tensor_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + load_tag
                  + '_group_decomp_' + str(trace_type) + '.npy')
    ids_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + load_tag_ids
                  + '_group_ids_' + str(trace_type) + '.npy')

    # load your data
    ensemble = np.load(tensor_path, allow_pickle=True)
    ensemble = ensemble.item()
    ids = np.load(ids_path)

    # re-balance your factors ()
    if verbose:
        print('{}: {}: Re-balancing factors.'.format(mouse, word))
    for r in ensemble[method].results:
        for i in range(len(ensemble[method].results[r])):
            ensemble[method].results[r][i].factors.rebalance()

    # force cell factors to be positive at the expense of trial factors
    if verbose:
        print('{}: {}: Re-nonneg-ing cell factors.'.format(mouse, word))
    ensemble = utils.correct_nonneg(ensemble)

    # sort neuron factors by component they belong to most
    sort_ensemble, my_sorts = utils.sortfactors(ensemble[method])

    cell_ids = {}  # keys are rank
    cell_clusters = {}
    itr_num = 0  # use only best iteration of TCA, index 0
    for k in sort_ensemble.results.keys():
        # factors are already sorted, so these will define
        # clusters, no need to sort again
        factors = sort_ensemble.results[k][itr_num].factors[0]
        max_fac = np.argmax(factors, axis=1)
        cell_clusters[k] = max_fac
        cell_ids[k] = ids[my_sorts[k - 1]]

    # Return either output for one rank, or everything for all ranks
    if unsorted:
        return ensemble[method], my_sorts
    else:
        if not full_output:
            return sort_ensemble, cell_ids[rank], cell_clusters[rank]
        else:
            return sort_ensemble, cell_ids, cell_clusters


def groupday_tca_meta(
        mouse='OA27',
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        word='tray',
        group_by='all3',
        nan_thresh=0.85,
        score_threshold=None):
    """
    Load existing tensor component analysis (TCA) metadata.

    Parameters
    ----------

    Returns
    -------

    """

    mouse = mouse
    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}
    group_pars = {'group_by': group_by}

    # if cells were removed with too many nan trials
    if nan_thresh:
        load_tag = '_nantrial' + str(nan_thresh)
    else:
        load_tag = ''

    # update saving tag if you used a cell score threshold
    if score_threshold:
        load_tag = '_score0pt' + str(int(score_threshold * 10)) + load_tag

    # load dir
    load_dir = paths.tca_path(
        mouse, 'group', pars=pars, word=word, group_pars=group_pars)
    meta_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + load_tag
                  + '_df_group_meta.pkl')

    # load your data
    meta = pd.read_pickle(meta_path)
    meta = utils.update_naive_meta(meta)

    return meta


def groupday_tca_input_tensor(
        mouse='OA27',
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        word='tray',
        group_by='all3',
        nan_thresh=0.95,
        score_threshold=0.8,
        train_test_split=0.8,
        cv=False):
    """
    Load existing input tensor from tensor component analysis (TCA).

    Parameters
    ----------

    Returns
    -------

    """

    mouse = mouse
    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}
    group_pars = {'group_by': group_by}

    # if cells were removed with too many nan trials
    if nan_thresh:
        load_tag = '_nantrial' + str(nan_thresh)
    else:
        load_tag = ''

    # update saving tag if you used a cell score threshold
    if score_threshold:
        load_tag = '_score0pt' + str(int(score_threshold * 10)) + load_tag

    # if train-test split was made
    if cv:
        load_tag = load_tag + '_cv' + str(train_test_split)

    # load dir
    load_dir = paths.tca_path(
        mouse, 'group', pars=pars, word=word, group_pars=group_pars)
    input_tensor_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + load_tag
                  + '_group_tensor_' + str(trace_type) + '.npy')

    # load your data
    input_tensor = np.load(input_tensor_path)

    return input_tensor


def groupday_tca_bhv(
        mouse='OA27',
        trace_type='zscore_day',
        method='ncp_hals',
        cs='',
        warp=False,
        word='determined',
        group_by='all3',
        nan_thresh=0.85,
        score_threshold=0.8):
    """
    Load existing behavioral tensor from tensor component analysis (TCA).

    Parameters
    ----------

    Returns
    -------

    """

    mouse = mouse
    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}
    group_pars = {'group_by': group_by}

    # if cells were removed with too many nan trials
    if nan_thresh:
        load_tag = '_nantrial' + str(nan_thresh)
    else:
        load_tag = ''

    # update saving tag if you used a cell score threshold
    if score_threshold:
        load_tag = '_score0pt' + str(int(score_threshold * 10)) + load_tag

    # load dir
    load_dir = paths.tca_path(
        mouse, 'group', pars=pars, word=word, group_pars=group_pars)
    input_tensor_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + load_tag
                  + '_group_bhv_' + str(trace_type) + '.npy')

    # load your data
    input_tensor = np.load(input_tensor_path)

    return input_tensor


def groupday_tca_cv_test_set_tensor(
        mouse='OA27',
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        word='tray',
        group_by='all3',
        nan_thresh=0.95,
        train_test_split=0.8,
        score_threshold=0.8):
    """
    Load existing input tensor from tensor component analysis (TCA).

    Parameters
    ----------

    Returns
    -------

    """

    mouse = mouse
    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}
    group_pars = {'group_by': group_by}

    # if cells were removed with too many nan trials
    if nan_thresh:
        load_tag = '_nantrial' + str(nan_thresh)
    else:
        load_tag = ''

    # update saving tag if you used a cell score threshold
    if score_threshold:
        load_tag = '_score0pt' + str(int(score_threshold * 10)) + load_tag

    # add cv tag always
    load_tag = load_tag + '_cv' + str(train_test_split)

    # load dir
    load_dir = paths.tca_path(
        mouse, 'group', pars=pars, word=word, group_pars=group_pars)
    input_tensor_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + load_tag
                  + '_test_tensor_' + str(trace_type) + '.npy')

    # load your data
    input_tensor = np.load(input_tensor_path)

    return input_tensor

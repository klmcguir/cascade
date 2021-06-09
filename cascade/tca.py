"""Functions for running tensor component analysis (TCA)."""

import tensortools as tt
from tensortools.tensors import KTensor

import numpy as np
import flow
import pool
import pandas as pd
import os
from . import utils
from . import paths
from . import lookups
from copy import deepcopy
from functools import reduce
import warnings

import matplotlib.pyplot as plt


def refit_naive_tempfac_tca_unwrapped(ensemble,
                                      data_dict,
                                      chosen_rank=9,
                                      mod='v4i10_norm_on_noT0',
                                      plot_please=True):
    """
    Refit your unwrapped TCA results (presumably using no naive data) to now include naive 
    data in the fitting of the temporal factors. 

    Note: relies on tensortools HALS, but is essentially a single round of least squares 
    fitting as it is implemented here.

    :param ensemble: dict of tensortools.Ensemble objects
        Fitting results from unwrapped TCA.
    :param data_dict: dict of numpy.ndarrays 
        Input data from unwrapped TCA fitting (must include "allstage" data if you want to fit naive)
    :param chosen rank: int
        Rank of model to refit naive temporal factor.
    :param mod: str
        Model name.
    :param plot_please: boolean
        Plot your newly fit temporal factor overlaid with the original model fitting. 

    """

    # fixed params
    method = 'ncp_hals'
    replicates = 3
    fit_options = {'tol': 0.0001, 'max_iter': 500, 'verbose': False, 'skip_modes': [0, 2]}

    # parse your data to get new input data with naive and fixed cell/tune factors
    existing_mod = ensemble[mod]
    mod_name = (mod + '_allstagesT0norm').replace('_norm', '') if 'norm' in mod else mod + '_allstages'
    new_init_data = data_dict[mod_name]
    new_rand_temp_factor = np.random.rand(new_init_data.shape[1], chosen_rank)
    existing_cell_factor = existing_mod.results[chosen_rank][0].factors[0]
    existing_tune_factor = existing_mod.results[chosen_rank][0].factors[2]

    # run TCA on this small
    init_factors = KTensor((existing_cell_factor, new_rand_temp_factor, existing_tune_factor))
    temp_fit_options = deepcopy(fit_options)
    mask = ~np.isnan(new_init_data)
    temp_fit_options['mask'] = mask
    temp_fit_options['init'] = init_factors
    temp_ensemble = tt.Ensemble(fit_method=method, fit_options=temp_fit_options)
    temp_ensemble.fit(new_init_data, ranks=chosen_rank, replicates=replicates, verbose=False)

    # ensure that nothing changed in your cell factors or tuning factors 
    assert np.array_equal(temp_ensemble.results[chosen_rank][0].factors[0], existing_cell_factor)
    assert np.array_equal(temp_ensemble.results[chosen_rank][0].factors[2], existing_tune_factor)

    if plot_please:
        _, ax = plt.subplots(chosen_rank,
                             1,
                             figsize=(4, chosen_rank*2+2),
                             sharex=True,
                             sharey=True)
        new = temp_ensemble.results[chosen_rank][0].factors[1]
        old = existing_mod.results[chosen_rank][0].factors[1]
        for i in range(chosen_rank):

            # calculate r_squared for your new fit
            correlation_matrix = np.corrcoef(new[47:, i], old[:, i])
            correlation_xy = correlation_matrix[0, 1]
            r_squared = correlation_xy**2

            ax[i].plot(np.arange(new.shape[0]),
                       new[:, i],
                       label=f'fit r_squared={round(r_squared, 4)}',
                       alpha=0.5,
                       color='red')
            ax[i].plot(np.arange(old.shape[0]) + 47,
                       old[:, i],
                       label='original',
                       alpha=0.5,
                       color='blue')
            ax[i].legend(bbox_to_anchor=(1.05, 1.0), loc=2)
            ax[i].set_ylabel(f'                 Component {i+1}',
                             size=16,
                             ha='right',
                             rotation=0)
        ax[0].set_title(f'{mod}, rank {chosen_rank}, refit with T0 Naive\n\n',
                        size=20)

        plt.savefig(paths.analysis_file(
            f'{mod}_refittingNAIVE_rank{chosen_rank}_facs.png',
            f'tca_dfs/TCA_factor_fitting/{mod}'),
                    bbox_inches='tight')
        plt.close('all')

    return temp_ensemble


def groupday_tca(
        mouse,
        tags=None,

        # TCA params
        rank=20,
        method=('ncp_hals', ),
        replicates=3,
        fit_options=None,
        skip_modes=[],
        negative_modes=[],
        tensor_init='rand',

        # grouping params
        group_by='all2',
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
        exclude_tags=('disengaged', 'orientation_mapping', 'contrast',
                      'retinotopy', 'sated'),
        exclude_conds=('blank', 'blank_reward', 'pavlovian'),
        driven=True,
        drive_css=('0', '135', '270'),
        drive_threshold=1.31,
        drive_type='trial',
        nan_trial_threshold=0.85,
        score_threshold=0.8,

        # other params
        update_meta=False,
        three_pt_tf=False,
        remove_stim_corr=False,

        # cross validation params
        cv=False,
        train_test_split=0.8,

        # force fit/save
        force=False):
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
        days = flow.DateSorter.frommeta(mice=[mouse],
                                        tags='naive',
                                        exclude_tags=['bad'])
        days.extend(
            flow.DateSorter.frommeta(mice=[mouse],
                                     tags='learning',
                                     exclude_tags=['bad']))
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
        days = flow.DateSorter.frommeta(mice=[mouse],
                                        tags='naive',
                                        exclude_tags=['bad'])
        days.extend(
            flow.DateSorter.frommeta(mice=[mouse],
                                     tags='learning',
                                     exclude_tags=['bad']))
        dates = [s.date for s in days]
        exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                        'retinotopy', 'sated', 'learning_start',
                        'reversal1_start')

    elif group_by.lower() == 'l_vs_r1':  # high dprime
        use_dprime = True
        up_or_down = 'up'
        tags = None
        days = flow.DateSorter.frommeta(mice=[mouse],
                                        tags='learning',
                                        exclude_tags=['bad'])
        days.extend(
            flow.DateSorter.frommeta(mice=[mouse],
                                     tags='reversal1',
                                     exclude_tags=['bad']))
        dates = [s.date for s in days]
        exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                        'retinotopy', 'sated', 'learning_start',
                        'reversal1_start')

    elif group_by.lower() == 'l_vs_r1_tight':  # focus on few days around rev
        use_dprime = False
        tags = None
        days = flow.DateSorter.frommeta(mice=[mouse],
                                        tags='learning',
                                        exclude_tags=['bad'])
        day_count = len(days) - 4
        dates = [s.date for c, s in enumerate(days) if c >= day_count]
        days = flow.DateSorter.frommeta(mice=[mouse],
                                        tags='reversal1',
                                        exclude_tags=['bad'])
        rev_dates = [s.date for c, s in enumerate(days) if c < 4]
        dates.extend(rev_dates)
        dates = list([int(s) for s in np.unique(dates)])
        exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                        'retinotopy', 'sated', 'learning_start')

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

    elif group_by.lower() == 'all2':
        tags = None
        use_dprime = False
        if mouse == 'OA27':
            exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                            'retinotopy', 'sated', 'learning_start',
                            'reversal1_start', 'reversal2_start', 'reversal2')
        else:
            exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                            'retinotopy', 'sated', 'reversal2_start',
                            'reversal2')

    elif group_by.lower() == 'all100':  #first 100 trials of every day
        tags = None
        use_dprime = False
        if mouse == 'OA27':
            exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                            'retinotopy', 'sated', 'learning_start',
                            'reversal2_start', 'reversal2')
        else:
            exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                            'retinotopy', 'sated', 'reversal2_start',
                            'reversal2')

    elif group_by.lower() == 'all3':
        tags = None
        use_dprime = False
        # if mouse == 'OA27':
        #     exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
        #                     'retinotopy', 'sated', 'learning_start',
        #                     'reversal2_start', 'reversal2')
        # else:
        #     exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
        #                     'retinotopy', 'sated', 'reversal2_start',
        #                     'reversal2')
        if mouse == 'OA27':
            exclude_tags = ('orientation_mapping', 'contrast',
                            'retinotopy', 'learning_start',
                            'reversal2_start', 'reversal2')
        else:
            exclude_tags = ('orientation_mapping', 'contrast',
                            'retinotopy', 'reversal2_start',
                            'reversal2')

    elif group_by.lower() == 'all_inc':
        tags = None
        use_dprime = False
        exclude_tags = ('orientation_mapping', 'contrast', 'retinotopy',
                        'reversal2_start', 'reversal2')

    else:
        print('Using input parameters without modification by group_by=...')

    # create folder structure and save dir
    if fit_options is None:
        fit_options = {'tol': 0.0001, 'max_iter': 500, 'verbose': False}
    pars = {
        'tags': tags,
        'rank': rank,
        'method': method,
        'replicates': replicates,
        'fit_options': fit_options,
        'trace_type': trace_type,
        'cs': cs,
        'downsample': downsample,
        'start_time': start_time,
        'end_time': end_time,
        'clean_artifacts': clean_artifacts,
        'thresh': thresh,
        'warp': warp,
        'smooth': smooth,
        'smooth_win': smooth_win,
        'exclude_tags': exclude_tags,
        'exclude_conds': exclude_conds,
        'driven': driven,
        'drive_css': drive_css,
        'drive_threshold': drive_threshold,
        'drive_type': drive_type
    }
    if three_pt_tf:
        pars['three_pt_trace'] = True
    if remove_stim_corr:
        pars['removed_stim_corr'] = True
    if len(negative_modes) > 0:
        pars['negative_modes'] = negative_modes,
    group_pars = {
        'group_by': group_by,
        'up_or_down': up_or_down,
        'use_dprime': use_dprime,
        'dprime_threshold': dprime_threshold
    }
    save_dir = paths.tca_path(mouse, 'group', pars=pars, group_pars=group_pars)

    # get DateSorter object
    if np.isin(group_by.lower(), [
            'naive_vs_high_dprime', 'l_vs_r1', 'l_vs_r1_tight',
            'naive_and_learning'
    ]):
        days = flow.DateSorter.frommeta(mice=[mouse],
                                        dates=dates,
                                        exclude_tags=['bad'])
    else:
        days = flow.DateSorter.frommeta(mice=[mouse],
                                        tags=tags,
                                        exclude_tags=['bad'])
    if mouse == 'OA26' and 'contrast' in exclude_tags:
        days = [s for s in days if s.date != 170302]

    # only include days with xday alignment
    days = [s for s in days if 'xday' in s.tags]

    # skip days with only "special" run types (i.e., these tags have been broadcast to DaySorter, rather than Run)
    days = [s for s in days if 'orientation_mapping' not in s.tags]

    # add monitor condition to exclusions
    exclude_conds += ('monitor', )

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
            days = [
                d for c, d in enumerate(days) if dprime[c] > dprime_threshold
            ]
        elif up_or_down.lower() == 'down':
            days = [
                d for c, d in enumerate(days) if dprime[c] <= dprime_threshold
            ]

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
            good_ids = _group_drive_ids(days,
                                        drive_css,
                                        drive_threshold,
                                        drive_type=drive_type)
            # filter for being able to check for quality of xday alignment
            if score_threshold > 0:
                orig_num_ids = len(good_ids)
                highscore_ids = _group_ids_score(days, score_threshold)
                good_ids = np.intersect1d(good_ids, highscore_ids)
                if verbose and c == 0:
                    print('Cell score threshold ' + str(score_threshold) +
                          ':' + ' ' + str(len(highscore_ids)) +
                          ' above threshold:' + ' good_ids updated to ' +
                          str(len(good_ids)) + '/' + str(orig_num_ids) +
                          ' cells.')
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
                    print('Cell score thresh ' + str(score_threshold) + ':' +
                          ' ' + str(len(highscore_ids)) + ' above thresh:' +
                          ' good_ids updated to ' + str(len(good_ids)) + '/' +
                          str(orig_num_ids) + ' cells.')
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
        d1_runs = [
            run for run in d1_runs if not any(np.isin(run.tags, exclude_tags))
        ]

        # build tensors for all correct runs and trials after filtering
        if d1_runs:
            d1_tensor_list = []
            d1_bhv_list = []
            d1_meta = []
            for run in d1_runs:
                t2p = run.trace2p()
                # trigger all trials around stimulus onsets
                run_traces = utils.getcstraces(run,
                                               cs=cs,
                                               trace_type=trace_type,
                                               start_time=start_time,
                                               end_time=end_time,
                                               downsample=downsample,
                                               clean_artifacts=clean_artifacts,
                                               thresh=thresh,
                                               warp=warp,
                                               smooth=smooth,
                                               smooth_win=smooth_win,
                                               exclude_tags=exclude_tags)
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
                        print(
                            'Skipping, only {} ori presented: '.format(
                                ori_wo_blanks), run)
                    continue
                # subselect metadata if you are only running certain cs
                if cs != '':
                    if cs == 'plus' or cs == 'minus' or cs == 'neutral':
                        run_traces = run_traces[:, :,
                                                (~dfr['condition'].isin([cs]))]
                        bhv_traces = bhv_traces[:, :,
                                                (~dfr['condition'].isin([cs]))]
                        dfr = dfr.loc[(dfr['condition'].isin([cs])), :]
                    elif cs == '0' or cs == '135' or cs == '270':
                        run_traces = run_traces[:, :, (
                            ~dfr['orientation'].isin([cs]))]
                        bhv_traces = bhv_traces[:, :, (
                            ~dfr['orientation'].isin([cs]))]
                        dfr = dfr.loc[(dfr['orientation'].isin([cs])), :]
                    else:
                        print('ERROR: cs called - "' + cs + '" - is not\
                              a valid option.'                                                                                                                                                                                                                                      )

                # subselect metadata to remove certain conditions
                if len(exclude_conds) > 0:
                    run_traces = run_traces[:, :, (
                        ~dfr['condition'].isin(exclude_conds))]
                    bhv_traces = bhv_traces[:, :, (
                        ~dfr['condition'].isin(exclude_conds))]
                    dfr = dfr.loc[(~dfr['condition'].isin(exclude_conds)), :]

                # assertions that your filtering worked
                if 'blank' in exclude_conds:
                    assert np.sum(dfr['orientation'].isin([-1])) == 0

                # drop trials with nans and add to lists
                keep = np.sum(np.sum(np.isnan(run_traces),
                                     axis=0,
                                     keepdims=True),
                              axis=1,
                              keepdims=True).flatten() == 0
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
            group_tensor[(id_union == k), :,
                         trial_start:trial_end] = celln_all_trials
        trial_start += np.shape(tensor_list[i])[2]

    # special case for focusing on reversal transition
    if group_by.lower() in ['l_vs_r1_tight', 'all100']:
        first_bool = _first100_bool_wdelta(meta)
        meta = meta.iloc[first_bool, :]
        group_bhv_tensor = group_bhv_tensor[:, :, first_bool]
        group_tensor = group_tensor[:, :, first_bool]

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

    # optionally remove 20% of trials for testing model success using variance explained later
    if cv:
        cv_tag = '_cv' + str(train_test_split)
        group_tensor, test_tensor = test_set_from_tensor(
            group_tensor, train_test_split=train_test_split)
        test_tensor_path = os.path.join(
            save_dir,
            str(day1.mouse) + '_' + str(group_by) + score_tag + nt_tag +
            cv_tag + '_test_tensor_' + str(trace_type) + '.npy')
        if not os.path.exists(test_tensor_path) or force:
            np.save(test_tensor_path, test_tensor)
    else:
        cv_tag = ''

    # just so you have a clue how big the tensor is
    if verbose:
        print('Tensor decomp about to begin: tensor shape = ' +
              str(np.shape(group_tensor)))

    # concatenate and save df for the day
    meta_path = os.path.join(
        save_dir,
        str(day1.mouse) + '_' + str(group_by) + score_tag + nt_tag +
        '_df_group_meta.pkl')
    input_tensor_path = os.path.join(
        save_dir,
        str(day1.mouse) + '_' + str(group_by) + score_tag + nt_tag + cv_tag +
        '_group_tensor_' + str(trace_type) + '.npy')
    input_bhv_path = os.path.join(
        save_dir,
        str(day1.mouse) + '_' + str(group_by) + score_tag + nt_tag +
        '_group_bhv_' + str(trace_type) + '.npy')
    input_ids_path = os.path.join(
        save_dir,
        str(day1.mouse) + '_' + str(group_by) + score_tag + nt_tag +
        '_group_ids_' + str(trace_type) + '.npy')
    output_tensor_path = os.path.join(
        save_dir,
        str(day1.mouse) + '_' + str(group_by) + score_tag + nt_tag + cv_tag +
        '_group_decomp_' + str(trace_type) + '.npy')
    all_paths = [
        meta_path, input_tensor_path, input_bhv_path, input_ids_path,
        output_tensor_path
    ]
    if all([os.path.exists(p) for p in all_paths]) and not force:
        print(
            f'All paths exist, force not set. Skipping save and decomposition.\n'
        )
        for p in all_paths:
            print(f'    {p}')
        print(' \n')
        return
    meta.to_pickle(meta_path)
    np.save(input_tensor_path, group_tensor)
    np.save(input_ids_path, id_union)
    np.save(input_bhv_path, group_bhv_tensor)

    # run TCA - iterate over different fitting methods
    if not update_meta:  # only run full TCA when not updating metadata
        if np.isin('mcp_als', method) | \
           np.isin('mncp_hals', method) | \
           np.isin('ncp_hals', method):
            mask = ~np.isnan(group_tensor)
            # allow for normal ncp_hals to run with no mask if the tensor
            # has no empties
            if np.sum(mask.flatten()) == len(mask.flatten()):
                fit_options['mask'] = None
            else:
                fit_options['mask'] = mask
            # allow for fitting with only certain negative dimesions
            if len(negative_modes) > 0:
                fit_options['negative_modes'] = negative_modes
            # allow for fitting with only certain negative dimesions
            if tensor_init.lower() != 'rand':
                fit_options['init'] = tensor_init
            if len(skip_modes) > 0:
                fit_options['skip_modes'] = skip_modes
        group_tensor[np.isnan(group_tensor)] = 0
        ensemble = {}
        for m in method:
            ensemble[m] = tt.Ensemble(fit_method=m,
                                      fit_options=deepcopy(fit_options))
            # Optionally take a list of particular ranks to run, default to first 20
            if isinstance(rank, list):
                ensemble[m].fit(group_tensor,
                                ranks=rank,
                                replicates=replicates,
                                verbose=verbose)
            else:
                ensemble[m].fit(group_tensor,
                                ranks=range(1, rank + 1),
                                replicates=replicates,
                                verbose=verbose)
        np.save(output_tensor_path, ensemble)

    # print output so you don't go crazy waiting
    if verbose:
        print(str(day1.mouse) + ': group_by=' + str(group_by) + ': done.')


def test_set_from_tensor(tensor, train_test_split=0.8):
    """
    Function to remove 20% of trials from each cell to be used as a test for model performance

    :param tensor: numpy_ndarray, 3D matrix cells x timepoints x trials
    :return: tensor, test_tensor
    """

    # train-test split
    frac_train = train_test_split
    np.random.seed(seed=42)  # seed random state
    ncells = tensor.shape[0]
    test_tensor = deepcopy(tensor)
    test_tensor[:] = np.nan

    # loop over cells to remove 20% of trials per cell
    for n in range(ncells):

        # randomly select 20% of existing (non-NaN) trials
        ntrials = np.sum(~np.isnan(tensor[n, 0, :]))
        trial_inds = np.where(~np.isnan(tensor[n, 0, :]))[0].astype(int)
        ntoselect = np.round(ntrials * (1. - frac_train)).astype(int)
        test_inds = np.random.choice(trial_inds, ntoselect, replace=False)

        # save test trials into new test_tensor
        test_tensor[n, :, test_inds] = tensor[n, :, test_inds]

        # remove test trials from the input tensor
        tensor[n, :, test_inds] = np.nan

    return tensor, test_tensor


def _get_speed_pupil_npil_traces(run,
                                 cs='',
                                 start_time=-1,
                                 end_time=6,
                                 downsample=True,
                                 cutoff_before_lick_ms=-1):
    """
    Helper function that makes a tensor stacking pupil and running speed
    into a single tensor. dpupil and dspeed are baseline subtracted from
    -1 to 0 sec before stimulus onset.
    
    Returns: array
    -------
    [pupil, dpupil, speed, dspeed] x timepoints x trials

    """
    pupil_traces = utils.getcsbehavior(
        run,
        cs=cs,
        trace_type='pupil',
        start_time=start_time,
        end_time=end_time,
        downsample=downsample,
        baseline=None,
        cutoff_before_lick_ms=cutoff_before_lick_ms)

    dpupil_traces = utils.getcsbehavior(run,
                                        cs=cs,
                                        trace_type='pupil',
                                        start_time=start_time,
                                        end_time=end_time,
                                        downsample=downsample,
                                        baseline=(-1, 0),
                                        cutoff_before_lick_ms=-1)

    dlick_traces = utils.getcsbehavior(run,
                                       cs=cs,
                                       trace_type='lick_rate',
                                       start_time=start_time,
                                       end_time=end_time,
                                       downsample=downsample,
                                       baseline=(-1, 0),
                                       cutoff_before_lick_ms=-1)

    lick_traces = utils.getcsbehavior(
        run,
        cs=cs,
        trace_type='lick_rate',
        start_time=start_time,
        end_time=end_time,
        downsample=downsample,
        baseline=None,
        cutoff_before_lick_ms=cutoff_before_lick_ms)

    speed_traces = utils.getcsbehavior(
        run,
        cs=cs,
        trace_type='speed',
        start_time=start_time,
        end_time=end_time,
        downsample=downsample,
        baseline=None,
        cutoff_before_lick_ms=cutoff_before_lick_ms)

    dspeed_traces = utils.getcsbehavior(run,
                                        cs=cs,
                                        trace_type='speed',
                                        start_time=start_time,
                                        end_time=end_time,
                                        downsample=downsample,
                                        baseline=(-1, 0),
                                        cutoff_before_lick_ms=-1)

    neuropil_traces = utils.getcsbehavior(
        run,
        cs=cs,
        trace_type='neuropil',
        start_time=start_time,
        end_time=end_time,
        downsample=downsample,
        baseline=None,
        cutoff_before_lick_ms=cutoff_before_lick_ms)

    dneuropil_traces = utils.getcsbehavior(run,
                                           cs=cs,
                                           trace_type='neuropil',
                                           start_time=start_time,
                                           end_time=end_time,
                                           downsample=downsample,
                                           baseline=(-1, 0),
                                           cutoff_before_lick_ms=-1)

    all_behaviors_list = (pupil_traces, dpupil_traces, lick_traces,
                          dlick_traces, speed_traces, dspeed_traces,
                          neuropil_traces, dneuropil_traces)
    bhv_traces = np.concatenate(all_behaviors_list, axis=0)

    return bhv_traces


def _remove_stimulus_corr(tensor, metadata):
    """
    Remove stimulus correlations from individual cells by taking average for 
    each CS and GO/NOGO combo and subtracting that from each corresponding
    set of trials. This will remove stimulus correlations (and a lot of motor
    differences which I don't love), but will also deal with the changing 
    proportion of GO/NOGO trials per day and force the differences in TCA
    components to focus less on motor differences. 

    This may also remove some of the slower changes in amplitude for cells.

    Currently this treats reversal days pre and post separately 
    """

    # preallocate
    new_tensor = np.empty(tensor.shape)
    new_tensor[:] = np.nan

    # take averages for CS, GO/NOGO combos per day and subtract from
    # corresponding trials
    for n_day in metadata.reset_index()['date'].unique():
        day_bool = metadata.reset_index()['date'].isin([n_day]).values
        for n_ori in metadata['orientation'].unique():
            ori_bool = metadata['orientation'].isin([n_ori]).values
            for n_te in metadata['trialerror'].unique():
                te_bool = metadata['trialerror'].isin([n_te]).values
                total_bool = te_bool & ori_bool & day_bool

                # if there are no trials of a given type skip
                if np.sum(total_bool) == 0:
                    continue

                # subtract mean trace from each cell
                new_tensor[:, :, total_bool] = (
                    tensor[:, :, total_bool] -
                    np.nanmean(tensor[:, :, total_bool], axis=2)[:, :, None])

    return new_tensor


def _three_point_temporal_trace(tensor, metadata):
    """
    Take average of baseline period, average of stimulus period, and
    average of response period to make a three point temporal trace
    """

    # set useful variables
    sample_rate = 15.5
    mouse = metadata.reset_index()['mouse'].unique()
    if mouse in ['OA32', 'OA34', 'OA36']:
        stim_start = int(np.floor(sample_rate))
        stim_end = int(np.floor((2 + 1) * sample_rate))
        resp_end = int(np.floor((2 + 2 + 1) * sample_rate))
    elif mouse in ['OA27', 'OA26', 'OA67', 'VF226', 'CC175']:
        stim_start = int(np.floor(sample_rate))
        stim_end = int(np.floor((3 + 1) * sample_rate))
        resp_end = int(np.floor((2 + 3 + 1) * sample_rate))
    else:
        print('Whose mouse is this!?')

    # preallocate
    new_tensor = np.empty((tensor.shape[0], 3, tensor.shape[2]))
    new_tensor[:] = np.nan

    # baseline mean
    new_tensor[:, 0, :] = np.nanmean(tensor[:, :stim_start, :], axis=1)

    # stimulus mean
    new_tensor[:, 1, :] = np.nanmean(tensor[:, stim_start:stim_end, :], axis=1)

    # baseline mean
    new_tensor[:, 2, :] = np.nanmean(tensor[:, stim_end:resp_end, :], axis=1)

    return new_tensor


def _group_drive_ids(days, drive_css, drive_threshold, drive_type='visual'):
    """
    Get an array of all unique ids driven on any day for a given DaySorter.
    """

    good_ids = []
    for day1 in days:
        # get cell_ids
        d1_ids = flow.xday._read_crossday_ids(day1.mouse, day1.date)
        # skip empty if there is no crossday alignment file
        if len(d1_ids) == 0:
            continue
        d1_ids = np.array([int(s) for s in d1_ids])
        # filter cells based on visual/trial drive across all cs
        d1_drive = []
        for dcs in drive_css:
            try:
                if drive_type.lower() == 'trial':
                    d1_drive.append(pool.calc.driven.trial(day1, dcs))
                elif drive_type.lower() == 'trial_abs':
                    d1_drive.append(pool.calc.driven.trial_abs(day1, dcs))
                elif drive_type.lower() == 'visual':
                    d1_drive.append(pool.calc.driven.visually(day1, dcs))
                elif drive_type.lower() == 'inhib':
                    d1_drive.append(pool.calc.driven.visually_inhib(day1, dcs))
            except KeyError:
                print(
                    str(day1) + ' requested ' + dcs + ' ' + drive_type +
                    ': no match to what was shown (probably pav only).')

        d1_drive = np.max(d1_drive, axis=0)
        d1_drive_ids = d1_ids[np.array(d1_drive) > drive_threshold]
        good_ids.extend(d1_drive_ids)

    return np.unique(good_ids)


def _group_ids_score(days, score_threshold):
    """
    Get an array of all unique ids with an alignment cell score above threshold
    on any day for a given DaySorter.
    """

    good_ids = []
    for day1 in days:
        # get cell_ids
        d1_ids = flow.xday._read_crossday_ids(day1.mouse, day1.date)
        d1_scores = flow.xday._read_crossday_scores(day1.mouse, day1.date)
        # skip empty if there is no crossday alignment file
        if len(d1_ids) == 0 or len(d1_scores) == 0:
            continue
        d1_ids = np.array([int(s) for s in d1_ids])
        d1_scores = np.array([float(s) for s in d1_scores])
        # filter cells based on visual/trial drive across all cs
        d1_highscore_ids = d1_ids[np.array(d1_scores) > score_threshold]
        good_ids.extend(d1_highscore_ids)

    return np.unique(good_ids)


def _triggerfromrun(run,
                    trace_type='zscore_day',
                    cs='',
                    downsample=True,
                    start_time=-1,
                    end_time=6,
                    clean_artifacts='interp',
                    thresh=20,
                    warp=False,
                    smooth=True,
                    smooth_win=6,
                    verbose=True):
    """
    Create a pandas dataframe of all of your triggered traces for a mouse.

    Parameters:
    -----------
    mouse : str
        Mouse name.
    cs : str
        Type of CS. e.g., plus, minus, neutral, 0, 135, 270, ...
    trace_type : str
        dff, zscore, zscore_iti, deconvolved
    downsample : bool
        Downsample from 31 to 15 Hz sampling rate
    clean_artifacts : str
        nan, interp; Remove huge artifacts in dff traces by interpolating
        or adding in nan values
        Note: setting either value here will cause z-scoring to nan artifacts
        before calculating mu/sigma
    thresh : int
        Threshold for removing artifacts
    warp : bool
        Warp the outcome to a particular time point using interpolation.
        Calls flow.Trace2P.warpcstraces()
    smooth : bool
        Smooth your signal by convolution
    smooth_win : int
        Window in sampling points over which to smooth
    smooth_win_dec : int
        Window in sampling points over which to smooth deconvolved data
        Note: this step follows downsampling so window should probably be
        smaller than that for smoothing z-score.

    Returns:
    --------
    Pandas dataframe of all triggered traces and saves to .../output folder.
    """

    # get t2p object
    t2p = run.trace2p()

    trial_list = []

    # get your t2p object
    t2p = run.trace2p()

    # get your cell# from xday alignment
    # use to index along axis=0 in cstraces/run_traces
    cell_ids = flow.xday._read_crossday_ids(run.mouse, run.date)
    cell_ids = [int(s) for s in cell_ids]

    # trigger all trials around stimulus onsets
    run_traces = utils.getcstraces(run,
                                   cs=cs,
                                   trace_type=trace_type,
                                   start_time=start_time,
                                   end_time=end_time,
                                   downsample=True,
                                   clean_artifacts=clean_artifacts,
                                   thresh=thresh,
                                   warp=warp,
                                   smooth=smooth,
                                   smooth_win=smooth_win)

    # make timestamps, downsample is necessary
    timestep = 1 / t2p.d['framerate']
    timestamps = np.arange(start_time, end_time, timestep)

    if (t2p.d['framerate'] > 30) and downsample:
        timestamps = timestamps[::2][:np.shape(run_traces)[1]]

    # check that you don't have extra cells
    if len(cell_ids) != np.shape(run_traces)[0]:
        run_traces = run_traces[range(0, len(cell_ids)), :, :]
        warnings.warn(
            str(run) +
            ': You have more cell traces than cell_idx: skipping extra cells.')

    # build matrices to match cell, trial, time variables to traces
    trial_mat = np.ones(np.shape(run_traces))
    for trial in range(np.shape(run_traces)[2]):
        trial_mat[:, :, trial] = trial

    cell_mat = np.ones(np.shape(run_traces))
    for cell in range(np.shape(run_traces)[0]):
        cell_mat[cell, :, :] = cell_ids[cell]

    time_mat = np.ones(np.shape(run_traces))
    for timept in range(np.shape(run_traces)[1]):
        time_mat[:, timept, :] = timestamps[timept]

    # reshape and build df
    vec_sz = run_traces.size
    index = pd.MultiIndex.from_arrays(
        [[run.mouse] * vec_sz, [run.date] * vec_sz, [run.run] * vec_sz,
         trial_mat.reshape(vec_sz),
         cell_mat.reshape(vec_sz),
         time_mat.reshape(vec_sz)],
        names=['mouse', 'date', 'run', 'trial_idx', 'cell_idx', 'timestamp'])

    # append all runs across a day together in a list
    df = pd.DataFrame({'trace': run_traces.reshape(vec_sz)}, index=index)

    return df


def _trialmetafromrun(run,
                      trace_type='dff',
                      start_time=-1,
                      end_time=6,
                      downsample=True,
                      add_p_cols=False,
                      verbose=True):
    """
    Create pandas dataframe of trial metadata from run.

    Parameters
    -------
    run, flow RunSorter object
        Run object to be parsed.
    trace_type, str
        Type of trace, to match loading of actual 2P imaging data.
    start_time, int
        Number of seconds before stimulus onset to include.
    end_time, int
        Number of seconds after stimulus onset to include.
    downsample, boolean
        Downsample by a factor of 2. Important for vectors that only record a 
        2P imaging frame number rather than trial by trial values. 
    add_p_cols, boolean
        Optionally, add in columns to metadata that include probability of 
        current trial since last trial of the same type. 
    verbose, boolean
        Print useful statement about progress of function and underlying data.

    Returns
    -------
    dfr, pandas.DataFrame
        DataFrame of all metadata as columns indexed by 'mouse', 'date', 'run',
        and 'trial_idx'. 
    """

    # get t2p object
    t2p = run.trace2p()

    # get the number of trials in your run
    try:
        ntrials = t2p.ntrials
        trial_idx = range(ntrials)
    except:
        run_traces = t2p.cstraces('',
                                  start_s=start_time,
                                  end_s=end_time,
                                  trace_type=trace_type,
                                  cutoff_before_lick_ms=-1,
                                  errortrials=-1,
                                  baseline=(-1, 0),
                                  baseline_to_stimulus=True)
        ntrials = np.shape(run_traces)[2]
        trial_idx = range(ntrials)

    # if there are no stimulus presentations skip "trials"
    if ntrials == 0:
        if verbose:
            print('No CS presentations on', run)
        return []

    # get your learning-state
    run_tags = [str(s) for s in run.tags]
    if 'naive' in run_tags:
        learning_state = 'naive'
    elif 'learning' in run_tags:
        learning_state = 'learning'
    elif 'reversal1' in run_tags:
        learning_state = 'reversal1'
    elif 'reversal2' in run_tags:
        learning_state = 'reversal2'
    else:
        learning_state = np.nan
    learning_state = [learning_state] * len(trial_idx)

    # get hunger-state for all trials, consider hungry if not sated
    if 'sated' in run_tags:
        hunger = 'sated'
    else:
        hunger = 'hungry'
    hunger = [hunger] * len(trial_idx)

    # get relevant trial-distinguishing tags excluding kelly, hunger-state, learning-state
    tags = [
        str(run_tags[s]) for s in range(len(run_tags))
        if run_tags[s] != hunger[0] and run_tags[s] != learning_state[0]
        and run_tags[s] != 'kelly' and run_tags[s] != 'learning_start' and
        run_tags[s] != 'reversal1_start' and run_tags[s] != 'reversal2_start'
    ]
    if tags == []:  # define as "standard" if the run is not another option
        tags = ['standard']
    tags = [tags[0]] * len(trial_idx)

    # get boolean of engagement per trial
    try:
        HMM_engaged = pool.calc.performance.engaged(
            run, across_run=False)[trial_idx]
        # if the run is naive check that it is disengaged
        if 'naive' in run_tags and np.any(HMM_engaged):
            HMM_engaged = np.zeros((len(trial_idx))) > 0
            if verbose:
                print('{} {} {}: HMM naive engagement reset.'.format(
                    run.mouse, run.date, run.run))
    except KeyError:
        HMM_engaged = np.zeros((len(trial_idx)))
        HMM_engaged[:] = np.nan
        if verbose:
            print('{} {} {}: HMM engagement failed.'.format(
                run.mouse, run.date, run.run))

    # get trialerror ensuring you don't include runthrough at end of trials
    trialerror = np.array(t2p.d['trialerror'][trial_idx])

    # prev trials responses
    prevtrialerror = np.array(trialerror, dtype=float)
    prevtrialerror = np.insert(prevtrialerror, 0,
                               np.nan)[trial_idx]  # frame shift

    # previous reward
    prev_reward = np.isin(prevtrialerror, [0, 8, 9])

    # previous punishment
    prev_punishment = np.isin(prevtrialerror, [5])

    # previous blank
    prev_blank = np.isin(prevtrialerror, [6, 7])

    # get cs and orientation info for each trial
    lookup = {v: k for k, v in t2p.d['codes'].items()}  # invert dict
    css = [lookup[s] for s in t2p.d['condition'][trial_idx]]
    try:
        oris = [
            t2p.d['orientations'][lookup[s]]
            for s in t2p.d['condition'][trial_idx]
        ]
    except KeyError:
        oris = [np.nan for s in t2p.d['condition'][trial_idx]]
        if verbose:
            print('{} {} {}: orientation lookup failed.'.format(
                run.mouse, run.date, run.run))

    # get ori according to which cs it was during initial learning
    initial_css = [lookups.lookup_ori[run.mouse][s] for s in oris]

    # previous cue (orientation) was the same as current cue, named by initial
    # learning cs
    assigned_prev_same_cue = {}
    for unk in ['plus', 'neutral', 'minus']:
        ori_is_unk = lookups.lookup[run.mouse][unk]
        te_to_reassign = np.unique(trialerror[np.isin(oris, ori_is_unk)])
        if np.all(np.isin(te_to_reassign, [0, 1, 8, 9])):
            assigned_prev_same_cue[unk] = (
                np.isin(trialerror, [0, 1, 8, 9])
                & np.isin(prevtrialerror, [0, 1, 8, 9]))
        elif np.all(np.isin(te_to_reassign, [2, 3])):
            assigned_prev_same_cue[unk] = (np.isin(trialerror, [2, 3])
                                           & np.isin(prevtrialerror, [2, 3]))
        elif np.all(np.isin(te_to_reassign, [4, 5])):
            assigned_prev_same_cue[unk] = (np.isin(trialerror, [4, 5])
                                           & np.isin(prevtrialerror, [4, 5]))
        else:
            assigned_prev_same_cue[unk] = np.ones((len(trialerror))) < 1

    # get mean running speed for time stim is on screen
    # use first 3 seconds after onset if there is no offset
    all_onsets = t2p.csonsets()
    try:
        all_offsets = t2p.d['offsets'][0:len(all_onsets)]
    except KeyError:
        all_offsets = all_onsets + (np.round(t2p.d['framerate']) * 3)
        if all_offsets[-1] > t2p.nframes:
            all_offsets[-1] = t2p.nframes
        if verbose:
            print('{} {} {}: offsets failed: hardcoding offsets to 3s.'.format(
                run.mouse, run.date, run.run))

    # calculate running speed during trial
    if t2p.d['running'].size > 0:
        speed_vec = t2p.speed()
        speed_vec = speed_vec.astype('float')
        speed = []
        for s in trial_idx:
            try:
                speed.append(
                    np.nanmean(speed_vec[all_onsets[s]:all_offsets[s]]))
            except:
                speed.append(np.nan)
        speed = np.array(speed)
    else:
        speed = np.full(len(trial_idx), np.nan)

    # calculate running speed preceding trial
    if t2p.d['running'].size > 0:
        pre_speed_vec = t2p.speed()
        pre_speed_vec = pre_speed_vec.astype('float')
        nframe_back = np.round(t2p.d['framerate'])
        pre_speed = []
        for s in trial_idx:
            try:
                pre_speed.append(
                    np.nanmean(pre_speed_vec[int(all_onsets[s] -
                                                 nframe_back):all_onsets[s]]))
            except:
                pre_speed.append(np.nan)
        pre_speed = np.array(pre_speed)
    else:
        pre_speed = np.full(len(trial_idx), np.nan)

    # calculate running speed during response window
    if t2p.d['running'].size > 0:
        post_speed_vec = t2p.speed()
        post_speed_vec = post_speed_vec.astype('float')
        nframe_fwd = np.round(t2p.d['framerate'] * 2)  # 2s response window
        post_speed = []
        for s in trial_idx:
            try:
                post_speed.append(
                    np.nanmean(
                        post_speed_vec[all_offsets[s]:int(all_offsets[s] +
                                                          nframe_fwd)]))
            except:
                post_speed.append(np.nan)
        post_speed = np.array(post_speed)
    else:
        post_speed = np.full(len(trial_idx), np.nan)

    # calculate running speed during trial
    if t2p.d['neuropil'].size > 0:
        npil_vec = t2p.d['neuropil']
        npil = []
        for s in trial_idx:
            try:
                npil.append(np.nanmean(npil_vec[all_onsets[s]:all_offsets[s]]))
            except:
                npil.append(np.nan)
        npil = np.array(npil)
    else:
        npil = np.full(len(trial_idx), np.nan)

    # calculate neuropil speed preceding trial
    if t2p.d['neuropil'].size > 0:
        pre_npil_vec = t2p.d['neuropil']
        nframe_back = np.round(t2p.d['framerate'])
        pre_npil = []
        for s in trial_idx:
            try:
                pre_npil.append(
                    np.nanmean(pre_npil_vec[int(all_onsets[s] -
                                                nframe_back):all_onsets[s]]))
            except:
                pre_npil.append(np.nan)
        pre_npil = np.array(pre_npil)
    else:
        pre_npil = np.full(len(trial_idx), np.nan)

    # get mean brainmotion for time stim is on screen
    # use first 3 seconds after onset if there is no offset
    if t2p.d['brainmotion'].size > 0:
        xy_vec = t2p.d['brainmotion']
        xy_vec = xy_vec.astype('float')
        brainmotion = []
        for s in trial_idx:
            try:
                brainmotion.append(
                    np.nanmean(xy_vec[all_onsets[s]:all_offsets[s]]))
            except:
                brainmotion.append(np.nan)
        brainmotion = np.array(speed)
    else:
        brainmotion = np.full(len(trial_idx), np.nan)

    # count anticipatory licks during stimulus presentation
    if 'licking' in t2p.d.keys() and (t2p.d['licking'].size > 0):
        lick_vec = t2p.d['licking']
        lick_vec = lick_vec.astype('float')
        antic_lick = []
        for s in trial_idx:
            try:
                lick_bool = (lick_vec > all_onsets[s]) & \
                            (lick_vec < all_offsets[s])
                antic_lick.append(np.sum(lick_bool))
            except:
                antic_lick.append(np.nan)
        antic_lick = np.array(antic_lick)
    else:
        antic_lick = np.full(len(trial_idx), np.nan)

    # count licks 1 second before stimulus presentation
    if 'licking' in t2p.d.keys() and (t2p.d['licking'].size > 0):
        pre_lick_vec = t2p.d['licking']
        pre_lick_vec = pre_lick_vec.astype('float')
        nframe_back = np.round(t2p.d['framerate'])
        pre_lick = []
        for s in trial_idx:
            try:
                pre_lick_bool = (pre_lick_vec < all_onsets[s]) & \
                                (pre_lick_vec > (all_onsets[s] - nframe_back))
                pre_lick.append(np.sum(pre_lick_bool))
            except:
                pre_lick.append(np.nan)
        pre_lick = np.array(pre_lick)
    else:
        pre_lick = np.full(len(trial_idx), np.nan)

    # count licks during response window
    if 'licking' in t2p.d.keys() and (t2p.d['licking'].size > 0):
        post_lick_vec = t2p.d['licking']
        post_lick_vec = post_lick_vec.astype('float')
        nframe_fwd = np.round(t2p.d['framerate'] * 2)  # 2s response window
        post_lick = []
        for s in trial_idx:
            try:
                post_lick_bool = (post_lick_vec > all_offsets[s]) & \
                                (post_lick_vec < (all_offsets[s] + nframe_fwd))
                post_lick.append(np.sum(post_lick_bool))
            except:
                post_lick.append(np.nan)
        post_lick = np.array(post_lick)
    else:
        post_lick = np.full(len(trial_idx), np.nan)

    # calculate pupil diameter preceding trial
    if 'pupil' in t2p.d.keys() and (t2p.d['pupil'].size > 0):
        pre_pupil_vec = t2p.d['pupil']
        pre_pupil_vec = pre_pupil_vec.astype('float')
        # account for fact that pupillometry is always at 15 Hz
        # divide onsets and offsets by 2 if the framerate is 30Hz
        if t2p.d['framerate'] > 30:
            nframe_back = np.round(t2p.d['framerate'] / 2)
            div = 2  # divisor for onsets and offsets
        else:
            nframe_back = np.round(t2p.d['framerate'])
            div = 1
        pre_pupil = []
        for s in trial_idx:
            try:
                nbefore = int(np.round(all_onsets[s] / div) - nframe_back)
                nafter = int(np.round(all_onsets[s] / div))
                pre_pupil.append(np.nanmean(pre_pupil_vec[nbefore:nafter]))
            except:
                pre_pupil.append(np.nan)
        pre_pupil = np.array(pre_pupil)
    else:
        pre_pupil = np.full(len(trial_idx), np.nan)

    # calculate pupil diameter during stimulus presentation
    if 'pupil' in t2p.d.keys() and (t2p.d['pupil'].size > 0):
        pupil_vec = t2p.d['pupil']
        pupil_vec = pupil_vec.astype('float')
        # account for fact that pupillometry is always at 15 Hz
        # divide onsets and offsets by 2 if the framerate is 30Hz
        if t2p.d['framerate'] > 30:
            nframe_back = np.round(t2p.d['framerate'] / 2)
            div = 2  # divisor for onsets and offsets
        else:
            nframe_back = np.round(t2p.d['framerate'])
            div = 1
        pupil = []
        for s in trial_idx:
            try:
                nbefore = int(np.round(all_onsets[s] / div))
                nafter = int(np.round(all_offsets[s] / div))
                pupil.append(np.nanmean(pupil_vec[nbefore:nafter]))
            except:
                pupil.append(np.nan)
        pupil = np.array(pupil)
    else:
        pupil = np.full(len(trial_idx), np.nan)

    # get ensure/ensure/firstlick relative to triggered data
    ensure = t2p.ensure()
    ensure = ensure.astype('float')
    ensure[ensure == 0] = np.nan
    ensure = ensure - all_onsets + (np.abs(start_time) *
                                    np.round(t2p.d['framerate']))

    quinine = t2p.quinine()
    quinine = quinine.astype('float')
    quinine[quinine == 0] = np.nan
    quinine = quinine - all_onsets + (np.abs(start_time) *
                                      np.round(t2p.d['framerate']))

    firstlick = t2p.firstlick('')[trial_idx]
    firstlick = firstlick + (np.abs(start_time) * np.round(t2p.d['framerate']))

    firstlickbout = t2p.firstlickbout('', lickbout_len=2,
                                      strict=True)[trial_idx]
    firstlickbout = firstlickbout + (np.abs(start_time) *
                                     np.round(t2p.d['framerate']))

    # downsample all timestamps to 15Hz if framerate is 31Hz
    if (t2p.d['framerate'] > 30) and downsample:
        ensure = ensure / 2
        quinine = quinine / 2
        firstlick = firstlick / 2
        firstlickbout = firstlickbout / 2

    # create your index out of relevant variables
    index = pd.MultiIndex.from_arrays(
        [[run.mouse] * len(trial_idx), [run.date] * len(trial_idx),
         [run.run] * len(trial_idx), trial_idx],
        names=['mouse', 'date', 'run', 'trial_idx'])

    data = {
        'orientation': oris,
        'condition': css,
        'initial_condition': initial_css,
        'trialerror': trialerror,
        'prev_reward': prev_reward,
        'prev_punish': prev_punishment,
        'prev_same_plus': assigned_prev_same_cue['plus'],
        'prev_same_neutral': assigned_prev_same_cue['neutral'],
        'prev_same_minus': assigned_prev_same_cue['minus'],
        'prev_blank': prev_blank,
        'hunger': hunger,
        'learning_state': learning_state,
        'tag': tags,
        'firstlick': firstlick,
        'firstlickbout': firstlickbout,
        'ensure': ensure,
        'quinine': quinine,
        'hmm_engaged': HMM_engaged,
        'speed': speed,
        'pre_speed': pre_speed,
        'post_speed': post_speed,
        'anticipatory_licks': antic_lick,
        'pre_licks': pre_lick,
        'post_licks': post_lick,
        'pupil': pupil,
        'pre_pupil': pre_pupil,
        'neuropil': npil,
        'pre_neuropil': pre_npil,
        'brainmotion': brainmotion
    }

    dfr = pd.DataFrame(data, index=index)

    # optionally add in columns of probabilities
    if add_p_cols:
        dfr = _add_prob_columns_trialmeta(run.mouse, dfr)

    return dfr


def _add_prob_columns_trialmeta(mouse, meta1_df):
    """
    Helper function to add in useful columns to metadata related to
    trial history. 
    """

    # add a binary column for choice, 1 for go 0 for nogo
    new_meta = {}
    new_meta['go'] = np.zeros(len(meta1_df))
    new_meta['go'][meta1_df['trialerror'].isin([0, 3, 5, 7]).values] = 1
    new_meta_df1 = pd.DataFrame(data=new_meta, index=meta1_df.index)
    # new_meta_df1 = pd.concat([new_meta_df1, new_meta_df], axis=1)

    # add a binary column for reward
    new_meta = {}
    new_meta['reward'] = np.zeros(len(new_meta_df1))
    new_meta['reward'][meta1_df['trialerror'].isin([0]).values] = 1
    new_meta_df = pd.DataFrame(data=new_meta, index=meta1_df.index)
    new_meta_df1 = pd.concat([new_meta_df1, new_meta_df], axis=1)

    # add a binary column for punishment
    new_meta = {}
    new_meta['punishment'] = np.zeros(len(new_meta_df1))
    new_meta['punishment'][meta1_df['trialerror'].isin([5]).values] = 1
    new_meta_df = pd.DataFrame(data=new_meta, index=meta1_df.index)
    new_meta_df1 = pd.concat([new_meta_df1, new_meta_df], axis=1)

    # rename oris according to their meaning during learning
    new_meta = {}
    for ori in ['plus', 'minus', 'neutral']:
        new_meta = {}
        new_meta['initial_{}'.format(ori)] = np.zeros(len(new_meta_df1))
        new_meta['initial_{}'.format(ori)][meta1_df['orientation'].isin(
            [lookups.lookup[mouse][ori]]).values] = 1
        new_meta_df = pd.DataFrame(data=new_meta, index=meta1_df.index)
        new_meta_df1 = pd.concat([new_meta_df1, new_meta_df], axis=1)

    # create epochs since last reward
    c = 0
    vec = []
    for s in new_meta_df1['reward'].values:
        if s == 0:
            vec.append(c)
        else:
            vec.append(c)
            c += 1
    new_meta_df1['reward_cum'] = vec

    # since last go
    c = 0
    vec = []
    for s in new_meta_df1['go'].values:
        if s == 0:
            vec.append(c)
        else:
            vec.append(c)
            c += 1
    new_meta_df1['go_cum'] = vec

    # since last of same cue type
    for ori in ['plus', 'minus', 'neutral']:
        c = 0
        vec = []
        for s in new_meta_df1['initial_{}'.format(ori)].values:
            if s == 0:
                vec.append(c)
            else:
                vec.append(c)
                c += 1
        new_meta_df1['initial_{}_cum'.format(ori)] = vec

    # vec of ones for finding denominator across a number of trials
    new_meta_df1['trial_number'] = np.ones((len(new_meta_df1)))

    # loop over different accumulators to get full length interaction terms
    p_cols = []
    for aci in [
            'initial_plus', 'initial_minus', 'initial_neutral', 'go', 'reward'
    ]:
        accumulated_df = new_meta_df1.groupby('{}_cum'.format(aci)).sum()
        prob_since_last = accumulated_df.divide(accumulated_df['trial_number'],
                                                axis=0)
        for vali in [
                'initial_plus', 'initial_minus', 'initial_neutral', 'go',
                'reward'
        ]:
            new_vec = np.zeros(len(new_meta_df1))
            new_vec[:] = np.nan
            new_bool = new_meta_df1[aci].gt(0).values
            new_vec[new_bool] = prob_since_last[vali].values[0:np.sum(
                new_bool)]  # use only matched trials
            new_meta_df1['p_{}_since_last_{}'.format(vali, aci)] = new_vec
            p_cols.append('p_{}_since_last_{}'.format(vali, aci))

    # turn go, reward, punishment into boolean
    grp_df = new_meta_df1.loc[:, ['go', 'reward', 'punishment']].gt(0)

    # get only columns for probability
    p_df = new_meta_df1.loc[:, p_cols]

    # add new columns to original df
    new_dfr = pd.concat([meta1_df, grp_df, p_df], axis=1)

    return new_dfr


def _getcstraces_filtered(runs_list,
                          d_ids_bool,
                          d_sorter,
                          cs='',
                          trace_type='zscore_day',
                          start_time=-1,
                          end_time=6,
                          downsample=True,
                          clean_artifacts=None,
                          thresh=20,
                          warp=False,
                          smooth=True,
                          smooth_win=6):
    """
    Build a tensor and matching metadata dataframe from list of run objects.

    """

    # preallocate lists
    tensor_list = []
    meta_list = []

    # loop over all runs
    for run in runs_list:
        t2p = run.trace2p()
        # trigger all trials around stimulus onsets
        run_traces = utils.getcstraces(run,
                                       cs=cs,
                                       trace_type=trace_type,
                                       start_time=start_time,
                                       end_time=end_time,
                                       downsample=True,
                                       clean_artifacts=clean_artifacts,
                                       thresh=thresh,
                                       warp=warp,
                                       smooth=smooth,
                                       smooth_win=smooth_win)
        # filter and sort
        run_traces = run_traces[d_ids_bool, :, :][d_sorter, :, :]
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
                      a valid option.'                                                                                                                                                                                              )
        # drop trials with nans and add to lists
        keep = np.sum(np.sum(np.isnan(run_traces), axis=0, keepdims=True),
                      axis=1,
                      keepdims=True).flatten() == 0
        dfr = dfr.iloc[keep, :]
        tensor_list.append(run_traces[:, :, keep])
        meta_list.append(dfr)

    # concatenate matched cells across trials 3rd dim (aka, 2)
    tensor = np.concatenate(tensor_list, axis=2)
    meta_df = pd.concat(meta_list, axis=0)

    return tensor, meta_df


def _first100_bool(meta):
    """
    Helper function to get a boolean vector of the first 100 trials for each day.
    If a day is shorter than 100 trials use the whole day. 
    """

    days = meta.reset_index()['date'].unique()

    first100 = np.zeros((len(meta)))
    for di in days:
        dboo = meta.reset_index()['date'].isin([di]).values
        daylength = np.sum(dboo)
        if daylength > 100:
            first100[np.where(dboo)[0][:100]] = 1
        else:
            first100[dboo] = 1
    firstboo = first100 > 0

    return firstboo


def _first100_bool_wdelta(meta):
    """
    Helper function to get a boolean vector of the first 100 trials for each day.
    Treats initial learning and reversal as a new "day."
    If a day is shorter than 100 trials use the whole day. 
    """

    day_vec = np.array(meta.reset_index()['date'].values, dtype='float')
    days = np.unique(day_vec)
    ls = meta['learning_state'].values
    for di in days:
        dboo = np.isin(day_vec, di)
        states_vec = ls[dboo]
        u_states = np.unique(states_vec)
        if len(u_states) > 1:
            second_state = dboo & (ls == u_states[1])
            day_vec[second_state] += 0.5

    first100 = np.zeros((len(meta)))
    days = np.unique(day_vec)
    for di in days:
        dboo = np.isin(day_vec, di)
        daylength = np.sum(dboo)
        if daylength > 100:
            first100[np.where(dboo)[0][:100]] = 1
        else:
            first100[dboo] = 1
    firstboo = first100 > 0

    return firstboo


def _last100_bool(meta):
    """
    Helper function to get a boolean vector of the last 100 trials for each day.
    If a day is shorter than 100 trials use the whole day. 
    """

    days = meta.reset_index()['date'].unique()

    first100 = np.zeros((len(meta)))
    for di in days:
        dboo = meta.reset_index()['date'].isin([di]).values
        daylength = np.sum(dboo)
        if daylength > 100:
            first100[np.where(dboo)[0][-100:]] = 1
        else:
            first100[dboo] = 1
    firstboo = first100 > 0

    return firstboo


def _firstlast100_bool(meta):
    """
    Helper function to get a boolean vector of the first and last 100 trials for each day.
    If a day is shorter than 200 trials use the whole day. 
    """

    days = meta.reset_index()['date'].unique()

    first100 = np.zeros((len(meta)))
    for di in days:
        dboo = meta.reset_index()['date'].isin([di]).values
        daylength = np.sum(dboo)
        if daylength > 100:
            first100[np.where(dboo)[0][-50:]] = 1
            first100[np.where(dboo)[0][:50]] = 1
        else:
            first100[dboo] = 1
    firstboo = first100 > 0

    return firstboo


def _mosttrials_bool(meta):
    """
    Helper function to get a boolean vector of the first n trials for each day.
    n = the number of trials on the shortest day.
    """

    day_lengths = []
    days = meta.reset_index()['date'].unique()
    for di in days:
        dboo = meta.reset_index()['date'].isin([di]).values
        day_lengths.append(np.sum(dboo))
    trialn = np.min(day_lengths)

    first100 = np.zeros((len(meta)))
    for di in days:
        dboo = meta.reset_index()['date'].isin([di]).values
        first100[np.where(dboo)[0][:trialn]] = 1
    firstboo = first100 > 0

    mouse = meta.reset_index()['mouse'].unique()[0]
    print('{}: {} trials are being used to maximize possible trials each day'.
          format(mouse, trialn))

    return firstboo


def _mostlatetrials_bool(meta):
    """
    Helper function to get a boolean vector of the first n trials for each day.
    n = the number of trials on the shortest day.
    """

    day_lengths = []
    days = meta.reset_index()['date'].unique()
    for di in days:
        dboo = meta.reset_index()['date'].isin([di]).values
        day_lengths.append(np.sum(dboo))
    trialn = np.min(day_lengths)

    first100 = np.zeros((len(meta)))
    for di in days:
        dboo = meta.reset_index()['date'].isin([di]).values
        first100[np.where(dboo)[0][-trialn:]] = 1
    firstboo = first100 > 0

    mouse = meta.reset_index()['mouse'].unique()[0]
    print('{}: {} trials are being used to maximize possible trials each day'.
          format(mouse, trialn))

    return firstboo
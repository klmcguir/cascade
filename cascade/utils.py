"""Functions for general calculations and data management."""
import flow
import pool
import numpy as np
import warnings
import pandas as pd
from . import tca

def correct_nonneg(ensemble):
    """
    Helper function that takes a tensortools ensemble and adds forces cell
    factors to be positive by flipping trial factors when needed. This is
    needed because .rebalance() from tensortools can flip sign when fitting
    has been done with negative modes allowed.
    """

    for method in ensemble:
        for r in ensemble[method].results:
            for i in range(len(ensemble[method].results[r])):
                neg_cellfac_vec = np.sum(
                    ensemble[method].results[r][i].factors[0], axis=0)
                if np.any(neg_cellfac_vec < 0):
                    flip_facs = list(np.where(neg_cellfac_vec < 0)[0])
                    for fac in flip_facs:
                        ensemble[method].results[r][i].factors[0][:, fac] = \
                           ensemble[method].results[r][i].factors[0][:, fac]*-1
                        ensemble[method].results[r][i].factors[2][:, fac] = \
                           ensemble[method].results[r][i].factors[2][:, fac]*-1
    
    return ensemble


def add_dprime_to_meta(meta):
    """
    Helper function that takes a pd metadata dataframe and adds in an extra
    column of the dprime calculated per day.
    """

    # meta can only be a data frame of a single mouse
    assert len(meta.reset_index()['mouse'].unique()) == 1

    # collect useful variables
    new_dprime = np.zeros(len(meta))
    days = meta.reset_index()['date']
    unique_days = days.unique()
    mouse = meta.reset_index()['mouse']

    # loop over unique days filling in dprime for all trial per day at once
    for di in unique_days:
        day_bool = days == di
        mi = mouse[day_bool].unique()[0]
        new_dprime[day_bool] = pool.calc.performance.dprime(
                                flow.Date(mouse=mi, date=di),
                                hmm_engaged=True)

    # save new_dprime into meta
    meta['dprime'] = new_dprime

    return meta


def update_naive_meta(meta, verbose=True):

    """
    Helper function that takes a pd metadata dataframe and makes sure that cses
    and trial error match between naive and learning.
    """
    meta = update_naive_cs(meta, verbose=verbose)
    meta = update_naive_trialerror(meta, verbose=verbose)

    return meta


def update_naive_cs(meta, verbose=True):
    """
    Helper function that takes a pd metadata dataframe and makes sure that cses
    match between naive and learning learning_state.
    """

    # cses to check, pavlovians etc. will remain the same
    cs_list = ['plus', 'minus', 'neutral']

    # original dataframe columns
    orientation = meta['orientation']
    condition = meta['condition']
    learning_state = meta['learning_state']

    # get correct cs-ori pairings
    learning_cs = condition[learning_state == 'learning']
    learning_ori = orientation[learning_state == 'learning']
    cs_codes = {}
    for cs in cs_list:
        ori = np.unique(learning_ori[learning_cs == cs])[0]
        cs_codes[ori] = cs

    # make sure not to mix in other run types (i.e., )
    naive_pmn = condition.isin(cs_list) & (learning_state == 'naive')

    # update metadate
    for ori, cs in cs_codes.items():
        meta.loc[naive_pmn & (orientation == ori), 'condition'] = cs

    if verbose:
        print('Updated naive cs-ori pairings to match learning.')
        for k, v in cs_codes.items():
            print('    ', k, v)

    return meta


def update_naive_trialerror(meta, verbose=True):
    """
    Helper function that takes a pd metadata dataframe and makes sure that 
    trialerror match between naive and learning learning_state.
    Note: CSs must be correct for naive data already otherwise it will not
    affect trialerror values.
    Note: Ignores pavlovians. 
    """

    # cses to check, pavlovians etc. will remain the same
    cs_list = ['plus', 'minus', 'neutral']

    # original dataframe columns
    meta['condition'] = meta['trialerror']
    condition = meta['condition']
    learning_state = meta['learning_state']

    # make sure not to mix in other run types (i.e., )
    naive_pmn = (condition.isin(cs_list) & (learning_state == 'naive')).values

    # create a corrected vector of trialerrors
    naive_te = meta['trialerror'].values[naive_pmn]
    naive_cs = meta['condition'].values[naive_pmn]
    new_te = []
    for te, cs in zip(naive_te, naive_cs):
        if cs == 'plus':
            if te % 2 == 0:
                new_te.append(0)
            else:
                new_te.append(1)
        elif cs == 'neutral':
            if te % 2 == 0:
                new_te.append(2)
            else:
                new_te.append(3)
        elif cs == 'minus':
            if te % 2 == 0:
                new_te.append(4)
            else:
                new_te.append(5)
        else:
            new_te.append(np.nan)
    meta.iloc[naive_pmn, 'trialerror'] = np.array(new_te)

    if verbose:
        print('Updated naive trialerror to match learning.')

    return meta


def getdailycstraces(
        # DateSorter params
        DateSorter,

        # cstrace params
        cs='',
        trace_type='zscore_day',
        start_time=-1,
        end_time=6,
        downsample=True,
        clean_artifacts=None,
        thresh=17.5,
        warp=False,
        smooth=False,
        smooth_win=5,
        smooth_win_dec=3):
    """
    Wrapper function for getcstraces. Gets cstraces for a DateSorter object.
    """

    if isinstance(DateSorter, flow.sorters.Date):
        runs = DateSorter.runs(
            run_types='training', tags='hungry', exclude_tags=['bad'])

    runlist = []
    for run in runs:
        trs = getcstraces(
                run, cs=cs, trace_type=trace_type,
                start_time=start_time, end_time=end_time,
                downsample=downsample, clean_artifacts=clean_artifacts,
                thresh=thresh, warp=warp, smooth=smooth,
                smooth_win=smooth_win, smooth_win_dec=smooth_win_dec)
        runlist.append(trs)
    cstraces = np.concatenate(runlist, axis=2)

    return cstraces


def getdailymeta(
        DateSorter,
        tags='hungry',
        run_types='training'):
    """
    Wrapper function for tca._trialmetafromrun(run). Gets trial metadata for a
    DateSorter object.
    """
    if isinstance(DateSorter, flow.sorters.Date):
        runs = DateSorter.runs(
            run_types=run_types, tags=tags, exclude_tags=['bad'])

    metalist = []
    for run in runs:
        metalist.append(tca._trialmetafromrun(run))
    meta = pd.concat(metalist, axis=0)

    return meta


def getcstraces(
        run,
        cs='',
        trace_type='zscore_day',
        start_time=-1,
        end_time=6,
        downsample=True,
        clean_artifacts=None,
        thresh=17.5,
        warp=False,
        smooth=True,
        smooth_win=6,
        smooth_win_dec=3,
        exclude_tags=('disengaged', 'orientation_mapping', 'contrast', 'retinotopy', 'sated')):
    """
    Wrapper function for flow.Trace2P.cstraces() or .warpsctraces().
    Adds in artifact removal, and multiple types of z-score calc.

    Parameters
    ----------
    run : Run object
    cs : str
        Type of CS. e.g., plus, minus, neutral, 0, 135, 270, ...
    trace_type : str
        dff, zscore, zscore_iti, deconvolved
    downsample : bool
        Downsample from 31 to 15 Hz sampling rate
    clean_artifacts : str
        nan, interp; Remove huge artifacts in dff traces by interpolating
        or adding in nan values
        Note: setting either value here will cause zscoring to nan artifacts
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

    Result
    ------
    np.ndarray
        ncells x frames x nstimuli/onsets

    """

    t2p = run.trace2p()
    date = run.parent
    date.set_subset(run.cells)

    # standardize: z-score
    if 'zscore' in trace_type.lower():

        # get dff for creation of alternative trace_types
        traces = t2p.trace('dff')

        # clean artifacts
        if clean_artifacts:
            nanpad = np.zeros(np.shape(traces))
            nanpad[np.abs(traces) > thresh] = 1
            print(np.sum(nanpad.flatten()))
            # dialate around threshold crossings
            for cell in range(np.shape(traces)[0]):
                nanpad[cell, :] = np.convolve(nanpad[cell, :], np.ones(3), mode='same')
            # clear with nans or interpolation
            if clean_artifacts.lower() == 'nan':
                traces[nanpad != 0] = np.nan
            elif clean_artifacts.lower() == 'interp':
                x = np.arange(0, np.shape(traces)[1])
                for cell in range(np.shape(traces)[0]):
                    # x = np.where(np.isfinite(run_traces[cell, :]))[0]
                    if np.nansum(nanpad[cell, :]) > 0:
                        blank = np.where(nanpad[cell, :] != 0)[0]
                        keep = np.where(nanpad[cell, :] == 0)[0]
                        traces[cell, blank] = np.interp(x[blank], x[keep],
                                                        traces[cell, keep])

        # z-score
        if 'zscore' in trace_type.lower():
            arti = False if clean_artifacts is None else True
            if 'zscore_day' in trace_type.lower():
                mu = pool.calc.zscore.mu(date, exclude_tags=exclude_tags, nan_artifacts=arti,
                                         thresh=thresh)
                sigma = pool.calc.zscore.sigma(date, exclude_tags=exclude_tags,
                                               nan_artifacts=arti,
                                               thresh=thresh)
            elif 'zscore_iti' in trace_type.lower():
                mu = pool.calc.zscore.iti_mu(date, exclude_tags=exclude_tags,
                                             window=4,
                                             nan_artifacts=arti, thresh=thresh)
                sigma = pool.calc.zscore.iti_sigma(date,
                                                   exclude_tags=exclude_tags,
                                                   window=4,
                                                   nan_artifacts=arti,
                                                   thresh=thresh)
            elif 'zscore_run' in trace_type.lower():
                mu = pool.calc.zscore.run_mu(run, nan_artifacts=arti,
                                             thresh=thresh)
                sigma = pool.calc.zscore.run_sigma(run, nan_artifacts=arti,
                                                   thresh=thresh)
            else:
                print('WARNING: did not recognize z-scoring method.')
            traces = ((traces.T - mu)/sigma).T

        # smooth data
        # should always be even to treat both 15 and 30 Hz data equivalently
        assert smooth_win % 2 == 0
        if smooth and (t2p.d['framerate'] > 30):
            for cell in range(np.shape(traces)[0]):
                traces[cell, :] = np.convolve(
                    traces[cell, :], np.ones(smooth_win,
                    dtype=np.float64)/smooth_win, 'same')
        elif smooth and (t2p.d['framerate'] < 16):
            for cell in range(np.shape(traces)[0]):
                traces[cell, :] = np.convolve(
                    traces[cell, :], np.ones(int(smooth_win/2),
                    dtype=np.float64)/(smooth_win/2), 'same')

        # add new trace type into t2p
        t2p.add_trace(trace_type, traces)

    # normalize: (X-min)/(max)
    elif '_norm' in trace_type.lower():
        arti = False if clean_artifacts is None else True  # artifact removal

        # get dff for creation of alternative trace_types
        traces = t2p.trace('dff')

        # subtract the min and divide by max of stimulus windows
        mx = pool.calc.zscore.stim_max(date, window=5, nan_artifacts=arti,
                                       thresh=thresh)
        mn = pool.calc.zscore.stim_min(date, window=5, nan_artifacts=arti,
                                       thresh=thresh)
        traces = ((traces.T - mn)/mx).T

        # smooth traces
        # should always be even to treat both 15 and 30 Hz data equivalently
        assert smooth_win % 2 == 0
        if smooth and (t2p.d['framerate'] > 30):
            for cell in range(np.shape(traces)[0]):
                traces[cell, :] = np.convolve(
                    traces[cell, :], np.ones(smooth_win,
                    dtype=np.float64)/smooth_win, 'same')
        elif smooth and (t2p.d['framerate'] < 16):
            for cell in range(np.shape(traces)[0]):
                traces[cell, :] = np.convolve(
                    traces[cell, :], np.ones(int(smooth_win/2),
                    dtype=np.float64)/(smooth_win/2), 'same')

        # add new trace type into t2p
        t2p.add_trace(trace_type, traces)

    # trigger all trials around stimulus onsets
    if warp:
        run_traces = t2p.warpcstraces(cs, start_s=start_time, end_s=end_time,
                                      trace_type=trace_type, cutoff_before_lick_ms=-1,
                                      errortrials=-1, baseline=(-1, 0),
                                      move_outcome_to=4, baseline_to_stimulus=True)
    else:
        run_traces = t2p.cstraces(cs, start_s=start_time, end_s=end_time,
                                  trace_type=trace_type, cutoff_before_lick_ms=-1,
                                  errortrials=-1, baseline=(-1, 0),
                                  baseline_to_stimulus=True)

    # downsample all traces/timestamps to 15Hz if framerate is 31Hz
    if (t2p.d['framerate'] > 30) and downsample:
        # make sure divisible by 2
        sz = np.shape(run_traces)  # dims: (cells, time, trials)
        if sz[1] % 2 == 1:
            run_traces = run_traces[:, :-1, :]
            sz = np.shape(run_traces)
        # downsample
        # ignore python and numpy divide by zero warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with np.errstate(invalid='ignore', divide='ignore'):
                ds_traces = np.zeros((sz[0], int(sz[1]/2), sz[2]))
                for trial in range(sz[2]):
                    a = run_traces[:, :, trial].reshape(sz[0], int(sz[1]/2), 2)
                    if trace_type.lower() == 'deconvolved':
                        ds_traces[:, :, trial] = np.nanmax(a, axis=2)
                    else:
                        ds_traces[:, :, trial] = np.nanmean(a, axis=2)

        run_traces = ds_traces

    # smooth deconvolved data
    if smooth and (trace_type.lower() == 'deconvolved'):
        sz = np.shape(run_traces)  # dims: (cells, time, trials)
        run_traces = run_traces.reshape((sz[0], sz[1]*sz[2]))
        for cell in range(sz[0]):
            run_traces[cell, :] = np.convolve(run_traces[cell, :],
                                              np.ones(smooth_win_dec,
                                              dtype=np.float64)/smooth_win_dec,
                                              'same')
        run_traces = run_traces.reshape((sz[0], sz[1], sz[2]))

    # truncate negative values (for NMF)
    if 'trunc_' in trace_type.lower():
        run_traces[run_traces < 0] = 0

    return run_traces


def build_tensor(
        mouse,
        tags=None,

        # overly specific params
        sloppy_OA27=False,  # prevents using days reversal days themselves

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
        smooth_win=6,
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
    Builds inputs for tensor component analysis (TCA) without running TCA.

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
        days = flow.DateSorter.frommeta(
            mice=[mouse], tags='naive', exclude_tags=['bad'])
        days.extend(
            flow.DateSorter.frommeta(
                mice=[mouse], tags='learning', exclude_tags=['bad']))
        dates = set(days)
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
        dates = set(days)
        exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                        'retinotopy', 'sated', 'learning_start',
                        'reversal1_start')

    elif group_by.lower() == 'all':
        tags = None
        use_dprime = False
        if mouse == 'OA27':
            if sloppy_OA27:
                exclude_tags = ('disengaged', 'orientation_mapping',
                                'contrast', 'retinotopy', 'sated')
            else:
                exclude_tags = ('disengaged', 'orientation_mapping',
                                'contrast', 'retinotopy', 'sated',
                                'learning_start',
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
        days = flow.DateSorter(
            dates=dates, exclude_tags=['bad'])
    else:
        days = flow.DateSorter.frommeta(
            mice=[mouse], tags=tags, exclude_tags=['bad'])

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
                run_traces = getcstraces(
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

                # skip if you had no trials of interest on this run
                if len(dfr) == 0:
                    continue

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

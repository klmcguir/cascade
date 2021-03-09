"""Functions for general calculations and data management."""
import flow
import pool
import numpy as np
import warnings
import pandas as pd
from . import lookups, bias, tca
from copy import deepcopy


def simple_mean_per_day(meta, tensor, meta_bool=None,
                        filter_running=None, filter_licking=None, filter_hmm_engaged=False):
    """
    Helper function to take the mean across days for a tensor.

    :param meta: pandas.DataFrame, trial metadata
    :param tensor: numpy.ndarray, a cells x times X trials
    :return: new_tensor: numpy.ndarray, a cells x times X days
    """

    # optionally filter
    if meta_bool is None:
        meta_bool = np.ones(len(meta)) > 0
    meta_bool = filter_meta_bool(meta, meta_bool,
                           filter_running=filter_running,
                           filter_licking=filter_licking,
                           filter_hmm_engaged=filter_hmm_engaged)

    # get average response per day
    days = meta.reset_index()['date'].unique()
    new_tensor = np.zeros((tensor.shape[0], tensor.shape[1], len(days)))
    new_tensor[:] = np.nan
    for c, di in enumerate(meta.reset_index()['date'].unique()):
        day_boo = meta.reset_index()['date'].isin([di]).values
        new_tensor[:, :, c] = np.nanmean(tensor[:, :, (day_boo & meta_bool)], axis=2)

    return new_tensor


def simple_mean_per_run(meta, tensor, meta_bool=None,
                        filter_running=None, filter_licking=None, filter_hmm_engaged=False):
    """
    Helper function to take the mean across runs for a tensor.

    :param meta: pandas.DataFrame, trial metadata
    :param tensor: numpy.ndarray, a cells x times X trials
    :return: new_tensor: numpy.ndarray, a cells x times X days
    """

    # optionally filter
    if meta_bool is None:
        meta_bool = np.ones(len(meta)) > 0
    meta_bool = filter_meta_bool(meta, meta_bool,
                           filter_running=filter_running,
                           filter_licking=filter_licking,
                           filter_hmm_engaged=filter_hmm_engaged)

    # get average response per day
    n_run_days = meta.groupby(['mouse', 'date', 'run']).nunique().shape[0]
    new_tensor = np.zeros((tensor.shape[0], tensor.shape[1], n_run_days))
    new_tensor[:] = np.nan

    for c, (mdr, _) in enumerate(meta.groupby(['mouse', 'date', 'run'])):
        day_boo = meta.reset_index()['date'].isin([mdr[1]]).values
        run_boo = meta.reset_index()['run'].isin([mdr[2]]).values
        new_tensor[:, :, c] = np.nanmean(tensor[:, :, (run_boo & day_boo & meta_bool)], axis=2)

    return new_tensor


def simple_mean_per_stage(meta, tensor, staging='parsed_11stage', meta_bool=None,
                          filter_running=None, filter_licking=None, filter_hmm_engaged=False):
    """
    Helper function to take the mean across a stage for a tensor.

    :param meta: pandas.DataFrame, trial metadata
    :param tensor: numpy.ndarray, a cells x times X trials
    :return: new_tensor: numpy.ndarray, a cells x times X stages
    """

    # staging must exist in your df
    meta = add_stages_to_meta(meta, staging)
    assert staging in meta.columns

    # optionally filter
    if meta_bool is None:
        meta_bool = np.ones(len(meta)) > 0
    meta_bool = filter_meta_bool(meta, meta_bool,
                           filter_running=filter_running,
                           filter_licking=filter_licking,
                           filter_hmm_engaged=filter_hmm_engaged)

    # get average response per stage
    stages = lookups.staging[staging]
    new_tensor = np.zeros((tensor.shape[0], tensor.shape[1], len(stages)))
    new_tensor[:] = np.nan
    for c, di in enumerate(stages):
        stage_boo = meta[staging].isin([di]).values
        new_tensor[:, :, c] = np.nanmean(tensor[:, :, stage_boo & meta_bool], axis=2)

    return new_tensor


def balanced_mean_per_stage(meta, tensor, staging='parsed_11stage', meta_bool=None,
                          filter_running=None, filter_licking=None, filter_hmm_engaged=False):
    """
    Helper function to take the mean across a stage (by day) for a tensor.

    :param meta: pandas.DataFrame, trial metadata
    :param tensor: numpy.ndarray, a cells x times X trials
    :return: new_tensor: numpy.ndarray, a cells x times X stages
    """

    # staging must exist in your df
    meta = add_stages_to_meta(meta, staging)
    assert staging in meta.columns

    # optionally filter
    if meta_bool is None:
        meta_bool = np.ones(len(meta)) > 0
    meta_bool = filter_meta_bool(meta, meta_bool,
                           filter_running=filter_running,
                           filter_licking=filter_licking,
                           filter_hmm_engaged=filter_hmm_engaged)

    # get average response per stage
    stages = lookups.staging[staging]
    new_tensor = np.zeros((tensor.shape[0], tensor.shape[1], len(stages)))
    new_tensor[:] = np.nan
    for c, di in enumerate(stages):
        stage_boo = meta[staging].isin([di]).values
        stage_days = meta.loc[stage_boo, :].reset_index()['date'].unique()
        day_means = np.zeros((tensor.shape[0], tensor.shape[1], len(stage_days)))
        day_means[:] = np.nan
        for c2, di2 in enumerate(stage_days):
            day_boo = meta.reset_index()['date'].isin([di2]).values
            day_means[:, :, c2] = np.nanmean(tensor[:, :, stage_boo & day_boo & meta_bool], axis=2)  ## stage added here
        new_tensor[:, :, c] = np.nanmean(day_means[:, :, :], axis=2)

    return new_tensor


def balanced_func_per_stageday(meta, tensor, staging='parsed_11stage', meta_bool=None, func=np.nanmean,
                          filter_running=None, filter_licking=None, filter_hmm_engaged=False):
    """
    Helper function to take the (func) across a stage (by day) for a tensor.

    :param meta: pandas.DataFrame, trial metadata
    :param tensor: numpy.ndarray, a cells x times X trials
    :return: new_tensor: numpy.ndarray, a cells x times X stages
    """

    # staging must exist in your df
    meta = add_stages_to_meta(meta, staging)
    assert staging in meta.columns

    # optionally filter
    if meta_bool is None:
        meta_bool = np.ones(len(meta)) > 0
    meta_bool = filter_meta_bool(meta, meta_bool,
                           filter_running=filter_running,
                           filter_licking=filter_licking,
                           filter_hmm_engaged=filter_hmm_engaged)

    # get average response per stage
    stages = lookups.staging[staging]
    new_tensor = np.zeros((tensor.shape[0], tensor.shape[1], len(stages)))
    new_tensor[:] = np.nan
    stage_mats = []
    for c, di in enumerate(stages):
        stage_boo = meta[staging].isin([di]).values
        stage_days = meta.loc[stage_boo, :].reset_index()['date'].unique()
        day_means = np.zeros((tensor.shape[0], tensor.shape[1], len(stage_days)))
        day_means[:] = np.nan
        for c2, di2 in enumerate(stage_days):
            day_boo = meta.reset_index()['date'].isin([di2]).values
            day_means[:, :, c2] = func(tensor[:, :, stage_boo & day_boo & meta_bool], axis=2)  # think about this impact.
        stage_mats.append(day_means)
        # new_tensor[:, :, c] = np.nanmean(day_means[:, :, :], axis=2)

    return stage_mats


def tensor_mean_per_day(meta, tensor, initial_cue=True, cue='plus', ignore_cue=False, nan_licking=False):
    """
    Helper function to calculate mean per day for axis of same length as meta for a single cue.
    """

    # meta can only be a DataFrame of a single mouse
    assert len(meta.reset_index()['mouse'].unique()) == 1

    # make sure that cue is a list, this allows string or list input
    if isinstance(cue, str):
        cue = [cue]
    assert isinstance(cue, list)

    # make sure that the date vec allows for 0.5 days at reversal and learning
    meta = update_meta_date_vec(meta)

    # choose to average over initial_condition (orientation) or condition (switches orientation at reversal)
    if initial_cue:
        cue_vec = meta['initial_condition']
    else:
        cue_vec = meta['condition']

    # optionally ignore cue, this is useful if you are passing a preferred tuning tensor
    if not ignore_cue:
        cue_bool = cue_vec.isin(cue)
    else:
        cue_bool = cue_vec.notna()

    # optionally nan all times after licking (median cs plus lick latency for non-lick trials)
    if nan_licking:
        mask = bias.get_lick_mask(meta, tensor)
        ablated_tensor = deepcopy(tensor)
        ablated_tensor[~mask] = np.nan
    else:
        ablated_tensor = tensor

    # get average response per day to a single cue
    days = meta.reset_index()['date'].unique()
    new_tensor = np.zeros((tensor.shape[0], tensor.shape[1], len(days)))
    new_tensor[:] = np.nan
    for c, di in enumerate(meta.reset_index()['date'].unique()):
        day_boo = meta.reset_index()['date'].isin([di]).values
        new_tensor[:, :, c] = np.nanmean(ablated_tensor[:, :, day_boo & cue_bool], axis=2)

    return new_tensor


def tensor_mean_per_trial(meta, tensor, nan_licking=False, account_for_offset=False, offset_bool=None,
                          stim_start_s=0, stim_off_s=None, off_start_s=0, off_off_s=None):
    """
    Helper function to calculate mean per trial for a single mouse, correctly accounting for stimulus length.
    Optionally buffer times around licking. Can also choose to automatically parse if a cell's peak activity is
    during the response window and use response window as the trial mean for that offset cells.

    :param meta: pandas.DataFrame
        Trial metadata.
    :param tensor: numpy.ndarray
        A cells x times X trials matrix of neural data.
    :param nan_licking: boolean
        Optionally remove licking by masking with nans.
    :param account_for_offset: boolean
        Take mean for offset or onset window depending on when a cell is driven.
    :param offset_bool: boolean
        Optionally pass Boolean vector, the same length and order as ids, and tensor --> [cells, :, :].
        True = Offset cells. False = not Offset cells.
    :param stim_start_s: float
        Time after stimulus onset start stimulus period. Seconds.
    :param stim_off_s: float
        Time after stimulus onset to end stimulus period. Seconds.
    :param off_start_s: float
        Time after stimulus offset to start response period. Seconds.
    :param off_off_s: float
        Time after stimulus offset to end response period. Seconds.
    :return:
    """

    # meta can only be a DataFrame of a single mouse
    assert len(meta.reset_index()['mouse'].unique()) == 1
    mouse = meta.reset_index()['mouse'].unique()[0]

    # assume 15.5 Hz sampling or downsampling for 7 seconds per trial (n = 108 timepoints)
    assert tensor.shape[1] == 108
    times = np.arange(-1, 6, 1 / 15.5)[:108]
    stim_end = lookups.stim_length[mouse] if stim_off_s is None else stim_off_s
    stim_bool = (times > 0 + stim_start_s) & (times < stim_end)
    resp_end = lookups.stim_length[mouse] + 2 if off_off_s is None else lookups.stim_length[mouse] + off_off_s
    response_bool = (times > lookups.stim_length[mouse] + off_start_s) & (times < resp_end)  # 0ms delay

    # make sure that the date vec allows for 0.5 days at reversal and learning
    meta = update_meta_date_vec(meta)

    # optionally determine cells with offset responses
    if account_for_offset:
        if offset_bool is None:
            offset_bool = get_offset_cells(meta, tensor)

    # optionally nan all times after licking (median cs plus lick latency for non-lick trials)
    if nan_licking:
        mask = bias.get_lick_mask(meta, tensor)
        ablated_tensor = deepcopy(tensor)
        ablated_tensor[~mask] = np.nan
    else:
        ablated_tensor = tensor

    # get average response per trial
    if account_for_offset:
        new_mat = np.zeros((ablated_tensor.shape[0], ablated_tensor.shape[2]))
        if np.sum(offset_bool) > 0:
            new_mat[offset_bool, :] = np.nanmean(ablated_tensor[offset_bool, :, :][:, response_bool, :], axis=1)
        if np.sum(~offset_bool) > 0:
            new_mat[~offset_bool, :] = np.nanmean(ablated_tensor[~offset_bool, :, :][:, stim_bool, :], axis=1)
    else:
        new_mat = np.nanmean(ablated_tensor[:, stim_bool, :], axis=1)

    return new_mat


def tensor_mean_baselines_per_trial(meta, tensor, nan_licking=False):
    """
    Helper function to calculate mean baseline per trial meta for a single mouse.
    Optionally buffer times around licking.
    """

    # meta can only be a DataFrame of a single mouse
    assert len(meta.reset_index()['mouse'].unique()) == 1

    # assume 15.5 Hz sampling or downsampling for 7 seconds per trial (n = 108 timepoints)
    assert tensor.shape[1] == 108
    times = np.arange(-1, 6, 1 / 15.5)[:108]
    base_bool = (times < 0)

    # make sure that the date vec allows for 0.5 days at reversal and learning
    meta = update_meta_date_vec(meta)

    # optionally nan all times after licking (median cs plus lick latency for non-lick trials)
    if nan_licking:
        mask = bias.get_lick_mask(meta, tensor)
        ablated_tensor = deepcopy(tensor)
        ablated_tensor[~mask] = np.nan
    else:
        ablated_tensor = tensor

    # get average response per trial
    new_mat = np.nanmean(ablated_tensor[:, base_bool, :], axis=1)

    return new_mat


def tensor_mean_per_stage(meta, tensor, initial_cue=True, cue='plus', nan_licking=False, staging='parsed_11stage'):
    """
    Helper function to calculate mean per stage for axis of same length as meta for a single cue.
    """

    # meta can only be a DataFrame of a single mouse
    assert len(meta.reset_index()['mouse'].unique()) == 1

    # make sure that cue is a list, this allows string or list input
    if isinstance(cue, str):
        cue = [cue]
    assert isinstance(cue, list)

    # make sure that the date vec allows for 0.5 days at reversal and learning
    meta = update_meta_date_vec(meta)

    # make sure parsed stage exists
    meta = add_stages_to_meta(meta, staging)

    # choose to average over initial_condition (orientation) or condition (switches orientation at reversal)
    if initial_cue:
        cue_vec = meta['initial_condition']
    else:
        cue_vec = meta['condition']
    cue_bool = cue_vec.isin(cue)

    # optionally nan all times after licking (median cs plus lick latency for non-lick trials)
    if nan_licking:
        mask = bias.get_lick_mask(meta, tensor)
        ablated_tensor = deepcopy(tensor)
        ablated_tensor[~mask] = np.nan
    else:
        ablated_tensor = tensor

    # get average response per stage to a single cue
    # stages = lookups.staging[staging]
    # new_tensor = np.zeros((tensor.shape[0], tensor.shape[1], len(stages)))
    # new_tensor[:] = np.nan
    # for c, di in enumerate(stages):
    #     stage_boo = meta[staging].isin([di]).values
    #     new_tensor[:, :, c] = np.nanmean(ablated_tensor[:, :, stage_boo & cue_bool], axis=2)

    new_tensor = balanced_mean_per_stage(meta.loc[cue_bool], tensor[:, :, cue_bool], staging=staging)

    return new_tensor


def tensor_mean_per_stage_filtered(meta, tensor, initial_cue=True, cue='plus', nan_licking=False,
                                      staging='parsed_11stage', filter=None):
    """
    Helper function to calculate mean per stage for axis of same length as meta for a single cue.
    Conditional aspect is that you can filter on metadata columns.
    """

    # meta can only be a DataFrame of a single mouse
    assert len(meta.reset_index()['mouse'].unique()) == 1

    # make sure that cue is a list, this allows string or list input
    if isinstance(cue, str):
        cue = [cue]
    assert isinstance(cue, list)

    # make sure that the date vec allows for 0.5 days at reversal and learning
    meta = update_meta_date_vec(meta)

    # make sure parsed stage exists
    meta = add_stages_to_meta(meta, staging)

    # choose to average over initial_condition (orientation) or condition (switches orientation at reversal)
    if initial_cue:
        cue_vec = meta['initial_condition']
    else:
        cue_vec = meta['condition']
    cue_bool = cue_vec.isin(cue)

    # additional filtering
    if filter is not None:
        if 'high_running' in filter:
            raise NotImplementedError

    # optionally nan all times after licking (median cs plus lick latency for non-lick trials)
    if nan_licking:
        mask = bias.get_lick_mask(meta, tensor)
        ablated_tensor = deepcopy(tensor)
        ablated_tensor[~mask] = np.nan
    else:
        ablated_tensor = tensor

    # get average response per day to a single cue
    stages = lookups.staging[staging]
    new_tensor = np.zeros((tensor.shape[0], tensor.shape[1], len(stages)))
    new_tensor[:] = np.nan
    for c, di in enumerate(stages):
        stage_boo = meta[staging].isin([di]).values
        new_tensor[:, :, c] = np.nanmean(ablated_tensor[:, :, stage_boo & cue_bool], axis=2)

    return new_tensor


def tensor_mean_per_stage_single_pt(meta, tensor, account_for_offset=True, **kwargs):
    """
    Single data point for each trial. Take the mean of the stimulus window, or preferred window for that cell
    (i.e., offset cells are averaged following the stimulus offset, during the response window).

    :param meta: pandas.DataFrame
        Tensor trial metadata.
    :param tensor: numpy.ndarray
        Matrix organized like this: tensor[cells, time points, trials].
    :param account_for_offset : boolean
        Optionally take mean of activity based on peak activity. Uses stimulus or response window.
    :param kwargs: takes kwargs for tensor_mean_per_stage, defaults:
        initial_cue=True, cue='plus', nan_licking=False, staging='parsed_11stage'
    :return: stage_matrix
        Matrix of cells x stages of learning. Organized like this: stage_matrix[cells, stages]
    """

    # meta can only be a DataFrame of a single mouse
    assert len(meta.reset_index()['mouse'].unique()) == 1
    mouse = meta.reset_index()['mouse'].unique()[0]

    # mean trace per stage
    mtensor = tensor_mean_per_stage(meta, tensor, **kwargs)

    # assume 15.5 Hz sampling or downsampling for 7 seconds per trial (n = 108 timepoints)
    assert mtensor.shape[1] == 108
    times = np.arange(-1, 6, 1 / 15.5)[:108]
    stim_bool = (times > 0) & (times < lookups.stim_length[mouse])
    response_bool = (times > lookups.stim_length[mouse] + 0.0) & (times < lookups.stim_length[mouse] + 2)  # 000ms delay

    # optionally determine cells with offset responses
    if account_for_offset:
        offset_bool = get_offset_cells(meta, tensor)
        # trace_mean = np.nanmean(mtensor, axis=2)
        # offset_bool = np.argmax(trace_mean, axis=1) > 15.5 * (1 + lookups.stim_length[mouse])

    # get average response per trial
    if account_for_offset:
        stage_matrix = np.zeros((mtensor.shape[0], mtensor.shape[2]))
        stage_matrix[offset_bool, :] = np.nanmean(mtensor[offset_bool, :, :][:, response_bool, :], axis=1)
        stage_matrix[~offset_bool, :] = np.nanmean(mtensor[~offset_bool, :, :][:, stim_bool, :], axis=1)
    else:
        stage_matrix = np.nanmean(mtensor[:, stim_bool, :], axis=1)

    return stage_matrix


def get_offset_cells(meta, tensor, buffer_s=0.300):
    """
    Determine cells with offset responses.

    Note: meta is only used to determine mouse name.

    :return: offset_bool: boolean
        Vector, true where offset cells exist
    """

    # meta can only be a DataFrame of a single mouse
    assert len(meta.reset_index()['mouse'].unique()) == 1
    mouse = meta.reset_index()['mouse'].unique()[0]

    # determine cells with offset responses
    trace_mean = np.nanmean(tensor, axis=2)
    stim_offset_buffer_start = int(np.floor(15.5 * (1 + lookups.stim_length[mouse] - buffer_s)))
    stim_offset_buffer_end = int(np.floor(15.5 * (1 + lookups.stim_length[mouse] + buffer_s)))
    trace_mean[:, stim_offset_buffer_start:stim_offset_buffer_end] = np.nan  # buffer around offset to avoid GECI tail
    trace_mean[:, :18] = np.nan  # buffer baseline with a few extra frames, ~200 ms

    # account for nans
    offset_bool = np.zeros(trace_mean.shape[0])
    offset_bool[:] = np.nan
    nan_boo = ~np.isnan(trace_mean[:, 18])
    offset_bool[nan_boo] = np.nanargmax(trace_mean[nan_boo, :], axis=1)
    offset_bool = offset_bool > 15.5 * (1 + lookups.stim_length[mouse])
    # WARNING using inverse will mean nans are counted as stimulus cells

    return offset_bool


def get_peak_times(tensor):
    """
    Determine peak response time

    Note: meta is only used to determine mouse name.

    :return: peak_times: float
        Peak time in seconds relative to stimulus onset
    """

    # tensor must be 15.5 Hz use a proxy assertion for now
    assert tensor.shape[1] == 108  # 108 is the number of frames for 15.5 Hz

    # determine cells with offset responses
    trace_mean = np.nanmean(tensor, axis=2)
    trace_mean[:, :16] = np.nan  # buffer baseline
    peak_frames = np.nanargmax(trace_mean, axis=1)
    peak_times = (peak_frames - 15.5)/15.5

    return peak_times


def correct_nonneg(ensemble):
    """
    Helper function that takes a tensortools ensemble and forces cell
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
                            ensemble[method].results[r][i].factors[0][:, fac] * -1
                        ensemble[method].results[r][i].factors[2][:, fac] = \
                            ensemble[method].results[r][i].factors[2][:, fac] * -1

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


def add_dprime_run_to_meta(meta):
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
    runs = meta.reset_index()['run']

    # loop over unique days filling in dprime for all trial per day at once
    for di in unique_days:
        day_bool = days == di
        mi = mouse[day_bool].unique()[0]
        for ri in runs[day_bool].unique():
            run_bool = runs == ri
            run_bool = day_bool & run_bool
            new_dprime[run_bool] = pool.calc.performance.dprime_run(
                flow.Run(mouse=mi, date=di, run=ri),
                hmm_engaged=True)

    # save new_dprime into meta
    meta['dprime_run'] = new_dprime

    return meta


def add_firstlick_wmedian_to_meta(meta):
    """
    Helper function that adds in the median lick time for all plus trials for trials with
    no, or slow licking. If the actual time to lick is shorter latency then that is used.
    """

    # make sure that the date vec allows for 0.5 days at reversal and learning
    meta = update_meta_date_vec(meta)
    mouse = meta.reset_index()['mouse'].unique()[0]

    # buffer = 1/15.5*2 = 129 ms
    buffer_ms = 2

    # get lick latency in terms of frames
    lick = meta['firstlick'].values
    cs = meta['condition'].values
    new_lick = np.zeros(len(lick))

    # find median lick latency for all plus trials (looking across stim and response window)
    last_trial_frame = (1 + 2 + lookups.stim_length[mouse]) * 15.5  # last frame of response window
    last_stim_frame = (1 + lookups.stim_length[mouse]) * 15.5  # last frame of stimulus window
    median_for_plus = np.nanmedian(lick[(lick < last_trial_frame) & np.isin(cs, 'plus')])
    new_lick[:] = median_for_plus

    # break if you will have less than 350 ms of datapoints for your bias calculation
    assert median_for_plus > 21

    # add in existing licks with 129 ms buffer before them, only update lick latency on plus trials
    # for all other trials use the median lick latency of plus trials or first lick, whichever is sooner
    lick_in_window_boo = lick < last_stim_frame
    lick_before_median = lick < median_for_plus
    new_lick[lick_in_window_boo & np.isin(cs, 'plus')] = lick[lick_in_window_boo & np.isin(cs, 'plus')] - buffer_ms
    new_lick[lick_before_median & ~np.isin(cs, 'plus')] = lick[lick_before_median & ~np.isin(cs, 'plus')]
    meta['firstlick_med'] = new_lick

    return meta


def add_firstlickbout_wmedian_to_meta(meta):
    """
    Helper function that adds in the median lick time for all plus trials for trials with
    no, or slow licking. If the actual time to lick is shorter latency then that is used.
    """

    # make sure that the date vec allows for 0.5 days at reversal and learning
    meta = update_meta_date_vec(meta)
    mouse = meta.reset_index()['mouse'].unique()[0]

    # buffer = 1/15.5*2 = 129 ms
    buffer_ms = 2

    # get lick latency in terms of frames
    lick = meta['firstlickbout'].values
    cs = meta['condition'].values
    new_lick = np.zeros(len(lick))

    # find median lick latency for all plus trials (looking across stim and response window)
    last_trial_frame = (1 + 2 + lookups.stim_length[mouse]) * 15.5  # last frame of response window
    last_stim_frame = (1 + lookups.stim_length[mouse]) * 15.5  # last frame of stimulus window
    median_for_plus = np.nanmedian(lick[(lick < last_trial_frame) & np.isin(cs, 'plus')])
    new_lick[:] = median_for_plus

    # break if you will have less than 350 ms of datapoints for your bias calculation
    assert median_for_plus > 21

    # add in existing licks with 129 ms buffer before them, only update lick latency on plus trials
    # for all other trials use the median lick latency of plus trials or first lick, whichever is sooner
    lick_in_window_boo = lick < last_stim_frame
    lick_before_median = lick < median_for_plus
    new_lick[lick_in_window_boo & np.isin(cs, 'plus')] = lick[lick_in_window_boo & np.isin(cs, 'plus')] - buffer_ms
    new_lick[lick_before_median & ~np.isin(cs, 'plus')] = lick[lick_before_median & ~np.isin(cs, 'plus')]
    meta['firstlickbout_med'] = new_lick

    return meta


def add_firstlick_wmedian_run_to_meta(meta):
    """
    Helper function that adds in the median lick time for all plus trials for trials with
    no, or slow licking. Median licking is calculated for each imaging session. In general this method
    seems like it punishes high dprime behavior because animals with incredibly fast lick latency on individual
    sessions have very little neural data left for neutral and minus trials calculated this way.
    """

    # make sure that the date vec allows for 0.5 days at reversal and learning
    meta = update_meta_date_vec(meta)
    mouse = meta.reset_index()['mouse'].unique()[0]

    # buffer = 1/15.5*2 = 129 ms
    buffer_ms = 2

    # get vector for looping over
    u_days = np.unique(meta.reset_index()['date'])
    all_days = meta.reset_index()['date'].values

    # get lick latency in terms of frames
    lick = meta['firstlick'].values
    cs = meta['condition'].values
    new_lick = np.zeros(len(lick))

    # find median lick latency for all plus trials
    median_for_plus = np.zeros(len(meta))
    last_trial_frame = (1 + 2 + lookups.stim_length[mouse]) * 15.5
    last_stim_frame = (1 + lookups.stim_length[mouse]) * 15.5
    day_vec = meta.reset_index()['date']
    run_vec = meta.reset_index()['run']
    for di in day_vec.unique():
        day_bool = day_vec.isin([di])
        curr_runs = run_vec.iloc[day_bool.values]
        for ri in curr_runs.unique():
            run_bool = day_bool & run_vec.isin([ri])
            median_for_plus[run_bool] = np.nanmedian(lick[(lick < last_trial_frame) & np.isin(cs, 'plus') & run_bool])
    new_lick = median_for_plus

    # break if you will have less than 300 ms of datapoints for your bias calculation
    assert np.nanmin(median_for_plus) > 21

    # add in existing licks with 129 ms buffer before them
    lick_in_window_boo = lick < last_stim_frame
    new_lick[lick_in_window_boo] = lick[lick_in_window_boo] - buffer_ms
    meta['firstlick_med_run'] = new_lick

    return meta


def add_prev_ori_cols_to_meta(meta):
    """
    Helper function to add in useful columns to metadata related to
    previous orientation presentations. This relies on prev_cue presentations
    in metadata, which will still correctly handle dropped pavlovians. Meaning
    a previous pavlovian cue will still be counted as a previous plus cue
    presentation even if pavlovians were exluded. This uses this CS info to
    create similar columns for orientation presentations. 
    """

    # meta can only be a data frame of a single mouse
    assert len(meta.reset_index()['mouse'].unique()) == 1

    # boolean for cues preceded by the same cue
    prev_same_boo = (meta['prev_same_plus'].gt(0)
                     | meta['prev_same_neutral'].gt(0)
                     | meta['prev_same_minus'].gt(0))

    # create column for each ori if it was preceded by the same ori
    for ori in [0, 135, 270]:
        curr_ori_bool = meta['orientation'].isin([ori]).values
        new_meta = {f'prev_same_{ori}': np.zeros(len(meta))}
        new_meta[f'prev_same_{ori}'][prev_same_boo & curr_ori_bool] = 1
        new_meta_df = pd.DataFrame(data=new_meta, index=meta.index)
        meta = pd.concat([meta, new_meta_df], axis=1)

    return meta


def add_reversal_mismatch_condition_to_meta(meta):
    """
    Helper function to add in useful columns to metadata. Renames orientations based on
    the type of mismatch that occurs at reversal.

    Possible types: 'becomes_rewarded', 'becomes_unrewarded', 'remains_unrewarded'.
    """

    # meta can only be a data frame of a single mouse
    assert len(meta.reset_index()['mouse'].unique()) == 1

    # must have a reversal to add a mismatch column
    if not meta['learning_state'].isin(['reversal1']).any():
        meta['mismatch_condition'] = ['none'] * len(meta)
        return meta

    # create column for each ori if it was preceded by the same ori
    new_mapping = {}
    for ori in [0, 135, 270]:
        curr_ori_bool = meta['orientation'].isin([ori]).values
        list_of_conds = meta['condition'].loc[curr_ori_bool].unique()  # pandas unique is in order of appearance
        list_of_conds = [s for s in list_of_conds if 'naive' != s]

        # get new naming convention
        if len(list_of_conds) == 2:
            if list_of_conds[0] == 'plus':
                new_mapping[ori] = 'becomes_unrewarded'
            else:
                if list_of_conds[1] == 'plus':
                    new_mapping[ori] = 'becomes_rewarded'
                else:
                    new_mapping[ori] = 'remains_unrewarded'
        elif len(list_of_conds) == 1:
            # deal with sub-case where Arthur's mice keep same minus cue across reversal
            assert list_of_conds[0] == 'minus'
            new_mapping[ori] = 'remains_unrewarded'
        else:
            raise NotImplementedError

    # add column to meta
    new_mismatch_conditions = [new_mapping[s] for s in meta['orientation'].values]
    meta['mismatch_condition'] = new_mismatch_conditions

    return meta


def add_cue_prob_to_meta(meta):
    """
    Helper function to add in useful columns to metadata related to
    trial history. 
    """

    # meta can only be a data frame of a single mouse
    assert len(meta.reset_index()['mouse'].unique()) == 1
    mouse = meta.reset_index()['mouse'].unique()[0]

    # add a binary column for choice, 1 for go 0 for nogo
    new_meta = {}
    new_meta['go'] = np.zeros(len(meta))
    new_meta['go'][meta['trialerror'].isin([0, 3, 5, 7]).values] = 1
    new_meta_df1 = pd.DataFrame(data=new_meta, index=meta.index)
    # new_meta_df1 = pd.concat([new_meta_df1, new_meta_df], axis=1)

    # add a binary column for reward
    new_meta = {}
    new_meta['reward'] = np.zeros(len(new_meta_df1))
    new_meta['reward'][meta['trialerror'].isin([0]).values] = 1
    new_meta_df = pd.DataFrame(data=new_meta, index=meta.index)
    new_meta_df1 = pd.concat([new_meta_df1, new_meta_df], axis=1)

    # add a binary column for punishment
    new_meta = {}
    new_meta['punishment'] = np.zeros(len(new_meta_df1))
    new_meta['punishment'][meta['trialerror'].isin([5]).values] = 1
    new_meta_df = pd.DataFrame(data=new_meta, index=meta.index)
    new_meta_df1 = pd.concat([new_meta_df1, new_meta_df], axis=1)

    # add a binary column for reward
    # new_meta = {}
    # new_meta['prev_reward'] = np.zeros(len(new_meta_df1))
    # new_meta['prev_reward'][meta['prev_reward'].values] = 1
    # new_meta_df = pd.DataFrame(data=new_meta, index=meta.index)
    # new_meta_df1 = pd.concat([new_meta_df1, new_meta_df], axis=1)
    #
    # # add a binary column for punishment
    # new_meta = {}
    # new_meta['prev_punishment'] = np.zeros(len(new_meta_df1))
    # new_meta['prev_punishment'][meta['prev_punishment'].values] = 1
    # new_meta_df = pd.DataFrame(data=new_meta, index=meta.index)
    # new_meta_df1 = pd.concat([new_meta_df1, new_meta_df], axis=1)

    # rename oris according to their meaning during learning
    for ori in ['plus', 'minus', 'neutral']:
        new_meta = {}
        new_meta['initial_{}'.format(ori)] = np.zeros(len(new_meta_df1))
        new_meta['initial_{}'.format(ori)][meta['orientation'].isin([lookups.lookup[mouse][ori]]).values] = 1
        new_meta_df = pd.DataFrame(data=new_meta, index=meta.index)
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
    for aci in ['initial_plus', 'initial_minus', 'initial_neutral', 'go', 'reward']:
        accumulated_df = new_meta_df1.groupby('{}_cum'.format(aci)).sum()
        prob_since_last = accumulated_df.divide(accumulated_df['trial_number'], axis=0)
        for vali in ['initial_plus', 'initial_minus', 'initial_neutral', 'go', 'reward']:
            new_vec = np.zeros(len(new_meta_df1))
            new_vec[:] = np.nan
            new_bool = new_meta_df1[aci].gt(0).values
            new_vec[new_bool] = prob_since_last[vali].values[0:np.sum(new_bool)]  # use only matched trials
            new_meta_df1['p_{}_since_last_{}'.format(vali, aci)] = new_vec
            p_cols.append('p_{}_since_last_{}'.format(vali, aci))

    # turn go, reward, punishment into boolean
    grp_df = new_meta_df1.loc[:, ['go', 'reward', 'punishment']].gt(0)

    # get only columns for probability
    p_df = new_meta_df1.loc[:, p_cols]

    # add new columns to original df
    new_dfr = pd.concat([meta, grp_df, p_df], axis=1)

    return new_dfr


def update_meta_date_vec(meta):
    """
    Helper function to get change date vector in metadata to be .5 for learning
    and reversal transitions in the middle of the day.
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

    # replace
    new_meta = meta.reset_index()
    new_meta['date'] = day_vec
    new_meta = new_meta.set_index(['mouse', 'date', 'run', 'trial_idx'])

    return new_meta


def add_sub_stages_to_meta(meta, staging='parsed_11stage', bins_per_stage=10):
    """
    Add a column that breaks each stage into 10 bins.

    :param meta: pandas.DataFrame, trial metadata
    :param staging: str, type of binning to be used to define stages of learning
    :param bins_per_stage: int, number of bins to break a stage into
    :return: meta, now with two new columns
    """
    if staging not in meta.columns:
        meta = add_stages_to_meta(meta, staging)

    bin_vec = np.zeros(len(meta))
    for stage in meta[staging].unique():
        stage_boo = meta[staging].isin([stage])
        stage_inds = np.where(stage_boo)[0]
        bin_size = int(np.floor(len(stage_inds)/bins_per_stage))
        # bins = np.arange(stage_inds[0], stage_inds[-1] + bin_size, bin_size)
        for bi in range(bins_per_stage):

            # account for floor rounding above for last bin
            if bi == bins_per_stage - 1:
                ind_chunk = stage_inds[bi * bin_size:]
            else:
                ind_chunk = stage_inds[bi * bin_size:(bi+1) * bin_size]
            bin_vec[ind_chunk] = bi

    meta['stage_bins'] = bin_vec
    return meta


def add_dprime_sub_stages_to_meta(meta, staging='parsed_11stage', bins_per_stage=10, dp_by_run=True):
    """
    Add a column that breaks each stage into 10 bins.

    :param meta: pandas.DataFrame, trial metadata
    :param staging: str, type of binning to be used to define stages of learning
    :param bins_per_stage: int, number of bins to break a stage into
    :return: meta, now with two new columns
    """
    if staging not in meta.columns:
        meta = add_stages_to_meta(meta, staging)

    if dp_by_run:
        if 'dprime_run' not in meta.columns:
            meta = add_dprime_run_to_meta(meta)
        dp = meta['dprime_run'].values
    else:
        if 'dprime' not in meta.columns:
            meta = add_dprime_to_meta(meta)
        dp = meta['dprime'].values

    bin_vec = np.zeros(len(meta))
    for stage in meta[staging].unique():
        stage_boo = meta[staging].isin([stage])
        stage_dp_sort = np.argsort(dp[stage_boo])
        stage_inds = np.where(stage_boo)[0][stage_dp_sort]
        bin_size = int(np.floor(len(stage_inds)/bins_per_stage))
        # bins = np.arange(stage_inds[0], stage_inds[-1] + bin_size, bin_size)
        for bi in range(bins_per_stage):

            # account for floor rounding above for last bin
            if bi == bins_per_stage - 1:
                ind_chunk = stage_inds[bi * bin_size:]
            else:
                ind_chunk = stage_inds[bi * bin_size:(bi+1) * bin_size]
            bin_vec[ind_chunk] = bi

    meta['stage_bins'] = bin_vec
    return meta


def add_numeric_stages_to_meta(meta, staging='parsed_11stage'):
    """
    Take you staging column and make it numeric easy quick visualizations etc.

    :param meta: pandas.DataFrame, trial metadata
    :param staging: str, type of binning to be used to define stages of learning
    :return: meta, now with one new column
    """
    if staging not in meta.columns:
        meta = add_stages_to_meta(meta, staging)

    bin_vec = np.zeros(len(meta))
    for c, stage in enumerate(lookups.staging[staging]):
        stage_boo = meta[staging].isin([stage])
        bin_vec[stage_boo] = c
    meta['numeric_stage'] = bin_vec

    return meta


def add_stages_to_meta(meta, staging, dp_by_run=True, simple=False, bin_scale=0.75, force=False):
    """
    Helper function to allow single function to check and create other staging vectors.

    :param meta: pandas.DataFrame, trial metadata
    :param staging: str, type of binning to be used to define stages of learning
    :param dp_by_run: boolean, use dprime calculated over runs (instead of over days)
    :param simple: boolean, for 10stage
            simple=True: Divide trials in a stage right down the middle.
            simple=False: Divide days. Single days are NOT broken in half. They are
            attributed to the later period. This is to prevent groups of cells that
            change dramatically within a day from affecting results if they are highly
            responsive at the beginning of the day but not at the end.
    :param bin_scale: float, for 11stage, width of dprime bins
    :param force: boolean, force recalculation of staging column
    :return: meta, now with one new column
    """
    # make sure parsed stage exists so you can loop over this.
    # add learning stages to meta
    if 'parsed_stage' in staging:
        if 'parsed_stage' not in meta.columns or force:
            meta = add_5stages_to_meta(meta, dp_by_run=dp_by_run)
    if 'parsed_10stage' in staging:
        if 'parsed_10stage' not in meta.columns or force:
            meta = add_10stages_to_meta(meta, dp_by_run=dp_by_run, simple=simple)
    if 'parsed_11stage' in staging:
        if 'parsed_11stage' not in meta.columns or force:
            meta = add_11stages_to_meta(meta, dp_by_run=dp_by_run, bin_scale=bin_scale)

    return meta


def add_5stages_to_meta(meta, dp_by_run=True):
    """
    Helper function to add the stage of learning to metadata.
    """

    if 'dprime' not in meta.columns:
        meta = add_dprime_to_meta(meta)
    if 'dprime_run' not in meta.columns:
        meta = add_dprime_run_to_meta(meta)
    meta = update_meta_date_vec(meta)

    ls = meta['learning_state'].values
    if dp_by_run:
        dp = meta['dprime_run'].values
    else:
        dp = meta['dprime'].values

    stage_vec = []
    for lsi, dpi in zip(ls, dp):

        if 'naive' in lsi:
            stage_vec.append('naive')
        elif 'learning' in lsi:
            if dpi < 2:
                stage_vec.append('low_dp learning')
            elif dpi >= 2:
                stage_vec.append('high_dp learning')
        elif 'reversal1' in lsi:
            if dpi < 2:
                stage_vec.append('low_dp reversal1')
            elif dpi >= 2:
                stage_vec.append('high_dp reversal1')

    meta['parsed_stage'] = stage_vec

    return meta


def add_11stages_to_meta(meta, dp_by_run=True, bin_scale=0.75):
    """
    Helper function to add the stage of learning to metadata. Uses evenly spaced dprime bins.
    Naive is treated as a single bin.
    """

    if 'dprime' not in meta.columns:
        meta = add_dprime_to_meta(meta)
    if 'dprime_run' not in meta.columns:
        meta = add_dprime_run_to_meta(meta)
    meta = update_meta_date_vec(meta)

    ls = meta['learning_state'].values
    if dp_by_run:
        dp = meta['dprime_run'].values
    else:
        dp = meta['dprime'].values

    stage_vec = []
    for lsi, dpi in zip(ls, dp):

        if 'naive' in lsi:
            stage_vec.append('L0 naive')
        elif 'learning' in lsi:
            if dpi <= 1 * bin_scale:
                stage_vec.append('L1 learning')
            elif 1 * bin_scale < dpi <= 2 * bin_scale:
                stage_vec.append('L2 learning')
            elif 2 * bin_scale < dpi <= 3 * bin_scale:
                stage_vec.append('L3 learning')
            elif 3 * bin_scale < dpi <= 4 * bin_scale:
                stage_vec.append('L4 learning')
            elif dpi > 4 * bin_scale:
                stage_vec.append('L5 learning')
        elif 'reversal1' in lsi:
            if dpi <= 1 * bin_scale:
                stage_vec.append('L1 reversal1')
            elif 1 * bin_scale < dpi <= 2 * bin_scale:
                stage_vec.append('L2 reversal1')
            elif 2 * bin_scale < dpi <= 3 * bin_scale:
                stage_vec.append('L3 reversal1')
            elif 3 * bin_scale < dpi <= 4 * bin_scale:
                stage_vec.append('L4 reversal1')
            elif dpi > 4 * bin_scale:
                stage_vec.append('L5 reversal1')

    meta['parsed_11stage'] = stage_vec

    return meta


def add_10stages_to_meta(meta, simple=False, dp_by_run=True):
    """
    Helper function to add the stage of learning to metadata breaking
    each of the 5 major stages ['naive', 'low_dp learning', 'high_dp learning',
    'low_dp reversal1', 'high_dp reversal1'] into an early and late stages.
    
    Can choose to divide trials in half or days in half.

    simple=True: Divide trials in a stage right down the middle.

    simple=False: Divide days. Single days are NOT broken in half. They are
    attributed to the later period. This is to prevent groups of cells that
    change dramatically within a day from affecting results if they are highly
    responsive at the beginning of the day but not at the end.

    """

    # make sure parsed stage exists so you can loop over this.
    if 'parsed_stage' not in meta.columns:
        meta = add_5stages_to_meta(meta, dp_by_run=dp_by_run)
    meta = update_meta_date_vec(meta)

    # get days and parsed stages of learning
    u_stages = meta['parsed_stage'].unique()
    parse = meta['parsed_stage']

    # get days or run-days if dp is being set by training run/session
    if dp_by_run:
        # run number must be less than 10 for decimals for runs to work (+.2 for run 2, etc)
        assert all(meta.reset_index()['run'].unique() < 10)
        all_days = meta.reset_index()['date'] + meta.reset_index()['run'] / 10
    else:
        all_days = meta.reset_index()['date']

    stage_vec = []
    for ic, istage in enumerate(u_stages):

        # simple=False; break a stage in half but assign shared days to later
        # period. i.e., if there are 3 days, 1 day is early and 2 are late.
        if simple:
            stage_bool = parse.isin([istage]).values
            stage_inds = np.where(stage_bool)[0]
            midpoint = int(np.ceil(len(stage_inds) / 2))
            first_half = stage_inds[:midpoint]
            last_half = stage_inds[midpoint:]

            # add the appropriate number of stage values to the list
            # s is not used, it is just important that the loop run this many iteration
            for s in first_half:
                stage_vec.append('early {}'.format(istage))
            for s in last_half:
                stage_vec.append('late {}'.format(istage))
        else:
            stage_bool = parse.isin([istage]).values
            stage_days = all_days.iloc[stage_bool].unique()
            day_bool = np.isin(all_days.values, stage_days)

            # if a stage only has one day consider it late
            if len(stage_days) == 1:
                stage_inds = np.where(day_bool)[0]
                for s in stage_inds:
                    stage_vec.append('late {}'.format(istage))
            else:
                day_mid = int(np.floor(len(stage_days) / 2))
                first_days = stage_days[:day_mid]
                last_days = stage_days[day_mid:]
                day_bool1 = np.isin(all_days.values, first_days)
                day_bool2 = np.isin(all_days.values, last_days)
                clean_first_inds = np.where(day_bool1)[0]
                clean_last_inds = np.where(day_bool2)[0]
                # s is not used, it is just important that the loop run this many iteration
                for s in clean_first_inds:
                    stage_vec.append('early {}'.format(istage))
                for s in clean_last_inds:
                    stage_vec.append('late {}'.format(istage))

    meta['parsed_10stage'] = stage_vec

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
    try:
        learning_cs = condition[learning_state == 'learning']
        learning_ori = orientation[learning_state == 'learning']
        cs_codes = {}
        for cs in cs_list:
            ori = np.unique(learning_ori[learning_cs == cs])[0]
            cs_codes[ori] = cs
    except IndexError:
        cs_codes = lookups.lookup_ori[meta_mouse(meta)]

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
    meta.at[naive_pmn, 'trialerror'] = new_te

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
        smooth_win_dec=12,
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

    # always exclude bad runs
    exclude_tags = exclude_tags + ('bad',)

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
            traces = ((traces.T - mu) / sigma).T

        # smooth data
        # should always be even to treat both 15 and 30 Hz data equivalently
        assert smooth_win % 2 == 0
        if smooth and (t2p.d['framerate'] > 30):
            for cell in range(np.shape(traces)[0]):
                traces[cell, :] = np.convolve(
                    traces[cell, :], np.ones(smooth_win,
                                             dtype=np.float64) / smooth_win, 'same')
        elif smooth and (t2p.d['framerate'] < 16):
            for cell in range(np.shape(traces)[0]):
                traces[cell, :] = np.convolve(
                    traces[cell, :], np.ones(int(smooth_win / 2),
                                             dtype=np.float64) / (smooth_win / 2), 'same')

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
        traces = ((traces.T - mn) / mx).T

        # smooth traces
        # should always be even to treat both 15 and 30 Hz data equivalently
        assert smooth_win % 2 == 0
        if smooth and (t2p.d['framerate'] > 30):
            for cell in range(np.shape(traces)[0]):
                traces[cell, :] = np.convolve(
                    traces[cell, :], np.ones(smooth_win,
                                             dtype=np.float64) / smooth_win, 'same')
        elif smooth and (t2p.d['framerate'] < 16):
            for cell in range(np.shape(traces)[0]):
                traces[cell, :] = np.convolve(
                    traces[cell, :], np.ones(int(smooth_win / 2),
                                             dtype=np.float64) / (smooth_win / 2), 'same')

        # add new trace type into t2p
        t2p.add_trace(trace_type, traces)

    # trigger all trials around stimulus onsets
    if warp:
        if 'deconvolved' in trace_type.lower() or '_nobs' in trace_type.lower():
            run_traces = t2p.warpcstraces(cs, start_s=start_time, end_s=end_time,
                                          trace_type=trace_type, cutoff_before_lick_ms=-1,
                                          errortrials=-1, baseline=None,
                                          move_outcome_to=5)
        else:
            run_traces = t2p.warpcstraces(cs, start_s=start_time, end_s=end_time,
                                          trace_type=trace_type, cutoff_before_lick_ms=-1,
                                          errortrials=-1, baseline=(-1, 0),
                                          move_outcome_to=5, baseline_to_stimulus=True)
    else:
        if 'deconvolved' in trace_type.lower() or '_nobs' in trace_type.lower():
            # double check that your deconvolved traces match your aligned data.
            if t2p.ncells == t2p.trace('deconvolved').shape[0]:
                run_traces = t2p.cstraces(cs, start_s=start_time, end_s=end_time,
                                          trace_type=trace_type, cutoff_before_lick_ms=-1,
                                          errortrials=-1, baseline=None)
            else:
                ntimes = 108 if (t2p.d['framerate'] < 30) else 216
                run_traces = np.zeros((t2p.ncells, ntimes, t2p.ntrials))
                run_traces[:] = np.nan
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
                ds_traces = np.zeros((sz[0], int(sz[1] / 2), sz[2]))
                for trial in range(sz[2]):
                    a = run_traces[:, :, trial].reshape(sz[0], int(sz[1] / 2), 2)
                    if 'deconvolved' in trace_type.lower():
                        ds_traces[:, :, trial] = np.nanmax(a, axis=2)
                    else:
                        ds_traces[:, :, trial] = np.nanmean(a, axis=2)

        run_traces = ds_traces

    # bin data
    if '_bin' in trace_type.lower():
        assert 'deconvolved' in trace_type.lower(), 'Only deconvolved traces can be binned'
        bin_factor = 4
        # make sure divisible by bin factor
        sz = np.shape(run_traces)  # dims: (cells, time, trials)
        if sz[1] % bin_factor > 0:
            mod = sz[1] % bin_factor
            run_traces = run_traces[:, :-mod, :]
            sz = np.shape(run_traces)
        # bin
        # ignore python and numpy divide by zero warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with np.errstate(invalid='ignore', divide='ignore'):
                ds_traces = np.zeros((sz[0], int(sz[1] / bin_factor), sz[2]))
                for trial in range(sz[2]):
                    a = run_traces[:, :, trial].reshape(
                        sz[0], int(sz[1] / bin_factor), bin_factor)
                    ds_traces[:, :, trial] = np.nansum(a, axis=2)

        run_traces = ds_traces

    # smooth deconvolved data
    if smooth and 'deconvolved' in trace_type.lower():
        sz = np.shape(run_traces)  # dims: (cells, time, trials)
        run_traces = run_traces.reshape((sz[0], sz[1] * sz[2]))
        for cell in range(sz[0]):
            run_traces[cell, :] = np.convolve(run_traces[cell, :],
                                              np.ones(smooth_win_dec,
                                                      dtype=np.float64) / smooth_win_dec,
                                              'same')
        run_traces = run_traces.reshape((sz[0], sz[1], sz[2]))

    # invert values (to explicitly model inhibition)
    if '_flip' in trace_type.lower():
        run_traces = run_traces * -1

    # shift baselines slightly positive
    if '_bump' in trace_type.lower():
        run_traces += 0.1

    # truncate negative values (for NMF)
    if '_trunc' in trace_type.lower():
        run_traces[run_traces < 0] = 0

    # only look at stimulus period
    if '_onset' in trace_type.lower():
        time_to_off = lookups.stim_length[run.mouse] + 1
        assert downsample
        frames_to_off = int(np.floor(time_to_off * 15.5))
        run_traces = run_traces[:, :frames_to_off, :]

    # cap positive values for deconvolution at 2
    if '_cap' in trace_type.lower():
        run_traces[run_traces > 2] = 2
        # if any entire trial now equals 2 set to nan
        trial_mins = run_traces.min(axis=1)
        for rt_cells in range(trial_mins.shape[0]):
            for rt_trials in range(trial_mins.shape[1]):
                if trial_mins[rt_cells, rt_trials] == 2:
                    run_traces[rt_cells, :, rt_trials] = np.nan

    return run_traces


def getcsbehavior(
        run,
        cs='',
        trace_type='pupil',
        start_time=-1,
        end_time=6,
        downsample=True,
        baseline=None,
        cutoff_before_lick_ms=-1):
    """
    Wrapper function for flow.Trace2P.csbehaviortraces().
    Downsamples traces which are not at the correct sampling frequency.

    Parameters
    ----------
    run : Run object
    cs : str
        Type of CS. e.g., plus, minus, neutral, 0, 135, 270, ...
    trace_type : str
        pupil, speed, etc.
    downsample : bool
        Downsample from 31 to 15 Hz sampling rate

    Result
    ------
    np.ndarray
        behavior x frames x nstimuli/onsets

    """

    t2p = run.trace2p()

    run_traces = t2p.csbehaviortraces(cs, start_s=start_time, end_s=end_time,
                                      trace_type=trace_type,
                                      cutoff_before_lick_ms=cutoff_before_lick_ms,
                                      errortrials=-1, baseline=baseline,
                                      baseline_to_stimulus=True)

    # downsample all traces/timestamps to 15Hz if framerate is 31Hz
    if (t2p.d['framerate'] > 30) and downsample and (trace_type.lower() not in ['pupil']):
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
                ds_traces = np.zeros((sz[0], int(sz[1] / 2), sz[2]))
                for trial in range(sz[2]):
                    a = run_traces[:, :, trial].reshape(sz[0], int(sz[1] / 2), 2)
                    if trace_type.lower() == 'deconvolved':
                        ds_traces[:, :, trial] = np.nanmax(a, axis=2)
                    else:
                        ds_traces[:, :, trial] = np.nanmean(a, axis=2)

        run_traces = ds_traces

    return run_traces


def sortfactors(my_method):
    """
    Sort a set of neuron factors by which factor they contribute
    to the most.

    Input
    -------
    Tensortools ensemble with method. (set of multiple initializations of TCA)
        i.e., _sortfactors(ensemble['ncp_bcd'])

    Returns
    -------
    my_method, copy of tensortools ensemble method now with neuron factors sorted.
    my_rank_sorts, sort indexes to keep track of cell identity

    """

    my_method = deepcopy(my_method)

    # only use the lowest error replicate, index 0, to define sort order
    rep_num = 0

    # keep sort indexes because these define original cell identity
    my_rank_sorts = []

    # sort each neuron factor and update the order based on strongest factor
    # reflecting prioritized sorting of earliest to latest factors
    for k in my_method.results.keys():

        full_sort = []
        # use the lowest index (and lowest error objective) to create sort order
        factors = my_method.results[k][rep_num].factors[0]

        # sort neuron factors according to which component had highest weight
        max_fac = np.argmax(factors, axis=1)
        sort_fac = np.argsort(max_fac)
        sort_max_fac = max_fac[sort_fac]
        first_sort = factors[sort_fac, :]

        # descending sort within each group of sorted neurons
        second_sort = []
        for i in np.unique(max_fac):
            second_inds = (np.where(sort_max_fac == i)[0])
            second_sub_sort = np.argsort(first_sort[sort_max_fac == i, i])
            second_sort.extend(second_inds[second_sub_sort][::-1])

        # apply the second sort
        full_sort = sort_fac[second_sort]
        sorted_factors = factors[full_sort, :]

        # check for zero-weight factors
        no_weight_binary = np.max(sorted_factors, axis=1) == 0
        no_weight_binary = np.max(sorted_factors, axis=1) == 0
        inds_to_end = full_sort[no_weight_binary]
        full_sort = np.concatenate(
            (full_sort[np.invert(no_weight_binary)], inds_to_end), axis=0)
        my_rank_sorts.append(full_sort)

        # reorder factors looping through each replicate and applying the same sort
        for i in range(0, len(my_method.results[k])):
            factors = my_method.results[k][i].factors[0]
            sorted_factors = factors[full_sort, :]
            my_method.results[k][i].factors[0] = sorted_factors

    return my_method, my_rank_sorts


def rescale_factors(factors):
    """Rescale factors so that last two modes max at 1.

    Parameters
    ----------
    factors : tensortools.KTensor, list, or tuple
        Factors from TCA, usually with 3 indices for each mode of TCA. 
    """
    factors = deepcopy(factors)
    temp_max = np.nanmax(factors[1], axis=0)
    tune_max = np.nanmax(factors[2], axis=0)
    scaled_cells = factors[0] * temp_max * tune_max
    scaled_traces = factors[1] / temp_max
    scaled_tune = factors[2] / tune_max
    scaled_factors = [scaled_cells, scaled_traces, scaled_tune]

    return scaled_factors


def sort_and_rescale_factors(mod_ensemble):
    """
    Sort a set of neuron factors by which factor they contribute
    to the most.

    Input
    -------
    Tensortools ensemble with method. (set of multiple initializations of TCA)
        i.e., _sortfactors(ensemble['ncp_bcd'])

    Returns
    -------
    my_method, copy of tensortools ensemble method now with neuron factors sorted.
    my_rank_sorts, sort indexes to keep track of cell identity

    """
    
    # copy so as not to change original model
    mod_ensemble = deepcopy(mod_ensemble)
    
    # rescale your factor so that the full weight is carried on the cell_factor
    for k in mod_ensemble.results.keys():
        for i in range(len(mod_ensemble.results[k])):
            
            # rescale
            factors = mod_ensemble.results[k][i].factors
            scaled_factors = rescale_factors(factors)

            # overwrite your copy of your model with rescaled factors
            for cj, j in enumerate(scaled_factors):
                mod_ensemble.results[k][i].factors[cj] = j
        
    # resort factors so that they are grouped by tuning
    my_tune_sorts = []
    for k in mod_ensemble.results.keys():
        for i in range(len(mod_ensemble.results[k])):
        
            factors = mod_ensemble.results[k][i].factors
            max_tune = np.argmax(factors[2], axis=0)
            sum_tune = np.nansum(factors[2], axis=0)  # sum == 1 if perfectly tuned, == 3 if broad
            
            # move joint and broadly tuned components to end
            max_tune[sum_tune > 1.5] = max_tune[sum_tune > 1.5] + factors[2].shape[1]
            tune_sorting = np.argsort(max_tune)
            if i == 0:
                my_tune_sorts.append(tune_sorting)  # use only top iterations
            
            # apply tune sort
            for j in range(3):
                mod_ensemble.results[k][i].factors[j] = mod_ensemble.results[k][i].factors[j][:, tune_sorting]
    
    
    # sort cells according to highest weight component, then order them highest to lowest within group
    my_rank_sorts = []
    for k in mod_ensemble.results.keys():
        
        # rescale your factor so that the full weight is carried on the cell_factor
        factors = mod_ensemble.results[k][0].factors

        # sort neuron factors according to which component had highest weight
        max_fac = np.argmax(factors[0], axis=1)
        sort_fac = np.argsort(max_fac)
        sort_max_fac = max_fac[sort_fac]
        first_sort = factors[0][sort_fac, :]

        # descending sort within each group of sorted neurons
        second_sort = []
        for i in np.unique(max_fac):
            second_inds = (np.where(sort_max_fac == i)[0])
            second_sub_sort = np.argsort(first_sort[sort_max_fac == i, i])
            second_sort.extend(second_inds[second_sub_sort][::-1])

        # apply the second sort
        full_sort = sort_fac[second_sort]
        sorted_factors = factors[0][full_sort, :]

        # check for zero-weight factors and move them to the end
        no_weight_binary = np.max(sorted_factors, axis=1) == 0
        inds_to_end = full_sort[no_weight_binary]
        full_sort = np.concatenate(
            (full_sort[np.invert(no_weight_binary)], inds_to_end), axis=0)
        
        # save your sort
        my_rank_sorts.append(full_sort)

        # reorder factors looping through each replicate and applying the same sort
        # sort is defined by ITERATION 0 
        for i in range(len(mod_ensemble.results[k])):
            factors =  mod_ensemble.results[k][i].factors[0]
            mod_ensemble.results[k][i].factors[0] = factors[full_sort, :]

    return mod_ensemble, my_rank_sorts, my_tune_sorts


def define_high_weight_cell_factors(model, rank, threshold=1):
    """
    Return the highest weight cluster for a cell. Note: this is an approximation. TCA suffers from
    a scalability issue (e.g., you can divide one factor by a value and multiply another factor by that value and
    the component remains the same). All models are rebalanced when they load.
    TODO: add more about what rebalance means intuitively

    def rebalance(self):
        '''Rescales factors across modes so that all norms match.'''

        # Compute norms along columns for each factor matrix
        norms = [sci.linalg.norm(f, axis=0) for f in self.factors]

        # Multiply norms across all modes
        lam = sci.multiply.reduce(norms) ** (1/self.ndim)

        # Update factors
        self.factors = [f * (lam / fn) for f, fn in zip(self.factors, norms)]
        return self

    :param model: tensortools.Ensemble, TCA model
    :param rank: int, TCA model rank
    :param threshold: int, standard deviation threshold
    :return: best_cluster: numpy.ndarray, vector of components that highest weight
    """

    # parse your model
    cell_factors = model.results[rank][0].factors[0][:, :]

    # find threshold for cells that were never high weight
    thresh = np.std(cell_factors, axis=0) * threshold
    weights = deepcopy(cell_factors)
    for i in range(cell_factors.shape[1]):
        weights[weights[:, i] < thresh[i], i] = np.nan
    above_thresh = ~np.isnan(np.nanmax(weights, axis=1))

    # find highest weight cluster and exclude cells deemed unworthy, above
    best_cluster = np.argmax(cell_factors, axis=1) + 1.  # to make clusters match component number
    best_cluster[~above_thresh] = np.nan

    return best_cluster


def count_high_weight_cell_factors(model, rank, threshold=1):
    """
    Return the count of high weight clusters for a cell as a list. Note: this is an approximation. TCA suffers from
    a scalability issue (e.g., you can divide one factor by a value and multiply another factor by that value and
    the component remains the same). All models are rebalanced when they load.
    TODO: add more about what rebalance means intuitively

    def rebalance(self):
        '''Rescales factors across modes so that all norms match.'''

        # Compute norms along columns for each factor matrix
        norms = [sci.linalg.norm(f, axis=0) for f in self.factors]

        # Multiply norms across all modes
        lam = sci.multiply.reduce(norms) ** (1/self.ndim)

        # Update factors
        self.factors = [f * (lam / fn) for f, fn in zip(self.factors, norms)]
        return self

    :param model: tensortools.Ensemble, TCA model
    :param rank: int, TCA model rank
    :param threshold: int, standard deviation threshold
    :return: participates_in_n_clusters: numpy.ndarray, vector of counts of above thresh clusters per cell
    """

    # parse your model
    cell_factors = model.results[rank][0].factors[0][:, :]

    # find threshold for cells that were never high weight
    thresh = np.std(cell_factors, axis=0) * threshold
    weights = deepcopy(cell_factors)
    for i in range(cell_factors.shape[1]):
        weights[weights[:, i] < thresh[i], i] = np.nan
    participates_in_n_clusters = np.nansum(~np.isnan(weights), axis=1)

    return participates_in_n_clusters


def does_cell_participate_in_offset_component(model, rank, mouse, threshold=1):
    """
    Return the if the cell had an above threshold component weight to any offset component.
    Note: this is an approximation. TCA suffers from a scalability issue (e.g., you can divide one factor by a value
    and multiply another factor by that value and the component remains the same). All models are rebalanced when
    they load.
    TODO: add more about what rebalance means intuitively

    def rebalance(self):
        '''Rescales factors across modes so that all norms match.'''

        # Compute norms along columns for each factor matrix
        norms = [sci.linalg.norm(f, axis=0) for f in self.factors]

        # Multiply norms across all modes
        lam = sci.multiply.reduce(norms) ** (1/self.ndim)

        # Update factors
        self.factors = [f * (lam / fn) for f, fn in zip(self.factors, norms)]
        return self

    :param model: tensortools.Ensemble, TCA model
    :param rank: int, TCA model rank
    :param mouse: str, name of mouse
    :param threshold: int, standard deviation threshold
    :return: participates_in_offset_component: numpy.ndarray, vector of counts of above thresh clusters per cell
    """

    # parse your model
    cell_factors = model.results[rank][0].factors[0][:, :]

    # offset starts at:
    offset = np.argmax(model.results[rank][0].factors[1][:, :], axis=0) > 15.5 * (1 + lookups.stim_length[mouse])

    # find threshold for cells that were never high weight
    thresh = np.std(cell_factors, axis=0) * threshold
    weights = deepcopy(cell_factors)
    for i in range(cell_factors.shape[1]):
        weights[weights[:, i] < thresh[i], i] = np.nan
    participates_in_offset_component = np.sum(~np.isnan(weights[:, offset]), axis=1) > 0

    return participates_in_offset_component


def unwrap_tensor(tensor):
    """
    Unwrap a tensor so that it is organized [cells x (trials/days/stages x times)]. Concatenates together trials/stages/
    days.

    :param tensor: numpy.ndarray
        Matrix organized like this: tensor[cells, time points, trials].
    :return: unwrapped_tensor: numpy.ndarray
        Matrix organized like this: tensor[cells, (time points x trials)].
    """

    epoch_list = []
    for epoch in range(tensor.shape[2]):
        epoch_list.append(tensor[:, :, epoch])
    unwrapped_tensor = np.concatenate(epoch_list, axis=1)

    return unwrapped_tensor


def filter_meta_bool(meta, meta_bool, filter_running=None, filter_licking=None, filter_hmm_engaged=True,
                     high_speed_thresh_cms=10,
                     low_speed_thresh_cms=4,
                     high_lick_thresh_ls=1.7,
                     low_lick_thresh_ls=1.7
                     ):
    """
        Helper function to take a boolean representing some set of trials from metadata and filtering for additional
    features.

    :param meta: pandas.DataFrame, trial metadata
    :param meta_bool: boolean, some set of trials for selection as a boolean

    :param filter_running: str
    :param filter_licking:
    :param filter_hmm_engaged:
    :param high_speed_thresh_cms:
    :param low_speed_thresh_cms:
    :param high_lick_thresh_ls:
    :param low_lick_thresh_ls:
    :return:
    """

    # copy boolean
    meta_bool = deepcopy(meta_bool)
    # TODO could make this start all true so it does not require a meta bool input at all

    # filter to include fixed running type: low or high
    if filter_running is not None:
        speed_cm_s = meta.speed.values
        pre_speed_cm_s = meta.pre_speed.values
        if filter_running == 'low_speed_only':
            meta_bool = meta_bool & (speed_cm_s <= low_speed_thresh_cms)
        elif filter_running == 'high_speed_only':
            meta_bool = meta_bool & (speed_cm_s > high_speed_thresh_cms)
        elif filter_running == 'low_pre_speed_only':
            meta_bool = meta_bool & (pre_speed_cm_s <= low_speed_thresh_cms)
        elif filter_running == 'high_pre_speed_only':
            meta_bool = meta_bool & (pre_speed_cm_s > high_speed_thresh_cms)
        else:
            raise NotImplementedError

    # filter to include fixed licking type: low or high
    if filter_licking is not None:
        mouse = meta.reset_index().mouse.unique()[0]
        # TODO this needs accounting for offset licking for offset cells
        mean_lick_rate = meta.anticipatory_licks.values / lookups.stim_length[mouse]
        pre_lick_rate = meta.pre_licks.values / 1  # one second baseline period
        post_lick_rate = meta.post_licks.values / 2  # two second response period
        if filter_licking == 'low_lick_only':
            meta_bool = meta_bool & (mean_lick_rate <= low_lick_thresh_ls)
        elif filter_licking == 'high_lick_only':
            meta_bool = meta_bool & (mean_lick_rate > high_lick_thresh_ls)
        elif filter_licking == 'low_pre_lick_only':
            meta_bool = meta_bool & (pre_lick_rate <= low_lick_thresh_ls)
        elif filter_licking == 'high_pre_lick_only':
            meta_bool = meta_bool & (pre_lick_rate > high_lick_thresh_ls)
        else:
            raise NotImplementedError

    # animal must be engaged in the task (or naive when it can't "engage")
    if filter_hmm_engaged:
        meta_bool = meta_bool & (meta.hmm_engaged.values | meta.learning_state.isin(['naive']).values)

    return meta_bool


def bin_running_calc(meta, trial_mean_tensor, set1, speed_type='speed', bin_width=3):
    """
    Take in one boolean vector that defines a set of trials. For example, this could be plus trials from a
    single day, that you want to compare to minus trials on the same day. Alternatively, you could pass plus trials on
    day one to be compared to plus trials on day two.

    # TODO could also match running calcs

    :param meta: pandas.DataFrame, DataFrame of trial metadata
    :param trial_mean_tensor: numpy.ndarray, a cells x trials
    :param set1: boolean ,of trials use for binning activity by running speed
    :param speed_type: str, 'speed', 'pre_speed', 'post_speed'
    :return:
    """

    # 3 cm/s bins evenly spaced from 0 to 1 m/s
    speed_bins = np.arange(0, 100, bin_width)

    # get running speed for trials of interest
    set1_speed = meta.loc[set1, speed_type]
    set1_bins = np.digitize(set1_speed, np.arange(0, 100, bin_width), right=True)

    # get tensor for trials of interest
    set1_tensor = trial_mean_tensor[:, set1]

    # Get mean cell responses per bin
    binned_set1 = np.zeros((trial_mean_tensor.shape[0], len(speed_bins)-1))
    binned_set1[:] = np.nan

    for bc, left_bin_edge in enumerate(speed_bins[:-1]):

        # get trials for a running bin
        bin_trials1 = set1_bins == bc

        # take mean of trials for running bin per cell
        binned_set1[:, bc] = np.nanmean(set1_tensor[:, bin_trials1], axis=1)

    return binned_set1


def bin_running_traces_calc(meta, full_tensor, set1, speed_type='speed', bin_width=3):
    """
    Take the mean of a set of trials broken into 3 cm/s bins across running speed.

    Could choose to only use same number of trials in each bin for best matched speeds?

    # TODO could also match running calcs

    :param meta: pandas.DataFrame, DataFrame of trial metadata
    :param full_tensor: numpy.ndarray, a cells x times x trials
    :param set1: boolean ,of trials use for binning activity by running speed
    :param speed_type: str, 'speed', 'pre_speed', 'post_speed'
    :return:
    """

    # 3 cm/s bins evenly spaced from 0 to 1 m/s
    speed_bins = np.arange(0, 100, bin_width)

    # get running speed for trials of interest
    set1_speed = meta.loc[set1, speed_type]
    set1_bins = np.digitize(set1_speed, np.arange(0, 100, bin_width), right=True)

    # get tensor for trials of interest
    set1_tensor = full_tensor[:, :, set1]

    # Get mean cell responses per bin
    binned_set1 = np.zeros((full_tensor.shape[0], full_tensor.shape[1], len(speed_bins)-1))
    binned_set1[:] = np.nan

    for bc, left_bin_edge in enumerate(speed_bins[:-1]):

        # get trials for a running bin
        bin_trials1 = set1_bins == bc

        # take mean of trials for running bin per cell
        binned_set1[:, :, bc] = np.nanmean(set1_tensor[:, :, bin_trials1], axis=2)

    return binned_set1


def df_split(df, split_on='mouse'):
    """
    Helper function to return a list grouping on index or columns using pandas.DataFrame.groupby()
    :param df: pandas.DataFrame
        pandas.DataFrame, probably with an index level that includes mouse
    :param split_on: str or list of str
        column(s) or index level (or levels) you want to use the unique entries of as list indices
    :return: a list of dataframe grouped by the split on category (or categories)
    """

    return [gr for _, gr in df.groupby(split_on)]


def meta_mouse(meta):
    """
    Helper function to get mouse name as str from meta DataFrame.

    :param meta: pandas.DataFrame
        Trial metadata.
    :return: mouse name as str
    """

    # meta can only be a DataFrame of a single mouse
    assert len(meta.reset_index()['mouse'].unique()) == 1
    return meta.reset_index()['mouse'].unique()[0]


def meta_last_learn_day_ind(meta):
    """
    Helper function to get the index of the last learning day.

    :param meta: pandas.DataFrame
        Trial metadata.
    :return: mouse name as str
    """

    # meta can only be a DataFrame of a single mouse
    assert len(meta.reset_index()['mouse'].unique()) == 1
    last_day = meta.loc[meta.learning_state.isin(['learning'])].reset_index()['date'].values[-1]
    days = meta.reset_index()['date'].unique()
    return np.where(days == last_day)[0]


def flip_cell_index(meta):
    """
    Helper function to flip cell indexing from cell_n indexed to cell_id or back
    """

    if 'cell_id' in meta.columns:
        return meta.reset_index().set_index(['mouse', 'cell_id']).sort_index()
    elif 'cell_n' in meta.columns:
        return meta.reset_index().set_index(['mouse', 'cell_n']).sort_index()

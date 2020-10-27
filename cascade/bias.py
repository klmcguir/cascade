"""Functions for calculating FC bias."""
import pandas as pd
import numpy as np
import flow
import pool
import os
from . import utils
from . import lookups
from copy import deepcopy
from tensortools.tensors import KTensor


def get_lick_mask(meta, tensor, buffer_frames=1):
    """
    Create a boolean mask in the shape of your tensor that is 1 wherever data preceded
    a lick and is 0 following licks. Trials with no licking during the stimulus
    window are set to the median lick latency on plus trials.

    :param buffer_frames: int, number of frames to buffer preceding first lick (single frame is ~64.5 ms)
    """

    # make sure that you have a full size 15.5 Hz tensor 
    assert tensor.shape[1] == 108

    # make sure that you have a matching tensor and meta
    assert tensor.shape[2] == len(meta)

    # get lick latency, adjusted with median for trials without licking
    meta = utils.add_firstlick_wmedian_to_meta(meta)
    mask = np.zeros(tensor.shape)
    lick_lat = meta['firstlick_med'].values  # see utils for other options for getting lick latency

    # loop over trials to create your licking mask, setting values to 1
    frame_number = np.arange(tensor.shape[1])
    for tri in range(tensor.shape[2]):
        lick_boo = ((frame_number < lick_lat[tri] - buffer_frames)  # keep frames less than firstlick
                    & (frame_number > 15.5))  # keep frames greater than baseline
        mask[:, lick_boo, tri] = 1

    return mask > 0


def get_bias_from_tensor(meta, tensor, staging='parsed_10stage'):
    """
    Calculate bias for stages of learning for a tensor and meta.
    """
    
    # make sure that you have a full size 15.5 Hz tensor 
    assert tensor.shape[1] == 108

    # make sure that you have a matching tensor and meta
    assert tensor.shape[2] == len(meta)

    # add learning stages to meta
    meta = utils.add_stages_to_meta(meta, staging)

    # make sure that the date includes half days for learning/reversal1
    meta = utils.update_meta_date_vec(meta)

    # get tensor-shaped mask of values that exclude licking 
    # this accounts for differences in stimulus length between mice
    # baseline is also set to false
    mask = get_lick_mask(meta, tensor)

    # baseline period boolean
    timepts = np.arange(tensor.shape[1])
    base_boo = timepts <= 15.5
    # baseline_mean = np.nanmean(tensor[:, base_boo, :], axis=1)

    # get mean for each trial
    ablated_tensor = deepcopy(tensor)
    ablated_tensor[~mask] = np.nan  # this sets baseline to NaN as well
    # stim_mean_nolick = np.nanmean(ablated_tensor, axis=1)

    # get amplitude compared to baseline
    # amplitude_nolick = stim_mean_nolick - baseline_mean
    # return amplitude_nolick, stim_mean_nolick, baseline_mean
    # amplitude_nolick[amplitude_nolick < checkup] = 0  # rectify

    # boolean of first 100 trials per day
    first100_bool = _first100_bool(meta)

    # loop over 5stages
    mean_per_stage = np.zeros((tensor.shape[0], 10, 3))
    stage = meta[staging]
    for cstagi, stagi in enumerate(meta[staging].unique()):
        stage_bool = stage.isin([stagi]).values
        for ccue, cue in enumerate(['plus', 'minus', 'neutral']):
            cue_bool = meta['condition'].isin([cue]).values
            # mean_per_stage[:, cstagi, ccue] = np.nanmean(
            #     amplitude_nolick[:, cue_bool & stage_bool & first100_bool], axis=1)
            # calculate mean response of each cell for each stage of learning
            # first average over trials to help clean up noise from dropped trials and NaNs.
            # next average over timepoints during the stimulus window.
            mean_resp = np.nanmean(
                np.nanmean(ablated_tensor[:, :, cue_bool & stage_bool & first100_bool], axis=2), axis=1)
            # calculate baseline as you would response period overaging over trials then timepoint
            mean_base = np.nanmean(
                np.nanmean(tensor[:, base_boo, :][:, :, cue_bool & stage_bool & first100_bool], axis=2), axis=1)
            mean_amplitude = mean_resp - mean_base
            mean_amplitude[mean_amplitude < 0] = 0  # rectify, little negatives break this calculation
            mean_per_stage[:, cstagi, ccue] = mean_amplitude

    # save mean_per_stage before normalization
    mean_per_stage_raw = deepcopy(mean_per_stage)

    # normalize by the max response
    max_response = np.nanmax(mean_per_stage, axis=2)
    for cue_n in range(mean_per_stage.shape[2]):
        mean_per_stage[:, :, cue_n] = mean_per_stage[:, :, cue_n]/max_response

    # return mean_per_stage

    # rewind ... unwrap your means for a DataFrame
    means_in = []
    means_raw_in = []
    stages_in = []
    cues_in = []
    bias_in = []
    for cstagi, stagi in enumerate(meta[staging].unique()):
        for ccue, cue in enumerate(['plus', 'minus', 'neutral']):
            mean_raw_vec = mean_per_stage_raw[:, cstagi, ccue]
            mean_vec = mean_per_stage[:, cstagi, ccue]
            bias_vec = mean_per_stage[:, cstagi, ccue]/np.nansum(mean_per_stage, axis=2)[:, cstagi]
            bias_in.extend(bias_vec)
            means_in.extend(mean_vec)
            means_raw_in.extend(mean_raw_vec)
            cues_in.extend([cue]*len(mean_vec))
            stages_in.extend([stagi]*len(mean_vec))

    data = {'cue': cues_in, 'learning stage': stages_in,
            'mean response': means_in, 'mean response raw': means_raw_in, 'bias': bias_in}
    mean_df = pd.DataFrame(data)

    return mean_df


def get_bias_from_model_simple(meta, input_tensor, model, model_rank, save_folder='', staging='parsed_10stage'):
    """
    Calculate bias for stages of learning for a tensor and meta on TCA model itself.
    """

    # loop over removing each component from TCA model, -1 is the whole model without ablation

    tensor = model.results[model_rank][0].factors.full()

    # mask values that are NaN in you original data
    mask = np.isnan(input_tensor)
    tensor[mask] = np.nan

    # get bias and mean from TCA model of data
    all_mean_df = get_bias_from_tensor(meta, tensor, staging=staging)

    # save your dataframe before returning
    mouse = meta.reset_index()['mouse'].unique()[0]
    all_mean_df.to_pickle(os.path.join(save_folder, f'{mouse}_rank{model_rank}_model_bias_and_mean_df_simp.pkl'))

    return all_mean_df


def get_bias_from_ablated_data_simple(meta, input_tensor, model, model_rank, save_folder='', staging='parsed_10stage'):
    """
    Calculate bias for stages of learning for a tensor and meta on TCA model itself.
    """

    # create a tensor removing one of you your TCA factors from your data
    tensor = input_tensor - model.results[model_rank][0].factors.full()

    # mask values that are NaN in you original data (this may be redundant for this version)
    mask = np.isnan(input_tensor)
    tensor[mask] = np.nan

    # get bias and mean from TCA model of data
    all_mean_df = get_bias_from_tensor(meta, tensor, staging=staging)

    # save your dataframe before returning
    mouse = meta.reset_index()['mouse'].unique()[0]
    all_mean_df.to_pickle(os.path.join(save_folder, f'{mouse}_rank{model_rank}_data_bias_and_mean_df_full_ablation.pkl'))

    return all_mean_df


def get_bias_from_model(meta, input_tensor, model, model_rank, save_folder='', staging='parsed_10stage'):
    """
    Calculate bias for stages of learning for a tensor and meta on TCA model itself.
    """

    # loop over removing each component from TCA model, -1 is the whole model without ablation
    mean_df_list = []
    for fac_num in range(-1, model_rank):

        # create a tensor from your TCA factors, for -1
        if fac_num == -1:
            tensor = model.results[model_rank][0].factors.full()
            fac_label = 'full'
        else:
            tensor = ablate_Ktensor(model.results[model_rank][0].factors, fac_num)
            fac_label = f'ablated component {fac_num + 1}'

        # mask values that are NaN in you original data
        mask = np.isnan(input_tensor)
        tensor[mask] = np.nan

        # get bias and mean from TCA model of data
        mean_df = get_bias_from_tensor(meta, tensor, staging=staging)
        mean_df['component'] = [fac_label]*len(mean_df)
        mean_df_list.append(mean_df)

    all_mean_df = pd.concat(mean_df_list, axis=0)

    # save your dataframe before returning
    mouse = meta.reset_index()['mouse'].unique()[0]
    all_mean_df.to_pickle(os.path.join(save_folder, f'{mouse}_rank{model_rank}_model_bias_and_mean_df_wablation.pkl'))

    return all_mean_df


def get_bias_from_ablated_data(meta, input_tensor, model, model_rank, save_folder='', staging='parsed_10stage'):
    """
    Calculate bias for stages of learning for a tensor and meta on TCA model itself.
    """

    # loop over removing each component from TCA model
    mean_df_list = []
    for fac_num in range(model_rank):

        # create a tensor removing one of you your TCA factors from your data
        tensor = ablate_data_with_Ktensor(input_tensor, model.results[model_rank][0].factors, fac_num)
        fac_label = f'ablated component {fac_num + 1} from data'

        # mask values that are NaN in you original data (this may be redundant for this version)
        mask = np.isnan(input_tensor)
        tensor[mask] = np.nan

        # get bias and mean from TCA model of data
        mean_df = get_bias_from_tensor(meta, tensor, staging=staging)
        mean_df['component'] = [fac_label]*len(mean_df)
        mean_df_list.append(mean_df)

    all_mean_df = pd.concat(mean_df_list, axis=0)

    # save your dataframe before returning
    mouse = meta.reset_index()['mouse'].unique()[0]
    all_mean_df.to_pickle(os.path.join(save_folder, f'{mouse}_rank{model_rank}_data_bias_and_mean_df_wablation.pkl'))

    return all_mean_df


def ablate_Ktensor(tt_factors, fac_num_to_remove):
    """Create full matrix from an ablated (one factor removed) KTensor from tensortool."""

    # turn factors into tuple, then remove factor from each mode's matrix
    factors = tuple(tt_factors)
    factors = tuple([np.delete(f, fac_num_to_remove, axis=1) for f in factors])

    # create a KTensor from tensortools to speed up some math
    kt = KTensor(factors)

    # create full tensor
    return kt.full()


def ablate_data_with_Ktensor(data_tensor, tt_factors, fac_num_to_keep):
    """Create full matrix removing a single tensor component from your data."""

    # turn factors into tuple, then select a single factor from each mode's matrix
    factors = tuple(tt_factors)
    factors = tuple([f[:, fac_num_to_keep][:, None] for f in factors])

    # create a KTensor from tensortools to speed up some math
    kt = KTensor(factors)
    full_factor = kt.full()

    # return a full tensor minus a single component
    return data_tensor - full_factor


def full_factor(tt_factors, fac_num_to_keep):
    """Create full matrix removing a single tensor component from your data."""

    # turn factors into tuple, then select a single factor from each mode's matrix
    factors = tuple(tt_factors)
    factors = tuple([f[:, fac_num_to_keep][:, None] for f in factors])

    # create a KTensor from tensortools to speed up some math
    kt = KTensor(factors)
    full_factor = kt.full()

    # return a full tensor component
    return full_factor


def get_bias(
        mouse,
        trace_type='zscore_day',
        drive_threshold=20,
        drive_type='visual',
        drop_licking=True):
    """
    Returns:
    --------
    FC_bias : ndarray
        bias per cell per day
    """

    # get tensor, metadata, and ids to get things rolling
    ten, met, id = utils.build_tensor(
        mouse, drive_threshold=drive_threshold, trace_type=trace_type)

    # get boolean indexer for period stim is on screen
    stim_window = np.arange(-1, 7, 1/15.5)[0:108]
    stim_window = (stim_window > 0) & (stim_window < lookups.stim_length[mouse])

    # get vector and count of dates for the loop
    met = utils.update_meta_date_vec(met)
    date_vec = met.reset_index()['date']
    date_num = len(np.unique(date_vec))

    # preallocate tensors
    FC_ten = np.zeros((np.shape(ten)[0], np.shape(ten)[1], date_num))
    QC_ten = np.zeros((np.shape(ten)[0], np.shape(ten)[1], date_num))
    NC_ten = np.zeros((np.shape(ten)[0], np.shape(ten)[1], date_num))
    FC_ten[:] = np.nan
    QC_ten[:] = np.nan
    NC_ten[:] = np.nan

    # preallocate lists
    ls_list = []
    dprime_list = []

    # boolean vecs for each CS
    FC_bool = met['condition'].isin(['plus']).values
    QC_bool = met['condition'].isin(['minus']).values
    NC_bool = met['condition'].isin(['neutral']).values

    # blank periods after first lick in trial 
    if drop_licking:
        mask = get_lick_mask(met, ten)
        ten[~mask] = np.nan

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
        dp = pool.calc.performance.dprime(
            flow.Date(mouse, date=day))
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
        drive_type='visual',
        drop_licking=True):

    # get tensor, metadata, and ids to get things rolling
    ten, met, id = utils.build_tensor(
        mouse, drive_threshold=drive_threshold, trace_type=trace_type)

    # get boolean indexer for period stim is on screen
    stim_window = np.arange(-1, 7, 1/15.5)[0:108]
    stim_window = (stim_window > 0) & (stim_window < 3)

    # get vector and count of dates for the loop
    met = utils.update_meta_date_vec(met)
    date_vec = met.reset_index()['date']
    date_num = len(np.unique(date_vec))

    # preallocate tensors
    FC_ten = np.zeros((np.shape(ten)[0], np.shape(ten)[1], date_num))
    QC_ten = np.zeros((np.shape(ten)[0], np.shape(ten)[1], date_num))
    NC_ten = np.zeros((np.shape(ten)[0], np.shape(ten)[1], date_num))
    FC_ten[:] = np.nan
    QC_ten[:] = np.nan
    NC_ten[:] = np.nan

    # preallocate lists
    ls_list = []
    dprime_list = []

    # boolean vecs for each CS
    FC_bool = met['condition'].isin(['plus']).values
    QC_bool = met['condition'].isin(['minus']).values
    NC_bool = met['condition'].isin(['neutral']).values

    # blank periods after first lick in trial 
    if drop_licking:
        mask = get_lick_mask(met, ten)
        ten[~mask] = np.nan

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
        dp = pool.calc.performance.dprime(
            flow.Date(mouse, date=day))
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


def _first100_bool(meta):
    """
    Helper function to get a boolean vector of the first 100 trials for each day.
    If a day is shorter than 100 trials use the whole day. 
    """

    days = meta.reset_index()['date'].unique()

    first100 = np.zeros((len(meta)))
    for di in days:
        dboo  = meta.reset_index()['date'].isin([di]).values
        daylength = np.sum(dboo)
        if daylength > 100:
            first100[np.where(dboo)[0][:100]] = 1
        else:
            first100[dboo] = 1
    firstboo = first100 > 0
    
    return firstboo

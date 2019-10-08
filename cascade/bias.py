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
    ten, met, id = utils.build_tensor(
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

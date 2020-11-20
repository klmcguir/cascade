from . import utils, tuning, lookups
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import os

# TODO template fitting
#  1. function to normalize responses for each day/stage and then get their mean shape --> template
#  2. function to least squares or NNLS fit f(x) = x*template
#  3. calculate goodness of fit as r-squared (variance explained)
#  4. return scalar from fitting and r-squared per stage.

# TODO stability
#  1. s_index needs updata to match running speeds
#  2. voluntary engagment index should be added, control for running


def trial_history_sensory(meta, pref_tensor, epoch='parsed_11stage', filter_running=None, filter_licking=None,
                          filter_hmm_engaged=True):

    # get mouse from metadata
    mouse = meta.reset_index()['mouse'].unique()[0]

    mean_t_tensor = utils.tensor_mean_per_trial(meta, pref_tensor, nan_licking=False, account_for_offset=True)

    # get vector saying if previous stimulus was the same or different
    # flipped_stim_order = meta.initial_condition.values[::-1]
    # prev_same = []
    # for c in range(len(flipped_stim_order)):
    #     if c + 1 < len(flipped_stim_order):
    #         prev_same.append(True if flipped_stim_order[c] == flipped_stim_order[c+1] else False)
    # prev_same.append(False)  # one more to account for first trial not having a previous trial
    # prev_same = np.array(prev_same[::-1])

    # get previous same accounting for dropped pavlovians
    prev_same = (meta.prev_same_plus | meta.prev_same_minus | meta.prev_same_neutral).values

    # get vector counting how many previous trials were the same
    # THIS DOES NOT ACCOUNT FOR PAVs
    same_in_a_row = np.zeros(len(meta))
    same_in_a_row[prev_same] = 1
    possible_double = np.diff(same_in_a_row, prepend=0) == 0
    possible_triple = np.diff(np.diff(same_in_a_row, prepend=0), prepend=0) == 0
    possible_quad = np.diff(np.diff(np.diff(same_in_a_row, prepend=0), prepend=0), prepend=0) == 0
    same_in_a_row[prev_same & possible_double] = 2
    same_in_a_row[prev_same & possible_double & possible_triple] = 3
    same_in_a_row[prev_same & possible_double & possible_triple & possible_quad] = 4
    # prev_same = same_in_a_row == 4

    # filter to include fixed running type: low or high
    if filter_running is not None:
        speed_cm_s = meta.speed.values
        if filter_running == 'low_speed_only':
            prev_same = prev_same & (speed_cm_s <= 4)
        elif filter_running == 'high_speed_only':
            prev_same = prev_same & (speed_cm_s > 10)
        else:
            raise NotImplementedError

    # filter to include fixed licking type: low or high
    if filter_licking is not None:
        # TODO this needs accounting for offset licking for offset cells
        mean_lick_rate = meta.anticipatory_licks.values / lookups.stim_length[mouse]
        if filter_licking == 'low_lick_only':
            prev_same = prev_same & (mean_lick_rate <= 1.7)
        elif filter_licking == 'high_lick_only':
            prev_same = prev_same & (mean_lick_rate > 1.7)
        else:
            raise NotImplementedError

    # animal must be engaged in the task (or naive when it can't "engage")
    if filter_hmm_engaged:
        prev_same = prev_same & (meta.hmm_engaged.values | meta.learning_state.isin(['naive']).values)

    if epoch == 'days':
        days = meta.reset_index()['date'].unique()
        # new_mat = np.zeros((pref_tensor.shape[0], len(days)))
        # new_mat[:] = np.nan
        # for c, di in enumerate(meta.reset_index()['date'].unique()):
        #     day_boo = meta.reset_index()['date'].isin([di]).values
        #     denom = np.nanmean(mean_t_tensor[:, day_boo], axis=1)
        #     fano = np.abs(np.nanvar(mean_t_tensor[:, day_boo], axis=1) / denom)
        #     fano[denom < min_denom] = np.nan
        #     new_mat[:, c] = fano

    elif epoch in ['parsed_stage', 'parsed_10stage', 'parsed_11stage']:
        # get average response per stage
        stages = lookups.staging[epoch]
        new_mat = np.zeros((pref_tensor.shape[0], len(stages)))
        new_mat[:] = np.nan
        for c, di in enumerate(stages):
            stage_boo = meta[epoch].isin([di]).values
            stage_days = meta.loc[stage_boo, :].reset_index()['date'].unique()
            day_sh_index = np.zeros((pref_tensor.shape[0], len(stage_days)))
            day_sh_index[:] = np.nan
            for c2, di2 in enumerate(stage_days):
                day_boo = meta.reset_index()['date'].isin([di2]).values
                prev_same_mean = np.nanmean(mean_t_tensor[:, stage_boo & day_boo & prev_same], axis=1)
                prev_diff_mean = np.nanmean(mean_t_tensor[:, stage_boo & day_boo & ~prev_same], axis=1)
                # rectify
                prev_same_mean[prev_same_mean < 0] = 0
                prev_diff_mean[prev_diff_mean < 0] = 0
                sensory_history_index = prev_same_mean - prev_diff_mean
                # sensory_history_index = (prev_diff_mean - prev_same_mean) / (prev_diff_mean + prev_same_mean)
                # sensory_history_index[(sensory_history_index < -1) | (sensory_history_index > 1)] = np.nan
                day_sh_index[:, c2] = sensory_history_index
            new_mat[:, c] = np.nanmean(day_sh_index, axis=1)
    else:
        raise NotImplementedError

    return new_mat


def lick_modulation(meta, pref_tensor, epoch='parsed_11stage', filter_running=None, filter_licking=None,
                          filter_hmm_engaged=True):

    # get mouse from metadata
    mouse = meta.reset_index()['mouse'].unique()[0]

    # get mean response per trial accounting for offset
    mean_t_tensor = utils.tensor_mean_per_trial(meta, pref_tensor, nan_licking=False, account_for_offset=True)

    # get previous same accounting for dropped pavlovians
    prev_same = meta.pre_licks.gt(1).values

    # get vector counting how many previous trials were the same
    # THIS DOES NOT ACCOUNT FOR PAVs
    same_in_a_row = np.zeros(len(meta))
    same_in_a_row[prev_same] = 1
    possible_double = np.diff(same_in_a_row, prepend=0) == 0
    possible_triple = np.diff(np.diff(same_in_a_row, prepend=0), prepend=0) == 0
    possible_quad = np.diff(np.diff(np.diff(same_in_a_row, prepend=0), prepend=0), prepend=0) == 0
    same_in_a_row[prev_same & possible_double] = 2
    same_in_a_row[prev_same & possible_double & possible_triple] = 3
    same_in_a_row[prev_same & possible_double & possible_triple & possible_quad] = 4
    # prev_same = same_in_a_row == 4

    # filter to include fixed running type: low or high
    if filter_running is not None:
        speed_cm_s = meta.speed.values
        if filter_running == 'low_speed_only':
            prev_same = prev_same & (speed_cm_s <= 4)
        elif filter_running == 'high_speed_only':
            prev_same = prev_same & (speed_cm_s > 10)
        else:
            raise NotImplementedError

    # filter to include fixed licking type: low or high
    if filter_licking is not None:
        # TODO this needs accounting for offset licking for offset cells
        mean_lick_rate = meta.anticipatory_licks.values / lookups.stim_length[mouse]
        if filter_licking == 'low_lick_only':
            prev_same = prev_same & (mean_lick_rate <= 1.7)
        elif filter_licking == 'high_lick_only':
            prev_same = prev_same & (mean_lick_rate > 1.7)
        else:
            raise NotImplementedError

    # animal must be engaged in the task (or naive when it can't "engage")
    if filter_hmm_engaged:
        prev_same = prev_same & (meta.hmm_engaged.values | meta.learning_state.isin(['naive']).values)

    if epoch == 'days':
        days = meta.reset_index()['date'].unique()
        # new_mat = np.zeros((pref_tensor.shape[0], len(days)))
        # new_mat[:] = np.nan
        # for c, di in enumerate(meta.reset_index()['date'].unique()):
        #     day_boo = meta.reset_index()['date'].isin([di]).values
        #     denom = np.nanmean(mean_t_tensor[:, day_boo], axis=1)
        #     fano = np.abs(np.nanvar(mean_t_tensor[:, day_boo], axis=1) / denom)
        #     fano[denom < min_denom] = np.nan
        #     new_mat[:, c] = fano

    elif epoch in ['parsed_stage', 'parsed_10stage', 'parsed_11stage']:
        # get average response per stage
        stages = lookups.staging[epoch]
        new_mat = np.zeros((pref_tensor.shape[0], len(stages)))
        new_mat[:] = np.nan
        for c, di in enumerate(stages):
            stage_boo = meta[epoch].isin([di]).values
            stage_days = meta.loc[stage_boo, :].reset_index()['date'].unique()
            day_sh_index = np.zeros((pref_tensor.shape[0], len(stage_days)))
            day_sh_index[:] = np.nan
            for c2, di2 in enumerate(stage_days):
                day_boo = meta.reset_index()['date'].isin([di2]).values
                prev_same_mean = np.nanmean(mean_t_tensor[:, stage_boo & day_boo & prev_same], axis=1)
                prev_diff_mean = np.nanmean(mean_t_tensor[:, stage_boo & day_boo & ~prev_same], axis=1)
                # rectify
                prev_same_mean[prev_same_mean < 0] = 0
                prev_diff_mean[prev_diff_mean < 0] = 0
                sensory_history_index = prev_same_mean - prev_diff_mean  # licking trials - not licking trials
                # sensory_history_index = (prev_diff_mean - prev_same_mean) / (prev_diff_mean + prev_same_mean)
                # sensory_history_index[(sensory_history_index < -1) | (sensory_history_index > 1)] = np.nan
                day_sh_index[:, c2] = sensory_history_index
            new_mat[:, c] = np.nanmean(day_sh_index, axis=1)
    else:
        raise NotImplementedError

    return new_mat


def run_modulation(meta, pref_tensor, epoch='parsed_11stage', filter_running=None, filter_licking=None,
                   filter_hmm_engaged=True):

    # get mouse from metadata
    mouse = meta.reset_index()['mouse'].unique()[0]

    # get mean response per trial accounting for offset
    mean_t_tensor = utils.tensor_mean_per_trial(meta, pref_tensor, nan_licking=False, account_for_offset=True)

    # get previous same accounting for dropped pavlovians
    prev_same = meta.pre_speed.gt(4).values

    # get vector counting how many previous trials were the same
    # THIS DOES NOT ACCOUNT FOR PAVs
    same_in_a_row = np.zeros(len(meta))
    same_in_a_row[prev_same] = 1
    possible_double = np.diff(same_in_a_row, prepend=0) == 0
    possible_triple = np.diff(np.diff(same_in_a_row, prepend=0), prepend=0) == 0
    possible_quad = np.diff(np.diff(np.diff(same_in_a_row, prepend=0), prepend=0), prepend=0) == 0
    same_in_a_row[prev_same & possible_double] = 2
    same_in_a_row[prev_same & possible_double & possible_triple] = 3
    same_in_a_row[prev_same & possible_double & possible_triple & possible_quad] = 4
    # prev_same = same_in_a_row == 4

    # filter to include fixed running type: low or high
    if filter_running is not None:
        speed_cm_s = meta.speed.values
        if filter_running == 'low_speed_only':
            prev_same = prev_same & (speed_cm_s <= 4)
        elif filter_running == 'high_speed_only':
            prev_same = prev_same & (speed_cm_s > 10)
        else:
            raise NotImplementedError

    # filter to include fixed licking type: low or high
    if filter_licking is not None:
        # TODO this needs accounting for offset licking for offset cells
        mean_lick_rate = meta.anticipatory_licks.values / lookups.stim_length[mouse]
        if filter_licking == 'low_lick_only':
            prev_same = prev_same & (mean_lick_rate <= 1.7)
        elif filter_licking == 'high_lick_only':
            prev_same = prev_same & (mean_lick_rate > 1.7)
        else:
            raise NotImplementedError

    # animal must be engaged in the task (or naive when it can't "engage")
    if filter_hmm_engaged:
        prev_same = prev_same & (meta.hmm_engaged.values | meta.learning_state.isin(['naive']).values)

    if epoch == 'days':
        days = meta.reset_index()['date'].unique()
        # new_mat = np.zeros((pref_tensor.shape[0], len(days)))
        # new_mat[:] = np.nan
        # for c, di in enumerate(meta.reset_index()['date'].unique()):
        #     day_boo = meta.reset_index()['date'].isin([di]).values
        #     denom = np.nanmean(mean_t_tensor[:, day_boo], axis=1)
        #     fano = np.abs(np.nanvar(mean_t_tensor[:, day_boo], axis=1) / denom)
        #     fano[denom < min_denom] = np.nan
        #     new_mat[:, c] = fano

    elif epoch in ['parsed_stage', 'parsed_10stage', 'parsed_11stage']:
        # get average response per stage
        stages = lookups.staging[epoch]
        new_mat = np.zeros((pref_tensor.shape[0], len(stages)))
        new_mat[:] = np.nan
        for c, di in enumerate(stages):
            stage_boo = meta[epoch].isin([di]).values
            stage_days = meta.loc[stage_boo, :].reset_index()['date'].unique()
            day_sh_index = np.zeros((pref_tensor.shape[0], len(stage_days)))
            day_sh_index[:] = np.nan
            for c2, di2 in enumerate(stage_days):
                day_boo = meta.reset_index()['date'].isin([di2]).values
                prev_same_mean = np.nanmean(mean_t_tensor[:, stage_boo & day_boo & prev_same], axis=1)
                prev_diff_mean = np.nanmean(mean_t_tensor[:, stage_boo & day_boo & ~prev_same], axis=1)
                # rectify
                prev_same_mean[prev_same_mean < 0] = 0
                prev_diff_mean[prev_diff_mean < 0] = 0
                sensory_history_index = (prev_diff_mean - prev_same_mean) / (prev_diff_mean + prev_same_mean)
                sensory_history_index[(sensory_history_index < -1) | (sensory_history_index > 1)] = np.nan
                day_sh_index[:, c2] = sensory_history_index
            new_mat[:, c] = np.nanmean(day_sh_index, axis=1)
    else:
        raise NotImplementedError

    return new_mat


def trial_history_sensory_blank(meta, pref_tensor, epoch='parsed_11stage', filter_running=None, filter_licking=None,
                          filter_hmm_engaged=True):

    # get mouse from metadata
    mouse = meta.reset_index()['mouse'].unique()[0]

    mean_t_tensor = utils.tensor_mean_per_trial(meta, pref_tensor, nan_licking=False, account_for_offset=True)

    # get vector of previous blanks
    prev_same = meta.prev_blank.values

    # get vector that checks if there were one or two previous blanks in a row
    double_blank = np.zeros(len(meta))
    double_blank[meta.prev_blank] = 1
    double_blank[(double_blank == 1) & (np.diff(double_blank, prepend=0) == 0)] = 2

    # filter to include fixed running type: low or high
    if filter_running is not None:
        speed_cm_s = meta.speed.values
        if filter_running == 'low_speed_only':
            prev_same = prev_same & (speed_cm_s <= 4)
        elif filter_running == 'high_speed_only':
            prev_same = prev_same & (speed_cm_s > 10)
        else:
            raise NotImplementedError

    # filter to include fixed licking type: low or high
    if filter_licking is not None:
        # TODO this needs accounting for offset licking for offset cells
        mean_lick_rate = meta.anticipatory_licks.values / lookups.stim_length[mouse]
        if filter_licking == 'low_lick_only':
            prev_same = prev_same & (mean_lick_rate <= 1.7)
        elif filter_licking == 'high_lick_only':
            prev_same = prev_same & (mean_lick_rate > 1.7)
        else:
            raise NotImplementedError

    # animal must be engaged in the task (or naive when it can't "engage")
    if filter_hmm_engaged:
        prev_same = prev_same & (meta.hmm_engaged.values | meta.learning_state.isin(['naive']).values)

    if epoch == 'days':
        days = meta.reset_index()['date'].unique()
        # new_mat = np.zeros((pref_tensor.shape[0], len(days)))
        # new_mat[:] = np.nan
        # for c, di in enumerate(meta.reset_index()['date'].unique()):
        #     day_boo = meta.reset_index()['date'].isin([di]).values
        #     denom = np.nanmean(mean_t_tensor[:, day_boo], axis=1)
        #     fano = np.abs(np.nanvar(mean_t_tensor[:, day_boo], axis=1) / denom)
        #     fano[denom < min_denom] = np.nan
        #     new_mat[:, c] = fano

    elif epoch in ['parsed_stage', 'parsed_10stage', 'parsed_11stage']:
        # get average response per stage
        stages = lookups.staging[epoch]
        new_mat = np.zeros((pref_tensor.shape[0], len(stages)))
        new_mat[:] = np.nan
        for c, di in enumerate(stages):
            stage_boo = meta[epoch].isin([di]).values
            stage_days = meta.loc[stage_boo, :].reset_index()['date'].unique()
            day_sh_index = np.zeros((pref_tensor.shape[0], len(stage_days)))
            day_sh_index[:] = np.nan
            for c2, di2 in enumerate(stage_days):
                day_boo = meta.reset_index()['date'].isin([di2]).values
                prev_same_mean = np.nanmean(mean_t_tensor[:, stage_boo & day_boo & prev_same], axis=1)
                prev_diff_mean = np.nanmean(mean_t_tensor[:, stage_boo & day_boo & ~prev_same], axis=1)
                # rectify
                prev_same_mean[prev_same_mean < 0] = 0
                prev_diff_mean[prev_diff_mean < 0] = 0
                sensory_history_index = (prev_diff_mean - prev_same_mean) / (prev_diff_mean + prev_same_mean)
                sensory_history_index[(sensory_history_index < -1) | (sensory_history_index > 1)] = np.nan
                day_sh_index[:, c2] = sensory_history_index
            new_mat[:, c] = np.nanmean(day_sh_index, axis=1)
    else:
        raise NotImplementedError

    return new_mat


def trial_history_reward(meta, pref_tensor, epoch='parsed_11stage', filter_running=None, filter_licking=None,
                          filter_hmm_engaged=True):

    # get mouse from metadata
    mouse = meta.reset_index()['mouse'].unique()[0]

    mean_t_tensor = utils.tensor_mean_per_trial(meta, pref_tensor, nan_licking=False, account_for_offset=True)

    # get vector saying if previous stimulus was the same or different
    # flipped_stim_order = meta.initial_condition.values[::-1]
    # prev_same = []
    # for c in range(len(flipped_stim_order)):
    #     if c + 1 < len(flipped_stim_order):
    #         prev_same.append(True if flipped_stim_order[c] == flipped_stim_order[c+1] else False)
    # prev_same.append(False)  # one more to account for first trial not having a previous trial
    # prev_same = np.array(prev_same[::-1])

    # get previous same accounting for dropped pavlovians
    prev_same = meta.prev_reward.values

    # get vector counting how many previous trials were the same
    # THIS DOES NOT ACCOUNT FOR PAVs
    same_in_a_row = np.zeros(len(meta))
    same_in_a_row[prev_same] = 1
    possible_double = np.diff(same_in_a_row, prepend=0) == 0
    possible_triple = np.diff(np.diff(same_in_a_row, prepend=0), prepend=0) == 0
    possible_quad = np.diff(np.diff(np.diff(same_in_a_row, prepend=0), prepend=0), prepend=0) == 0
    same_in_a_row[prev_same & possible_double] = 2
    same_in_a_row[prev_same & possible_double & possible_triple] = 3
    same_in_a_row[prev_same & possible_double & possible_triple & possible_quad] = 4
    # prev_same = same_in_a_row == 4

    # filter to include fixed running type: low or high
    if filter_running is not None:
        speed_cm_s = meta.speed.values
        if filter_running == 'low_speed_only':
            prev_same = prev_same & (speed_cm_s <= 4)
        elif filter_running == 'high_speed_only':
            prev_same = prev_same & (speed_cm_s > 10)
        else:
            raise NotImplementedError

    # filter to include fixed licking type: low or high
    if filter_licking is not None:
        # TODO this needs accounting for offset licking for offset cells
        mean_lick_rate = meta.anticipatory_licks.values / lookups.stim_length[mouse]
        if filter_licking == 'low_lick_only':
            prev_same = prev_same & (mean_lick_rate <= 1.7)
        elif filter_licking == 'high_lick_only':
            prev_same = prev_same & (mean_lick_rate > 1.7)
        else:
            raise NotImplementedError

    # animal must be engaged in the task (or naive when it can't "engage")
    if filter_hmm_engaged:
        prev_same = prev_same & (meta.hmm_engaged.values | meta.learning_state.isin(['naive']).values)

    if epoch == 'days':
        days = meta.reset_index()['date'].unique()
        # new_mat = np.zeros((pref_tensor.shape[0], len(days)))
        # new_mat[:] = np.nan
        # for c, di in enumerate(meta.reset_index()['date'].unique()):
        #     day_boo = meta.reset_index()['date'].isin([di]).values
        #     denom = np.nanmean(mean_t_tensor[:, day_boo], axis=1)
        #     fano = np.abs(np.nanvar(mean_t_tensor[:, day_boo], axis=1) / denom)
        #     fano[denom < min_denom] = np.nan
        #     new_mat[:, c] = fano

    elif epoch in ['parsed_stage', 'parsed_10stage', 'parsed_11stage']:
        # get average response per stage
        stages = lookups.staging[epoch]
        new_mat = np.zeros((pref_tensor.shape[0], len(stages)))
        new_mat[:] = np.nan
        for c, di in enumerate(stages):
            stage_boo = meta[epoch].isin([di]).values
            stage_days = meta.loc[stage_boo, :].reset_index()['date'].unique()
            day_sh_index = np.zeros((pref_tensor.shape[0], len(stage_days)))
            day_sh_index[:] = np.nan
            for c2, di2 in enumerate(stage_days):
                day_boo = meta.reset_index()['date'].isin([di2]).values
                prev_same_mean = np.nanmean(mean_t_tensor[:, stage_boo & day_boo & prev_same], axis=1)
                prev_diff_mean = np.nanmean(mean_t_tensor[:, stage_boo & day_boo & ~prev_same], axis=1)
                # rectify
                prev_same_mean[prev_same_mean < 0] = 0
                prev_diff_mean[prev_diff_mean < 0] = 0
                sensory_history_index = (prev_diff_mean - prev_same_mean) / (prev_diff_mean + prev_same_mean)
                sensory_history_index[(sensory_history_index < -1) | (sensory_history_index > 1)] = np.nan
                day_sh_index[:, c2] = sensory_history_index
            new_mat[:, c] = np.nanmean(day_sh_index, axis=1)
    else:
        raise NotImplementedError

    return new_mat


def trial_history_punishment(meta, pref_tensor, epoch='parsed_11stage', filter_running=None, filter_licking=None,
                          filter_hmm_engaged=True):

    # get mouse from metadata
    mouse = meta.reset_index()['mouse'].unique()[0]

    mean_t_tensor = utils.tensor_mean_per_trial(meta, pref_tensor, nan_licking=False, account_for_offset=True)

    # get vector saying if previous stimulus was the same or different
    # flipped_stim_order = meta.initial_condition.values[::-1]
    # prev_same = []
    # for c in range(len(flipped_stim_order)):
    #     if c + 1 < len(flipped_stim_order):
    #         prev_same.append(True if flipped_stim_order[c] == flipped_stim_order[c+1] else False)
    # prev_same.append(False)  # one more to account for first trial not having a previous trial
    # prev_same = np.array(prev_same[::-1])

    # get previous same accounting for dropped pavlovians
    prev_same = meta.prev_punish.values

    # get vector counting how many previous trials were the same
    # THIS DOES NOT ACCOUNT FOR PAVs
    same_in_a_row = np.zeros(len(meta))
    same_in_a_row[prev_same] = 1
    possible_double = np.diff(same_in_a_row, prepend=0) == 0
    possible_triple = np.diff(np.diff(same_in_a_row, prepend=0), prepend=0) == 0
    possible_quad = np.diff(np.diff(np.diff(same_in_a_row, prepend=0), prepend=0), prepend=0) == 0
    same_in_a_row[prev_same & possible_double] = 2
    same_in_a_row[prev_same & possible_double & possible_triple] = 3
    same_in_a_row[prev_same & possible_double & possible_triple & possible_quad] = 4
    # prev_same = same_in_a_row == 4

    # filter to include fixed running type: low or high
    if filter_running is not None:
        speed_cm_s = meta.speed.values
        if filter_running == 'low_speed_only':
            prev_same = prev_same & (speed_cm_s <= 4)
        elif filter_running == 'high_speed_only':
            prev_same = prev_same & (speed_cm_s > 10)
        else:
            raise NotImplementedError

    # filter to include fixed licking type: low or high
    if filter_licking is not None:
        # TODO this needs accounting for offset licking for offset cells
        mean_lick_rate = meta.anticipatory_licks.values / lookups.stim_length[mouse]
        if filter_licking == 'low_lick_only':
            prev_same = prev_same & (mean_lick_rate <= 1.7)
        elif filter_licking == 'high_lick_only':
            prev_same = prev_same & (mean_lick_rate > 1.7)
        else:
            raise NotImplementedError

    # animal must be engaged in the task (or naive when it can't "engage")
    if filter_hmm_engaged:
        prev_same = prev_same & (meta.hmm_engaged.values | meta.learning_state.isin(['naive']).values)

    if epoch == 'days':
        days = meta.reset_index()['date'].unique()
        # new_mat = np.zeros((pref_tensor.shape[0], len(days)))
        # new_mat[:] = np.nan
        # for c, di in enumerate(meta.reset_index()['date'].unique()):
        #     day_boo = meta.reset_index()['date'].isin([di]).values
        #     denom = np.nanmean(mean_t_tensor[:, day_boo], axis=1)
        #     fano = np.abs(np.nanvar(mean_t_tensor[:, day_boo], axis=1) / denom)
        #     fano[denom < min_denom] = np.nan
        #     new_mat[:, c] = fano

    elif epoch in ['parsed_stage', 'parsed_10stage', 'parsed_11stage']:
        # get average response per stage
        stages = lookups.staging[epoch]
        new_mat = np.zeros((pref_tensor.shape[0], len(stages)))
        new_mat[:] = np.nan
        for c, di in enumerate(stages):
            stage_boo = meta[epoch].isin([di]).values
            stage_days = meta.loc[stage_boo, :].reset_index()['date'].unique()
            day_sh_index = np.zeros((pref_tensor.shape[0], len(stage_days)))
            day_sh_index[:] = np.nan
            for c2, di2 in enumerate(stage_days):
                day_boo = meta.reset_index()['date'].isin([di2]).values
                prev_same_mean = np.nanmean(mean_t_tensor[:, stage_boo & day_boo & prev_same], axis=1)
                prev_diff_mean = np.nanmean(mean_t_tensor[:, stage_boo & day_boo & ~prev_same], axis=1)
                # rectify
                prev_same_mean[prev_same_mean < 0] = 0
                prev_diff_mean[prev_diff_mean < 0] = 0
                sensory_history_index = prev_same_mean - prev_diff_mean
                # sensory_history_index = (prev_diff_mean - prev_same_mean) / (prev_diff_mean + prev_same_mean)
                # sensory_history_index[(sensory_history_index < -1) | (sensory_history_index > 1)] = np.nan
                day_sh_index[:, c2] = sensory_history_index
            new_mat[:, c] = np.nanmean(day_sh_index, axis=1)
    else:
        raise NotImplementedError

    return new_mat


def latency_to_half_peak(meta, pref_tensor, epoch='day'):
    """
    Get cells peak and half peak latency relative to the onset or offset of the stimulus. Only relative to offset for
    cells that are offset responsive. Units are still in frames (i.e. 15.5 Hz frames), so conversion to seconds still
    requires division by 15.5.

    :param meta: pandas.DataFrame, DataFrame of trial metadata
    :param pref_tensor: numpy.ndarray
        A tensor the exact size of the tensor input, now containing nans for un-preferred stimulus presentations.
    :param epoch: str, 'day' or staging type to take averages overs
    :return: peak_time, half_peak_time, numpy.ndarrays
        cells x stages/days/epochs matrices of peak latency RELATIVE TO STIMULUS ONSET OR OFFSET depending
        on type of cell.
    """

    # determine cells with offset responses
    offset_bool = utils.get_offset_cells(meta, pref_tensor)

    # define windows for checking peak
    mouse = meta.reset_index()['mouse'].unique()[0]
    response_window = np.arange(np.floor(15.5 * (1 + lookups.stim_length[mouse])),
                                np.floor(15.5 * (3 + lookups.stim_length[mouse])), dtype='int')
    stimulus_window = np.arange(16, np.floor(15.5 * (1 + lookups.stim_length[mouse])), dtype='int')

    # get mean across epoch
    if epoch == 'day':
        mean_tensor = utils.simple_mean_per_day(meta, pref_tensor)
    else:
        assert epoch in ['parsed_stage', 'parsed_10stage', 'parsed_11stage']
        mean_tensor = utils.simple_mean_per_stage(meta, pref_tensor, staging=epoch)

    # simple peak latency accounting for noisy baselines and nans
    peak_time = np.zeros((mean_tensor.shape[0], mean_tensor.shape[2]))
    peak_time[:] = np.nan
    half_peak_time = np.zeros((mean_tensor.shape[0], mean_tensor.shape[2]))
    half_peak_time[:] = np.nan
    for ti in range(mean_tensor.shape[2]):
        for ci in range(mean_tensor.shape[0]):
            cell_vec = mean_tensor[ci, :, ti]

            # get only the chunk of time for that cells preferred response
            if offset_bool[ci]:
                cell_vec = cell_vec[response_window]

            else:
                cell_vec = cell_vec[stimulus_window]

            # skip missing trials or data
            if all(np.isnan(cell_vec)):
                continue

            # get peak time, only include if the peak is not on the edge of the window, and max must be above 0
            peak_lat = np.nanargmax(cell_vec)
            if (peak_lat > 0) and (peak_lat < (len(cell_vec) - 1)) and (np.nanmax(cell_vec) > 0):
                peak_time[ci, ti] = peak_lat

                # get half peak time (if there is a peak), only include if the half peak is not on the edge
                half_peak_bool = cell_vec >= np.nanmax(cell_vec) / 2
                if np.sum(half_peak_bool) > 0 and (np.where(half_peak_bool)[0][0] > 0):
                    half_peak_lat = np.where(half_peak_bool)[0][0]
                    half_peak_time[ci, ti] = half_peak_lat

    return peak_time, half_peak_time


def sustainedness95(meta, pref_tensor, epoch='day', full_trial_window=False,
                    filter_running='low_speed_only', filter_licking=None, filter_hmm_engaged=True):
    """
    Get cell's sustainedness, SI = (mean response / 95th percentile response). Accounts offset or can make calc for
    onset to response window close. Benefit of this method is that it is not making any assumptions about the
    timing/dynamics of a cell's response.

    # TODO Consider passing tensor to get_offset_cells()

    :param meta: pandas.DataFrame
        DataFrame of trial metadata
    :param pref_tensor: numpy.ndarray
        A tensor the exact size of the tensor input, now containing nans for un-preferred stimulus presentations.
    :param epoch: str
        'day' or staging type to take averages overs
    :param full_trial_window: boolean
        Use the stimulus window + response window to calculate transientness.
    :return: peak_time, half_peak_time, numpy.ndarrays
        cells x stages/days/epochs matrices of peak latency RELATIVE TO STIMULUS ONSET OR OFFSET depending
        on type of cell.
    """

    # determine cells with offset responses
    offset_bool = utils.get_offset_cells(meta, pref_tensor)

    # define windows for checking peak
    mouse = meta.reset_index()['mouse'].unique()[0]
    if full_trial_window:
        response_window = np.arange(16, np.floor(15.5 * (3 + lookups.stim_length[mouse])), dtype='int')
        stimulus_window = np.arange(16, np.floor(15.5 * (3 + lookups.stim_length[mouse])), dtype='int')
    else:
        response_window = np.arange(np.floor(15.5 * (1 + lookups.stim_length[mouse])),
                                    np.floor(15.5 * (3 + lookups.stim_length[mouse])), dtype='int')
        stimulus_window = np.arange(16, np.floor(15.5 * (1 + lookups.stim_length[mouse])), dtype='int')

    # get mean across epoch
    if epoch == 'day':
        mean_tensor = utils.simple_mean_per_day(meta, pref_tensor)
    else:
        assert epoch in ['parsed_stage', 'parsed_10stage', 'parsed_11stage']
        mean_tensor = utils.simple_mean_per_stage(meta, pref_tensor, staging=epoch)

    # utils.filter_meta_bool(meta, meta_bool,
    #                        filter_running=filter_running,
    #                        filter_licking=filter_licking,
    #                        filter_hmm_engaged=filter_hmm_engaged)

    # simple peak latency accounting for noisy baselines and nans
    s_index = np.zeros((mean_tensor.shape[0], mean_tensor.shape[2]))
    s_index[:] = np.nan
    for ti in range(mean_tensor.shape[2]):
        for ci in range(mean_tensor.shape[0]):
            cell_vec = mean_tensor[ci, :, ti]

            # get only the chunk of time for that cells preferred response
            if offset_bool[ci]:
                cell_vec = cell_vec[response_window]

            else:
                cell_vec = cell_vec[stimulus_window]

            # nan rectify
            cell_vec[cell_vec < 0] = np.nan

            # skip missing trials or data
            if all(np.isnan(cell_vec)):
                continue

            # get mean and 95th percentile response
            mean_response = np.nanmean(cell_vec)

            # flip non-NaN array values [0, 1, 10, np.nan] --> [10, 1, 0, np.nan]
            sorted_vals = np.sort(cell_vec)  # nans appended to end
            ind95 = int(np.ceil(len(cell_vec)*.05))
            sorted_vals[~np.isnan(sorted_vals)] = sorted_vals[~np.isnan(sorted_vals)][::-1]
            percentile95 = sorted_vals[ind95]

            # add index
            if percentile95 > 0.001 and mean_response > 0.001 and percentile95 >= mean_response:
                s_index[ci, ti] = mean_response / percentile95

    return s_index


def fano_factor_mean_days(meta, pref_tensor, min_denom=0.01):
    """

    :param meta: pandas.DataFrame, trial metadata
    :param pref_tensor: numpy.ndarray, a cells x times X trials; should contain NaNs for trials that are not a cells
    preferred stimulus.

    :return:
    """

    # meta can only be a DataFrame of a single mouse
    assert len(meta.reset_index()['mouse'].unique()) == 1
    mouse = meta.reset_index()['mouse'].unique()[0]

    # get mean per day for preferred tuning
    mean_tensor = utils.simple_mean_per_day(meta, pref_tensor)

    # assume 15.5 Hz sampling or downsampling for 7 seconds per trial (n = 108 timepoints)
    assert pref_tensor.shape[1] == 108
    times = np.arange(-1, 6, 1 / 15.5)[:108]
    stim_bool = (times > 0) & (times < lookups.stim_length[mouse])
    response_bool = (times > lookups.stim_length[mouse] + 0.3) & (times < lookups.stim_length[mouse] + 2)

    # determine cells with offset responses
    offset_bool = utils.get_offset_cells(meta, pref_tensor)

    # get mean across epoch
    mean_tensor = utils.simple_mean_per_day(meta, pref_tensor)

    # get mean of response window or stimulus window depending on peak response time
    new_mat = np.zeros((mean_tensor.shape[0], mean_tensor.shape[2]))
    if np.sum(offset_bool) > 0:
        new_mat[offset_bool, :] = np.nanmean(mean_tensor[offset_bool, :, :][:, response_bool, :], axis=1)
    if np.sum(~offset_bool) > 0:
        new_mat[~offset_bool, :] = np.nanmean(mean_tensor[~offset_bool, :, :][:, stim_bool, :], axis=1)

    # get fano factor
    denom = np.nanmean(new_mat, axis=1)
    fano = np.abs(np.nanvar(new_mat, axis=1) / denom)
    fano[denom < min_denom] = np.nan

    return fano


def fano_factor_trials(meta, pref_tensor, epoch='parsed_11stage', min_denom=0.01):

    mean_t_tensor = utils.tensor_mean_per_trial(meta, pref_tensor, nan_licking=False, account_for_offset=True)

    if epoch == 'days':
        days = meta.reset_index()['date'].unique()
        new_mat = np.zeros((pref_tensor.shape[0], len(days)))
        new_mat[:] = np.nan
        for c, di in enumerate(meta.reset_index()['date'].unique()):
            day_boo = meta.reset_index()['date'].isin([di]).values
            denom = np.nanmean(mean_t_tensor[:, day_boo], axis=1)
            fano = np.abs(np.nanvar(mean_t_tensor[:, day_boo], axis=1) / denom)
            fano[denom < min_denom] = np.nan
            new_mat[:, c] = fano
    elif epoch in ['parsed_stage', 'parsed_10stage', 'parsed_11stage']:
        # get average response per stage
        stages = lookups.staging[epoch]
        new_mat = np.zeros((pref_tensor.shape[0], len(stages)))
        new_mat[:] = np.nan
        for c, di in enumerate(stages):
            stage_boo = meta[epoch].isin([di]).values
            stage_days = meta.loc[stage_boo, :].reset_index()['date'].unique()
            day_fano = np.zeros((pref_tensor.shape[0], len(stage_days)))
            day_fano[:] = np.nan
            for c2, di2 in enumerate(stage_days):
                day_boo = meta.reset_index()['date'].isin([di2]).values
                denom = np.nanmean(mean_t_tensor[:, stage_boo & day_boo], axis=1)
                fano = np.abs(np.nanvar(mean_t_tensor[:, stage_boo & day_boo], axis=1) / denom)
                fano[denom < min_denom] = np.nan
                day_fano[:, c2] = fano
            new_mat[:, c] = np.nanmean(day_fano, axis=1)
    else:
        raise NotImplementedError

    return new_mat


def correlate_epochs_wearly(meta, pref_tensor, epoch='parsed_11stage'):
    """
    Correlate each cell's average temporal trace per epoch with it's average on early stages of learning.

    :param meta:
    :param pref_tensor:
    :param epoch:
    :return:
    """
    # get mean across epoch
    if epoch == 'day':
        mean_tensor = utils.simple_mean_per_day(meta, pref_tensor)
        assert NotImplementedError  # right now use of mean_early_traces is thinking about stages
    else:
        assert epoch in ['parsed_stage', 'parsed_10stage', 'parsed_11stage']
        mean_tensor = utils.simple_mean_per_stage(meta, pref_tensor, staging=epoch)

    # get average of L0 and L1 learning
    mean_early_traces = np.nanmean(mean_tensor[:, :, 0:2], axis=2)

    # average pairwise corr across stages, same cell
    corr1_avgs = np.zeros((mean_tensor.shape[0], mean_tensor.shape[2]))
    corr1_avgs[:] = np.nan
    for cell_n in range(mean_tensor.shape[0]):

        # only use cells that we have early L0 or L1
        # TODO could add other stages here
        if all(np.isnan(mean_tensor[cell_n, 0, 0:2])):
            continue

        # build a timepoints x stage matrix for each cell, only look beyond stage L0/L1
        corr = np.zeros((mean_tensor.shape[1], len(range(mean_tensor.shape[2])[2:]) + 1))
        for c, s in enumerate(range(mean_tensor.shape[2])[2:]):
            corr[:, c] = mean_tensor[cell_n, :, s]
        corr[:, -1] = mean_early_traces[cell_n, :]  # add your early stage to the front

        # account for nan stages
        fill_inds = np.where(~np.isnan(corr[0, :]))[0]
        corrmat = np.corrcoef(corr[:, fill_inds].T)
        if corrmat.size <= 1:
            continue  # avoid only correlations with self

        # get pairwise correlation of all stages with early stages, excluding self (early-early)
        early_stage_corr_nonnan = corrmat[-1, :-1]
        #         np.fill_diagonal(corrmat, np.nan)  # inplace
        #         mean_corr_vec_same_cell_other_stages = np.nanmean(corrmat, axis=0)

        # add to matrix for all cells
        corr1_avgs[cell_n, fill_inds[:-1] + 2] = early_stage_corr_nonnan  # + 2 will keep stages aligned with nan early

    return corr1_avgs


def correlate_epochs_wlate(meta, pref_tensor, epoch='parsed_11stage'):
    """
    Correlate each cell's average temporal trace per epoch with it's average on L5 learning.

    :param meta:
    :param pref_tensor:
    :param epoch:
    :return:
    """
    # get mean across epoch
    if epoch == 'day':
        mean_tensor = utils.simple_mean_per_day(meta, pref_tensor)
        assert NotImplementedError  # right now use of mean_early_traces is thinking about stages
    else:
        assert epoch in ['parsed_stage', 'parsed_10stage', 'parsed_11stage']
        mean_tensor = utils.simple_mean_per_stage(meta, pref_tensor, staging=epoch)

    # average pairwise corr across stages, same cell
    corr1_avgs = np.zeros((mean_tensor.shape[0], mean_tensor.shape[2]))
    corr1_avgs[:] = np.nan
    for cell_n in range(mean_tensor.shape[0]):

        # only use cells that we have on L5
        if np.isnan(mean_tensor[cell_n, 0, 5]):
            continue

        # build a timepoints x stage matrix for each cell
        corr = np.zeros((mean_tensor.shape[1], mean_tensor.shape[2]))
        for c in range(mean_tensor.shape[2]):
            corr[:, c] = mean_tensor[cell_n, :, c]

        # account for nan stages
        fill_inds = np.where(~np.isnan(corr[0, :]))[0]
        corrmat = np.corrcoef(corr[:, fill_inds].T)
        if corrmat.size <= 1:
            continue  # avoid only correlations with self

        # get pairwise correlation of all stages with early stages
        np.fill_diagonal(corrmat, np.nan)  # inplace
        learning_L5 = corrmat[fill_inds == 5, :]

        # add to matrix for all cells
        corr1_avgs[cell_n, fill_inds] = learning_L5

    return corr1_avgs


def correlate_epochs_wall_mean(meta, pref_tensor, epoch='parsed_11stage'):
    """
    Correlate each cell's average temporal trace per epoch pairwise with all other stages/epoch. Take mean for
    each stage, excluding self-self correlation.

    :param meta:
    :param pref_tensor:
    :param epoch:
    :return:
    """
    # get mean across epoch
    if epoch == 'day':
        mean_tensor = utils.simple_mean_per_day(meta, pref_tensor)
        assert NotImplementedError  # right now use of mean_early_traces is thinking about stages
    else:
        assert epoch in ['parsed_stage', 'parsed_10stage', 'parsed_11stage']
        mean_tensor = utils.simple_mean_per_stage(meta, pref_tensor, staging=epoch)

    # average pairwise corr across stages, same cell
    corr1_avgs = np.zeros((mean_tensor.shape[0], mean_tensor.shape[2]))
    corr1_avgs[:] = np.nan
    for cell_n in range(mean_tensor.shape[0]):

        # build a timepoints x stage matrix for each cell
        corr = np.zeros((mean_tensor.shape[1], mean_tensor.shape[2]))
        for c in range(mean_tensor.shape[2]):
            corr[:, c] = mean_tensor[cell_n, :, c]

        # account for nan stages
        fill_inds = np.where(~np.isnan(corr[0, :]))[0]
        corrmat = np.corrcoef(corr[:, fill_inds].T)
        if corrmat.size <= 1:
            continue  # avoid only correlations with self

        # get mean pairwise correlation of all stages, excluding self
        np.fill_diagonal(corrmat, np.nan)  # inplace
        mean_corr_same_cell_other_stages = np.nanmean(corrmat, axis=0)

        # add to matrix for all cells
        corr1_avgs[cell_n, fill_inds] = mean_corr_same_cell_other_stages

    return corr1_avgs


def correlate_epochs_wall_control(meta, pref_tensor, epoch='parsed_11stage'):
        """
        Correlate each cell's pairwise with every other cell for the same stages/epoch. Take mean for
        each stage, excluding self-self correlation.

        :param meta:
        :param pref_tensor:
        :param epoch:
        :return:
        """
        # get mean across epoch
        if epoch == 'day':
            mean_tensor = utils.simple_mean_per_day(meta, pref_tensor)
            assert NotImplementedError  # right now use of mean_early_traces is thinking about stages
        else:
            assert epoch in ['parsed_stage', 'parsed_10stage', 'parsed_11stage']
            mean_tensor = utils.simple_mean_per_stage(meta, pref_tensor, staging=epoch)

        # average pairwise corr across given stage for each cell ("population" correlation)
        corr2_avgs = np.zeros((mean_tensor.shape[0], mean_tensor.shape[2]))
        corr2_avgs[:] = np.nan
        for c in range(mean_tensor.shape[2]):

            # build a cells x timepoints matrix for each stage
            corr = mean_tensor[:, :, c]

            # account for nan cells
            cell_inds = np.where(~np.isnan(corr[:, 0]))[0]
            corrmat = np.corrcoef(corr[cell_inds, :])
            if corrmat.size <= 1:
                continue  # avoid only correlations with self
            np.fill_diagonal(corrmat, np.nan)  # inplace
            mean_corr_vec_same_stage_other_cells = np.nanmean(corrmat, axis=0)

            # add to matrix for all cells
            corr2_avgs[cell_inds, c] = mean_corr_vec_same_stage_other_cells

        return corr2_avgs


def correlation_noise(meta, pref_tensor, epoch='parsed_11stage', min_denom=0.01):

    mean_t_tensor = utils.tensor_mean_per_trial(meta, pref_tensor, nan_licking=False, account_for_offset=True)

    if epoch == 'days':
        days = meta.reset_index()['date'].unique()
        new_mat = np.zeros((pref_tensor.shape[0], len(days)))
        new_mat[:] = np.nan
        for c, di in enumerate(meta.reset_index()['date'].unique()):
            day_boo = meta.reset_index()['date'].isin([di]).values
            day_mat = np.nanmean(mean_t_tensor[:, day_boo])
            noise_mat = day_mat - np.nanmean(day_mat, axis=1)  # mean subtract
            # TODO loop over cells with same pmn trials
            new_mat[:, c] = fano
    elif epoch in ['parsed_stage', 'parsed_10stage', 'parsed_11stage']:
        # get average response per stage
        stages = lookups.staging[epoch]
        new_mat = np.zeros((pref_tensor.shape[0], len(stages)))
        new_mat[:] = np.nan
        for c, di in enumerate(stages):
            stage_boo = meta[epoch].isin([di]).values
            stage_days = meta.loc[stage_boo, :].reset_index()['date'].unique()
            day_fano = np.zeros((pref_tensor.shape[0], len(stage_days)))
            day_fano[:] = np.nan
            for c2, di2 in enumerate(stage_days):
                day_boo = meta.reset_index()['date'].isin([di2]).values
                denom = np.nanmean(mean_t_tensor[:, stage_boo & day_boo], axis=1)
                fano = np.abs(np.nanvar(mean_t_tensor[:, stage_boo & day_boo], axis=1) / denom)
                fano[denom < min_denom] = np.nan
                day_fano[:, c2] = fano
            new_mat[:, c] = np.nanmean(day_fano, axis=1)
    else:
        raise NotImplementedError

    return new_mat


def correlate_wrunning_per_trial(meta, pref_tensor, epoch='parsed_11stage', account_for_offset=True):

    mean_t_tensor = utils.tensor_mean_per_trial(meta, pref_tensor, nan_licking=False,
                                                account_for_offset=account_for_offset)

    if epoch == 'days':
        days = meta.reset_index()['date'].unique()
        new_mat = np.zeros((pref_tensor.shape[0], len(days)))
        new_mat[:] = np.nan
        for c, di in enumerate(meta.reset_index()['date'].unique()):
            day_boo = meta.reset_index()['date'].isin([di]).values
            day_mat = mean_t_tensor[:, day_boo]
            day_speed = meta.loc[day_boo, 'speed'].values

            # correlate each cell with running speed dropping nans (aka un-preferred cues)
            for celli in range(day_mat.shape[0]):
                cell_and_run = np.concatenate([day_mat[celli][:, None], day_speed[:, None]], axis=1)
                cell_and_run = cell_and_run[~np.isnan(np.mean(cell_and_run, axis=1))]
                if len(cell_and_run < 10): # arbitrary use of 10, just make sure that the cell is not empty, ~30-120 trials per cue per day
                    continue
                # correlate mean trial values and running speed
                corr = np.corrcoef(cell_and_run)
                assert corr.shape[0] == 2 and corr.shape[0] == 2
                new_mat[celli, c] = corr[0, 0]

    elif epoch in ['parsed_stage', 'parsed_10stage', 'parsed_11stage']:
        # get average response per stage
        stages = lookups.staging[epoch]
        new_mat = np.zeros((pref_tensor.shape[0], len(stages)))
        new_mat[:] = np.nan

        for c, di in enumerate(stages):
            stage_boo = meta[epoch].isin([di]).values
            stage_days = meta.loc[stage_boo, :].reset_index()['date'].unique()
            corr_days = np.zeros((pref_tensor.shape[0], len(stage_days)))
            corr_days[:] = np.nan

            for c2, di2 in enumerate(stage_days):
                day_boo = meta.reset_index()['date'].isin([di2]).values
                day_mat = mean_t_tensor[:, day_boo]
                day_speed = meta.loc[day_boo, 'speed'].values

                # correlate each cell with running speed dropping nans (aka un-preferred cues)
                for celli in range(day_mat.shape[0]):

                    # stick cell dff values and running speed together
                    cell_and_run = np.concatenate([day_mat[celli, :][:, None], day_speed[:, None]], axis=1)

                    # drop nans for any trial missing cell value (aka un-preferred cue) or speed
                    cell_and_run = cell_and_run[~np.isnan(np.mean(cell_and_run, axis=1))]

                    # check there is still a cell or speed to correlate
                    # arbitrary use of 10, just make sure that the cell is not empty, ~30-120 trials per cue per day
                    if len(cell_and_run) < 10:
                        continue

                    # correlate mean trial values and running speed
                    corr = np.corrcoef(cell_and_run.T)

                    # double check that you had the inputs about .T-ed correctly
                    assert corr.shape[0] == 2 and corr.shape[0] == 2

                    # hold onto values for each day of a stage, grabbing off diag value 
                    corr_days[celli, c2] = corr[1, 0]

            # take mean correlation across days for a stage
            new_mat[:, c] = np.nanmean(corr_days, axis=1)
    else:
        raise NotImplementedError

    return new_mat


def correlate_pre_running_per_trial(meta, pref_tensor, epoch='parsed_11stage', account_for_offset=True):

    mean_t_tensor = utils.tensor_mean_per_trial(meta, pref_tensor, nan_licking=False,
                                                account_for_offset=account_for_offset)

    if epoch == 'days':
        days = meta.reset_index()['date'].unique()
        new_mat = np.zeros((pref_tensor.shape[0], len(days)))
        new_mat[:] = np.nan
        for c, di in enumerate(meta.reset_index()['date'].unique()):
            day_boo = meta.reset_index()['date'].isin([di]).values
            day_mat = mean_t_tensor[:, day_boo]
            day_speed = meta.loc[day_boo, 'pre_speed'].values

            # correlate each cell with running speed dropping nans (aka un-preferred cues)
            for celli in range(day_mat.shape[0]):
                cell_and_run = np.concatenate([day_mat[celli][:, None], day_speed[:, None]], axis=1)
                cell_and_run = cell_and_run[~np.isnan(np.mean(cell_and_run, axis=1))]
                if len(cell_and_run < 10): # arbitrary use of 10, just make sure that the cell is not empty, ~30-120 trials per cue per day
                    continue
                # correlate mean trial values and running speed
                corr = np.corrcoef(cell_and_run)
                assert corr.shape[0] == 2 and corr.shape[0] == 2
                new_mat[celli, c] = corr[0, 0]

    elif epoch in ['parsed_stage', 'parsed_10stage', 'parsed_11stage']:
        # get average response per stage
        stages = lookups.staging[epoch]
        new_mat = np.zeros((pref_tensor.shape[0], len(stages)))
        new_mat[:] = np.nan

        for c, di in enumerate(stages):
            stage_boo = meta[epoch].isin([di]).values
            stage_days = meta.loc[stage_boo, :].reset_index()['date'].unique()
            corr_days = np.zeros((pref_tensor.shape[0], len(stage_days)))
            corr_days[:] = np.nan

            for c2, di2 in enumerate(stage_days):
                day_boo = meta.reset_index()['date'].isin([di2]).values
                day_mat = mean_t_tensor[:, day_boo]
                day_speed = meta.loc[day_boo, 'pre_speed'].values

                # correlate each cell with running speed dropping nans (aka un-preferred cues)
                for celli in range(day_mat.shape[0]):

                    # stick cell dff values and running speed together
                    cell_and_run = np.concatenate([day_mat[celli, :][:, None], day_speed[:, None]], axis=1)

                    # drop nans for any trial missing cell value (aka un-preferred cue) or speed
                    cell_and_run = cell_and_run[~np.isnan(np.mean(cell_and_run, axis=1))]

                    # check there is still a cell or speed to correlate
                    # arbitrary use of 10, just make sure that the cell is not empty, ~30-120 trials per cue per day
                    if len(cell_and_run) < 10:
                        continue

                    # correlate mean trial values and running speed
                    corr = np.corrcoef(cell_and_run.T)

                    # double check that you had the inputs about .T-ed correctly
                    assert corr.shape[0] == 2 and corr.shape[0] == 2

                    # hold onto values for each day of a stage, grabbing off diag value
                    corr_days[celli, c2] = corr[1, 0]

            # take mean correlation across days for a stage
            new_mat[:, c] = np.nanmean(corr_days, axis=1)
    else:
        raise NotImplementedError

    return new_mat


def correlate_wlicking_per_trial(meta, pref_tensor, epoch='parsed_11stage'):

    mean_t_tensor = utils.tensor_mean_per_trial(meta, pref_tensor, nan_licking=False, account_for_offset=True)

    if epoch == 'days':
        days = meta.reset_index()['date'].unique()
        new_mat = np.zeros((pref_tensor.shape[0], len(days)))
        new_mat[:] = np.nan
        for c, di in enumerate(meta.reset_index()['date'].unique()):
            day_boo = meta.reset_index()['date'].isin([di]).values
            day_mat = mean_t_tensor[:, day_boo]
            day_speed = meta.loc[day_boo, 'anticipatory_licks'].values

            # correlate each cell with running speed dropping nans (aka un-preferred cues)
            for celli in range(day_mat.shape[0]):
                cell_and_run = np.concatenate([day_mat[celli][:, None], day_speed[:, None]], axis=1)
                cell_and_run = cell_and_run[~np.isnan(np.mean(cell_and_run, axis=1))]
                if len(cell_and_run < 10): # arbitrary use of 10, just make sure that the cell is not empty, ~30-120 trials per cue per day
                    continue
                # correlate mean trial values and running speed
                corr = np.corrcoef(cell_and_run)
                assert corr.shape[0] == 2 and corr.shape[0] == 2
                new_mat[celli, c] = corr[0, 0]

    elif epoch in ['parsed_stage', 'parsed_10stage', 'parsed_11stage']:
        # get average response per stage
        stages = lookups.staging[epoch]
        new_mat = np.zeros((pref_tensor.shape[0], len(stages)))
        new_mat[:] = np.nan

        for c, di in enumerate(stages):
            stage_boo = meta[epoch].isin([di]).values
            stage_days = meta.loc[stage_boo, :].reset_index()['date'].unique()
            corr_days = np.zeros((pref_tensor.shape[0], len(stage_days)))
            corr_days[:] = np.nan

            for c2, di2 in enumerate(stage_days):
                day_boo = meta.reset_index()['date'].isin([di2]).values
                day_mat = mean_t_tensor[:, day_boo]
                day_speed = meta.loc[day_boo, 'anticipatory_licks'].values

                # correlate each cell with running speed dropping nans (aka un-preferred cues)
                for celli in range(day_mat.shape[0]):

                    # stick cell dff values and running speed together
                    cell_and_run = np.concatenate([day_mat[celli, :][:, None], day_speed[:, None]], axis=1)

                    # drop nans for any trial missing cell value (aka un-preferred cue) or speed
                    cell_and_run = cell_and_run[~np.isnan(np.mean(cell_and_run, axis=1))]

                    # check there is still a cell or speed to correlate
                    # arbitrary use of 10, just make sure that the cell is not empty, ~30-120 trials per cue per day
                    if len(cell_and_run) < 10:
                        continue

                    # correlate mean trial values and running speed
                    corr = np.corrcoef(cell_and_run.T)

                    # double check that you had the inputs about .T-ed correctly
                    assert corr.shape[0] == 2 and corr.shape[0] == 2

                    # hold onto values for each day of a stage, grabbing off diag value
                    corr_days[celli, c2] = corr[1, 0]

            # take mean correlation across days for a stage
            new_mat[:, c] = np.nanmean(corr_days, axis=1)
    else:
        raise NotImplementedError

    return new_mat


def correlate_wlickingonset_per_trial(meta, pref_tensor, epoch='parsed_11stage'):

    mean_t_tensor = utils.tensor_mean_per_trial(meta, pref_tensor, nan_licking=False, account_for_offset=True)

    if epoch == 'days':
        days = meta.reset_index()['date'].unique()
        new_mat = np.zeros((pref_tensor.shape[0], len(days)))
        new_mat[:] = np.nan
        for c, di in enumerate(meta.reset_index()['date'].unique()):
            day_boo = meta.reset_index()['date'].isin([di]).values
            day_mat = mean_t_tensor[:, day_boo]
            day_speed = meta.loc[day_boo, 'firstlickbout'].values

            # correlate each cell with running speed dropping nans (aka un-preferred cues)
            for celli in range(day_mat.shape[0]):
                cell_and_run = np.concatenate([day_mat[celli][:, None], day_speed[:, None]], axis=1)
                cell_and_run = cell_and_run[~np.isnan(np.mean(cell_and_run, axis=1))]
                if len(cell_and_run < 10): # arbitrary use of 10, just make sure that the cell is not empty, ~30-120 trials per cue per day
                    continue
                # correlate mean trial values and running speed
                corr = np.corrcoef(cell_and_run)
                assert corr.shape[0] == 2 and corr.shape[0] == 2
                new_mat[celli, c] = corr[0, 0]

    elif epoch in ['parsed_stage', 'parsed_10stage', 'parsed_11stage']:
        # get average response per stage
        stages = lookups.staging[epoch]
        new_mat = np.zeros((pref_tensor.shape[0], len(stages)))
        new_mat[:] = np.nan

        for c, di in enumerate(stages):
            stage_boo = meta[epoch].isin([di]).values
            stage_days = meta.loc[stage_boo, :].reset_index()['date'].unique()
            corr_days = np.zeros((pref_tensor.shape[0], len(stage_days)))
            corr_days[:] = np.nan

            for c2, di2 in enumerate(stage_days):
                day_boo = meta.reset_index()['date'].isin([di2]).values
                day_mat = mean_t_tensor[:, day_boo]
                day_speed = meta.loc[day_boo, 'firstlickbout'].values

                # correlate each cell with running speed dropping nans (aka un-preferred cues)
                for celli in range(day_mat.shape[0]):

                    # stick cell dff values and running speed together
                    cell_and_run = np.concatenate([day_mat[celli, :][:, None], day_speed[:, None]], axis=1)

                    # drop nans for any trial missing cell value (aka un-preferred cue) or speed
                    cell_and_run = cell_and_run[~np.isnan(np.mean(cell_and_run, axis=1))]

                    # check there is still a cell or speed to correlate
                    # arbitrary use of 10, just make sure that the cell is not empty, ~30-120 trials per cue per day
                    if len(cell_and_run) < 10:
                        continue

                    # correlate mean trial values and running speed
                    corr = np.corrcoef(cell_and_run.T)

                    # double check that you had the inputs about .T-ed correctly
                    assert corr.shape[0] == 2 and corr.shape[0] == 2

                    # hold onto values for each day of a stage, grabbing off diag value
                    corr_days[celli, c2] = corr[1, 0]

            # take mean correlation across days for a stage
            new_mat[:, c] = np.nanmean(corr_days, axis=1)
    else:
        raise NotImplementedError

    return new_mat


def mean_stim_and_response(meta, pref_tensor, epoch='parsed_11stage', full_trial_window=False):
    """
    Get cell's mean response during the stimulus and during the response window.

    :param meta: pandas.DataFrame
        DataFrame of trial metadata
    :param pref_tensor: numpy.ndarray
        A tensor the exact size of the tensor input, now containing nans for un-preferred stimulus presentations.
    :param epoch: str
        'day' or staging type to take averages overs
    :param full_trial_window: boolean
        Use the stimulus window + response window to calculate transientness.
    :return: peak_time, half_peak_time, numpy.ndarrays
        cells x stages/days/epochs matrices of peak latency RELATIVE TO STIMULUS ONSET OR OFFSET depending
        on type of cell.
    """

    # define windows for checking peak
    mouse = meta.reset_index()['mouse'].unique()[0]
    if full_trial_window:
        response_window = np.arange(16, np.floor(15.5 * (3 + lookups.stim_length[mouse])), dtype='int')
        stimulus_window = np.arange(16, np.floor(15.5 * (3 + lookups.stim_length[mouse])), dtype='int')
    else:
        response_window = np.arange(np.floor(15.5 * (1 + lookups.stim_length[mouse])),
                                    np.floor(15.5 * (3 + lookups.stim_length[mouse])), dtype='int')
        stimulus_window = np.arange(16, np.floor(15.5 * (1 + lookups.stim_length[mouse])), dtype='int')

    # get mean across epoch
    if epoch == 'day':
        mean_tensor = utils.simple_mean_per_day(meta, pref_tensor)
    else:
        assert epoch in ['parsed_stage', 'parsed_10stage', 'parsed_11stage']
        mean_tensor = utils.simple_mean_per_stage(meta, pref_tensor, staging=epoch)

    offset_response = np.nanmean(mean_tensor[:, response_window, :], axis=1)
    stimulus_response = np.nanmean(mean_tensor[:, stimulus_window, :], axis=1)

    return stimulus_response, offset_response


def sine_fit_stage_response(meta, pref_tensor, epoch='parsed_11stage', plot_please=False):
    """
    Fit a sinusoidal function to the sustained persiod of a stimulus driven cell's response.

    :param meta: pandas.DataFrame
        DataFrame of trial metadata
    :param pref_tensor: numpy.ndarray
        A tensor the exact size of the tensor input, now containing nans for un-preferred stimulus presentations.
    :param epoch: str
        'day' or staging type to take averages overs
    :param full_trial_window: boolean
        Use the stimulus window + response window to calculate transientness.
    :return: 
    """

    # determine cells with offset responses
    offset_bool = utils.get_offset_cells(meta, pref_tensor)

    # get mean across epoch
    if epoch == 'day':
        mean_tensor = utils.simple_mean_per_day(meta, pref_tensor)
        xorder = np.arange(mean_tensor.shape[2])
    else:
        assert epoch in ['parsed_stage', 'parsed_10stage', 'parsed_11stage']
        mean_tensor = utils.simple_mean_per_stage(meta, pref_tensor, staging=epoch)
        xorder = lookups.staging[epoch]

    # for each cell fit each stage that you have data for
    fit_mat = np.zeros((mean_tensor.shape[0], len(xorder), 7))  # 5 + 1 + 1 = 7, params plus varex, plus a mean
    fit_mat[:] = np.nan
    for celli in range(mean_tensor.shape[0]):
        if plot_please:
            plt_flag = False
            plt.figure(figsize=(6, 4))
        for c_s, s in enumerate(xorder):

            # define chunk of cell data
            pref_chunk = mean_tensor[celli, 27:47, c_s]

            # skip missing data
            if np.isnan(pref_chunk).any():
                continue

            # skip cells with a peak after stimulus offset
            if offset_bool[celli]:
                continue

            # skip supressed cells, checking first 1.5 s of stim
            chunk_mean = np.nanmean(mean_tensor[celli, 16:38, c_s])
            if chunk_mean < 0:
                continue
            # fit_mat[celli, c_s, -1] = chunk_mean

            # define x range of stimulus to fit over (in units of seconds)
            x_data = (np.arange(27, 47, 1) - 15.5) / 15.5

            # fit variation on sin function to each cell
            # if norm_please:
            #     params, params_covariance = optimize.curve_fit(_sin_func, x_data, pref_chunk,
            #                                                    p0=[0.1, 12, 0.2, 0.1, 0.1],
            #                                                    bounds=(
            #                                                    (0.01, 11, -0.3, -1, -0.5), (0.3, 14.3, 1, 1, 1.5)))
            # c changed to -0.3 as lower bound to improve fit speed. (previous fits have almost no datapoint below this)
            # else:
            params, params_covariance = optimize.curve_fit(_sin_func, x_data, pref_chunk,
                                                           p0=[0.1, 12, 0.2, 0.1, 0.1],
                                                           bounds=((0.01, 10, 0, -10, 0), (10, 15, 5, 10, 10)))
            fit_mat[celli, c_s, :-2] = params

            # calculate variance explained
            y_fit = _sin_func(x_data, params[0], params[1], params[2], params[3], params[4])
            fit_mat[celli, c_s, -1] = np.nanmean(y_fit)  # save mean "firing rate" F0 according to fit

            # get r-squared
            ss_res = np.sum((pref_chunk - y_fit) ** 2)  # residual sum of squares
            ss_tot = np.sum((pref_chunk - np.mean(pref_chunk)) ** 2)  # total sum of squares
            r2 = 1 - (ss_res / ss_tot)  # r-squared

            fit_mat[celli, c_s, 5] = r2

            # have to hit this at least once for a cell to save a plot
            plt_flag = True

            if plot_please:
                # have to hit this at least once for a cell to save a plot
                plt_flag = True
                plt.plot(x_data, y_fit,
                     label=f'{s}: {[round(ss, 2) for ss in params]} --> {round(r2, 2)}')
                full_x = (np.arange(0, 108, 1) - 15.5) / 15.5
                plt.scatter(full_x, mean_tensor[celli, :, c_s])

        if plot_please:
            if plt_flag:
                plt.legend(title='parameters --> variance explained\n' +
                         '      a*sin(b*(x + c)) + d*x + e', loc='best', bbox_to_anchor=(1.05, 1))
                mouse = meta.reset_index()['mouse'].unique()[0]
                plt.title(f'{mouse}: cell index {celli}\nMean response per stage with fits')
                plt.xlabel('time from stimulus onset (sec)')
                plt.ylabel('\u0394F/F (z-score)')
                folderpath = f'/twophoton_analysis/Data/analysis/Group-attractive/sine_fits/{mouse}/'
                if not os.path.isdir(folderpath):
                    os.mkdir(folderpath)
                plt.savefig(os.path.join(folderpath, f'cell{celli}_sine.png'), bbox_inches='tight')
    if plot_please:
        plt.close('all')

    return fit_mat


def _sin_func(x, a, b, c, d, e):
    return a * np.sin(b * (x + c)) + d*x + e

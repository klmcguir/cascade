import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .. import utils, lookups

def run_controlled_reversal_mismatch_traces(meta, pref_tensor, filter_licking=None, filter_running=None,
                                    filter_hmm_engaged=True, force_same_day_reversal=False,
                                    use_stages_for_reversal=False, skew_stages_for_reversal=True,
                                     boot=False):
    """
    Calculate a mismatch binning running and calculating between matched bins, then averaging across bins

    :param meta:
    :param pref_tensor:
    :param filter_running:
    :param filter_licking:
    :param filter_hmm_engaged:
    :param force_same_day_reversal:
    :return:
    reversal_mismatch
    """

    # get mouse from metadata
    mouse = meta.reset_index()['mouse'].unique()[0]

    # mean_t_tensor = utils.tensor_mean_per_trial(meta, pref_tensor, nan_licking=False, account_for_offset=True)

    # get day of reversal
    if force_same_day_reversal:
        post_rev = meta.reset_index()['date'].mod(1).isin([0.5]).values
        if np.sum(post_rev) == 0:
            out = np.zeros(pref_tensor.shape[0])
            out[:] = np.nan
            print(f'Mouse {mouse} did not have single-day reversal.')
            return out
        rev_date = meta.reset_index().loc[post_rev, 'date'].unique()[0] - 0.5
        pre_rev = meta.reset_index()['date'].isin([rev_date]).values
    elif use_stages_for_reversal:
        assert 'parsed_11stage' in meta.columns
        pre_rev = meta.parsed_11stage.isin(['L5 learning']).values
        post_rev = meta.parsed_11stage.isin(['L1 reversal1']).values
    elif skew_stages_for_reversal:
        pre_rev = meta.parsed_11stage.isin(['L5 learning']).values
        rev_vec = meta.reset_index()['learning_state'].isin(['reversal1']).values
        if all(~rev_vec):
            out = np.zeros(pref_tensor.shape[0])
            out[:] = np.nan
            print(f'Mouse {mouse} did not have any reversal.')
            if boot:
                return out, out
            return out
        post_rev_date = meta.reset_index().loc[rev_vec, 'date'].iloc[0]
        post_rev = meta.reset_index()['date'].isin([post_rev_date]).values
        post_rev[np.where(post_rev)[0][100:]] = False
    else:
        learning_vec = meta.reset_index()['learning_state'].isin(['learning']).values
        rev_date = meta.reset_index().loc[learning_vec, 'date'].iloc[-1]
        pre_rev = meta.reset_index()['date'].isin([rev_date]).values
        rev_vec = meta.reset_index()['learning_state'].isin(['reversal1']).values
        if all(~rev_vec):
            out = np.zeros(pref_tensor.shape[0])
            out[:] = np.nan
            print(f'Mouse {mouse} did not have any reversal.')
            return out
        post_rev_date = meta.reset_index().loc[rev_vec, 'date'].iloc[0]
        post_rev = meta.reset_index()['date'].isin([post_rev_date]).values
    print(f'{mouse}: pre-rev: {np.sum(pre_rev)}, post-rev: {np.sum(post_rev)}')
    rev_day = (pre_rev | post_rev)

    # filter to include fixed running type: low or high
    if filter_running is not None:
        speed_cm_s = meta.speed.values
        if filter_running == 'low_speed_only':
            rev_day = rev_day & (speed_cm_s <= 6)  # was 4
            print('WARNING low_speed_only set to 6 cm/s')
        elif filter_running == 'high_speed_only':
            rev_day = rev_day & (speed_cm_s > 20)  # was 10
            print('WARNING high_speed_only set to 20 cm/s')
        else:
            raise NotImplementedError

    # filter to include fixed licking type: low or high
    if filter_licking is not None:
        # TODO this needs accounting for offset licking for offset cells
        # TODO could also do a grid of lick and run bins to make comparisons
        mean_lick_rate = meta.anticipatory_licks.values / lookups.stim_length[mouse]
        if filter_licking == 'low_lick_only':
            rev_day = rev_day & (mean_lick_rate <= 1.7)
        elif filter_licking == 'high_lick_only':
            rev_day = rev_day & (mean_lick_rate > 1.7)
        else:
            raise NotImplementedError

    # animal must be engaged in the task (or naive when it can't "engage")
    if filter_hmm_engaged:
        rev_day = rev_day & (meta.hmm_engaged.values | meta.learning_state.isin(['naive']).values)

    print(f'  -->  {mouse}: pre-rev: {np.sum(pre_rev & rev_day)}, post-rev: {np.sum(post_rev & rev_day)}')

    # calculate mean for preferred cue for each cell across reversal
    pre_bins = utils.bin_running_traces_calc(meta, pref_tensor, (rev_day & pre_rev))
    post_bins = utils.bin_running_traces_calc(meta, pref_tensor, (rev_day & post_rev))
    # TODO may want to NOT do this across running always so you can get a trace with SEM error bars

    pre_sem = np.nanstd(pref_tensor[:, :, (rev_day & pre_rev)], axis=2) \
        / np.sqrt(np.sum(~np.isnan(pref_tensor[:, :, (rev_day & pre_rev)]), axis=2))
    post_sem = np.nanstd(pref_tensor[:, :, (rev_day & post_rev)], axis=2) \
        / np.sqrt(np.sum(~np.isnan(pref_tensor[:, :, (rev_day & post_rev)]), axis=2))
    simple_pre_mean = np.nanmean(pref_tensor[:, :, (rev_day & pre_rev)], axis=2)
    simple_post_mean = np.nanmean(pref_tensor[:, :, (rev_day & post_rev)], axis=2)
    reversal_mismatch = np.nanmean(post_bins - pre_bins, axis=2)
    pre_collapse_bins = np.nanmean(pre_bins, axis=2)
    post_collapse_bins = np.nanmean(post_bins, axis=2)

    return [reversal_mismatch, pre_collapse_bins, post_collapse_bins,
            pre_sem, post_sem, simple_pre_mean, simple_post_mean]
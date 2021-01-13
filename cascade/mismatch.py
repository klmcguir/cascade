from copy import deepcopy

import numpy as np
import pandas as pd

from . import utils, tuning, lookups, drive

"""
In V1 cells, that respond with a depolarizing mismatch signal when visual flow is halted, also respond with a slight 
inhibition to visual flow (stimulus) presented alone. These responses also have an offset response. 
"""


def match_trials(meta, target_epoch='L1 reversal1', search_epoch='L5 learning', match_on='speed', tolerance=1):
    """
    Generate matched sets of trials based on metadata variables. Always matches for different cue presentations
    independently. i.e., Running speeds are matched for the same cue pre and post reversal, never across cues.

    # TODO could filter target and search trials on other variables before search.

    :param meta: pandas.DataFrame
        Trial metadata.
    :param target_epoch: str
        Set of trials to try and match.
    :param search_epoch: str
        Set of trials to search over.
    :param match_on: str
        Name of column in trial meta DataFrame to try and match on
    :param tolerance: float or int
        Threshold in units of [match_on]. Forces: target trial - tol < best matched trial < target trial + tol.
    :return: two vectors, one for each epoch, with integers that match trials between vectors
        s_had_match --> target trials [nan nan 1 5 nan ...] where 1s and 5s are paired trials
        matched_s --> searched trials [5 nan nan nan 1 ...]
    """

    # set up trials post reversal as target
    possible_post = meta.parsed_11stage.isin([target_epoch]).values
    if np.sum(possible_post) == 0:
        return [], []  # return empty if no reversal
    speed_copy_post = deepcopy(meta[match_on].values)
    speed_copy_post[~possible_post] = np.nan  # nan all values that you can't draw from
    post_inds_to_check = np.where(possible_post)[0]
    np.random.shuffle(post_inds_to_check)  # shuffle inds for for loop

    # set up trials pre reversal to search over
    possible_pre = meta.parsed_11stage.isin([search_epoch]).values
    speed_copy_pre = deepcopy(meta[match_on].values)
    speed_copy_pre[~possible_pre] = np.nan  # nan all values that you can't draw from

    # get copies of speed vectors for each cue
    speed_copy_plus_pre = deepcopy(speed_copy_pre)
    speed_copy_plus_pre[~meta.initial_condition.isin(['plus'])] = np.nan
    speed_copy_minus_pre = deepcopy(speed_copy_pre)
    speed_copy_minus_pre[~meta.initial_condition.isin(['minus'])] = np.nan
    speed_copy_neut_pre = deepcopy(speed_copy_pre)
    speed_copy_neut_pre[~meta.initial_condition.isin(['neutral'])] = np.nan
    speed_dict_pre = {'plus': speed_copy_plus_pre, 'minus': speed_copy_minus_pre, 'neutral': speed_copy_neut_pre, }

    # preallocate
    matched_s = np.zeros(len(meta))
    s_had_match = np.zeros(len(meta))
    match_counter = 1
    for indi in post_inds_to_check:

        # for each index match speed and cue type
        c_to_match = meta.initial_condition.values[indi]
        s_to_match = meta[match_on].values[indi]

        # find closest matched speed
        difference_to_target = speed_dict_pre[c_to_match] - s_to_match
        if all(np.isnan(difference_to_target)):
            continue
        closest_matched_ind = np.nanargmin(np.abs(difference_to_target))
        closest_matched_speed = speed_dict_pre[c_to_match][closest_matched_ind]

        # only use closest matched speed within tolerance (i.e., 1 cm/s)
        if (closest_matched_speed < s_to_match + tolerance) & (closest_matched_speed > s_to_match - tolerance):
            matched_s[closest_matched_ind] = match_counter
            s_had_match[indi] = match_counter
            speed_dict_pre[c_to_match][closest_matched_ind] = np.nan  # no replacement, blank trial
            match_counter += 1

    return s_had_match, matched_s


def trial_match_diff_over_stages(meta, pref_tensor, search_epoch='L5 learning'):

    # get mean trial response matrix, cells x trials
    mean_t_tensor = utils.tensor_mean_per_trial(meta, pref_tensor, nan_licking=False, account_for_offset=True)

    # preallocate
    diff_mat = np.zeros((mean_t_tensor.shape[0], len(lookups.staging['parsed_11stage']))) + np.nan
    for si, stage in enumerate(lookups.staging['parsed_11stage']):

        # calculate change between two stages, matching distribution of running speeds per cue
        s_had_match, matched_s = match_trials(meta, target_epoch=stage, search_epoch=search_epoch,
                                              match_on='speed', tolerance=1)

        # if a mouse is missing a stage skip calculation
        if len(s_had_match) == 0 or len(matched_s) == 0:
            continue

        # create booleans of possible trials to use
        pre_rev = matched_s > 0
        post_rev = s_had_match > 0

        # for each cell only use matched trials. If a cell is missing data in one period, drop matching trials in other.
        pre_mean = np.zeros(mean_t_tensor.shape[0]) + np.nan
        for celli in range(mean_t_tensor.shape[0]):
            existing_post_trials = s_had_match[~np.isnan(mean_t_tensor[celli, :])]
            existing_pre_trials = np.isin(matched_s, existing_post_trials)
            pre_mean[celli] = np.nanmean(mean_t_tensor[celli, pre_rev & existing_pre_trials])
            if np.sum(existing_pre_trials) < 10:
                print(f'WARNING: cell_n {celli}, only {np.sum(existing_pre_trials)} trials matched' +
                      ' pre_reversal after accounting for missing data post_reversal')
        post_mean = np.nanmean(mean_t_tensor[:, post_rev], axis=1)
        reversal_mismatch = post_mean - pre_mean
        diff_mat[:, si] = reversal_mismatch

    return diff_mat


def run_controlled_reversal_mismatch_traces(meta, pref_tensor, filter_licking=None, filter_running=None,
                                            filter_hmm_engaged=True, force_same_day_reversal=False,
                                            match_trials=True,
                                            use_stages_for_reversal=False, skew_stages_for_reversal=False,
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
    elif match_trials:
        # match_distribution of running values by matching trials
        post_rev = meta.parsed_11stage.isin(['L1 reversal']).values
        possible_pre = meta.parsed_11stage.isin(['L5 learning']).values
        speed_copy = deepcopy(meta.speed.values)
        speed_copy[~possible_pre] = np.nan  # nan all values that you can't draw from
        running_dist_post = sorted(meta.speed.loc[post_rev].values)
        pool_pre = np.sort(speed_copy)
        pool_pos_pre = np.argsort(speed_copy)  # actual position in the full vector

        matched_s = np.zeros(len(meta))
        available = np.zeros(len(meta)) == 0
        for s_to_match in running_dist_post:
            best_match_pos = pool_pos_pre[(pool_pre <= s_to_match) & available]
            best_match_speed = pool_pre[pool_pre <= s_to_match & available]

            # must be within 10% of original speed
            if (len(best_match_pos) > 0) & (best_match_speed < s_to_match * 1.1) & (
                    best_match_speed > s_to_match * 0.9):
                matched_s[best_match_pos] = 1
                available[pool_pos_pre == best_match_pos] = False  # set used position to unavailable
            else:
                print(f'Unmatched trial speed: {s_to_match} cm/s, best match {best_match_speed} cm/s')
        pre_rev = matched_s == 1
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


def run_controlled_reversal_mismatch(meta, pref_tensor, filter_licking=None, filter_running=None,
                                     filter_hmm_engaged=True, force_same_day_reversal=False,
                                     match_trials=True,
                                     use_stages_for_reversal=False, skew_stages_for_reversal=False,
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

    mean_t_tensor = utils.tensor_mean_per_trial(meta, pref_tensor, nan_licking=False, account_for_offset=True)

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
        # TODO limit pre rev to the first 1000 trials
    elif match_trials:
        # set up trials post reversal as target
        possible_post = meta.parsed_11stage.isin(['L1 reversal1']).values
        if np.sum(possible_post) == 0:
            return np.zeros(mean_t_tensor.shape[0]) + np.nan  # return np.nan vec if no reversal
        speed_copy_post = deepcopy(meta.speed.values)
        speed_copy_post[~possible_post] = np.nan  # nan all values that you can't draw from
        post_inds_to_check = np.where(possible_post)[0]
        np.random.shuffle(post_inds_to_check)  # shuffle inds for for loop

        # set up trials pre reversal to search over
        possible_pre = meta.parsed_11stage.isin(['L5 learning']).values
        speed_copy_pre = deepcopy(meta.speed.values)
        speed_copy_pre[~possible_pre] = np.nan  # nan all values that you can't draw from

        # get copies of speed vectors for each cue
        speed_copy_plus_pre = deepcopy(speed_copy_pre)
        speed_copy_plus_pre[~meta.initial_condition.isin(['plus'])] = np.nan
        speed_copy_minus_pre = deepcopy(speed_copy_pre)
        speed_copy_minus_pre[~meta.initial_condition.isin(['minus'])] = np.nan
        speed_copy_neut_pre = deepcopy(speed_copy_pre)
        speed_copy_neut_pre[~meta.initial_condition.isin(['neutral'])] = np.nan
        speed_dict_pre = {'plus': speed_copy_plus_pre, 'minus': speed_copy_minus_pre, 'neutral': speed_copy_neut_pre, }

        # preallocate
        matched_s = np.zeros(len(meta))
        s_had_match = np.zeros(len(meta))
        match_counter = 1
        for indi in post_inds_to_check:

            # for each index match speed and cue type
            c_to_match = meta.initial_condition.values[indi]
            s_to_match = meta.speed.values[indi]

            # find closest matched speed
            difference_to_target = speed_dict_pre[c_to_match] - s_to_match
            if all(np.isnan(difference_to_target)):
                continue
            closest_matched_ind = np.nanargmin(np.abs(difference_to_target))
            closest_matched_speed = speed_dict_pre[c_to_match][closest_matched_ind]

            # only use closest matched speed within 1 cm/s
            if (closest_matched_speed < s_to_match + 1) & (closest_matched_speed > s_to_match - 1):
                # TODO could make this a counter so trials are matched so you could match cell by cell for trial count 
                matched_s[closest_matched_ind] = match_counter
                s_had_match[indi] = match_counter
                speed_dict_pre[c_to_match][closest_matched_ind] = np.nan  # no replacement, blank trial
                match_counter += 1
            # else:
            #     print(f'Unmatched trial speed: {s_to_match} cm/s, best match {closest_matched_speed} cm/s')

        pre_rev = matched_s > 0
        post_rev = s_had_match > 0
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
    if match_trials:

        # for each cell only use matched trials. If a cell is missing data in one period, drop matching trials in other.
        pre_mean = np.zeros(mean_t_tensor.shape[0]) + np.nan
        # post_mean = np.zeros(mean_t_tensor.shape[0]) + np.nan
        for celli in range(mean_t_tensor.shape[0]):
            existing_post_trials = s_had_match[~np.isnan(mean_t_tensor[celli, :])]
            existing_pre_trials = np.isin(matched_s, existing_post_trials)
            pre_mean[celli] = np.nanmean(mean_t_tensor[celli, pre_rev & existing_pre_trials])
            if np.sum(existing_pre_trials) < 10:
                print(f'WARNING: cell_n {celli}, only {np.sum(existing_pre_trials)} trials matched' +
                      ' pre_reversal after accounting for missing data post_reversal')
        # pre_mean = np.nanmean(mean_t_tensor[:, pre_rev], axis=1)
        post_mean = np.nanmean(mean_t_tensor[:, post_rev], axis=1)
        reversal_mismatch = post_mean - pre_mean
    else:
        pre_bins = utils.bin_running_calc(meta, mean_t_tensor, (rev_day & pre_rev))
        post_bins = utils.bin_running_calc(meta, mean_t_tensor, (rev_day & post_rev))
        reversal_mismatch = np.nanmean(post_bins - pre_bins, axis=1)

    if boot:
        all_trials = (rev_day & (pre_rev | post_rev))
        trial_inds = np.where(all_trials)[0]
        boot_mat = np.zeros((mean_t_tensor.shape[0], 1000))
        boot_mat[:] = np.nan

        for booti in range(boot_mat.shape[1]):
            set1 = deepcopy(all_trials)
            it_inds = np.random.choice(trial_inds, 100)
            set1[it_inds] = False
            set2 = np.zeros(len(set1)) > 1
            set2[it_inds] = True
            pre_bins = utils.bin_running_calc(meta, mean_t_tensor, set1)
            post_bins = utils.bin_running_calc(meta, mean_t_tensor, set2)
            boot_mat[:, booti] = np.nanmean(post_bins - pre_bins, axis=1)
        pvals = []
        # import pdb; pdb.set_trace()
        for celli in range(boot_mat.shape[0]):
            rev_thresh = reversal_mismatch[celli]
            denom = np.sum(~np.isnan(boot_mat[celli]))
            if rev_thresh > 0:
                pvals.append(np.nansum(boot_mat[celli, :] > rev_thresh) / denom)
            elif np.isnan(rev_thresh):
                pvals.append(np.nan)
            else:
                pvals.append(np.nansum(boot_mat[celli, :] < rev_thresh) / denom)
        pvals = np.array(pvals)
        print(f'  -->  {mouse}: significant mismatch: n = {np.sum(pvals < 0.05)}')

        return reversal_mismatch, pvals

    return reversal_mismatch


def run_controlled_naive_mismatch(meta, pref_tensor, filter_licking=None, filter_running=None,
                                  filter_hmm_engaged=False, boot=True, account_for_offset=True):
    """
    Calculate a mismatch binning running and calculating between matched bins, then averaging across bins. Look at the
    period surround initial learning onset and reversal.

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

    # get mean response per cue
    mean_t_tensor = utils.tensor_mean_per_trial(meta, pref_tensor, nan_licking=False,
                                                account_for_offset=account_for_offset)

    # 1000 trials pre, 100 trials post reversal
    pre_rev = meta.parsed_11stage.isin(['L5 learning']).values
    if np.sum(pre_rev) > 1000:
        pre_rev[np.where(pre_rev)[0][:-1000]] = False
    rev_vec = meta.reset_index()['learning_state'].isin(['reversal1']).values
    if all(~rev_vec):
        out = np.zeros(pref_tensor.shape[0])
        out[:] = np.nan
        print(f'Mouse {mouse} did not have any reversal.')
        if boot:
            rev_pvals = deepcopy(out)
        reversal_mismatch = deepcopy(out)
    else:
        post_rev_date = meta.reset_index().loc[rev_vec, 'date'].iloc[0]
        post_rev = meta.reset_index()['date'].isin([post_rev_date]).values
        post_rev[np.where(post_rev)[0][100:]] = False
        reversal_set = (pre_rev | post_rev)

        # filter based on running, licking, engagement
        reversal_set = _trial_filter_set(meta, reversal_set, filter_running=filter_running,
                                         filter_licking=filter_licking, filter_hmm_engaged=filter_hmm_engaged)
        print(f'  -->  {mouse}: pre-rev: {np.sum(pre_rev & reversal_set)}, ' +
              f'post-rev: {np.sum(post_rev & reversal_set)}')

        # calculate mean for preferred cue for each cell across reversal
        pre_bins = utils.bin_running_calc(meta, mean_t_tensor, (reversal_set & pre_rev))
        post_bins = utils.bin_running_calc(meta, mean_t_tensor, (reversal_set & post_rev))
        reversal_mismatch = np.nanmean(post_bins - pre_bins, axis=1)

        if boot:
            all_trials = (reversal_set & (pre_rev | post_rev))
            trial_inds = np.where(all_trials)[0]
            boot_mat = np.zeros((mean_t_tensor.shape[0], 1000))
            boot_mat[:] = np.nan

            for booti in range(boot_mat.shape[1]):
                set1 = deepcopy(all_trials)
                it_inds = np.random.choice(trial_inds, 100)
                set1[it_inds] = False
                set2 = np.zeros(len(set1)) > 1
                set2[it_inds] = True
                pre_bins = utils.bin_running_calc(meta, mean_t_tensor, set1)
                post_bins = utils.bin_running_calc(meta, mean_t_tensor, set2)
                boot_mat[:, booti] = np.nanmean(post_bins - pre_bins, axis=1)
            pvals = []
            # import pdb; pdb.set_trace()
            for celli in range(boot_mat.shape[0]):
                rev_thresh = reversal_mismatch[celli]
                denom = np.sum(~np.isnan(boot_mat[celli]))
                if rev_thresh > 0:
                    pvals.append(np.nansum(boot_mat[celli, :] > rev_thresh) / denom)
                elif np.isnan(rev_thresh):
                    pvals.append(np.nan)
                else:
                    pvals.append(np.nansum(boot_mat[celli, :] < rev_thresh) / denom)
            rev_pvals = np.array(pvals)
            print(f'  -->  {mouse}: significant mismatch: n = {np.sum(rev_pvals < 0.05)}')

    # ----- Repeat process for naive --> learning ------
    # 1000 trials pre, 100 trials post learning initiation

    pre_learn = meta.parsed_11stage.isin(['L0 naive']).values
    if np.sum(pre_learn) > 1000:
        pre_learn[np.where(pre_learn)[0][:-1000]] = False
    l_vec = meta.reset_index()['learning_state'].isin(['learning']).values
    # import pdb; pdb.set_trace()
    if all(~l_vec) or np.sum(pre_learn) == 0:
        out = np.zeros(pref_tensor.shape[0])
        out[:] = np.nan
        print(f'Mouse {mouse} did not have L0 naive.')
        if boot:
            learn_pvals = deepcopy(out)
        learn_mismatch = deepcopy(out)
    else:
        post_learn_date = meta.reset_index().loc[l_vec, 'date'].iloc[0]
        post_learn = meta.reset_index()['date'].isin([post_learn_date]).values
        post_learn[np.where(post_learn)[0][100:]] = False
        learning_set = (pre_learn | post_learn)

        # filter based on running, licking, engagement
        learning_set = _trial_filter_set(meta, learning_set, filter_running=filter_running,
                                         filter_licking=filter_licking, filter_hmm_engaged=filter_hmm_engaged)
        print(f'  -->  {mouse}: naive: {np.sum(pre_learn & learning_set)}, ' +
              f'learning-start: {np.sum(post_learn & learning_set)}')

        # calculate mismatch binned by running speed
        pre_learn_bins = utils.bin_running_calc(meta, mean_t_tensor, (learning_set & pre_learn))
        post_learn_bins = utils.bin_running_calc(meta, mean_t_tensor, (learning_set & post_learn))
        learn_mismatch = np.nanmean(post_learn_bins - pre_learn_bins, axis=1)

        if boot:
            all_trials = (learning_set & (pre_learn | post_learn))
            trial_inds = np.where(all_trials)[0]
            boot_mat = np.zeros((mean_t_tensor.shape[0], 1000))
            boot_mat[:] = np.nan

            for booti in range(boot_mat.shape[1]):
                set1 = deepcopy(all_trials)
                it_inds = np.random.choice(trial_inds, 100)
                set1[it_inds] = False
                set2 = np.zeros(len(set1)) > 1
                set2[it_inds] = True
                pre_bins = utils.bin_running_calc(meta, mean_t_tensor, set1)
                post_bins = utils.bin_running_calc(meta, mean_t_tensor, set2)
                boot_mat[:, booti] = np.nanmean(post_bins - pre_bins, axis=1)
            pvals = []
            # import pdb; pdb.set_trace()
            for celli in range(boot_mat.shape[0]):
                rev_thresh = reversal_mismatch[celli]
                denom = np.sum(~np.isnan(boot_mat[celli]))
                if rev_thresh > 0:
                    pvals.append(np.nansum(boot_mat[celli, :] > rev_thresh) / denom)
                elif np.isnan(rev_thresh):
                    pvals.append(np.nan)
                else:
                    pvals.append(np.nansum(boot_mat[celli, :] < rev_thresh) / denom)
            learn_pvals = np.array(pvals)
            print(f'  -->  {mouse}: significant mismatch: n = {np.sum(learn_pvals < 0.05)}')

    if boot:
        # return learn_mismatch, learn_pvals, reversal_mismatch, rev_pvals
        return pd.DataFrame({'mouse': [mouse] * mean_t_tensor.shape[0],
                             'cell_n': np.arange(mean_t_tensor.shape[0]) + 1,
                             'lMM_response': learn_mismatch,
                             'lMM_pvals': learn_pvals,
                             'rMM_response': reversal_mismatch,
                             'rMM_pvals': rev_pvals,
                             }
                            )
    else:
        # return learn_mismatch, reversal_mismatch
        return pd.DataFrame({'mouse': [mouse] * mean_t_tensor.shape[0],
                             'cell_n': np.arange(mean_t_tensor.shape[0]) + 1,
                             'lMM_response': learn_mismatch,
                             'rMM_response': reversal_mismatch,
                             }
                            )


def calculate_reversal_mismatch(meta, pref_tensor, filter_running=None, filter_licking=None,
                                filter_hmm_engaged=False, force_same_day_reversal=False,
                                use_stages_for_reversal=False, skew_stages_for_reversal=False,
                                account_for_offset=True):
    """
    Calculate a mismatch score (difference in zscore) for cells that we have across reversal.

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

    mean_t_tensor = utils.tensor_mean_per_trial(meta, pref_tensor, nan_licking=False,
                                                account_for_offset=account_for_offset)

    # get day of reversal
    if force_same_day_reversal:
        post_rev = meta.reset_index()['date'].mod(1).isin([0.5]).values
        if np.sum(post_rev) == 0:
            out = np.zeros(pref_tensor.shape[0])
            out[:] = np.nan
            print(f'Mouse {mouse} did not have single-day reversal.')
            # return out
            return pd.DataFrame({'mouse': [mouse] * len(out),
                                 'cell_n': np.arange(len(out)) + 1,
                                 'rMM_response': out,
                                 }
                                )
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
            # return out
            return pd.DataFrame({'mouse': [mouse] * len(out),
                                 'cell_n': np.arange(len(out)) + 1,
                                 'rMM_response': out,
                                 }
                                )
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
            # return out
            return pd.DataFrame({'mouse': [mouse] * len(out),
                                 'cell_n': np.arange(len(out)) + 1,
                                 'rMM_response': out,
                                 }
                                )
        post_rev_date = meta.reset_index().loc[rev_vec, 'date'].iloc[0]
        post_rev = meta.reset_index()['date'].isin([post_rev_date]).values
    rev_day = (pre_rev | post_rev)

    # filter to include fixed running type: low or high
    if filter_running is not None:
        speed_cm_s = meta.speed.values
        pre_speed_cm_s = meta.pre_speed.values
        if filter_running == 'low_speed_only':
            rev_day = rev_day & (speed_cm_s <= 3)
        elif filter_running == 'high_speed_only':
            rev_day = rev_day & (speed_cm_s > 10)
        elif filter_running == 'low_pre_speed_only':
            rev_day = rev_day & (pre_speed_cm_s <= 3)
        elif filter_running == 'high_pre_speed_only':
            rev_day = rev_day & (pre_speed_cm_s > 10)
        else:
            raise NotImplementedError

    # filter to include fixed licking type: low or high
    if filter_licking is not None:
        # TODO this needs accounting for offset licking for offset cells
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

    # calculate mean for preferred cue for each cell across reversal
    pre_rev_day_mean = np.nanmean(mean_t_tensor[:, rev_day & pre_rev], axis=1)
    post_rev_day_mean = np.nanmean(mean_t_tensor[:, rev_day & post_rev], axis=1)
    reversal_mismatch = post_rev_day_mean - pre_rev_day_mean

    # return reversal_mismatch
    return pd.DataFrame({'mouse': [mouse] * len(reversal_mismatch),
                         'cell_n': np.arange(len(reversal_mismatch)) + 1,
                         'rMM_response': reversal_mismatch,
                         }
                        )


def reversal_mismatch_from_tensor(meta, tensor, model, tune_staging='staging_LR',
                                  best_tuning_only=True, drop_broad_tuning=True,
                                  staging='parsed_11stage', tuning_type='initial',
                                  filter_running=None, filter_licking=None,
                                  filter_hmm_engaged=False, force_same_day_reversal=False,
                                  use_stages_for_reversal=False,
                                  tight_reversal=True, n_trials=20):
    """
    Create a tensor where you have NaNed all trials that are not of a cells preferred type. Preference can be calculated
    either once pre and post reversal or for each behavioral stage of learning.

    :param meta: pandas.DataFrame, DataFrame of trial metadata
    :param tensor: numpy.ndarray, a cells x times X trials
    :param model: tensortools.ensemble, TCA results
    :param tune_staging: str --> 'parsed_11stage' or 'staging_LR'
        Determine how to pick preferred tuning, as only the preferred responses of cells are returned. By default this
        will be evaluate for each stage, but you can also pass a tuning_df that had other tuning calculations.
        For example, 'staging_LR' in tuning_df calculates preferred tuning only for pre and post reversal (not by
        dprime bin).
    :param best_tuning_only: boolean
        Tuning using cosine distances allows for cells to be broadly and joint tuned. best_tuning_only will use the
        "preferred" tuning of a joint tuned neuron (i.e., minus-plus --> minus). Otherwise both trial types of a joint
        tuned neuron are kept.
    :param drop_broad_tuning: boolean
        Optionally NaN any cells that are broadly tuned.
    :param staging: str, way to bin stages of learning
    :param tuning_type: str, way to define stimulus type. 'initial', 'orientation', or defaults to 'condition'
    :param filter_running:
    :param filter_licking:
    :param filter_hmm_engaged:
    :param force_same_day_reversal:

    # TODO right now it is necessary to pass model, but this is only to match cells to their preferred TCA components
    #  TODO rank is hard coded to 15, this is arbitrary and can be removed after model passing is optional

    :return: pref_tensor, numpy.ndarray
        A tensor the exact size of the tensor input, now containing nans for un-preferred stimulus presentations.
    """

    # parse params
    if 'staging_LR' in tune_staging:
        by_reversal = True
        by_stage = False
    elif staging in tune_staging:
        raise NotImplementedError
        # by_reversal = False
        # by_stage = True
    else:
        raise NotImplementedError

    # get initial conditions
    if 'initial' in tuning_type:
        cond_type = 'initial_condition'
    elif 'ori' in tuning_type:
        raise NotImplementedError
        # cond_type = 'orientation'
    elif tuning_type.lower() == 'cond' or tuning_type.lower() == 'condition':
        cond_type = 'condition'
    else:
        raise NotImplementedError

    # get mouse name
    mouse = meta.reset_index()['mouse'].unique()[0]

    # add mismatch condition to meta if it is not there
    if 'mismatch_condition' not in meta.columns:
        meta = utils.add_reversal_mismatch_condition_to_meta(meta)

    # do useful pre-precheck so skip mice that calculation will fail on
    # get day of reversal
    if force_same_day_reversal:
        post_rev = meta.reset_index()['date'].mod(1).isin([0.5]).values
        if np.sum(post_rev) == 0:
            out = np.zeros(tensor.shape[0])
            out[:] = np.nan
            print(f'Mouse {mouse} did not have single-day reversal.')
            return out
    else:
        rev_vec = meta.reset_index()['learning_state'].isin(['reversal1']).values
        if all(~rev_vec):
            out = np.zeros(tensor.shape[0])
            out[:] = np.nan
            print(f'Mouse {mouse} did not have any reversal.')
            return out

    # get tuning df, accounting for offsets (i.e., offset cell's tuning is calculated on offset response itself)
    tuning_df = tuning.cell_tuning(meta, tensor, model, 15,
                                   by_stage=by_stage, by_reversal=by_reversal, nan_lick=False,
                                   staging=staging, tuning_type=tuning_type, force_stim_avg=False)

    # define tuning based on pre-reversal condition
    pre_rev_tuning_df = tuning_df.loc[tuning_df['staging_LR'].isin(['learning']), :]

    # loop over cells
    mismatch_tuning_list = ['none'] * tensor.shape[0]
    best_tuning_list = ['empty'] * tensor.shape[0]
    pref_tensor = deepcopy(tensor)
    for ind, row in pre_rev_tuning_df.iterrows():

        # get cell ind subtracting one to zero-index
        cell_index = ind[1] - 1

        # cell pref tuning
        cell_pref = row['preferred tuning']
        best_tuning_list[cell_index] = cell_pref

        # define trials to blank
        if cell_pref == 'broad':
            if drop_broad_tuning:
                cues_to_drop = meta[cond_type].isin(['plus', 'minus', 'neutral']).values
            else:
                cues_to_drop = ~meta[cond_type].isin(['plus', 'minus', 'neutral']).values
        elif cell_pref.lower() == 'none':
            cues_to_drop = meta[cond_type].isin(['plus', 'minus', 'neutral']).values
        elif '-' in cell_pref:  # meaning the cell is joint tuned
            hyphind = cell_pref.find('-')
            if best_tuning_only:
                cues_to_drop = ~meta[cond_type].isin([cell_pref[:hyphind]]).values

                # additionally get the mismatch type for each cell
                mtuning = meta.loc[
                    (meta[cond_type].isin([cell_pref[:hyphind]]).values & meta['learning_state'].isin(
                        ['learning']).values),
                    'mismatch_condition'
                ].unique()
                assert len(mtuning) == 1
                mismatch_tuning_list[int(cell_index)] = mtuning[0]

                # # also grab the tuning assigned to that cell
                # best_tuning_list[cell_index] = cell_pref[:hyphind]

            else:
                cues_to_drop = ~meta[cond_type].isin([cell_pref[:hyphind], cell_pref[hyphind + 1:]]).values
        else:
            assert cell_pref in ['plus', 'minus', 'neutral']
            cues_to_drop = ~meta[cond_type].isin([cell_pref]).values

            # additionally get the mismatch type for each cell
            mtuning = meta.loc[
                (meta[cond_type].isin([cell_pref]).values & meta['learning_state'].isin(['learning']).values),
                'mismatch_condition'
            ].unique()
            assert len(mtuning) == 1
            mismatch_tuning_list[int(cell_index)] = mtuning[0]

            # # also grab the tuning assigned to that cell
            # best_tuning_list[cell_index] = cell_pref[:hyphind]

        trials_to_nan = cues_to_drop

        # clear non-preferred trials
        pref_tensor[cell_index, :, trials_to_nan] = np.nan

    # optionally only check the last N or the first N trials across reversal
    # this is because of the possibility that changes are emphasized or underrepresented due to large daily changes
    if tight_reversal:
        n_trials = n_trials
        rev_vec = meta.reset_index()['learning_state'].isin(['reversal1']).values
        first_rev_ind = np.where(rev_vec)[0][0]
        pref_tensor[:, :, int(first_rev_ind + n_trials):] = np.nan
        pref_tensor[:, :, :int(first_rev_ind - n_trials)] = np.nan

    # pass your truncated meta and tensor to your calculation
    reversal_mismatch = calculate_reversal_mismatch(meta, pref_tensor,
                                                    filter_running=filter_running, filter_licking=filter_licking,
                                                    filter_hmm_engaged=filter_hmm_engaged,
                                                    force_same_day_reversal=force_same_day_reversal,
                                                    use_stages_for_reversal=use_stages_for_reversal)

    rmm_df = pd.DataFrame(data={'mismatch_response': reversal_mismatch,
                                'mismatch_condition': mismatch_tuning_list,
                                'preferred_tuning': best_tuning_list,
                                'mouse': [mouse] * tensor.shape[0],
                                'cell_n': [int(s) for s in np.arange(0, tensor.shape[0])]
                                }
                          ).set_index(['mouse', 'cell_n'])
    return rmm_df


def _trial_filter_set(meta, bool_to_filter, filter_running=None, filter_licking=None, filter_hmm_engaged=False):
    # filter to include fixed running type: low or high
    if filter_running is not None:
        speed_cm_s = meta.speed.values
        if filter_running == 'low_speed_only':
            bool_to_filter = bool_to_filter & (speed_cm_s <= 6)  # was 4
            print('WARNING low_speed_only set to 6 cm/s')
        elif filter_running == 'high_speed_only':
            bool_to_filter = bool_to_filter & (speed_cm_s > 20)  # was 10
            print('WARNING high_speed_only set to 20 cm/s')
        else:
            raise NotImplementedError

    # filter to include fixed licking type: low or high
    if filter_licking is not None:
        # TODO this needs accounting for offset licking for offset cells
        # TODO could also do a grid of lick and run bins to make comparisons
        mean_lick_rate = meta.anticipatory_licks.values / lookups.stim_length[mouse]
        if filter_licking == 'low_lick_only':
            bool_to_filter = bool_to_filter & (mean_lick_rate <= 1.7)
        elif filter_licking == 'high_lick_only':
            bool_to_filter = bool_to_filter & (mean_lick_rate > 1.7)
        else:
            raise NotImplementedError

    # animal must be engaged in the task (or naive when it can't "engage")
    if filter_hmm_engaged:
        bool_to_filter = bool_to_filter & (meta.hmm_engaged.values | meta.learning_state.isin(['naive']).values)

    return bool_to_filter

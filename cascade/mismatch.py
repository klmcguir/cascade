from copy import deepcopy

import numpy as np
import pandas as pd
from scipy import stats
import os
import seaborn as sns
import matplotlib.pyplot as plt

from . import utils, tuning, lookups

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

    #      target:      search:
    return s_had_match, matched_s


def mismatch_stat(meta, tensor, ids, search_epoch='L5 learning', offset_bool=None,
                  stim_calc_start_s=0.2, stim_calc_end_s=0.700, off_calc_start_s=0.200, off_calc_end_s=0.700,
                  plot_please=False, plot_w='heatmap', neg_log10_pv_thresh=4, alternative='less'):
    """
    Calculate mismatch on trials matched for running speed. Also calculated p-values on the drivenness of a cell
    for those trials. This is in an attempt to post-hoc throw out cells that were not driven
    during any part of the calculation.

    :param meta: pandas.DataFrame
        Trial metadata.
    :param tensor: numpy.ndarray
        cells x times x trials matrix of neural data.
    :param ids: numpy.ndarray (1,)
        Vector of cell IDs.
    :param search_epoch: str
        Period of learning to compare all other periods to. i.e. 'L5 learning'
    :param offset_bool: boolean
        Optionally pass boolean vector defining offset driven cells. True is an Offset cells. False is other.
    :param stim_calc_start_s: float
        When to start the stim period calculation relative to stim onset, units of seconds.
    :param stim_calc_end_s: float
        When to end the stim period calculation relative to stim onset, units of seconds. If None, defaults to full
        stimulus length
    :param off_calc_start_s: float
        When to start the offset period calculation relative to stim offset, units of seconds.
    :param off_calc_end_s: float
        When to end the offset period calculation relative to stim offset, units of seconds. If None, defaults to full
        response window (2 seconds after offset).
    :param plot_please: boolean
        Optionally plot, mostly for testing and debugging. WARNING: Will generate THOUSANDS of plots.
    :param plot_w: str
        'heatmap' or 'traces', plot mean traces or include heatmap. None defaults to 'traces'.
    :param neg_log10_pv_thresh: float
        -log10(p-value) threshold for calling a cell driven. This adds an extra column, but does no filtering.
    :param alternative: str
        'less' or 'two-sided', not 'greater'. kwarg for scipy.stats.wilcoxon(). 'less' is a one sided test, testing for
        an increase over baseline.

    :return: pandas.DataFrame with mismatch calculations, and p-values for drivenness of the trials included in each
    calculation
    """

    # loop over cues and epochs, match trials, calculate differences between matched trials for each cell
    cue_dfs = []
    for cue in ['minus', 'neutral', 'plus']:
        meta_bool = meta.initial_condition.isin([cue]).values
        cue_meta = meta.loc[meta_bool]
        cue_tensor = tensor[:, :, meta_bool]

        # offset_bool = cas.utils.get_offset_cells(meta, pref_tensor)
        if offset_bool is None:
            offset_bool = utils.get_offset_cells(meta, tensor)

        # assume data is the usual 15.5 hz 7 sec 108 frame vector
        assert cue_tensor.shape[1] == 108

        # get windows for additional comparisons --> 500 ms with a 200 ms delay to account for GECI tau
        stim_length = lookups.stim_length[utils.meta_mouse(meta)]
        time_vec = np.arange(-1, 6, 1 / 15.5)[:108]
        ms200 = np.where(time_vec > stim_calc_start_s)[0][0]
        ms700 = np.where(time_vec > stim_calc_end_s)[0][0]
        off_ms200 = np.where(time_vec > stim_length + off_calc_start_s)[0][0]
        off_ms700 = np.where(time_vec > stim_length + off_calc_end_s)[0][0]
        stim_off_frame = np.where(time_vec > stim_length)[0][0]
        stim_off_frame_minus1s = np.where(time_vec > stim_length - 1)[0][0]
        # for sustainedness
        sus0 = np.where(time_vec > 0)[0][0]
        sus2 = np.where(time_vec > 2)[0][0]  # use 2 seconds for all mice

        # get mean trial response matriices, cells x trials
        mean_t_tensor = utils.tensor_mean_per_trial(cue_meta, cue_tensor, nan_licking=False, account_for_offset=True,
                                                    offset_bool=offset_bool,
                                                    stim_start_s=stim_calc_start_s, stim_off_s=stim_calc_end_s,
                                                    off_start_s=off_calc_start_s, off_off_s=off_calc_end_s)
        mean_t_baselines = np.nanmean(cue_tensor[:, :15, :], axis=1)
        mean_t_1stsec = np.nanmean(cue_tensor[:, ms200:ms700, :], axis=1)

        # for offset cells it is useful to have averages of 1sec before offset and 1 sec after as well
        mean_t_off_baselines = np.nanmean(cue_tensor[:, stim_off_frame_minus1s:stim_off_frame, :], axis=1)
        mean_t_off_1st_sec = np.nanmean(cue_tensor[:, off_ms200:off_ms700, :], axis=1)

        # preallocate
        diff_mat = np.zeros((mean_t_tensor.shape[0], len(lookups.staging['parsed_11stage']))) + np.nan
        frac_diff_mat = np.zeros((mean_t_tensor.shape[0], len(lookups.staging['parsed_11stage']))) + np.nan
        pv_pre_mat = np.zeros((mean_t_tensor.shape[0], len(lookups.staging['parsed_11stage']))) + np.nan
        pv_post_mat = np.zeros((mean_t_tensor.shape[0], len(lookups.staging['parsed_11stage']))) + np.nan
        driven_to_one_mat = np.zeros((mean_t_tensor.shape[0], len(lookups.staging['parsed_11stage']))) + np.nan
        delta_sus_mat = np.zeros((mean_t_tensor.shape[0], len(lookups.staging['parsed_11stage']))) + np.nan
        frac_delta_sus_mat = np.zeros((mean_t_tensor.shape[0], len(lookups.staging['parsed_11stage']))) + np.nan
        trial_inds_relrev = np.zeros((mean_t_tensor.shape[0], len(lookups.staging['parsed_11stage']))) + np.nan
        for si, stage in enumerate(lookups.staging['parsed_11stage']):

            # don't compare to self. Skip and set to 0 where cells were found.
            if stage == search_epoch:
                diff_mat[~np.isnan(mean_t_tensor[:, 0]), si] = 0
                frac_diff_mat[~np.isnan(mean_t_tensor[:, 0]), si] = 0
                delta_sus_mat[~np.isnan(mean_t_tensor[:, 0]), si] = 0
                frac_delta_sus_mat[~np.isnan(mean_t_tensor[:, 0]), si] = 0
                continue

            # calculate change between two stages, matching distribution of running speeds per cue
            # s_had_match --> target --> post
            # matched_s --> search --> pre
            s_had_match, matched_s = match_trials(cue_meta, target_epoch=stage, search_epoch=search_epoch,
                                                  match_on='speed', tolerance=5)
            print(f'{utils.meta_mouse(cue_meta)}: {stage} vs {search_epoch}, '
                  f'n={np.sum(np.array(s_had_match) > 0)} trials possible')

            # if a mouse is missing a stage skip calculation
            if len(s_had_match) == 0 or len(matched_s) == 0:
                continue

            # for each cell only use matched trials. If a cell is missing data in one period,
            # drop matching trials in other.
            pre_mean = np.zeros(mean_t_tensor.shape[0]) + np.nan
            post_mean = np.zeros(mean_t_tensor.shape[0]) + np.nan
            for celli in range(mean_t_tensor.shape[0]):

                # get possible trials that were matched given cell existence in each epoch
                existing_post_trial_nums = s_had_match[~np.isnan(mean_t_tensor[celli, :])]
                existing_pre_trial_nums = matched_s[~np.isnan(mean_t_tensor[celli, :])]
                still_matched_in_both = existing_pre_trial_nums[
                    np.isin(existing_pre_trial_nums, existing_post_trial_nums) &
                    (existing_pre_trial_nums > 0)]  # 0 is an unmatched trial

                # get boolean for both stages for your subset of existing matched trials
                existing_post_trial_bool = np.isin(s_had_match, still_matched_in_both)
                existing_pre_trial_bool = np.isin(matched_s, still_matched_in_both)
                if np.sum(existing_post_trial_bool) == 0 or np.sum(
                        existing_pre_trial_bool) == 0:  # no matched trials for cell
                    continue

                # get the median trial number relative to reversal
                last_learning_ind = np.where(meta.learning_state.isin(['learning']).values)[0][-1]
                trials = np.arange(len(meta)) - last_learning_ind
                trial_inds_relrev[celli, si] = np.nanmedian(trials[meta_bool][existing_post_trial_bool])

                # get baseline vectors for both stages
                # mean_t_tensor accounts for offset, so this is the first test for both cases full-stim or full-offset
                post_bases = mean_t_baselines[celli, existing_post_trial_bool]
                pre_bases = mean_t_baselines[celli, existing_pre_trial_bool]
                post_means = mean_t_tensor[celli, existing_post_trial_bool]
                pre_means = mean_t_tensor[celli, existing_pre_trial_bool]

                # different calcs if a cell is a stimulus peaking or offset peaking cell
                if not offset_bool[celli]:

                    # additional test, check 1st second of stim period to not punish tranient responses
                    post_1s = mean_t_1stsec[celli, existing_post_trial_bool]
                    pre_1s = mean_t_1stsec[celli, existing_pre_trial_bool]

                elif offset_bool[celli]:

                    # additional baseline values defined from last second of the stimulus
                    off_post_bases = mean_t_off_baselines[celli, existing_post_trial_bool]  # last second of stimulus
                    off_pre_bases = mean_t_off_baselines[celli, existing_pre_trial_bool]

                    # additional data for 1st second following offset
                    post_1s = mean_t_off_1st_sec[celli, existing_post_trial_bool]  # first second following offset
                    pre_1s = mean_t_off_1st_sec[celli, existing_pre_trial_bool]

                # check for drivenness in both stages
                # this compares baseline to the full stim or offset period as well as to the first second of both
                # tests bases-means delta is negative, H0: symmetric
                # 'two-sided'  # 'less' --> for one_tailed in the right direction
                pv_post = stats.wilcoxon(post_bases, post_means, alternative=alternative).pvalue
                pv_pre = stats.wilcoxon(pre_bases, pre_means, alternative=alternative).pvalue
                pv_post_1s = stats.wilcoxon(post_bases, post_1s, alternative=alternative).pvalue
                pv_pre_1s = stats.wilcoxon(pre_bases, pre_1s, alternative=alternative).pvalue
                pv_post = np.nanmin([pv_post, pv_post_1s]) * 2  # reset pv_post to be best with bonferroni
                pv_pre = np.nanmin([pv_pre, pv_pre_1s]) * 2

                if offset_bool[
                    celli]:  # additional tests specific to full offset period as well as to the first second of offset
                    # this compares 1s pre offset to stim or offset period as well as to the first second of both
                    off_pv_post = stats.wilcoxon(off_post_bases, post_means, alternative=alternative).pvalue
                    off_pv_pre = stats.wilcoxon(off_pre_bases, pre_means, alternative=alternative).pvalue
                    off_pv_post_1s = stats.wilcoxon(off_post_bases, post_1s, alternative=alternative).pvalue
                    off_pv_pre_1s = stats.wilcoxon(off_pre_bases, pre_1s, alternative=alternative).pvalue
                    off_pv_post = np.nanmin(
                        [off_pv_post, off_pv_post_1s]) * 2  # reset pv_post to be best with bonferroni
                    off_pv_pre = np.nanmin([off_pv_pre, off_pv_pre_1s]) * 2
                    # select your final p-value for offset cells to be the worst of the two comparisons.
                    # 1. baseline-(full or 1s)
                    # 2. last_second_of_stim-(full or 1s)
                    # both must be significant for a cell to pass so select the worst of the two.
                    pv_post = np.nanmax([pv_post, off_pv_post])  # worst of the two
                    pv_pre = np.nanmax([pv_pre, off_pv_pre])

                # get mean trace
                post_trace = np.nanmean(cue_tensor[celli, :, existing_post_trial_bool], axis=0)
                pre_trace = np.nanmean(cue_tensor[celli, :, existing_pre_trial_bool], axis=0)

                # plot traces (and other optional plots for checking metrics)
                if plot_please:
                    if np.sum(existing_pre_trial_bool) > 0:
                        offtag = 'OFFSET' if offset_bool[celli] else 'STIM'
                        drivetag = (-np.log10(pv_pre) >= neg_log10_pv_thresh) | (
                                -np.log10(pv_post) >= neg_log10_pv_thresh)
                        if plot_w is None or 'traces' in plot_w:
                            plt.figure()
                            plt.plot(pre_trace, label=f'target {round(pv_pre, 4)}')
                            plt.plot(post_trace, label=f'search {round(pv_post, 4)}')
                            plt.title(f'Driven, ntrials={np.sum(existing_pre_trial_bool)}' if drivetag
                                      else f'NOT driven , ntrials={np.sum(existing_pre_trial_bool)}')
                            plt.legend()
                        else:
                            fig, ax = plt.subplots(1, 3, figsize=(11, 3))
                            ax[0].plot(pre_trace, label=f'target {round(pv_pre, 4)}')
                            ax[0].plot(post_trace, label=f'search {round(pv_post, 4)}')
                            ax[0].set_title(f'{offtag} Driven, ntrials={np.sum(existing_pre_trial_bool)}' if drivetag
                                            else f'{offtag} NOT driven , ntrials={np.sum(existing_pre_trial_bool)}')
                            ax[0].legend()
                            ax[0].set_ylabel('mean response')
                            ax[0].set_xlabel('time from stim onset')
                            ax[0].set_xticks(np.arange(0, 108, 15.5))
                            ax[0].set_xticklabels(np.arange(-1, 6, 1))

                            if plot_w.lower() == 'heatmap':
                                sns.heatmap(cue_tensor[celli, :, existing_pre_trial_bool], ax=ax[1])
                            else:
                                sns.histplot(pre_means, ax=ax[1], color='blue', label='response')
                                sns.histplot(pre_bases, ax=ax[1], color='gray', label='baseline', alpha=0.5)
                                ax[1].legend()
                            ax[1].set_title('Target set mean responses')
                            ax[1].set_xlabel('mean response')
                            ax[1].set_ylabel('trial count')

                            if plot_w.lower() == 'heatmap':
                                sns.heatmap(cue_tensor[celli, :, existing_post_trial_bool], ax=ax[2])
                            else:
                                sns.histplot(post_means, ax=ax[2], color='orange', label='response')
                                sns.histplot(post_bases, ax=ax[2], color='gray', label='baseline', alpha=0.5)
                                ax[2].legend()
                            ax[2].set_title('Search set mean responses')
                            ax[2].set_xlabel('mean response')
                            ax[2].set_ylabel('trial count')
                        dtag = '_DRIVEN' if drivetag else ''
                        bpv = int(round(
                            np.max([-np.log10(pv_post), -np.log10(pv_pre)])))  # highest -log10 pvalue for that pair
                        plt.savefig(os.path.join('/twophoton_analysis/Data/analysis/pilot_mm_plots_v8_stat/',
                                                 f'{utils.meta_mouse(meta)}{dtag}_bpv{bpv}_n{celli}_{search_epoch}_v_{stage}.png'),
                                    bbox_inches='tight')
                    plt.close('all')

                # save if cells passed criteria
                if (-np.log10(pv_pre) >= neg_log10_pv_thresh) | (-np.log10(pv_post) >= neg_log10_pv_thresh):
                    driven_to_one_mat[celli, si] = 1
                else:
                    driven_to_one_mat[celli, si] = 0

                # means regardless of p-values
                pre_mean[celli] = np.nanmean(mean_t_tensor[celli, existing_pre_trial_bool])
                post_mean[celli] = np.nanmean(mean_t_tensor[celli, existing_post_trial_bool])

                # matrix of drivenness p-values for cell --> for each stage
                pv_pre_mat[celli, si] = -np.log10(pv_pre)

                # matrix of drivenness p-values for cell --> for search epoch, (compared to a given stage)
                pv_post_mat[celli, si] = -np.log10(pv_post)

                # change in sustainedness, mean and 95th percentile response
                sus_pair = []
                for susi in [pre_trace[sus0:sus2], post_trace[sus0:sus2]]:
                    mean_response = np.nanmean(susi)

                    # flip non-NaN array values [0, 1, 10, np.nan] --> [10, 1, 0, np.nan]
                    sorted_vals = np.sort(susi)  # nans appended to end
                    ind95 = int(np.ceil(len(susi) * .05))
                    sorted_vals[~np.isnan(sorted_vals)] = sorted_vals[~np.isnan(sorted_vals)][::-1]
                    percentile95 = sorted_vals[ind95]
                    if percentile95 > 0.001 and mean_response > 0.001 and percentile95 >= mean_response:
                        sus_pair.append(mean_response / percentile95)
                    else:
                        sus_pair.append(np.nan)

                delta_sus_mat[celli, si] = sus_pair[1] - sus_pair[0]
                frac_delta_sus_mat[celli, si] = (sus_pair[1] - sus_pair[0]) / np.nanmax(sus_pair)

            # diff as zscore
            diff_mat[:, si] = post_mean - pre_mean

            # diff as fraction
            post_mean[post_mean < 0] = 0
            pre_mean[pre_mean < 0] = 0
            denom = np.nanmax(np.vstack([post_mean, pre_mean]), axis=0)
            frac_diff_mat[:, si] = (post_mean - pre_mean) / denom


        # unwrap and create df
        stages = lookups.staging['parsed_11stage_T']
        search_epoch_num = [s for s, sn in enumerate(lookups.staging['parsed_11stage']) if sn == search_epoch][0]
        updated_search_epoch = stages[search_epoch_num]
        all_pairs_drive_dfs = []
        for c, s in enumerate(stages):

            stage_diff = diff_mat[:, c]
            frac_stage_diff = frac_diff_mat[:, c]
            post_vec = pv_post_mat[:, c]
            pre_vec = pv_pre_mat[:, c]
            dt_col = driven_to_one_mat[:, c]
            d_sus = delta_sus_mat[:, c]
            d_frac_sus = frac_delta_sus_mat[:, c]
            tr_num = trial_inds_relrev[:, c]

            all_pairs_drive_dfs.append(
                pd.DataFrame(
                    data={'mouse': [utils.meta_mouse(meta)] * len(stage_diff),
                          'parsed_11stage': [s] * len(stage_diff),
                          'search_stage': [updated_search_epoch] * len(stage_diff),
                          'median_trial': tr_num,
                          'initial_cue': [cue] * len(stage_diff),
                          'mm_type': [lookups.lookup_mm[utils.meta_mouse(meta)][cue]] * len(stage_diff),
                          'mm_frac': frac_stage_diff,
                          'mm_amp': stage_diff,
                          'mm_neglogpv_target': post_vec,
                          'mm_neglogpv_search': pre_vec,  # this is your "fixed" search epoch
                          'delta_sustainedness': d_sus,
                          'delta_frac_sustainedness': d_frac_sus,
                          'cell_id': ids,
                          'cell_n': np.arange(len(stage_diff)) + 1,
                          f'driven_{neg_log10_pv_thresh}': dt_col,
                          'offset_cell': offset_bool
                          }
                ).set_index(['mouse', 'cell_id']).sort_index()
            )

        learning_mm_wtype = pd.concat(all_pairs_drive_dfs, axis=0)
        cue_dfs.append(learning_mm_wtype)
    cue_dfs_all_cues = pd.concat(cue_dfs, axis=0)

    return cue_dfs_all_cues


def trial_match_diff_over_stages(meta, pref_tensor, search_epoch='L5 learning', min_trials=10):
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
            if np.sum(existing_pre_trials) < min_trials:
                print(f'WARNING: cell_n {celli}, only {np.sum(existing_pre_trials)} trials matched' +
                      ' pre_reversal after accounting for missing data post_reversal')
        post_mean = np.nanmean(mean_t_tensor[:, post_rev], axis=1)
        reversal_mismatch = post_mean - pre_mean
        diff_mat[:, si] = reversal_mismatch

    return diff_mat


def trial_match_frac_over_stages(meta, pref_tensor, search_epoch='L5 learning'):
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

        # max of post or pre if both had a value
        # denom = []
        # for celli in range(mean_t_tensor.shape[0]):
        #     if np.isnan(post_mean[celli]) | np.isnan(pre_mean[celli]):
        #         denom.append(np.nan)
        #     else:
        #         denom.append(np.max([post_mean[celli], pre_mean[celli]]))

        post_mean[post_mean < 0] = 0
        pre_mean[pre_mean < 0] = 0
        denom = np.nanmax(np.vstack([post_mean, pre_mean]), axis=0)

        reversal_mismatch = (post_mean - pre_mean) / denom
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

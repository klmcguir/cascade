"""Functions for general calculations and data management."""
import flow
import pool
import numpy as np
import warnings
import pandas as pd
from . import lookups, bias, tca
from copy import deepcopy


def tensor_mean_per_day(meta, tensor, initial_cue=True, cue='plus', nan_licking=False):
    """
    Helper function to calculate mean per day for axis of same length as meta for a single cue.
    """

    # meta can only be a DataFrame of a single mouse
    assert len(meta.reset_index()['mouse'].unique()) == 1

    # make sure that the date vec allows for 0.5 days at reversal and learning
    meta = update_meta_date_vec(meta)

    # choose to average over initial_condition (orientation) or condition (switches orientation at reversal)
    if initial_cue:
        cue_vec = meta['initial_condition']
    else:
        cue_vec = meta['condition']
    cue_bool = cue_vec.isin([cue])

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


def tensor_mean_per_trial(meta, tensor, nan_licking=False):
    """
    Helper function to calculate mean per trial meta for a single mouse, correctly accounting for stimulus length.
    Optionally only use
    """

    # meta can only be a DataFrame of a single mouse
    assert len(meta.reset_index()['mouse'].unique()) == 1
    mouse = meta.reset_index()['mouse'].unique()[0]

    # assume 15.5 Hz sampling or downsampling for 7 seconds per trial (n = 108 timepoints)
    assert tensor.shape[1] == 108
    times = np.arange(-1, 6, 1 / 15.5)[:108]
    stim_bool = (times > 0) & (times < lookups.stim_length[mouse])

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
    new_mat = np.nanmean(ablated_tensor[:, stim_bool, :], axis=1)

    return new_mat


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


def add_stages_to_meta(meta, staging, dp_by_run=True, simple=False, bin_scale=0.75):
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
    :return: meta, now with one new column
    """
    # make sure parsed stage exists so you can loop over this.
    # add learning stages to meta
    if 'parsed_stage' not in meta.columns and 'parsed_stage' in staging:
        meta = add_5stages_to_meta(meta, dp_by_run=dp_by_run)
    if 'parsed_10stage' not in meta.columns and 'parsed_10stage' in staging:
        meta = add_10stages_to_meta(meta, dp_by_run=dp_by_run, simple=simple)
    if 'parsed_11stage' not in meta.columns and 'parsed_11stage' in staging:
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
            run_traces = t2p.cstraces(cs, start_s=start_time, end_s=end_time,
                                      trace_type=trace_type, cutoff_before_lick_ms=-1,
                                      errortrials=-1, baseline=None)
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
            mod = z[1] % bin_factor
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
    :param use_dprime:

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
    if mouse == 'OA26' and 'contrast' in exclude_tags:
        days = [s for s in days if s.date != 170302]

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
        badtrialratio = nbadtrials / ntrials
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

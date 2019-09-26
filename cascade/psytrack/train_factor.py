import numpy as np
from pprint import pprint
import time
import pool
import flow
import pandas as pd
from .. import utils, load

from psytrack.helper.invBlkTriDiag import getCredibleInterval
from psytrack.hyperOpt import hyperOpt

"""
Define hyperparameters that are annoying to calculate and worth
keeping track of.
"""
default_pars = {
    'fixed_sigmas':
        [0.098, 0.185, 0.185, 0.185, 0.0166, 0.1128, 0.0457],
    'fixed_sigma_day':
        [1.3003, 2.1746, 2.1746, 2.1746, 0.1195, 0.3035, 0.6393]
                }


def train(
        mouse,
        runs,
        weights,
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        word='tray',
        group_by='all',
        nan_thresh=0.85,
        score_threshold=None,
        rank_num=18,
        comp_num=1,
        include_pavlovian=False,
        separate_day_var=True,
        fixed_sigma=None,
        fixed_sigma_day=None,
        TCA_inputs=True,
        verbose=False):
    """Main function use to train a bastardized PsyTracker."""

    k = np.sum([weights[i] for i in weights.keys()])
    if verbose:
        print('* Beginning training for {} *'.format(runs.mouse))
        print(' Fitting weights:')
        pprint(weights)
        print(' Fitting {} total hyper-parameters'.format(k))

    # Initialize hyperparameters
    hyper = {'sigInit': 2**4.}
    opt_list = []
    if fixed_sigma is None:
        hyper['sigma'] = [2**-4.]*k
        opt_list.append('sigma')
    else:
        hyper['sigma'] = fixed_sigma

    # Add in parameters for each day if needed
    if separate_day_var:
        if fixed_sigma_day is None:
            hyper['sigDay'] = [2**-4.]*k
            opt_list.append('sigDay')
        else:
            hyper['sigDay'] = fixed_sigma_day

    # Extract data from our simpcells and convert to format for PsyTrack
    if verbose:
        print('- Collecting data')
    data = _gather_data(
        runs, include_pavlovian=include_pavlovian, weights=weights)
    if verbose:
        print(' Data keys:\n  {}'.format(sorted(data.keys())))
        print(' Inputs:\n  {}'.format(sorted(data['inputs'].keys())))
        print(' Total trials:\n  {}'.format(len(data['y'])))

    if TCA_inputs:
        # add in TCA factors as inputs to pillow
        data = _splice_data_inputs(
            data,
            mouse,
            trace_type=trace_type,
            method=method,
            cs=cs,
            warp=warp,
            word=word,
            group_by=group_by,
            nan_thresh=nan_thresh,
            score_threshold=score_threshold,
            rank_num=rank_num,
            verbose=verbose)
        # update weights
        weights = {}
        for ci in range(1, rank_num + 1):
            weights['factor_' + str(ci)] = 1
        # update sigmas
        hyper['sigma'] = [2**-4.]*rank_num
        hyper['sigDay'] = [2**-4.]*rank_num
    else:
        # add 'y' but now it is a 1-2 binary vector of a TCA trial factor
        data = _splice_data_y(
            data,
            mouse,
            trace_type=trace_type,
            method=method,
            cs=cs,
            warp=warp,
            word=word,
            group_by=group_by,
            nan_thresh=nan_thresh,
            score_threshold=score_threshold,
            rank_num=rank_num,
            comp_num=comp_num,
            verbose=verbose)

    # Fit model
    if verbose:
        print('- Fitting model')
        start_t = time.time()
    hyp, evd, wMode, hess = hyperOpt(
        data, hyper, weights, opt_list,
        showOpt=0 if not verbose else int(verbose) - 1)
    if verbose:
        stop_t = time.time()
        print(' Model fit in {:.2f} minutes.'.format((stop_t - start_t) / 60.))

    # Calculate confidence intervals
    if verbose:
        print('- Determining confidence intervals')
    credible_int = getCredibleInterval(hess)

    results = {'hyper': hyp,
               'evidence': evd,
               'model_weights': wMode,
               'hessian': hess,
               'credible_intervals': credible_int}

    init = {'hyper': hyper, 'opt_list': opt_list}

    return data, results, init


def _parse_weights(weights):
    """Parse the weights dict to determine what data needs to be collected."""
    oris = []
    for key in weights:
        if key[:4] == 'ori_':
            oris.append(int(key[4:]))

    return oris


def _gather_data(runs, weights, include_pavlovian=True):
    orientations = _parse_weights(weights)
    day_length, run_length = [], []  # Number of trials for each day/run
    days = []  # Dates for all days
    dateRuns = []  # (date, run) for all runs
    dateRunTrials = []  # (date, run, trial_idx) for all trials
    y = []  # 2 for lick, 1 for no lick
    correct = []  # correct trials, boolean
    answer = []  # The correct choice, 2 for lick, 1 for no lick

    if not include_pavlovian:
        # All the pavlovian trials, so they can be excluded at the end
        pav_trials = []
    if 'prev_reward' in weights or 'cum_reward' in weights:
        reward = []  # rewarded trials, boolean
    if 'prev_punish' in weights or 'cum_punish' in weights:
        punish = []  # punished trials, boolean
    if 'cum_reward' in weights:
        cum_reward = []  # cumulative number of rewarded trials per day
    if 'cum_punish' in weights:
        cum_punish = []  # cumulative number of punish trials per day
    oris = {ori: [] for ori in orientations}
    last_date = runs[0].date
    date_ntrials = 0
    date_reward, date_punish = [], []
    for run in runs:
        if run.date != last_date:
            if date_ntrials > 0:
                day_length.append(date_ntrials)
                days.append(last_date)

                reward.extend(date_reward)
                punish.extend(date_punish)

                if 'cum_reward' in weights:
                    cum_reward.extend(np.cumsum(date_reward))
                if 'cum_punish' in weights:
                    cum_punish.extend(np.cumsum(date_punish))

            last_date = run.date
            date_ntrials = 0
            date_reward, date_punish = [], []

        t2p = run.trace2p()
        ntrials = t2p.ntrials
        # If dropping pavlovian trials, still need to keep them around
        # for all of the trial history parameters, then drop at very end.
        if not include_pavlovian:
            run_pav_trials = ['pavlovian' in x for x in
                              t2p.conditions(return_as_strings=True)]
            n_run_pav_trials = sum(run_pav_trials)

            if not (ntrials - n_run_pav_trials > 0):
                continue

            pav_trials.extend(run_pav_trials)
            date_ntrials += ntrials - n_run_pav_trials
            run_length.append(ntrials - n_run_pav_trials)
        else:
            if not ntrials > 0:
                continue

            date_ntrials += ntrials
            run_length.append(ntrials)

        dateRuns.append((run.date, run.run))
        dateRunTrials.extend(
            zip([run.date]*ntrials, [run.run]*ntrials, range(ntrials)))

        run_choice = t2p.choice()
        assert(len(run_choice) == ntrials)
        y.extend(run_choice)

        run_errs = t2p.errors()
        assert(len(~run_errs) == ntrials)
        correct.extend(~run_errs)

        run_answer = np.logical_xor(
            run_choice, run_errs)  # This should be the correct action
        assert(len(run_answer) == ntrials)
        answer.extend(run_answer)

        if 'prev_reward' in weights or 'cum_reward' in weights:
            run_rew = t2p.reward() > 0
            assert(len(run_rew) == ntrials)
            date_reward.extend(run_rew)

        if 'prev_punish' in weights or 'cum_punish' in weights:
            run_punish = t2p.punishment() > 0
            assert(len(run_punish) == ntrials)
            date_punish.extend(run_punish)

        for ori in oris:
            ori_trials = [o == ori for o in t2p.orientations]
            assert(len(ori_trials) == ntrials)
            oris[ori].extend(ori_trials)

    if date_ntrials > 0:
        day_length.append(date_ntrials)
        days.append(last_date)

        reward.extend(date_reward)
        punish.extend(date_punish)

        if 'cum_reward' in weights:
            cum_reward.extend(np.cumsum(date_reward))
        if 'cum_punish' in weights:
            cum_punish.extend(np.cumsum(date_punish))

    out = {'name': runs.mouse}
    out['dayLength'] = np.array(day_length)
    out['runLength'] = np.array(run_length)
    out['days'] = np.array(days)
    out['dateRuns'] = np.array(dateRuns)
    out['dateRunTrials'] = np.array(dateRunTrials)
    out['y'] = np.array([2 if val else 1 for val in y])
    assert(len(out['dateRunTrials']) == len(out['y']))
    out['correct'] = np.array(correct)
    out['answer'] = np.array([2 if val else 1 for val in answer])

    out['inputs'] = {}
    for ori in oris:
        key = 'ori_{}'.format(ori)
        ori_arr = np.array(oris[ori])
        out['inputs'][key] = np.zeros((len(oris[ori]), 2))
        out['inputs'][key][:, 0] += ori_arr
        out['inputs'][key][1:, 1] += ori_arr[:-1]

    if 'prev_choice' in weights:
        out['inputs']['prev_choice'] = np.zeros((len(out['y']), 2))
        out['inputs']['prev_choice'][1:, 0] = out['y'][:-1]
        out['inputs']['prev_choice'][2:, 1] = out['y'][:-2]
    if 'prev_answer' in weights:
        out['inputs']['prev_answer'] = np.zeros((len(out['answer']), 2))
        out['inputs']['prev_answer'][1:, 0] = out['answer'][:-1]
        out['inputs']['prev_answer'][2:, 1] = out['answer'][:-2]

    if 'prev_reward' in weights:
        out['inputs']['prev_reward'] = np.zeros((len(reward), 2))
        out['inputs']['prev_reward'][1:, 0] += reward[:-1]
        out['inputs']['prev_reward'][2:, 1] += reward[:-2]
    if 'prev_punish' in weights:
        out['inputs']['prev_punish'] = np.zeros((len(punish), 2))
        out['inputs']['prev_punish'][1:, 0] += punish[:-1]
        out['inputs']['prev_punish'][2:, 1] += punish[:-2]

    if 'cum_reward' in weights:
        # Convert to array and add a second dimension
        out['inputs']['cum_reward'] = np.array(cum_reward)[:, None]
    if 'cum_punish' in weights:
        out['inputs']['cum_punish'] = np.array(cum_punish)[:, None]

    if not include_pavlovian:
        pav_trials = np.array(pav_trials)
        for key in ['y', 'correct', 'answer', 'dateRunTrials']:
            out[key] = out[key][~pav_trials]
        for key in out['inputs']:
            out['inputs'][key] = out['inputs'][key][~pav_trials]

    return out


def sync_tca_pillow(
        mouse,
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        word=None,
        group_by='all',
        nan_thresh=0.85,
        score_threshold=0.8,
        rank_num=18,
        verbose=True):
    """
    Create a pandas dataframe of pillow and TCA results with indices
    updated to match.
    """

    # default TCA params to use
    if not word:
        if mouse == 'OA27':
            word = 'restaurant'
        else:
            word = 'whale'  # should be updated to 'obligations'
        if verbose:
            print('Creating dataframe for ' + mouse + '-' + word)

    ms = flow.Mouse(mouse)
    psy = ms.psytracker(verbose=True)
    dateRuns = psy.data['dateRuns']
    trialRuns = psy.data['runLength']

    # create your trial indices per day and run
    trial_idx = []
    for i in trialRuns:
        trial_idx.extend(range(i))

    # get date and run vectors
    date_vec = []
    run_vec = []
    for c, i in enumerate(dateRuns):
        date_vec.extend([i[0]]*trialRuns[c])
        run_vec.extend([i[1]]*trialRuns[c])

    # create your data dict, transform from log odds to odds ratio
    data = {}
    for c, i in enumerate(psy.weight_labels):
        # adding multiplication step here with binary vector !!!!!!
        data[i + '_interaction'] = np.exp(psy.fits[c, :])*psy.inputs[:, c].T
        data[i] = np.exp(psy.fits[c, :])
        data[i + '_input'] = psy.inputs[:, c].T
    ori_0_in = [i[0] for i in psy.data['inputs']['ori_0']]
    ori_135_in = [i[0] for i in psy.data['inputs']['ori_135']]
    ori_270_in = [i[0] for i in psy.data['inputs']['ori_270']]
    blank_in = [
        0 if i == 1 else 1 for i in
        np.sum((ori_0_in, ori_135_in, ori_270_in), axis=0)]

    # loop through psy data create a binary vectors for trial history
    binary_cat = ['ori_0', 'ori_135', 'ori_270', 'prev_reward', 'prev_punish']
    for cat in binary_cat:
        data[cat + '_th'] = [i[0] for i in psy.data['inputs'][cat]]
        data[cat + '_th_prev'] = [i[1] for i in psy.data['inputs'][cat]]

    # create a single list of orientations to match format of meta
    ori_order = [0, 135, 270, -1]
    data['orientation'] = [
        ori_order[np.where(np.isin(i, 1))[0][0]]
        for i in zip(ori_0_in, ori_135_in, ori_270_in, blank_in)]

    # create your index out of relevant variables
    index = pd.MultiIndex.from_arrays([
                [mouse]*len(trial_idx),
                date_vec,
                run_vec,
                trial_idx
                ],
                names=['mouse', 'date', 'run', 'trial_idx'])

    # make master dataframe
    dfr = pd.DataFrame(data, index=index)

    # load TCA data
    load_kwargs = {'mouse': mouse,
                   'method': method,
                   'cs': cs,
                   'warp': warp,
                   'word': word,
                   'group_by': group_by,
                   'nan_thresh': nan_thresh,
                   'score_threshold': score_threshold,
                   'rank': rank_num}
    tensor, _, _ = load.groupday_tca_model(**load_kwargs)
    meta = load.groupday_tca_meta(**load_kwargs)

    # add in continuous dprime so psytracker data frame
    dp = pool.calc.psytrack.dprime(flow.Mouse(mouse))
    dfr['dprime'] = dp

    # add in non continuous dprime to meta dataframe
    meta = utils.add_dprime_to_meta(meta)

    # filter out blank trials
    blank_trials_bool = (dfr['orientation'] >= 0)
    psy_df = dfr.loc[blank_trials_bool, :]

    # check that all runs have matched trial orientations
    new_psy_df_list = []
    new_meta_df_list = []
    drop_trials_bin = np.zeros((len(psy_df)))
    dates = meta.reset_index()['date'].unique()
    for d in dates:
        psy_day_bool = psy_df.reset_index()['date'].isin([d]).values
        meta_day_bool = meta.reset_index()['date'].isin([d]).values
        psy_day_df = psy_df.iloc[psy_day_bool, :]
        meta_day_df = meta.iloc[meta_day_bool, :]
        runs = meta_day_df.reset_index()['run'].unique()
        drop_pos_day = np.where(psy_day_bool)[0]
        for r in runs:
            psy_run_bool = psy_day_df.reset_index()['run'].isin([r]).values
            meta_run_bool = meta_day_df.reset_index()['run'].isin([r]).values
            psy_run_df = psy_day_df.iloc[psy_run_bool, :]
            meta_run_df = meta_day_df.iloc[meta_run_bool, :]
            psy_run_idx = psy_run_df.reset_index()['trial_idx'].values
            meta_run_idx = meta_run_df.reset_index()['trial_idx'].values


            # drop extra trials from trace2P that don't have associated imaging
            max_trials = np.min([len(psy_run_idx), len(meta_run_idx)])

            # get just your orientations for checking that trials are matched
            meta_ori = meta_run_df['orientation'].iloc[:max_trials]
            psy_ori = psy_run_df['orientation'].iloc[:max_trials]

            # make sure all oris match between vectors of the same length each day
            assert np.all(psy_ori.values == meta_ori.values)

            # check which trials are dropped
            drop_pos_run = drop_pos_day[psy_run_bool][:max_trials]
            drop_trials_bin[drop_pos_run] = 1

            # if everything looks good, copy meta index into psy
            meta_new = meta_run_df.iloc[:max_trials]
            psy_new = psy_run_df.iloc[:max_trials]
            data = {}
            for i in psy_new.columns:
                data[i] = psy_new[i].values
            new_psy_df_list.append(pd.DataFrame(data=data, index=meta_new.index))
            new_meta_df_list.append(meta_new)

    meta1 = pd.concat(new_meta_df_list, axis=0)
    psy1 = pd.concat(new_psy_df_list, axis=0)

    tca_data = {}
    for comp_num in range(1, rank_num + 1):
        fac = tensor.results[rank_num][0].factors[2][:, comp_num-1]
        tca_data['factor_' + str(comp_num)] = fac
    fac_df = pd.DataFrame(data=tca_data, index=meta1.index)

    return psy1, meta1, fac_df


def _splice_data_y(
        psydata,
        mouse,
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        word=None,
        group_by='all',
        nan_thresh=0.85,
        score_threshold=0.8,
        rank_num=18,
        comp_num=1,
        verbose=True):
    """
    Create a pandas dataframe of trial history modulation indices for one
    mouse. Only looks at initial learning stage.
    """

    # default TCA params to use
    if not word:
        if mouse == 'OA27':
            word = 'tray'
        else:
            word = 'obligations'  # should be updated to 'obligations'
        if verbose:
            print('Creating dataframe for ' + mouse + '-' + word)

    ms = flow.Mouse(mouse)
    psy = ms.psytracker(verbose=True)
    dateRuns = psy.data['dateRuns']
    trialRuns = psy.data['runLength']

    # create your trial indices per day and run
    trial_idx = []
    for i in trialRuns:
        trial_idx.extend(range(i))

    # get date and run vectors
    date_vec = []
    run_vec = []
    for c, i in enumerate(dateRuns):
        date_vec.extend([i[0]]*trialRuns[c])
        run_vec.extend([i[1]]*trialRuns[c])

    # create your data dict, transform from log odds to odds ratio
    data = {}
    for c, i in enumerate(psy.weight_labels):
        # adding multiplication step here with binary vector !!!!!!
        data[i] = np.exp(psy.fits[c, :])*psy.inputs[:, c].T
    ori_0_in = [i[0] for i in psy.data['inputs']['ori_0']]
    ori_135_in = [i[0] for i in psy.data['inputs']['ori_135']]
    ori_270_in = [i[0] for i in psy.data['inputs']['ori_270']]
    blank_in = [
        0 if i == 1 else 1 for i in
        np.sum((ori_0_in, ori_135_in, ori_270_in), axis=0)]

    # loop through psy data create a binary vectors for trial history
    binary_cat = ['ori_0', 'ori_135', 'ori_270', 'prev_reward', 'prev_punish']
    for cat in binary_cat:
        data[cat + '_th'] = [i[0] for i in psy.data['inputs'][cat]]
        data[cat + '_th_prev'] = [i[1] for i in psy.data['inputs'][cat]]

    # create a single list of orientations to match format of meta
    ori_order = [0, 135, 270, -1]
    data['orientation'] = [
        ori_order[np.where(np.isin(i, 1))[0][0]]
        for i in zip(ori_0_in, ori_135_in, ori_270_in, blank_in)]

    # create your index out of relevant variables
    index = pd.MultiIndex.from_arrays([
                [mouse]*len(trial_idx),
                date_vec,
                run_vec,
                trial_idx
                ],
                names=['mouse', 'date', 'run', 'trial_idx'])

    # make master dataframe
    dfr = pd.DataFrame(data, index=index)

    # load TCA data
    load_kwargs = {'mouse': mouse,
                   'method': method,
                   'cs': cs,
                   'warp': warp,
                   'word': word,
                   'group_by': group_by,
                   'nan_thresh': nan_thresh,
                   'score_threshold': score_threshold,
                   'rank': rank_num}
    tensor, _, _ = load.groupday_tca_model(**load_kwargs)
    meta = load.groupday_tca_meta(**load_kwargs)

    # add in continuous dprime so psytracker data frame
    dp = pool.calc.psytrack.dprime(flow.Mouse(mouse))
    dfr['dprime'] = dp

    # add in non continuous dprime to meta dataframe
    meta = utils.add_dprime_to_meta(meta)

    # filter out blank trials
    blank_trials_bool = (dfr['orientation'] >= 0)
    psy_df = dfr.loc[blank_trials_bool, :]

    # check that all runs have matched trial orientations
    new_psy_df_list = []
    new_meta_df_list = []
    drop_trials_bin = np.zeros((len(psy_df)))
    dates = meta.reset_index()['date'].unique()
    for d in dates:
        psy_day_bool = psy_df.reset_index()['date'].isin([d]).values
        meta_day_bool = meta.reset_index()['date'].isin([d]).values
        psy_day_df = psy_df.iloc[psy_day_bool, :]
        meta_day_df = meta.iloc[meta_day_bool, :]
        runs = meta_day_df.reset_index()['run'].unique()
        drop_pos_day = np.where(psy_day_bool)[0]
        for r in runs:
            psy_run_bool = psy_day_df.reset_index()['run'].isin([r]).values
            meta_run_bool = meta_day_df.reset_index()['run'].isin([r]).values
            psy_run_df = psy_day_df.iloc[psy_run_bool, :]
            meta_run_df = meta_day_df.iloc[meta_run_bool, :]
            psy_run_idx = psy_run_df.reset_index()['trial_idx'].values
            meta_run_idx = meta_run_df.reset_index()['trial_idx'].values


            # drop extra trials from trace2P that don't have associated imaging
            max_trials = np.min([len(psy_run_idx), len(meta_run_idx)])

            # get just your orientations for checking that trials are matched
            meta_ori = meta_run_df['orientation'].iloc[:max_trials]
            psy_ori = psy_run_df['orientation'].iloc[:max_trials]

            # make sure all oris match between vectors of the same length each day
            assert np.all(psy_ori.values == meta_ori.values)

            # check which trials are dropped
            drop_pos_run = drop_pos_day[psy_run_bool][:max_trials]
            drop_trials_bin[drop_pos_run] = 1

            # if everything looks good, copy meta index into psy
            meta_new = meta_run_df.iloc[:max_trials]
            psy_new = psy_run_df.iloc[:max_trials]
            data = {}
            for i in psy_new.columns:
                data[i] = psy_new[i].values
            new_psy_df_list.append(pd.DataFrame(data=data, index=meta_new.index))
            new_meta_df_list.append(meta_new)

    meta1 = pd.concat(new_meta_df_list, axis=0)
    psy1 = pd.concat(new_psy_df_list, axis=0)

    tca_data = {}
    fac = tensor.results[rank_num][0].factors[2][:, comp_num-1]
    tca_data['factor_' + str(comp_num)] = fac
    fac_df = pd.DataFrame(data=tca_data, index=meta1.index)

    # threshold your data in a clever way so that you are not only
    # looking at orientation trials
    clever_binary = np.ones((len(fac)))
    thresh = np.nanstd(fac)*1
    clever_binary[fac > thresh] = 2

    # which values were dropped from the psydata. Use this to update psydata
    blank_trials_bool[blank_trials_bool] = (drop_trials_bin == 1)
    keep_bool = blank_trials_bool
    drop_bool = blank_trials_bool == False
    # keep_bool = np.logical_and(
    #     drop_trials_bin == 0, blank_trials_bool.values == True)
    # drop_bool = np.logical_or(
    #     drop_trials_bin == 1, blank_trials_bool.values == False)

    # you don't have any blank trials to avoid using them.
    psydata['y'][drop_bool] = 1
    psydata['answer'][drop_bool] = 1  # 1-2 binary not 0-1
    psydata['y'][keep_bool] = clever_binary
    psydata['answer'][keep_bool] = clever_binary

    print('cleared :)')

    return psydata


def _splice_data_inputs(
        psydata,
        mouse,
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        word=None,
        group_by='all',
        nan_thresh=0.85,
        score_threshold=0.8,
        rank_num=18,
        verbose=True):
    """
    Create dict used for fitting Pillow model. Main purpose of function
    is to align indices from Pillow and TCA since they often sub-select
    different trials. This forces Pillow 'y' and 'answer' to have same
    trials as TCA and uses TCA trial factors as 'inputs'.
    """

    # default TCA params to use
    if not word:
        if mouse == 'OA27':
            word = 'tray'
        else:
            word = 'obligations'  # should be updated to 'obligations'
        if verbose:
            print('Creating dataframe for ' + mouse + '-' + word)

    ms = flow.Mouse(mouse)
    psy = ms.psytracker(verbose=True)
    dateRuns = psy.data['dateRuns']
    trialRuns = psy.data['runLength']

    # create your trial indices per day and run
    trial_idx = []
    for i in trialRuns:
        trial_idx.extend(range(i))

    # get date and run vectors
    date_vec = []
    run_vec = []
    for c, i in enumerate(dateRuns):
        date_vec.extend([i[0]]*trialRuns[c])
        run_vec.extend([i[1]]*trialRuns[c])

    # create your data dict, transform from log odds to odds ratio
    data = {}
    for c, i in enumerate(psy.weight_labels):
        # adding multiplication step here with binary vector !!!!!!
        data[i] = np.exp(psy.fits[c, :])*psy.inputs[:, c].T
    ori_0_in = [i[0] for i in psy.data['inputs']['ori_0']]
    ori_135_in = [i[0] for i in psy.data['inputs']['ori_135']]
    ori_270_in = [i[0] for i in psy.data['inputs']['ori_270']]
    blank_in = [
        0 if i == 1 else 1 for i in
        np.sum((ori_0_in, ori_135_in, ori_270_in), axis=0)]

    # loop through psy data create a binary vectors for trial history
    binary_cat = ['ori_0', 'ori_135', 'ori_270', 'prev_reward', 'prev_punish']
    for cat in binary_cat:
        data[cat + '_th'] = [i[0] for i in psy.data['inputs'][cat]]
        data[cat + '_th_prev'] = [i[1] for i in psy.data['inputs'][cat]]

    # create a single list of orientations to match format of meta
    ori_order = [0, 135, 270, -1]
    data['orientation'] = [
        ori_order[np.where(np.isin(i, 1))[0][0]]
        for i in zip(ori_0_in, ori_135_in, ori_270_in, blank_in)]

    # create your index out of relevant variables
    index = pd.MultiIndex.from_arrays([
                [mouse]*len(trial_idx),
                date_vec,
                run_vec,
                trial_idx
                ],
                names=['mouse', 'date', 'run', 'trial_idx'])

    # make master dataframe
    dfr = pd.DataFrame(data, index=index)

    # load TCA data
    load_kwargs = {'mouse': mouse,
                   'method': method,
                   'cs': cs,
                   'warp': warp,
                   'word': word,
                   'group_by': group_by,
                   'nan_thresh': nan_thresh,
                   'score_threshold': score_threshold,
                   'rank': rank_num}
    tensor, _, _ = load.groupday_tca_model(**load_kwargs)
    meta = load.groupday_tca_meta(**load_kwargs)

    # add in continuous dprime so psytracker data frame
    dp = pool.calc.psytrack.dprime(flow.Mouse(mouse))
    dfr['dprime'] = dp

    # add in non continuous dprime to meta dataframe
    meta = utils.add_dprime_to_meta(meta)

    # filter out blank trials
    blank_trials_bool = (dfr['orientation'] >= 0)
    psy_df = dfr.loc[blank_trials_bool, :]

    # check that all runs have matched trial orientations
    new_psy_df_list = []
    new_meta_df_list = []
    drop_trials_bin = np.zeros((len(psy_df)))
    dates = meta.reset_index()['date'].unique()
    for d in dates:
        psy_day_bool = psy_df.reset_index()['date'].isin([d]).values
        meta_day_bool = meta.reset_index()['date'].isin([d]).values
        psy_day_df = psy_df.iloc[psy_day_bool, :]
        meta_day_df = meta.iloc[meta_day_bool, :]
        runs = meta_day_df.reset_index()['run'].unique()
        drop_pos_day = np.where(psy_day_bool)[0]
        for r in runs:
            psy_run_bool = psy_day_df.reset_index()['run'].isin([r]).values
            meta_run_bool = meta_day_df.reset_index()['run'].isin([r]).values
            psy_run_df = psy_day_df.iloc[psy_run_bool, :]
            meta_run_df = meta_day_df.iloc[meta_run_bool, :]
            psy_run_idx = psy_run_df.reset_index()['trial_idx'].values
            meta_run_idx = meta_run_df.reset_index()['trial_idx'].values


            # drop extra trials from trace2P that don't have associated imaging
            max_trials = np.min([len(psy_run_idx), len(meta_run_idx)])

            # get just your orientations for checking that trials are matched
            meta_ori = meta_run_df['orientation'].iloc[:max_trials]
            psy_ori = psy_run_df['orientation'].iloc[:max_trials]

            # make sure all oris match between vectors of the same length each day
            assert np.all(psy_ori.values == meta_ori.values)

            # check which trials are dropped
            drop_pos_run = drop_pos_day[psy_run_bool][:max_trials]
            drop_trials_bin[drop_pos_run] = 1

            # if everything looks good, copy meta index into psy
            meta_new = meta_run_df.iloc[:max_trials]
            psy_new = psy_run_df.iloc[:max_trials]
            data = {}
            for i in psy_new.columns:
                data[i] = psy_new[i].values
            new_psy_df_list.append(pd.DataFrame(data=data, index=meta_new.index))
            new_meta_df_list.append(meta_new)

    meta1 = pd.concat(new_meta_df_list, axis=0)
    psy1 = pd.concat(new_psy_df_list, axis=0)

    tca_data = {}
    for comp_num in range(1, rank_num+1):
        fac = tensor.results[rank_num][0].factors[2][:, comp_num-1]
        tca_data['factor_' + str(comp_num)] = fac[:, None]

    # which values were dropped from the psydata. Use this to update psydata
    blank_trials_bool[blank_trials_bool] = (drop_trials_bin == 1)
    keep_bool = blank_trials_bool

    # you don't have any blank trials so drop them.
    psydata['y'] = psydata['y'][keep_bool]  # 1-2 binary not 0-1
    psydata['answer'] = psydata['answer'][keep_bool]
    psydata['correct'] = psydata['correct'][keep_bool]
    psydata['dateRunTrials'] = psydata['dateRunTrials'][keep_bool]

    # recalculate dayLength and runLength
    new_runLength = []
    new_dayLength = []
    for di in np.unique(psydata['dateRunTrials'][:, 0]):
        day_bool = psydata['dateRunTrials'][:, 0] == di
        new_dayLength.append(np.sum(day_bool))
        day_runs = psydata['dateRunTrials'][day_bool, 1]
        for ri in np.unique(day_runs):
            run_bool = psydata['dateRunTrials'][:, 1] == ri
            new_runLength.append(np.sum(run_bool))
    psydata['dayLength'] = new_dayLength
    psydata['runLength'] = new_runLength

    # update dateRuns and days
    clean_days = np.unique(psydata['dateRunTrials'][:, 0])
    clean_day_bool = np.isin(psydata['days'], clean_days)
    psydata['days'] = psydata['days'][clean_day_bool]
    clean_run_bool = np.isin(psydata['dateRuns'][:, 0], psydata['days'])
    psydata['dateRuns'] = psydata['dateRuns'][clean_run_bool]

    # ensure that you still have the same number of runs
    assert len(psydata['runLength']) == len(psydata['dateRuns'])

    # ensure that you still have the same number of days
    assert len(psydata['dayLength']) == len(psydata['days'])

    # reset inputs
    psydata['inputs'] = tca_data

    if verbose:
        print('Successful sync of psytracker and TCA data :)')
        print(' Fitting {} days'.format(psydata['days']))
        print(' Fitting {} runs'.format(psydata['dateRuns']))
        print(' Fitting {} trials'.format(psydata['dateRunTrials']))
        print(' Fitting {} total hyper-parameters'.format(rank_num))

    return psydata

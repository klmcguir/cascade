"""Functions for fitting generalized linear models (GLM)."""
import flow
import pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from . import paths
from . import utils
from flow.misc import regression


def fit_trial_factors(
        mouse='OA27',
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        word='orlando',
        group_by='all',
        nan_thresh=0.85,
        rank_num=18,
        historical_memory=5,
        rectified=False,
        verbose=False):

    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}
    group_pars = {'group_by': group_by}

    # if cells were removed with too many nan trials
    if nan_thresh:
        nt_tag = '_nantrial' + str(nan_thresh)
    else:
        nt_tag = ''

    # load dir
    load_dir = paths.tca_path(
        mouse, 'group', pars=pars, word=word, group_pars=group_pars)
    tensor_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_group_decomp_' + str(trace_type) + '.npy')
    input_tensor_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_group_tensor_' + str(trace_type) + '.npy')
    meta_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_df_group_meta.pkl')

    # load your data
    ensemble = np.load(tensor_path)
    ensemble = ensemble.item()
    # re-balance your factors ()
    print('Re-balancing factors.')
    for r in ensemble[method].results:
        for i in range(len(ensemble[method].results[r])):
            ensemble[method].results[r][i].factors.rebalance()
    V = ensemble[method]
    meta = pd.read_pickle(meta_path)
    meta = utils.update_naive_cs(meta)
    orientation = meta.reset_index()['orientation']
    condition = meta.reset_index()['condition']
    speed = meta.reset_index()['speed']
    dates = meta.reset_index()['date']
    total_time = pd.DataFrame(data={'total_time': np.arange(len(time_in_trial))}, index=time_in_trial.index)
    learning_state = meta['learning_state']
    trialerror = meta['trialerror']

    # create dataframe of dprime values
    dprime_vec = []
    for date in dates:
        date_obj = flow.Date(mouse, date=date)
        dprime_vec.append(pool.calc.performance.dprime(date_obj))
    data = {'dprime': dprime_vec}
    dprime = pd.DataFrame(data=data, index=speed.index)
    dprime = dprime['dprime']  # make indices match to meta

    # get time per day sawtooth
    time_per_day = []
    counter = 0
    prev_date = dates[0]
    for date in dates:
        if prev_date != date:
            counter = 0
            prev_date = date
        time_per_day.append(counter)
        counter += 1
    data = {'time_day': time_per_day}
    time_day = pd.DataFrame(data=data, index=speed.index)
    time_day = time_day['time_day']  # make indices match to meta

    # learning times
    time_naive, time_learning, time_reversal = [], [], []
    counter, ncounter, rcounter = 0, 0, 0
    for stage in learning_state:
        if stage == 'naive':
            time_naive.append(ncounter)
            ncounter += 1
        else:
            time_naive.append(0)
        if stage == 'learning':
            time_learning.append(counter)
            counter += 1
        else:
            time_learning.append(0)
        if stage == 'reversal1':
            time_reversal.append(rcounter)
            rcounter += 1
        else:
            time_reversal.append(0)
    data = {'time_learning': time_learning, 'time_naive': time_naive, 'time_reversal': time_reversal}
    time_learning = pd.DataFrame(data=data, index=speed.index)

    # choose which learning stage to run GLM on
    stage_indexer = learning_state.isin(['learning']).values

    # ------------- GET Condition TUNING
    trial_weights = V.results[rank_num][0].factors[2][:, :]
    conds_to_check = ['plus', 'minus', 'neutral']
    conds_weights = np.zeros((len(conds_to_check), rank_num))
    for c, conds in enumerate(conds_to_check):
        conds_weights[c, :] = np.nanmean(
            trial_weights[(condition.values == conds) & stage_indexer, :], axis=0)
    # normalize using summed mean response to all three
    conds_total = np.nansum(conds_weights, axis=0)
    for c in range(len(conds_to_check)):
        conds_weights[c, :] = np.divide(
            conds_weights[c, :], conds_total)
    pref_cs_idx = np.argmax(conds_weights, axis=0)

    # loop through components and fit linear model
    models = []
    model_fits = []
    for fac_num in range(np.shape(V.results[rank_num][0].factors[2][:, :])[1]):
        trial_weights = V.results[rank_num][0].factors[2][:, fac_num]
        cond_indexer = (condition.values == conds_to_check[pref_cs_idx[fac_num]]) & stage_indexer
        # cond_indexer = (condition.isin(['plus', 'minus', 'neutral']).values) & stage_indexer
        # plus, minus, neutral = np.zeros(len(trial_weights)), np.zeros(len(trial_weights)), np.zeros(len(trial_weights))
        # plus[condition.isin(['plus']).values] = 1
        # minus[condition.isin(['minus']).values] = 1
        # neutral[condition.isin(['neutral']).values] = 1

        trial_fac = trial_weights[cond_indexer]

        # create df of trial history (sliding window of when similar cue
        # appeared in the last 5 trials)
        # create df of reward history (sliding window of recieved reward
        # in the last 5 trials)
        trial_pos = np.where(cond_indexer)[0]
        reward_pos = (trialerror.values == 0)
        trial_history, reward_history = [], []
        for i in trial_pos:
            trial_history.append(np.sum((trial_pos >= i-historical_memory) & (trial_pos < i)))
            ind = i-historical_memory if (i-historical_memory > 0) else 0
            reward_history.append(np.sum(reward_pos[ind:i]))

        data = {'trial_fac': trial_fac.flatten(),
                'dprime': dprime.values[cond_indexer].flatten(),
                'time_day': time_day.values[cond_indexer].flatten(),
                'time': total_time.values[cond_indexer].flatten(),
                'speed': speed.values[cond_indexer].flatten(),
                # 'time_naive': time_learning['time_naive'].values[cond_indexer].flatten(),
                'time_learning': time_learning['time_learning'].values[cond_indexer].flatten(),
                # 'time_reversal': time_learning['time_reversal'].values[cond_indexer].flatten(),
                'reward_history': reward_history,
                'trial_history': trial_history,
                # 'plus': plus[cond_indexer],
                # 'minus': minus[cond_indexer],
                # 'neutral': neutral[cond_indexer]
               }

        # z-score
        for k in data.keys():
            data[k] = (data[k] - np.nanmean(data[k]))/np.nanstd(data[k])
        fac_df = pd.DataFrame(data=data).dropna()

        # fit GLM
        print('Component ' + str(fac_num+1))
        formula = 'trial_fac ~ time + time_day + time_learning + speed + dprime + trial_history + reward_history'
        model = regression.glm(formula, fac_df, dropzeros=False, link='identity')
        models.append(model)
        res = model.fit()
        model_fits.append(res)
        print('Total deviance explained: ', 1 - res.deviance/res.null_deviance)
        for k in res.params.keys():
            if k.lower() != 'intercept':
                beta = res.params[k]
                print(k + ' deviance explained: ', 1 - (np.sum(np.square(beta*data[k] - model.endog))/res.null_deviance))
        print('')
        plt.figure(figsize=(20,4))
        plt.plot(model.endog, label='trial factor')
        plt.plot(res.mu, label='model')
        plt.xlabel('trials')
        plt.legend()
        plt.title('Component ' + str(fac_num+1))


def fit_cells(
        mouse='OA27',
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        word='orlando',
        group_by='all',
        nan_thresh=0.85,
        rank_num=18,
        historical_memory=5,
        rectified=False,
        verbose=False):

    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}
    group_pars = {'group_by': group_by}

    # if cells were removed with too many nan trials
    if nan_thresh:
        nt_tag = '_nantrial' + str(nan_thresh)
    else:
        nt_tag = ''

    # load dir
    load_dir = paths.tca_path(
        mouse, 'group', pars=pars, word=word, group_pars=group_pars)
    tensor_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_group_decomp_' + str(trace_type) + '.npy')
    input_tensor_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_group_tensor_' + str(trace_type) + '.npy')
    meta_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_df_group_meta.pkl')

    # load your data
    X = np.load(input_tensor_path)
    meta = pd.read_pickle(meta_path)
    meta = utils.update_naive_cs(meta)
    orientation = meta.reset_index()['orientation']
    condition = meta.reset_index()['condition']
    speed = meta.reset_index()['speed']
    dates = meta.reset_index()['date']
    total_time = pd.DataFrame(data={'total_time': np.arange(len(time_in_trial))}, index=time_in_trial.index)
    learning_state = meta['learning_state']
    trialerror = meta['trialerror']

    # create dataframe of dprime values
    dprime_vec = []
    for date in dates:
        date_obj = flow.Date(mouse, date=date)
        dprime_vec.append(pool.calc.performance.dprime(date_obj))
    data = {'dprime': dprime_vec}
    dprime = pd.DataFrame(data=data, index=speed.index)
    dprime = dprime['dprime']  # make indices match to meta

    # get time per day sawtooth
    time_per_day = []
    counter = 0
    prev_date = dates[0]
    for date in dates:
        if prev_date != date:
            counter = 0
            prev_date = date
        time_per_day.append(counter)
        counter += 1
    data = {'time_day': time_per_day}
    time_day = pd.DataFrame(data=data, index=speed.index)
    time_day = time_day['time_day']  # make indices match to meta

    # learning times
    time_naive, time_learning, time_reversal = [], [], []
    counter, ncounter, rcounter = 0, 0, 0
    for stage in learning_state:
        if stage == 'naive':
            time_naive.append(ncounter)
            ncounter += 1
        else:
            time_naive.append(0)
        if stage == 'learning':
            time_learning.append(counter)
            counter += 1
        else:
            time_learning.append(0)
        if stage == 'reversal1':
            time_reversal.append(rcounter)
            rcounter += 1
        else:
            time_reversal.append(0)
    data = {'time_learning': time_learning, 'time_naive': time_naive, 'time_reversal': time_reversal}
    time_learning = pd.DataFrame(data=data, index=speed.index)

    # choose which learning stage to run GLM on
    stage_indexer = learning_state.isin(['learning']).values

    # ------------- GET Condition TUNING
    trial_weights = V.results[rank_num][0].factors[2][:, :]
    conds_to_check = ['plus', 'minus', 'neutral']
    conds_weights = np.zeros((len(conds_to_check), rank_num))
    for c, conds in enumerate(conds_to_check):
        conds_weights[c, :] = np.nanmean(
            trial_weights[(condition.values == conds) & stage_indexer, :], axis=0)
    # normalize using summed mean response to all three
    conds_total = np.nansum(conds_weights, axis=0)
    for c in range(len(conds_to_check)):
        conds_weights[c, :] = np.divide(
            conds_weights[c, :], conds_total)
    pref_cs_idx = np.argmax(conds_weights, axis=0)

    # loop through components and fit linear model
    models = []
    model_fits = []
    for fac_num in range(np.shape(V.results[rank_num][0].factors[2][:, :])[1]):
        trial_weights = V.results[rank_num][0].factors[2][:, fac_num]
        cond_indexer = (condition.values == conds_to_check[pref_cs_idx[fac_num]]) & stage_indexer
        # cond_indexer = (condition.isin(['plus', 'minus', 'neutral']).values) & stage_indexer
        # plus, minus, neutral = np.zeros(len(trial_weights)), np.zeros(len(trial_weights)), np.zeros(len(trial_weights))
        # plus[condition.isin(['plus']).values] = 1
        # minus[condition.isin(['minus']).values] = 1
        # neutral[condition.isin(['neutral']).values] = 1

        trial_fac = trial_weights[cond_indexer]

        # create df of trial history (sliding window of when similar cue
        # appeared in the last 5 trials)
        # create df of reward history (sliding window of recieved reward
        # in the last 5 trials)
        trial_pos = np.where(cond_indexer)[0]
        reward_pos = (trialerror.values == 0)
        trial_history, reward_history = [], []
        for i in trial_pos:
            trial_history.append(np.sum((trial_pos >= i-historical_memory) & (trial_pos < i)))
            ind = i-historical_memory if (i-historical_memory > 0) else 0
            reward_history.append(np.sum(reward_pos[ind:i]))

        data = {'trial_fac': trial_fac.flatten(),
                'dprime': dprime.values[cond_indexer].flatten(),
                'time_day': time_day.values[cond_indexer].flatten(),
                'time': total_time.values[cond_indexer].flatten(),
                'speed': speed.values[cond_indexer].flatten(),
                # 'time_naive': time_learning['time_naive'].values[cond_indexer].flatten(),
                'time_learning': time_learning['time_learning'].values[cond_indexer].flatten(),
                # 'time_reversal': time_learning['time_reversal'].values[cond_indexer].flatten(),
                'reward_history': reward_history,
                'trial_history': trial_history,
                # 'plus': plus[cond_indexer],
                # 'minus': minus[cond_indexer],
                # 'neutral': neutral[cond_indexer]
               }

        # z-score
        for k in data.keys():
            data[k] = (data[k] - np.nanmean(data[k]))/np.nanstd(data[k])
        fac_df = pd.DataFrame(data=data).dropna()

        # fit GLM
        print('Component ' + str(fac_num+1))
        formula = 'trial_fac ~ time + time_day + time_learning + speed + dprime + trial_history + reward_history'
        model = regression.glm(formula, fac_df, dropzeros=False, link='identity')
        models.append(model)
        res = model.fit()
        model_fits.append(res)
        print('Total deviance explained: ', 1 - res.deviance/res.null_deviance)
        for k in res.params.keys():
            if k.lower() != 'intercept':
                beta = res.params[k]
                print(k + ' deviance explained: ', 1 - (np.sum(np.square(beta*data[k] - model.endog))/res.null_deviance))
        print('')
        plt.figure(figsize=(20,4))
        plt.plot(model.endog, label='trial factor')
        plt.plot(res.mu, label='model')
        plt.xlabel('trials')
        plt.legend()
        plt.title('Component ' + str(fac_num+1))

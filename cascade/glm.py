"""Functions for fitting generalized linear models (GLM)."""
import flow
import pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from . import paths, utils, load
from . psytrack.train_factor import sync_tca_pillow
from flow.misc import regression
from . import calc
from copy import deepcopy

# default values (from mean of all model sigmas when allowing fitting)
default_sigmas = {
    'fixed_sigma':
        np.array([0.098, 0.185, 0.185, 0.185, 0.0166, 0.1128, 0.0457]),
    'fixed_sigma_day':
        np.array([1.3003, 2.1746, 2.1746, 2.1746, 0.1195, 0.3035, 0.6393])
                }


def groupmouse_fit_poisson(
        mice=['OA27', 'OA26', 'OA67', 'VF226', 'OA32', 'OA34', 'OA36'],
        words=None,
        hitmiss=False,
        verbose=True, **kwargs):
    """
    Create a pandas dataframe of GLM results fit across mice.
    """
    kwargs_defaults = {
        'trace_type': 'zscore_day',
        'method': 'mncp_hals',
        'cs': '',
        'warp': False,
        'word': 'restaurant',
        'group_by': 'all',
        'nan_thresh': 0.85,
        'score_threshold': 0.8,
        'rank_num': 15,
        'fixed_sigma': default_sigmas['fixed_sigma'],
        'fixed_sigma_day': default_sigmas['fixed_sigma_day']}
    if kwargs is None:
        kwargs = {}
    kwargs_defaults.update(kwargs)

    # default TCA params to use
    if not words:
        w1 = 'restaurant'
        w2 = 'whale'
        words = [w1 if s == 'OA27' else w2 for s in mice]

    # get all single mouse dataframes
    df_list = []
    for m, w in zip(mice, words):
        kwargs_defaults['word'] = w
        if hitmiss:
            # includes preferred ori and relevant hit/miss, CR/FA
            th_df = fit_trial_factors_poisson_hitmiss(
                        m,
                        verbose=verbose,
                        **kwargs_defaults)
        else:
            # includes preferred ori and relevant plus, minus, neutral
            th_df = fit_trial_factors_poisson(
                        m,
                        verbose=verbose,
                        **kwargs_defaults)

        df_list.append(th_df)
    all_dfs = pd.concat(df_list, axis=0)

    return all_dfs


def fit_trial_factors_poisson(mouse, verbose=True, **kwargs):
    kwargs_defaults = {
        'trace_type': 'zscore_day',
        'method': 'mncp_hals',
        'cs': '',
        'warp': False,
        'word': 'restaurant',
        'group_by': 'all',
        'nan_thresh': 0.85,
        'score_threshold': 0.8,
        'rank_num': 15,
        'fixed_sigma': default_sigmas['fixed_sigma'],
        'fixed_sigma_day': default_sigmas['fixed_sigma_day']}
    if kwargs is None:
        kwargs = {}
    kwargs_defaults.update(kwargs)

    # load your TCA and pillow data, matching their trial indices
    psy1, meta1, fac_df, psydata = sync_tca_pillow(
        mouse, verbose=verbose, **kwargs_defaults)

    # load in the tuning of your factors
    tuning_kwargs = deepcopy(kwargs_defaults)
    tuning_kwargs.pop('fixed_sigma')
    tuning_kwargs.pop('fixed_sigma_day')
    tuning_df = calc.tca.trial_factor_tuning(
            mouse=flow.Mouse(mouse=mouse),
            verbose=verbose, **tuning_kwargs)

    # drop unused columns
    filters_df = psy1.drop(columns=['orientation'])

    # z-score meta1 by day if you want a within day normalization
    meta1_z_byday = (
        meta1
        .groupby('date')
        .transform(lambda x: (x - x.mean()) / x.std()))

    # add in biometrics during stimulus presentation
    filters_df = filters_df.join(meta1['speed'])
    filters_df = filters_df.join(meta1['anticipatory_licks'])
    filters_df = filters_df.join(meta1_z_byday['pupil'])

    # add in biometrics one second before stimulus presentation
    filters_df = filters_df.join(meta1['pre_speed'])
    filters_df = filters_df.join(meta1['pre_licks'])
    filters_df = filters_df.join(meta1_z_byday['pre_pupil'])

    # add in plus, minus, neutral
    cs_to_add = ['plus', 'minus', 'neutral']
    for csi in cs_to_add:
        cs_tuning_vec = np.zeros(len(filters_df))
        cs_tuning_vec[meta1['condition'].isin([csi])] = 1
        filters_df[csi] = cs_tuning_vec

    # z-score to get all filters on a similar scale
    zfilters_df = (
        filters_df
        .transform(lambda x: (x - x.mean()) / x.std()))

    # choose your filters from Pillow to fit to TCA
    cols = ['ori_270', 'ori_135', 'ori_0', 'prev_reward_interaction',
            'prev_punish_interaction', 'prev_choice_interaction',
            'bias', 'speed']
    cols.extend(['anticipatory_licks', 'pupil'])
    cols.extend(['ori_270_input', 'ori_135_input',
                 'ori_0_input'])
    cols.extend(['ori_270_interaction', 'ori_135_interaction',
                 'ori_0_interaction'])
    cols.extend(['prev_reward_input', 'prev_punish_input',
                 'prev_choice_input'])
    cols.extend(['ori_270_th_prev', 'ori_135_th_prev',
                 'ori_0_th_prev'])
    cols.extend(['plus', 'minus', 'neutral'])
    filters_subset = zfilters_df.loc[:, cols]

    # fit GLM and a GLM dropping out each filter to test deviance explained
    models, model_fits, dev_exp_full_list = [], [], []
    delta_aic_full_list, sub_aic_full_list = [], []
    total_aic_full_list = []
    total_dev_full_list = []
    fac_list = []
    for fac_num in range(1, kwargs_defaults['rank_num']+1):
        # original formula
        fac_tuning = tuning_df.loc[(mouse, fac_num), 'preferred_tuning']
        fac_cs = tuning_df.loc[(mouse, fac_num), 'preferred_tuning_cs']
        if fac_tuning == '0':
            formula = 'y ~ ori_0_input +{}'.format(' {} +'.format(fac_cs)) \
             + ' prev_reward_input + prev_punish_input + prev_choice_input +' \
             + ' speed + pupil + anticipatory_licks'
            drop_list = [
                ' ori_0_input +',
                ' {} +'.format(fac_cs),
                ' prev_reward_input +',
                ' prev_punish_input +',
                ' prev_choice_input +',
                ' speed +',
                ' pupil +',
                ' + anticipatory_licks']
        elif fac_tuning == '135':
            formula = 'y ~ ori_135_input +{}'.format(' {} +'.format(fac_cs)) \
             + ' prev_reward_input + prev_punish_input + prev_choice_input +' \
             + ' speed + pupil + anticipatory_licks'
            drop_list = [
                ' ori_135_input +',
                ' {} +'.format(fac_cs),
                ' prev_reward_input +',
                ' prev_punish_input +',
                ' prev_choice_input +',
                ' speed +',
                ' pupil +',
                ' + anticipatory_licks']
        elif fac_tuning == '270':
            formula = 'y ~ ori_270_input +{}'.format(' {} +'.format(fac_cs)) \
             + ' prev_reward_input + prev_punish_input + prev_choice_input +' \
             + ' speed + pupil + anticipatory_licks'
            drop_list = [
                ' ori_270_input +',
                ' {} +'.format(fac_cs),
                ' prev_reward_input +',
                ' prev_punish_input +',
                ' prev_choice_input +',
                ' speed +',
                ' pupil +',
                ' + anticipatory_licks']
        elif fac_tuning == 'broad':
            formula = 'y ~ ori_270_input + ori_135_input + ori_0_input +' \
             + ' plus + minus + neutral + prev_reward_input +' \
             + ' prev_punish_input + prev_choice_input + speed + pupil +' \
             + ' anticipatory_licks'
            drop_list = [
                ' ori_270_input +',
                ' ori_135_input +',
                ' ori_0_input +',
                ' {} +'.format('plus'),
                ' {} +'.format('minus'),
                ' {} +'.format('neutral'),
                ' prev_reward_input +',
                ' prev_punish_input +',
                ' prev_choice_input +',
                ' speed +',
                ' pupil +',
                ' + anticipatory_licks']

        # add your factor for fitting as the y variable
        fac = 'factor_' + str(fac_num)
        sub_xy = filters_subset.join(fac_df)
        # scale and round to make it Poisson-friendly
        sub_xy['y'] = deepcopy((sub_xy[fac]*100).apply(np.floor))
        sub_xy = sub_xy.reset_index()

        # if a filter is totally empty remove it from the formula and the drop
        for col in sub_xy.columns:
            total_nan = np.sum(sub_xy[col].isna().values)
            total_vals = len(sub_xy[col].values)
            if total_nan == total_vals:
                sub_xy = sub_xy.drop(columns=[col])
                drop_list = [s for s in drop_list if col not in s]
                if verbose:
                    print('{}: dropped column/filter: {}'.format(mouse, col))

        # make sure you don't have any nans
        sub_xy = sub_xy.replace([np.inf, -np.inf], np.nan).dropna()

        try:
            model = regression.glm(
                formula, sub_xy, dropzeros=False,
                link='log', family='Poisson', verbose=False)
        except ValueError:
            print('!!!!!!')
            print('{}: Skipped {}'.format(mouse, fac))
            print('!!!!!!')
            continue

        models.append(model)
        res = model.fit()
        model_fits.append(res)
        total_dev_exp = 1 - res.deviance/res.null_deviance
        total_aic = res.aic
        if verbose:
            print('{}: Component {}'.format(mouse, fac_num))
            print('    Total deviance explained: ',
                  1 - res.deviance/res.null_deviance)

        # get deviance explained per filter
        # add NaN for the internal intercept
        dev_explained_drop = []
        delta_aic = []
        aic_drop = []
        for c, dl in enumerate(drop_list):
            drop_formula = formula.replace(dl, '')
            try:
                model = regression.glm(
                    drop_formula, sub_xy, dropzeros=False,
                    link='log', family='Poisson', verbose=False)
            except:
                dev_explained_drop.append(np.nan)
                delta_aic.append(np.nan)
                aic_drop.append(np.nan)
                continue
            # make up for the intercept beta
            if c == 0:
                dev_explained_drop.append(np.nan)
                delta_aic.append(np.nan)
                aic_drop.append(np.nan)
            res = model.fit()
            aic_drop.append(res.aic)
            delta_aic.append(res.aic - total_aic)
            drop_dev_exp = 1 - res.deviance/res.null_deviance
            dev_explained_drop.append(total_dev_exp - drop_dev_exp)
            delta_aic
        dev_exp_full_list.extend(dev_explained_drop)
        sub_aic_full_list.extend(aic_drop)
        delta_aic_full_list.extend(delta_aic)
        total_aic_full_list.extend([total_aic]*len(delta_aic))
        total_dev_full_list.extend([total_dev_exp]*len(delta_aic))
        fac_list.append(fac_num)

    # aggregate all of your fit results
    df_list = []
    for c, mod in zip(fac_list, model_fits):
        mod_df = mod.summary2().tables[1]
        mod_df['component'] = [c]*len(mod_df)
        mod_df['x'] = mod_df.index
        df_list.append(mod_df)
    all_model_df = pd.concat(df_list, axis=0)
    all_model_df['sub_deviance_explained'] = dev_exp_full_list
    all_model_df['frac_deviance_explained'] = (
        np.array(dev_exp_full_list)/np.array(total_dev_full_list))
    all_model_df['sub_model_aic'] = sub_aic_full_list
    all_model_df['delta_aic'] = delta_aic_full_list
    all_model_df['full_model_aic'] = total_aic_full_list
    all_model_df['full_deviance_explained'] = total_dev_full_list

    all_model_df['mouse'] = [mouse]*len(all_model_df)
    all_model_df = (
        all_model_df
        .reset_index()
        .drop(columns=['index'])
        .set_index(['mouse', 'component', 'x']))

    return all_model_df


def fit_trial_factors_poisson_hitmiss(mouse, verbose=True, **kwargs):
    kwargs_defaults = {
        'trace_type': 'zscore_day',
        'method': 'mncp_hals',
        'cs': '',
        'warp': False,
        'word': 'restaurant',
        'group_by': 'all',
        'nan_thresh': 0.85,
        'score_threshold': 0.8,
        'rank_num': 15,
        'fixed_sigma': default_sigmas['fixed_sigma'],
        'fixed_sigma_day': default_sigmas['fixed_sigma_day']}
    if kwargs is None:
        kwargs = {}
    kwargs_defaults.update(kwargs)

    # load your TCA and pillow data, matching their trial indices
    psy1, meta1, fac_df, psydata = sync_tca_pillow(
        mouse, verbose=verbose, **kwargs_defaults)

    # load in the tuning of your factors
    tuning_kwargs = deepcopy(kwargs_defaults)
    tuning_kwargs.pop('fixed_sigma')
    tuning_kwargs.pop('fixed_sigma_day')
    tuning_df = calc.tca.trial_factor_tuning(
            mouse=flow.Mouse(mouse=mouse),
            verbose=verbose, **tuning_kwargs)

    # drop unused columns
    filters_df = psy1.drop(columns=['orientation'])

    # z-score meta1 by day if you want a within day normalization
    meta1_z_byday = (
        meta1
        .groupby('date')
        .transform(lambda x: (x - x.mean()) / x.std()))

    # add in biometrics during stimulus presentation
    filters_df = filters_df.join(meta1['speed'])
    filters_df = filters_df.join(meta1['anticipatory_licks'])
    filters_df = filters_df.join(meta1_z_byday['pupil'])

    # add in biometrics one second before stimulus presentation
    filters_df = filters_df.join(meta1['pre_speed'])
    filters_df = filters_df.join(meta1['pre_licks'])
    filters_df = filters_df.join(meta1_z_byday['pre_pupil'])

    # add in plus, minus, neutral
    cs_to_add = ['plus', 'minus', 'neutral']
    for csi in cs_to_add:
        cs_tuning_vec = np.zeros(len(filters_df))
        cs_tuning_vec[meta1['condition'].isin([csi])] = 1
        filters_df[csi] = cs_tuning_vec

    # add in hit, miss, CRm, FAm, CRn, FAn
    trialerror_codes = [0, 1, 2, 3, 4, 5]  # , 6, 7, 8, 9]
    trialerror_labels = ['hit',
                         'miss',
                         'neutral_CR',
                         'neutral_FA',
                         'minus_CR',
                         'minus_FA',
                         'blank_CR',
                         'blank_FA',
                         'pav_early_licking',
                         'pav_late_licking']
    for tei in trialerror_codes:
        te_vec = np.zeros(len(filters_df))
        te_vec[meta1['trialerror'].isin([tei])] = 1
        filters_df[trialerror_labels[tei]] = te_vec

    # z-score to get all filters on a similar scale
    zfilters_df = (
        filters_df
        .transform(lambda x: (x - x.mean()) / x.std()))

    # choose your filters from Pillow to fit to TCA
    cols = ['ori_270', 'ori_135', 'ori_0', 'prev_reward_interaction',
            'prev_punish_interaction', 'prev_choice_interaction',
            'bias', 'speed']
    cols.extend(['anticipatory_licks', 'pupil'])
    cols.extend(['ori_270_input', 'ori_135_input',
                 'ori_0_input'])
    cols.extend(['ori_270_interaction', 'ori_135_interaction',
                 'ori_0_interaction'])
    cols.extend(['prev_reward_input', 'prev_punish_input',
                 'prev_choice_input'])
    cols.extend(['ori_270_th_prev', 'ori_135_th_prev',
                 'ori_0_th_prev'])
    cols.extend(['plus', 'minus', 'neutral'])
    cols.extend(['hit', 'miss', 'neutral_CR', 'neutral_FA',
                 'minus_CR', 'minus_FA'])
    filters_subset = zfilters_df.loc[:, cols]

    # fit GLM and a GLM dropping out each filter to test deviance explained
    models, model_fits, dev_exp_full_list = [], [], []
    delta_aic_full_list, sub_aic_full_list = [], []
    total_aic_full_list = []
    total_dev_full_list = []
    fac_list = []
    for fac_num in range(1, kwargs_defaults['rank_num']+1):

        # get your tuning (ori), cs, or trialerror vectors for each factor
        fac_tuning = tuning_df.loc[(mouse, fac_num), 'preferred_tuning']
        fac_cs = tuning_df.loc[(mouse, fac_num), 'preferred_tuning_cs']

        if fac_cs == 'plus':
            te_pair = ['hit', 'miss']
        elif fac_cs == 'neutral':
            te_pair = ['neutral_CR', 'neutral_FA']
        elif fac_cs == 'minus':
            te_pair = ['minus_CR', 'minus_FA']

        if fac_tuning == '0':
            formula = 'y ~ ori_0_input +' \
             + ' {} + {} +'.format(te_pair[0], te_pair[1]) \
             + ' prev_reward_input + prev_punish_input + prev_choice_input +' \
             + ' speed + pupil + anticipatory_licks'
            drop_list = [
                ' ori_0_input +',
                ' {} +'.format(te_pair[0]),
                ' {} +'.format(te_pair[1]),
                ' prev_reward_input +',
                ' prev_punish_input +',
                ' prev_choice_input +',
                ' speed +',
                ' pupil +',
                ' + anticipatory_licks']
        elif fac_tuning == '135':
            formula = 'y ~ ori_135_input +' \
             + ' {} + {} +'.format(te_pair[0], te_pair[1]) \
             + ' prev_reward_input + prev_punish_input + prev_choice_input +' \
             + ' speed + pupil + anticipatory_licks'
            drop_list = [
                ' ori_135_input +',
                ' {} +'.format(te_pair[0]),
                ' {} +'.format(te_pair[1]),
                ' prev_reward_input +',
                ' prev_punish_input +',
                ' prev_choice_input +',
                ' speed +',
                ' pupil +',
                ' + anticipatory_licks']
        elif fac_tuning == '270':
            formula = 'y ~ ori_270_input +' \
             + ' {} + {} +'.format(te_pair[0], te_pair[1]) \
             + ' prev_reward_input + prev_punish_input + prev_choice_input +' \
             + ' speed + pupil + anticipatory_licks'
            drop_list = [
                ' ori_270_input +',
                ' {} +'.format(te_pair[0]),
                ' {} +'.format(te_pair[1]),
                ' prev_reward_input +',
                ' prev_punish_input +',
                ' prev_choice_input +',
                ' speed +',
                ' pupil +',
                ' + anticipatory_licks']
        elif fac_tuning == 'broad':
            formula = 'y ~ ori_270_input + ori_135_input + ori_0_input +' \
             + ' hit + miss +' \
             + ' neutral_CR + neutral_FA +' \
             + ' minus_CR + minus_FA +' \
             + ' prev_reward_input +' \
             + ' prev_punish_input + prev_choice_input + speed + pupil +' \
             + ' anticipatory_licks'
            drop_list = [
                ' ori_270_input +',
                ' ori_135_input +',
                ' ori_0_input +',
                ' {} +'.format('hit'),
                ' {} +'.format('neutral_CR'),
                ' {} +'.format('minus_CR'),
                ' {} +'.format('miss'),
                ' {} +'.format('neutral_FA'),
                ' {} +'.format('minus_FA'),
                ' prev_reward_input +',
                ' prev_punish_input +',
                ' prev_choice_input +',
                ' speed +',
                ' pupil +',
                ' + anticipatory_licks']

        # add your factor for fitting as the y variable
        fac = 'factor_' + str(fac_num)
        sub_xy = filters_subset.join(fac_df)
        # scale and round to make it Poisson-friendly
        sub_xy['y'] = deepcopy((sub_xy[fac]*100).apply(np.floor))
        sub_xy = sub_xy.reset_index()

        # if a filter is totally empty remove it from the formula and the drop
        for col in sub_xy.columns:
            total_nan = np.sum(sub_xy[col].isna().values)
            total_vals = len(sub_xy[col].values)
            if total_nan == total_vals:
                sub_xy = sub_xy.drop(columns=[col])
                drop_list = [s for s in drop_list if col not in s]
                if verbose:
                    print('{}: dropped column/filter: {}'.format(mouse, col))

        # make sure you don't have any nans
        sub_xy = sub_xy.replace([np.inf, -np.inf], np.nan).dropna()

        try:
            model = regression.glm(
                formula, sub_xy, dropzeros=False,
                link='log', family='Poisson', verbose=False)
        except ValueError:
            print('!!!!!!')
            print('{}: Skipped {}'.format(mouse, fac))
            print('!!!!!!')
            continue

        models.append(model)
        res = model.fit()
        model_fits.append(res)
        total_dev_exp = 1 - res.deviance/res.null_deviance
        total_aic = res.aic
        if verbose:
            print('{}: Component {}'.format(mouse, fac_num))
            print('    Total deviance explained: ',
                  1 - res.deviance/res.null_deviance)

        # get deviance explained per filter
        # add NaN for the internal intercept
        dev_explained_drop = []
        delta_aic = []
        aic_drop = []
        for c, dl in enumerate(drop_list):
            drop_formula = formula.replace(dl, '')
            try:
                model = regression.glm(
                    drop_formula, sub_xy, dropzeros=False,
                    link='log', family='Poisson', verbose=False)
            except:
                dev_explained_drop.append(np.nan)
                delta_aic.append(np.nan)
                aic_drop.append(np.nan)
                continue
            # make up for the intercept beta
            if c == 0:
                dev_explained_drop.append(np.nan)
                delta_aic.append(np.nan)
                aic_drop.append(np.nan)
            res = model.fit()
            aic_drop.append(res.aic)
            delta_aic.append(res.aic - total_aic)
            drop_dev_exp = 1 - res.deviance/res.null_deviance
            dev_explained_drop.append(total_dev_exp - drop_dev_exp)
            delta_aic
        dev_exp_full_list.extend(dev_explained_drop)
        sub_aic_full_list.extend(aic_drop)
        delta_aic_full_list.extend(delta_aic)
        total_aic_full_list.extend([total_aic]*len(delta_aic))
        total_dev_full_list.extend([total_dev_exp]*len(delta_aic))
        fac_list.append(fac_num)

    # aggregate all of your fit results
    df_list = []
    for c, mod in zip(fac_list, model_fits):
        mod_df = mod.summary2().tables[1]
        mod_df['component'] = [c]*len(mod_df)
        mod_df['x'] = mod_df.index
        df_list.append(mod_df)
    all_model_df = pd.concat(df_list, axis=0)
    all_model_df['sub_deviance_explained'] = dev_exp_full_list
    all_model_df['frac_deviance_explained'] = (
        np.array(dev_exp_full_list)/np.array(total_dev_full_list))
    all_model_df['sub_model_aic'] = sub_aic_full_list
    all_model_df['delta_aic'] = delta_aic_full_list
    all_model_df['full_model_aic'] = total_aic_full_list
    all_model_df['full_deviance_explained'] = total_dev_full_list

    all_model_df['mouse'] = [mouse]*len(all_model_df)
    all_model_df = (
        all_model_df
        .reset_index()
        .drop(columns=['index'])
        .set_index(['mouse', 'component', 'x']))

    return all_model_df


def fit_trial_factors(
        mouse='OA27',
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        word='orlando',
        group_by='all',
        nan_thresh=0.85,
        score_threshold=None,
        rank_num=18,
        historical_memory=5,
        rectified=False,
        verbose=False):

    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}
    group_pars = {'group_by': group_by}
    load_kwargs = {'mouse': mouse,
                   'method': method,
                   'cs': cs,
                   'warp': warp,
                   'word': word,
                   'group_by': group_by,
                   'nan_thresh': nan_thresh,
                   'score_threshold': score_threshold,
                   'rank': rank_num}

    # load your data
    ensemble, cell_ids, cell_clusters = load.groupday_tca_model(
            **load_kwargs, full_output=True)
    meta = load.groupday_tca_meta(**load_kwargs)
    V = ensemble[method]
    orientation = meta.reset_index()['orientation']
    condition = meta.reset_index()['condition']
    speed = meta.reset_index()['speed']
    dates = meta.reset_index()['date']
    total_time = pd.DataFrame(
        data={'total_time': np.arange(len(time_in_trial))},
        index=time_in_trial.index)
    learning_state = meta['learning_state']
    trialerror = meta['trialerror']

    # create dataframe of dprime values
    dprime_vec = []
    for date in dates:
        date_obj = flow.Date(mouse, date=date, exclude_tags=['bad'])
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

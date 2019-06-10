""" Functions for plotting variance calculations for TCA. """
from .. import paths, calc
from copy import deepcopy
import os
import numpy as np
import pandas as pd
import flow
import pool
import seaborn as sns
import matplotlib.pyplot as plt


def varex_norm_bycomp_byday(
        mouse,
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        word='orlando',
        group_by='all',
        nan_thresh=0.85,
        rank_num=18,
        rectified=True,
        verbose=False):

    # necessary parameters for determining type of analysis
    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}
    group_pars = {'group_by': group_by}

    # save dir
    # if cells were removed with too many nan trials
    if nan_thresh:
        nt_tag = '_nantrial' + str(nan_thresh)
        nt_save_tag = ' nantrial ' + str(nan_thresh)
    else:
        nt_tag = ''
        nt_save_tag = ''
    # save tag for rectification
    if rectified:
        r_tag = ' rectified'
        r_save_tag = '_rectified'
    else:
        r_tag = ''
        r_save_tag = ''
    save_dir = paths.tca_plots(
        mouse, 'group', pars=pars, word=word, group_pars=group_pars)
    save_dir = os.path.join(save_dir, 'varex' + nt_save_tag + r_tag)
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, 'byday_bycomp')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    var_path = os.path.join(
        save_dir, str(mouse) + '_rank' + str(rank_num) +
        '_norm_varex_by_day' + r_save_tag + nt_save_tag + '.pdf')

    # load dir
    load_dir = paths.tca_path(
        mouse, 'group', pars=pars, word=word, group_pars=group_pars)
    tensor_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_group_decomp_' + str(trace_type) + '.npy')
    input_tensor_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_group_tensor_' + str(trace_type) + '.npy')
    ids_tensor_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_group_ids_' + str(trace_type) + '.npy')
    meta_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_df_group_meta.pkl')

    # load your data
    # ensemble = np.load(tensor_path)
    # ensemble = ensemble.item()
    # # re-balance your factors ()
    # print('Re-balancing factors.')
    # for r in ensemble[method].results:
    #     for i in range(len(ensemble[method].results[r])):
    #         ensemble[method].results[r][i].factors.rebalance()
    # V = ensemble[method]
    # X = np.load(input_tensor_path)
    # meta = pd.read_pickle(meta_path)
    # meta = utils.update_naive_cs(meta)
    # orientation = meta.reset_index()['orientation']
    # condition = meta.reset_index()['condition']
    # speed = meta.reset_index()['speed']
    # dates = meta.reset_index()['date']
    # # time_in_trial = meta.reset_index()['trial_idx']
    # # total_time = pd.DataFrame(data={'total_time': np.arange(len(time_in_trial))}, index=time_in_trial.index)
    # learning_state = meta['learning_state']
    # trialerror = meta['trialerror']
    # ids = np.load(ids_tensor_path)

    # create dataframe of dprime values
    # dprime_vec = []
    # for date in dates:
    #     date_obj = flow.Date(mouse, date=date)
    #     dprime_vec.append(pool.calc.performance.dprime(date_obj))
    # data = {'dprime': dprime_vec}
    # dprime = pd.DataFrame(data=data, index=speed.index)
    # dprime = dprime['dprime']  # make indices match to meta

    test = calc.var.groupday_varex_byday_bycomp(flow.Mouse(mouse='OA27'), word='orlando')
    test3 = calc.var.groupday_varex_byday(flow.Mouse(mouse='OA27'), word='orlando')
    # test = cas.calc.var.groupday_varex_byday_bycomp(flow.Mouse(mouse='VF226'), word='already')
    # test3 = cas.calc.var.groupday_varex_byday(flow.Mouse(mouse='VF226'), word='already')

    R = rank_num
    comp_var_df = test.loc[test['rank'] == R, :]
    col_var = deepcopy(comp_var_df['variance_explained_tcamodel'].values)
    new_col_var = deepcopy(comp_var_df['variance_explained_tcamodel'].values)
    new_col_dates = deepcopy(comp_var_df['date'].values)

    daily_var_df = test3.loc[test3['rank'] == R, :]
    daily_var_lookup = deepcopy(daily_var_df['variance_explained_tcamodel'].values)

    for c, day in enumerate(np.unique(new_col_dates)):
        new_col_dates[new_col_dates == day] = c
        new_col_var[comp_var_df['date'].values == day] = (col_var[comp_var_df['date'].values == day]/daily_var_lookup[daily_var_df['date'].values == day])
    comp_var_df['norm_varex'] = new_col_var
    comp_var_df['day_num'] = new_col_dates

    g = sns.relplot(
        x="day_num", y="norm_varex", hue="component",
        data=comp_var_df.loc[(comp_var_df['day_num'] >= 0)
                             & (comp_var_df['day_num'] <= 100), :],
        legend='full', kind='line',
        palette=sns.color_palette('muted', R), marker='o')
    plt.savefig(var_path, bbox_inches='tight')

""" Functions for plotting behavioral variables directly from metadata or similar outputs"""
import seaborn as sns
import matplotlib.pyplot as plt
from .. import utils, load, paths, lookups
import os
import numpy as np
import pandas as pd


def hmm_engaged_from_meta(meta, save_folder=''):
    """
    Function for plotting hmm engagement, contained in your metadata DataFrame, across learning.

    :param meta: pandas.DataFrame, metadata DataFrame where each index is a unique trial
    :param save_folder: str, directory to save plots into
    :return: saves plot of HMM performance over time
    """

    # get mouse from metadata index, must have only one mouse
    assert len(meta.reset_index()['mouse'].unique()) == 1
    mouse = meta.reset_index()['mouse'].unique()

    plt.figure(figsize=(20, 4))
    plt.xlabel('trial number', size=14)
    plt.ylabel('engagment', size=14)
    plt.yticks([0, 1], labels=['disengaged', 'engaged'], size=12)
    plt.title(f'{mouse}: HMM Engagement', size=16)

    # calculate change indices for days and reversal/learning
    dates = meta.reset_index()['date']
    ndays = np.diff(dates.values)
    day_x = np.where(ndays)[0] + 0.5

    # get your learning and reversal start indices
    rev_ind = np.where(meta['learning_state'].isin(['learning']).values)[0][-1]
    lear_ind = np.where(meta['learning_state'].isin(['learning']).values)[0][0]

    # plot date a d rev
    y_min = 0
    y_max = 1
    first_day = True
    if len(day_x) > 0:
        for k in day_x:
            if first_day:
                plt.plot([k, k], [y_min, y_max], color=lookups.color_dict['gray'], linewidth=2, label='day transitions')
                first_day = False
            else:
                plt.plot([k, k], [y_min, y_max], color=lookups.color_dict['gray'], linewidth=2)
    plt.plot([lear_ind, lear_ind], [y_min, y_max], '--', color=lookups.color_dict['learning'], linewidth=3,
             label='learning starts')
    plt.plot([rev_ind, rev_ind], [y_min, y_max], '--', color=lookups.color_dict['reversal'], linewidth=3,
             label='reversal starts')

    # plot hmm on top
    plt.plot(meta['hmm_engaged'].rolling(30).mean().values, label='30-trial smoothing', color='orange')
    plt.plot(meta['hmm_engaged'].values, 'o', markeredgecolor='w', markeredgewidth=0.1, markerfacecolor='#734f96',
             label='individual trials')

    # add legend
    plt.legend(bbox_to_anchor=(1.01, 1.03), loc='upper left')

    # save
    dp_save_folder = save_folder + ' hmm'
    if not os.path.isdir(dp_save_folder):
        os.mkdir(dp_save_folder)
    plt.savefig(os.path.join(dp_save_folder, f'{mouse}_hmm_engagement_all_trials.pdf'), bbox_inches='tight')


def hmm_summary_groupmouse(
        mice,
        words=None,
        method='ncp_hals',
        cs='',
        warp=False,
        trace_type='zscore_day',
        group_by='all3',
        nan_thresh=0.95,
        score_threshold=0.8):
    """
    Function for plotting hmm engagement across stages of learning.
    Creates plots for multiple mice.

    :param mice: list of str, names of mice for analysis
    :param words: list of str, associated parameter hash words
    :param method: str, fit method from tensortools package
        'cp_als', fits CP Decomposition using Alternating
            Least Squares (ALS).
        'ncp_bcd', fits nonnegative CP Decomposition using
            the Block Coordinate Descent (BCD) Method.
        'ncp_hals', fits nonnegative CP Decomposition using
            the Hierarchical Alternating Least Squares
            (HALS) Method.
        'mncp_hals', fits nonnegative CP Decomposition using
            the Hierarchical Alternating Least Squares
            (HALS) Method with missing data.
        'mcp_als', fits CP Decomposition with missing data using
            Alternating Least Squares (ALS).
    :param cs: str, conditioned stimuli, '' defaults to all CSes
    :param warp: boolean, warp trace offset so ensure delivery is a single point in time
    :param trace_type: str, type of calcium imaging trace being used in associated analysis
    :param group_by: str, period of time being analyzed across animal training
    :param nan_thresh: float, fraction of trials that must contain non-NaN entries
    :param score_threshold: float, score threshold for CellReg package alignment score
    """

    # load your metadata
    for mouse, word in zip(mice, words):
        load_kwargs = {'mouse': mouse,
                       'method': method,
                       'cs': cs,
                       'warp': warp,
                       'word': word,
                       'trace_type': trace_type,
                       'group_by': group_by,
                       'nan_thresh': nan_thresh,
                       'score_threshold': score_threshold}
        meta = load.groupday_tca_meta(**load_kwargs)

        # create a new analysis directory for your mouse named 'behavior'
        save_path = paths.groupmouse_analysis_path('behavior', mice=mice, words=words, **load_kwargs)

        # create plots of your hmm engagement
        hmm_engaged_from_meta(meta,  save_folder=save_path)


def barplots_group_summary_from_meta(meta, staging='parsed_10stage', save_folder='', sharey='row', pupil_scale='norm'):
    """
    Function for plotting behavioral data contained in your metadata DataFrame across mice.

    :param staging: str, the way you want to bin stages of learning
    :param save_folder: str, name of sub-folder to put plots in
    :param sharey: boolean, share the y axis across rows of subplots
    :param pupil_scale: str, 'norm' or 'zscore', normalize or standardize the pupil data
    :return: saves plot of behavioral variables
    """

    # get mice from metadata index, must have more than one mouse
    assert len(meta.reset_index()['mouse'].unique()) > 1
    mice = meta.reset_index()['mouse'].unique()

    # make sure your learning stage column was added
    assert staging in meta.columns

    # add acceleration and change in lick rate to meta
    if 'delta_speed' not in meta.columns:
        meta['delta_speed'] = meta['speed'] - meta['pre_speed']
    if 'pre_stim_speed' not in meta.columns:
        meta['pre_stim_speed'] = meta['pre_speed']
    if 'stim_speed' not in meta.columns:
        meta['stim_speed'] = meta['speed']
    # preallocate and deal with different stim lengths in next step
    if 'pre_lickrate' not in meta.columns:
        meta['pre_stim_lickrate'] = meta['pre_licks']  # /1 because pre is already for one second
    if 'delta_lickrate' not in meta.columns:
        meta['delta_lickrate'] = np.nan
    if 'stim_lickrate' not in meta.columns:
        meta['stim_lickrate'] = np.nan
    # add pupil
    if 'delta_pupil' not in meta.columns:
        meta['delta_pupil'] = np.nan
    if pupil_scale == 'zscore':
        if 'pre_pupil_zscore' not in meta.columns:
            meta['pre_pupil_zscore'] = np.nan
        if 'pupil_zscore' not in meta.columns:
            meta['pupil_zscore'] = np.nan
    elif pupil_scale == 'norm':
        if 'pre_pupil_norm' not in meta.columns:
            meta['pre_pupil_norm'] = np.nan
        if 'pupil_norm' not in meta.columns:
            meta['pupil_norm'] = np.nan

    # loop over and add lickrate to meta, accounting for differences in stimulus duration
    for mouse in mice:
        mbool = meta.reset_index()['mouse'].isin([mouse]).values
        meta['delta_lickrate'].iloc[mbool] = (
                meta['anticipatory_licks'].iloc[mbool] / lookups.stim_length[mouse] - meta['pre_licks'].iloc[mbool])
        meta['stim_lickrate'].iloc[mbool] = (
                meta['anticipatory_licks'].iloc[mbool] / lookups.stim_length[mouse])
        if pupil_scale == 'zscore':
            meta['pupil_zscore'].iloc[mbool] = (
                (meta['pupil'].iloc[mbool] - meta['pupil'].iloc[mbool].mean()) / meta['pupil'].iloc[mbool].std())
            meta['pre_pupil_zscore'].iloc[mbool] = (
                (meta['pre_pupil'].iloc[mbool] - meta['pre_pupil'].iloc[mbool].mean()) / meta['pre_pupil'].iloc[mbool].std())
        elif pupil_scale == 'norm':
            meta['pupil_norm'].iloc[mbool] = (
                (meta['pupil'].iloc[mbool] - meta['pupil'].iloc[mbool].min()) / meta['pupil'].iloc[mbool].max())
            meta['pre_pupil_norm'].iloc[mbool] = (
                (meta['pre_pupil'].iloc[mbool] - meta['pre_pupil'].iloc[mbool].min()) / meta['pre_pupil'].iloc[mbool].max())

    # in delta pupil post-zscore
    if pupil_scale == 'zscore':
        meta['delta_pupil'] = meta['pupil_zscore'] - meta['pre_pupil_zscore']
    elif pupil_scale == 'norm':
        meta['delta_pupil'] = meta['pupil_norm'] - meta['pre_pupil_norm']

    # get your DataFrame to take average for mice, set naive dp to NaN
    all_df = meta.groupby(['mouse', 'condition', staging]).mean().reset_index([staging, 'condition'])

    # get your dprime DataFrame to take average across runs, set naive dp to NaN
    dprime_df = meta.groupby(['mouse', staging]).mean().reset_index(staging)
    dprime_df.loc[dprime_df[staging].isin(['naive', 'early naive', 'late naive', 'L0 naive']), 'dprime_run'] = np.nan

    # add additional condition column for additional colors in swarm plots
    all_df['condition_swarm'] = [s + '1' for s in all_df['condition']]

    # columns to plot
    col_to_plot = ['dprime_run', 'dprime_run', 'dprime_run',
                   'pre_stim_speed', 'stim_speed', 'delta_speed',
                   'pre_stim_lickrate', 'stim_lickrate', 'delta_lickrate']
    if pupil_scale == 'zscore':
        col_to_plot = col_to_plot + ['pre_pupil_zscore', 'pupil_zscore', 'delta_pupil']
    elif pupil_scale == 'norm':
        col_to_plot = col_to_plot + ['pre_pupil_norm', 'pupil_norm', 'delta_pupil']

    # plot all behavior variables as a single long plot with subplot
    # fig, axes = plt.subplots(3, int(len(col_to_plot) / 3), figsize=(7 * len(col_to_plot) / 3, 6 * 3),
    #                          sharey=sharey, sharex='all')
    fig, axes = plt.subplots(int(len(col_to_plot) / 3), 3, figsize=(7 * 3, 6 * len(col_to_plot) / 3),
                             sharey=sharey, sharex='all')

    # col order gets mixed up sometimes with sns defaults, so set order for plotting
    stage_order = utils.lookups.staging[staging]

    # loop over and plot each sublot column of behavior data
    for c, var in enumerate(col_to_plot):

        # set ax row (cr) and col (cc)
        cr = int(np.floor(c / 3 % 3) + 3*np.floor(c / 9 % 9))
        cc = int(c % 3)

        axes[cr, cc].axhline(0, ls='--', color='#737373', linewidth=1)
        if var == 'dprime_run':
            sns.barplot(data=dprime_df,
                        y=var,
                        x=staging,
                        order=stage_order,
                        color=lookups.color_dict['dprime'], ax=axes[cr, cc])
            sns.swarmplot(data=dprime_df,
                          y=var,
                          x=staging,
                          order=stage_order,
                          color=lookups.color_dict['dprime1'], ax=axes[cr, cc])
        elif 'delta' in var:
            for mouse in mice:
                mbool = all_df.reset_index()['mouse'].isin([mouse]).values
                # skip mice that have no values at for the var of interest
                if all_df.iloc[mbool, :][var].isna().all():
                    continue
                sns.pointplot(data=all_df.iloc[mbool, :],
                              y=var,
                              x=staging,
                              order=stage_order,
                              hue='condition_swarm',
                              palette=lookups.color_dict, ax=axes[cr, cc],
                              dashes=False, legend=False, ci=False, scale=0.5, alpha=0.3, markers='')
            g = sns.pointplot(data=all_df,
                              y=var,
                              x=staging,
                              order=stage_order,
                              hue='condition',
                              palette=lookups.color_dict, ax=axes[cr, cc], dodge=True)
            handles, labels = g.get_legend_handles_labels()
            axes[cr, cc].legend(handles[-3:], labels[-3:], loc='upper right')  # lake last 3 things plotted
        else:
            g = sns.barplot(data=all_df,
                            y=var,
                            x=staging,
                            order=stage_order,
                            hue='condition',
                            palette=lookups.color_dict, ax=axes[cr, cc])
            handles, labels = g.get_legend_handles_labels()
            sns.stripplot(data=all_df,
                          y=var,
                          x=staging,
                          order=stage_order,
                          hue='condition_swarm',
                          palette=lookups.color_dict, ax=axes[cr, cc],
                          dodge=True)
            axes[cr, cc].legend(handles, labels, loc='upper right')

        axes[cr, cc].set_xticklabels(axes[cr, cc].get_xticklabels(), rotation=45, ha='right')
        axes[cr, cc].set_title(f'{var}', size=18)
        sns.despine()

        # remove x labels between subplots
        if cr < int(np.floor(len(col_to_plot) / 3 % 3) + 3*np.floor(len(col_to_plot) / 9 % 9)):
            axes[cr, cc].set_xlabel('')

        # change y label based on name
        curr_y_lbl = axes[cr, cc].get_ylabel()
        delt = ['\u0394 ' if 'delta' in cyl else '' for cyl in [curr_y_lbl]][0]
        if 'speed' in curr_y_lbl:
            axes[cr, cc].set_ylabel(f'{delt}speed (cm / s)', size=16)
        elif 'lick' in curr_y_lbl:
            axes[cr, cc].set_ylabel(f'{delt}lick rate (licks / s)', size=16)
        elif 'dprime' in curr_y_lbl:
            axes[cr, cc].set_ylabel("d\u2032", size=16)
        elif 'pupil' in curr_y_lbl:
            if pupil_scale == 'zscore':
                axes[cr, cc].set_ylabel(f'{delt}pupil diameter (z-score)', size=16)
            elif pupil_scale == 'norm':
                axes[cr, cc].set_ylabel(f'{delt}normalized pupil diameter', size=16)


    mice_word = paths.groupmouse_word(mice)
    plt.suptitle(f'{mice_word}: behavior summary', position=(0.5, 1.03), ha='center', size=22)
    sum_save_folder = save_folder + ' sum'
    if not os.path.isdir(sum_save_folder):
        os.mkdir(sum_save_folder)
    tag = ' sharey' if sharey == 'row' else ''
    tag = tag + ' pup-' + pupil_scale
    plt.savefig(os.path.join(sum_save_folder, f'{mice_word} behavior {staging}{tag}.png'), bbox_inches='tight')
    plt.savefig(os.path.join(sum_save_folder, f'{mice_word} behavior {staging}{tag}.pdf'), bbox_inches='tight')


def barplots_summary_from_meta_wdp(meta, staging='parsed_10stage', save_folder=''):
    """
    Function for plotting behavioral data contained in your metadata DataFrame, with dprime.

    :param staging: str, the way you want to set stages of learning
    :return: saves plot of behavioral variables
    """

    # get mouse from metadata index
    assert len(meta.reset_index()['mouse'].unique()) == 1  # only one mouse
    mouse = meta.reset_index()['mouse'].unique()[0]

    # add your learning stage column if it doesn't exist
    meta = utils.add_stages_to_meta(meta, staging)

    # add your dprime if it doesn't exist
    if 'dprime_run' not in meta.columns:
        meta = utils.add_dprime_run_to_meta(meta)

    # get your dprime DataFrame to take average across runs, set naive dp to NaN
    dprime_df = meta.groupby(['mouse', 'date', 'run', staging]).mean().reset_index(staging)
    dprime_df.loc[dprime_df[staging].isin(['naive', 'early naive', 'late naive', 'L0 naive']), 'dprime_run'] = np.nan

    # add acceleration and change in lick rate to model
    if 'delta_speed' not in meta.columns:
        meta['delta_speed'] = meta['speed'] - meta['pre_speed']
    if 'delta_lickrate' not in meta.columns:
        meta['delta_lickrate'] = meta['anticipatory_licks'] / lookups.stim_length[mouse] - meta['pre_licks']
    if 'pre_lickrate' not in meta.columns:
        meta['pre_lickrate'] = meta['pre_licks'] / 1
    if 'stim_lickrate' not in meta.columns:
        meta['stim_lickrate'] = meta['anticipatory_licks'] / lookups.stim_length[mouse]

    # columns to plot
    col_to_plot = ['dprime_run', 'dprime_run', 'dprime_run',
                   'pre_speed', 'speed', 'delta_speed',
                   'pre_lickrate', 'stim_lickrate', 'delta_lickrate']

    # plot all behavior variables as a single long plot with subplot
    fig, axes = plt.subplots(3, int(len(col_to_plot) / 3), figsize=(7 * len(col_to_plot) / 3, 6 * 3),
                             sharey='row', sharex='all')

    # col order gets mixed up sometimes with sns defaults, so set order for plotting
    stage_order = utils.lookups.staging[staging]

    # loop over and plot each sublot column of behavior data
    for c, var in enumerate(col_to_plot):

        # set ax row (cr) and col (cc)
        cr = int(np.floor(c / 3 % 3))
        cc = int(c % 3)

        axes[cr, cc].axhline(0, ls='--', color='#737373', linewidth=1)
        if var == 'dprime_run':
            sns.barplot(data=dprime_df,
                        y=var,
                        x=staging,
                        order=stage_order,
                        color=lookups.color_dict['dprime'], ax=axes[cr, cc])
        else:
            sns.barplot(data=meta,
                        y=var,
                        x=staging,
                        order=stage_order,
                        hue='condition',
                        palette=lookups.color_dict, ax=axes[cr, cc])
            axes[cr, cc].legend(loc='upper right')

        axes[cr, cc].set_xticklabels(axes[cr, cc].get_xticklabels(), rotation=45, ha='right')
        axes[cr, cc].set_title(f'{var}', size=18)
        sns.despine()

        # remove x labels between subplots
        if cr < 2:
            axes[cr, cc].set_xlabel('')

        # change y label based on name
        curr_y_lbl = axes[cr, cc].get_ylabel()
        delt = ['\u0394 ' if 'delta' in cyl else '' for cyl in [curr_y_lbl]][0]
        if 'speed' in curr_y_lbl:
            axes[cr, cc].set_ylabel(f'{delt}speed (cm / s)', size=16)
        elif 'lick' in curr_y_lbl:
            axes[cr, cc].set_ylabel(f'{delt}lick rate (licks / s)', size=16)
        elif 'dprime' in curr_y_lbl:
            axes[cr, cc].set_ylabel("d\u2032", size=16)

    plt.suptitle(f'{mouse}: behavior summary', position=(0.5, 0.95), ha='center', size=22)
    dp_save_folder = save_folder + ' dp'
    if not os.path.isdir(dp_save_folder):
        os.mkdir(dp_save_folder)
    plt.savefig(os.path.join(dp_save_folder, f'{mouse} behavior {staging} dp.png'), bbox_inches='tight')


def barplots_summary_from_meta(meta, staging='parsed_10stage', save_folder=''):
    """
    Function for plotting behavioral data contained in your metadata DataFrame

    :param staging: str, the way you want to set stages of learning
    :return: saves plot of behavioral variables
    """

    # get mouse from metadata index
    assert len(meta.reset_index()['mouse'].unique()) == 1  # only one mouse
    mouse = meta.reset_index()['mouse'].unique()[0]

    # add your learning stage column if it doesn't exist
    meta = utils.add_stages_to_meta(meta, staging)

    # add acceleration and change in lick rate to model
    if 'delta_speed' not in meta.columns:
        meta['delta_speed'] = meta['speed'] - meta['pre_speed']
    if 'delta_lickrate' not in meta.columns:
        meta['delta_lickrate'] = meta['anticipatory_licks'] / lookups.stim_length[mouse] - meta['pre_licks']
    if 'pre_lickrate' not in meta.columns:
        meta['pre_lickrate'] = meta['pre_licks'] / 1
    if 'stim_lickrate' not in meta.columns:
        meta['stim_lickrate'] = meta['anticipatory_licks'] / lookups.stim_length[mouse]

    # columns to plot
    col_to_plot = ['pre_speed', 'speed', 'delta_speed', 'pre_lickrate', 'stim_lickrate', 'delta_lickrate']

    # plot all behavior variables as a single long plot with subplot
    fig, axes = plt.subplots(2, int(len(col_to_plot) / 2), figsize=(7 * len(col_to_plot) / 2, 6 * 2),
                             sharey='row', sharex='all')

    # col order gets mixed up sometimes with sns defaults, so set order for plotting
    stage_order = utils.lookups.staging[staging]

    # loop over and plot each sublot column of behavior data
    for c, var in enumerate(col_to_plot):

        # set ax row (cr) and col (cc)
        cr = int(np.floor(c / 3 % 3))
        cc = int(c % 3)

        axes[cr, cc].axhline(0, ls='--', color='#737373', linewidth=1)
        sns.barplot(data=meta,
                    y=var,
                    x=staging,
                    order=stage_order,
                    hue='condition',
                    palette=lookups.color_dict, ax=axes[cr, cc])
        sns.despine()
        axes[cr, cc].set_xticklabels(axes[cr, cc].get_xticklabels(), rotation=45, ha='right')
        axes[cr, cc].set_title(f'{var}', size=18)
        axes[cr, cc].legend(loc='upper right')

        # remove x labels between subplots
        if cr == 0:
            axes[cr, cc].set_xlabel('')

        # change y label based on name
        curr_y_lbl = axes[cr, cc].get_ylabel()
        delt = ['\u0394 ' if 'delta' in cyl else '' for cyl in [curr_y_lbl]][0]
        if 'speed' in curr_y_lbl:
            axes[cr, cc].set_ylabel(f'{delt}speed (cm / s)', size=16)
        elif 'lick' in curr_y_lbl:
            axes[cr, cc].set_ylabel(f'{delt}lick rate (licks / s)', size=16)

    plt.suptitle(f'{mouse}: behavior summary', position=(0.5, 1.03), ha='center', size=22)
    plt.savefig(os.path.join(save_folder, f'{mouse} behavior {staging}.png'), bbox_inches='tight')


def barplots_summary_mouse(
        mouse,
        word=None,
        method='ncp_hals',
        cs='',
        warp=False,
        trace_type='zscore_day',
        group_by='all3',
        nan_thresh=0.95,
        score_threshold=0.8,
        staging='parsed_10stage',
        with_dp=False):
    """
    Function for plotting barplots of behavioral variables across stages of learning.
    Creates plots for one mouse.

    :param mouse: list of str, names of mice for analysis
    :param word: list of str, associated parameter hash words
    :param method: str, fit method from tensortools package
        'cp_als', fits CP Decomposition using Alternating
            Least Squares (ALS).
        'ncp_bcd', fits nonnegative CP Decomposition using
            the Block Coordinate Descent (BCD) Method.
        'ncp_hals', fits nonnegative CP Decomposition using
            the Hierarchical Alternating Least Squares
            (HALS) Method.
        'mncp_hals', fits nonnegative CP Decomposition using
            the Hierarchical Alternating Least Squares
            (HALS) Method with missing data.
        'mcp_als', fits CP Decomposition with missing data using
            Alternating Least Squares (ALS).
    :param cs: str, conditioned stimuli, '' defaults to all CSes
    :param warp: boolean, warp trace offset so ensure delivery is a single point in time
    :param trace_type: str, type of calcium imaging trace being used in associated analysis
    :param group_by: str, period of time being analyzed across animal training
    :param nan_thresh: float, fraction of trials that must contain non-NaN entries
    :param score_threshold: float, score threshold for CellReg package alignment score
    :param staging: str, assign predetermined binning method for associated analysis
    :param with_dp: boolean, include a row for dprime in plots
    """

    # load your metadata
    load_kwargs = {'mouse': mouse,
                   'method': method,
                   'cs': cs,
                   'warp': warp,
                   'word': word,
                   'trace_type': trace_type,
                   'group_by': group_by,
                   'nan_thresh': nan_thresh,
                   'score_threshold': score_threshold}
    meta = load.groupday_tca_meta(**load_kwargs)

    # create a new analysis directory for your mouse named 'behavior'
    save_path = paths.mouse_analysis_path('behavior', **load_kwargs)

    # create subplots of your
    if with_dp:
        barplots_summary_from_meta_wdp(meta, staging=staging, save_folder=save_path)
    else:
        barplots_summary_from_meta(meta, staging=staging, save_folder=save_path)


def barplots_summary_groupmouse(
        mice,
        words=None,
        method='ncp_hals',
        cs='',
        warp=False,
        trace_type='zscore_day',
        group_by='all3',
        nan_thresh=0.95,
        score_threshold=0.8,
        staging='parsed_10stage',
        with_dp=False):
    """
    Function for plotting barplots of behavioral variables across stages of learning.
    Creates plots for multiple mice.

    :param mice: list of str, names of mice for analysis
    :param words: list of str, associated parameter hash words
    :param method: str, fit method from tensortools package
        'cp_als', fits CP Decomposition using Alternating
            Least Squares (ALS).
        'ncp_bcd', fits nonnegative CP Decomposition using
            the Block Coordinate Descent (BCD) Method.
        'ncp_hals', fits nonnegative CP Decomposition using
            the Hierarchical Alternating Least Squares
            (HALS) Method.
        'mncp_hals', fits nonnegative CP Decomposition using
            the Hierarchical Alternating Least Squares
            (HALS) Method with missing data.
        'mcp_als', fits CP Decomposition with missing data using
            Alternating Least Squares (ALS).
    :param cs: str, conditioned stimuli, '' defaults to all CSes
    :param warp: boolean, warp trace offset so ensure delivery is a single point in time
    :param trace_type: str, type of calcium imaging trace being used in associated analysis
    :param group_by: str, period of time being analyzed across animal training
    :param nan_thresh: float, fraction of trials that must contain non-NaN entries
    :param score_threshold: float, score threshold for CellReg package alignment score
    :param staging: str, assign predetermined binning method for associated analysis
    :param with_dp: boolean, include a row for dprime in plots
    """

    # load your metadata
    for mouse, word in zip(mice, words):
        load_kwargs = {'mouse': mouse,
                       'method': method,
                       'cs': cs,
                       'warp': warp,
                       'word': word,
                       'trace_type': trace_type,
                       'group_by': group_by,
                       'nan_thresh': nan_thresh,
                       'score_threshold': score_threshold}
        meta = load.groupday_tca_meta(**load_kwargs)

        # create a new analysis directory for your mouse named 'behavior'
        save_path = paths.groupmouse_analysis_path('behavior', mice=mice, words=words, **load_kwargs)

        # create subplots of your
        if with_dp:
            barplots_summary_from_meta_wdp(meta, staging=staging, save_folder=save_path)
        else:
            barplots_summary_from_meta(meta, staging=staging, save_folder=save_path)


def barplots_summary_aggregate(
        mice,
        words=None,
        method='ncp_hals',
        cs='',
        warp=False,
        trace_type='zscore_day',
        group_by='all3',
        nan_thresh=0.95,
        score_threshold=0.8,
        staging='parsed_10stage',
        with_dp=False):
    """
    Function for plotting barplots of behavioral variables across stages of learning.
    Creates plots for multiple mice and save a duplicate in that single mouse's directory.

    :param mice: list of str, names of mice for analysis
    :param words: list of str, associated parameter hash words
    :param method: str, fit method from tensortools package
        'cp_als', fits CP Decomposition using Alternating
            Least Squares (ALS).
        'ncp_bcd', fits nonnegative CP Decomposition using
            the Block Coordinate Descent (BCD) Method.
        'ncp_hals', fits nonnegative CP Decomposition using
            the Hierarchical Alternating Least Squares
            (HALS) Method.
        'mncp_hals', fits nonnegative CP Decomposition using
            the Hierarchical Alternating Least Squares
            (HALS) Method with missing data.
        'mcp_als', fits CP Decomposition with missing data using
            Alternating Least Squares (ALS).
    :param cs: str, conditioned stimuli, '' defaults to all CSes
    :param warp: boolean, warp trace offset so ensure delivery is a single point in time
    :param trace_type: str, type of calcium imaging trace being used in associated analysis
    :param group_by: str, period of time being analyzed across animal training
    :param nan_thresh: float, fraction of trials that must contain non-NaN entries
    :param score_threshold: float, score threshold for CellReg package alignment score
    :param staging: str, assign predetermined binning method for associated analysis
    :param with_dp: boolean, include a row for dprime in plots
    """

    # load your metadata
    for mouse, word in zip(mice, words):
        load_kwargs = {'mouse': mouse,
                       'method': method,
                       'cs': cs,
                       'warp': warp,
                       'word': word,
                       'trace_type': trace_type,
                       'group_by': group_by,
                       'nan_thresh': nan_thresh,
                       'score_threshold': score_threshold}
        meta = load.groupday_tca_meta(**load_kwargs)

        # create a new analysis directory for your groupmouse and single mouse named 'behavior'
        save_path = paths.groupmouse_analysis_path('behavior', mice=mice, words=words, **load_kwargs)
        save_path_single = paths.mouse_analysis_path('behavior', **load_kwargs)

        # create duplicate subplots for your groupmouse and single mouse
        if with_dp:
            barplots_summary_from_meta_wdp(meta, staging=staging, save_folder=save_path)
            plt.close()  # only show a single plot per mouse
            barplots_summary_from_meta_wdp(meta, staging=staging, save_folder=save_path_single)
        else:
            barplots_summary_from_meta(meta, staging=staging, save_folder=save_path)
            plt.close()  # only show a single plot per mouse
            barplots_summary_from_meta(meta, staging=staging, save_folder=save_path_single)


def summary_across_mice(
        mice,
        words=None,
        method='ncp_hals',
        cs='',
        warp=False,
        trace_type='zscore_day',
        group_by='all3',
        nan_thresh=0.95,
        score_threshold=0.8,
        staging='parsed_10stage'):
    """
    Function for plotting summary of behavioral variables across stages of learning.
    Creates single plot for multiple mice.

    :param mice: list of str, names of mice for analysis
    :param words: list of str, associated parameter hash words
    :param method: str, fit method from tensortools package
        'cp_als', fits CP Decomposition using Alternating
            Least Squares (ALS).
        'ncp_bcd', fits nonnegative CP Decomposition using
            the Block Coordinate Descent (BCD) Method.
        'ncp_hals', fits nonnegative CP Decomposition using
            the Hierarchical Alternating Least Squares
            (HALS) Method.
        'mncp_hals', fits nonnegative CP Decomposition using
            the Hierarchical Alternating Least Squares
            (HALS) Method with missing data.
        'mcp_als', fits CP Decomposition with missing data using
            Alternating Least Squares (ALS).
    :param cs: str, conditioned stimuli, '' defaults to all CSes
    :param warp: boolean, warp trace offset so ensure delivery is a single point in time
    :param trace_type: str, type of calcium imaging trace being used in associated analysis
    :param group_by: str, period of time being analyzed across animal training
    :param nan_thresh: float, fraction of trials that must contain non-NaN entries
    :param score_threshold: float, score threshold for CellReg package alignment score
    :param staging: str, assign predetermined binning method for associated analysis
    """

    # load your metadata
    meta_list = []
    for mouse, word in zip(mice, words):
        load_kwargs = {'mouse': mouse,
                       'method': method,
                       'cs': cs,
                       'warp': warp,
                       'word': word,
                       'trace_type': trace_type,
                       'group_by': group_by,
                       'nan_thresh': nan_thresh,
                       'score_threshold': score_threshold}
        meta = load.groupday_tca_meta(**load_kwargs)

        # add your learning stage column if it doesn't exist
        meta = utils.add_stages_to_meta(meta, staging)

        # add your dprime if it doesn't exist
        if 'dprime_run' not in meta.columns:
            meta = utils.add_dprime_run_to_meta(meta);

        meta_list.append(meta)

    meta_all = pd.concat(meta_list, axis=0)

    # create a new analysis directory for your groupmouse and single mouse named 'behavior'
    save_path = paths.groupmouse_analysis_path('behavior', mice=mice, words=words, **load_kwargs)

    # create subplot for all mice
    barplots_group_summary_from_meta(meta_all, staging=staging, save_folder=save_path)
    barplots_group_summary_from_meta(meta_all, staging=staging, save_folder=save_path, sharey=False)


def dprime_df_across_mice(
        mice,
        words=None,
        method='ncp_hals',
        cs='',
        warp=False,
        trace_type='zscore_day',
        group_by='all3',
        nan_thresh=0.95,
        score_threshold=0.8,
        staging='parsed_10stage',
        dp_by='day'):
    """
    Function for creating DataFrame of dprime across stages of learning.
    Creates single DataFrame for multiple mice.

    :param mice: list of str, names of mice for analysis
    :param words: list of str, associated parameter hash words
    :param method: str, fit method from tensortools package
        'cp_als', fits CP Decomposition using Alternating
            Least Squares (ALS).
        'ncp_bcd', fits nonnegative CP Decomposition using
            the Block Coordinate Descent (BCD) Method.
        'ncp_hals', fits nonnegative CP Decomposition using
            the Hierarchical Alternating Least Squares
            (HALS) Method.
        'mncp_hals', fits nonnegative CP Decomposition using
            the Hierarchical Alternating Least Squares
            (HALS) Method with missing data.
        'mcp_als', fits CP Decomposition with missing data using
            Alternating Least Squares (ALS).
    :param cs: str, conditioned stimuli, '' defaults to all CSes
    :param warp: boolean, warp trace offset so ensure delivery is a single point in time
    :param trace_type: str, type of calcium imaging trace being used in associated analysis
    :param group_by: str, period of time being analyzed across animal training
    :param nan_thresh: float, fraction of trials that must contain non-NaN entries
    :param score_threshold: float, score threshold for CellReg package alignment score
    :param staging: str, assign predetermined binning method for associated analysis
    :param dp_by: str, 'day' or 'run' TODO add pillow
    :return dprime_df: pandas.DataFrame, DataFrame with dprime values grouped by day or run
    """

    # load your metadata
    meta_list = []
    for mouse, word in zip(mice, words):
        load_kwargs = {'mouse': mouse,
                       'method': method,
                       'cs': cs,
                       'warp': warp,
                       'word': word,
                       'trace_type': trace_type,
                       'group_by': group_by,
                       'nan_thresh': nan_thresh,
                       'score_threshold': score_threshold}
        meta = load.groupday_tca_meta(**load_kwargs)

        # add your learning stage column if it doesn't exist
        meta = utils.add_stages_to_meta(meta, staging)

        # add your dprime if it doesn't exist
        if dp_by.lower() == 'run':
            if 'dprime_run' not in meta.columns:
                meta = utils.add_dprime_run_to_meta(meta)
            gb = ['mouse', 'date', 'run']
            dp_type = 'dprime_run'
        elif dp_by.lower() == 'day':
            if 'dprime' not in meta.columns:
                meta = utils.add_dprime_to_meta(meta)
            gb = ['mouse', 'date']
            dp_type = 'dprime'

        meta_list.append(meta)

    meta_all = pd.concat(meta_list, axis=0)

    # get dprime calculated across day or runs, use .min() so categorical values are not lost
    stages_to_nan = ['naive', 'early naive', 'late naive']
    dprime_df = meta_all.groupby(gb).min()
    dprime_df.loc[dprime_df[staging].isin(stages_to_nan), dp_type] = np.nan

    return dprime_df


def plot_dp(dprime_df, dp_by='run', save_folder='', scale_by=[1, .7, .5], ftype='png', staging='parsed_10stage'):
    """
    Create summary plot of all mice for dprime aligned to reversal.

    :param dprime_df: pandas DataFrame, dataframe with single dp value per day or per run for each mouse
    :param dp_by: str, 'run' or 'day' TODO add pillow
    :param save_folder: str, folder to save to
    :param scale_by: list of float, scale factor applied to
    :param ftype: str, file suffix to use for saving 'eps', 'png', 'pdf', etc
    :return: saves a number of plots to the specified folder
    """

    # add your dprime if it doesn't exist
    if dp_by.lower() == 'run':
        dp_type = 'dprime_run'
    elif dp_by.lower() == 'day':
        dp_type = 'dprime'

    # get mice in DataFrame
    mice = dprime_df.reset_index()['mouse'].unique()
    mice_word = paths.groupmouse_word(mice)

    # greate a couple different scales of plots for ease of presentation/figures
    for scale in scale_by:
        fig, ax = plt.subplots(1, 1, figsize=(int(np.ceil(15 * scale)), 4))
        cmap = sns.color_palette('tab20', len(mice))[::-1]
        # cmap = sns.cubehelix_palette(len(mice), start=1, rot=30, dark=0, light=.8, reverse=False)
        ax.axhline(0, ls='--', color='#737373', linewidth=1)
        ax.axvline(0, ls='--', color='#737373', linewidth=1)
        for c, mouse in enumerate(mice):
            mboo = dprime_df.reset_index()['mouse'].isin([mouse]).values
            mouse_df = dprime_df.iloc[mboo, :]
            rev_ind = np.where(mouse_df[staging].isin(['early low_dp learning',
                                                       'late low_dp learning',
                                                       'early high_dp learning',
                                                       'late high_dp learning',
                                                       'low_dp learning',
                                                       'high_dp learning']))[0][-1]
            y = mouse_df[dp_type].values
            x = np.arange(len(y)) - rev_ind - 1  # last learning ind, 0 is your last day of learning
            sns.lineplot(x=x, y=y, label=mouse, color=cmap[c], alpha=0.7, linewidth=2, ax=ax)

            # add dashed line for naive
            if any(np.isnan(y)):
                first_train_ind = np.where(np.isnan(y))[0][-1] + 1
                x2 = x[:first_train_ind + 1]
                scale_by = y[first_train_ind]
                if scale_by < 0.05 and scale_by > -0.05:
                    scale_by += 0.02
                y2 = np.ones(len(x2)) * scale_by
                plt.plot(x2, y2, linestyle=':', color=cmap[c])

        # despine
        sns.despine()

        # add plot labels and title
        ax.legend(title='mice\n', bbox_to_anchor=(1.01, 1.0), loc='upper left', borderaxespad=0)
        if dp_by.lower() == 'run':
            ax.set_xlabel('Imaging sessions aligned to task reversal', size=14)
        elif dp_by.lower() == 'day':
            ax.set_xlabel('Imaging days aligned to task reversal', size=14)
        ax.set_ylabel("Behavioral performance (d\u2032)", size=14)
        ax.set_title('Orientation discrimination task performance\n', size=16)

        # save your plots
        dp_save_folder = save_folder + ' dp'
        if not os.path.isdir(dp_save_folder):
            os.mkdir(dp_save_folder)
        plt.savefig(
            os.path.join(dp_save_folder, f'{mice_word} dp {dp_by} behavior {scale}sc.{ftype}'),
            bbox_inches='tight')


def plot_dprime_across_mice(
        mice,
        words=None,
        method='ncp_hals',
        cs='',
        warp=False,
        trace_type='zscore_day',
        group_by='all3',
        nan_thresh=0.95,
        score_threshold=0.8,
        staging='parsed_10stage',
        dp_by='day',
        scale_by=[1, .7, .5],
        ftype='png'):
    """
    Function for plotting summary of dprime across stages of learning.
    Creates single plot for multiple mice.

    :param mice: list of str, names of mice for analysis
    :param words: list of str, associated parameter hash words
    :param method: str, fit method from tensortools package
        'cp_als', fits CP Decomposition using Alternating
            Least Squares (ALS).
        'ncp_bcd', fits nonnegative CP Decomposition using
            the Block Coordinate Descent (BCD) Method.
        'ncp_hals', fits nonnegative CP Decomposition using
            the Hierarchical Alternating Least Squares
            (HALS) Method.
        'mncp_hals', fits nonnegative CP Decomposition using
            the Hierarchical Alternating Least Squares
            (HALS) Method with missing data.
        'mcp_als', fits CP Decomposition with missing data using
            Alternating Least Squares (ALS).
    :param cs: str, conditioned stimuli, '' defaults to all CSes
    :param warp: boolean, warp trace offset so ensure delivery is a single point in time
    :param trace_type: str, type of calcium imaging trace being used in associated analysis
    :param group_by: str, period of time being analyzed across animal training
    :param nan_thresh: float, fraction of trials that must contain non-NaN entries
    :param score_threshold: float, score threshold for CellReg package alignment score
    :param staging: str, assign predetermined binning method for associated analysis
    :param dp_by: str, 'day' or 'run' TODO add pillow
    """

    # load your metadata
    load_kwargs = {'mice': mice,
                   'method': method,
                   'cs': cs,
                   'warp': warp,
                   'words': words,
                   'trace_type': trace_type,
                   'group_by': group_by,
                   'score_threshold': score_threshold,
                   'nan_thresh': nan_thresh}

    # create a new analysis directory for your mice named 'behavior'
    save_path = paths.groupmouse_analysis_path('behavior', **load_kwargs)

    # load your dprime DataFrame
    dprime_df = dprime_df_across_mice(staging=staging, dp_by=dp_by, **load_kwargs)

    # plot your dprime summary figures and save
    plot_dp(dprime_df, dp_by=dp_by, save_folder=save_path, scale_by=scale_by, ftype=ftype, staging=staging)


def bhv_df_across_mice(
        mice,
        words=None,
        method='ncp_hals',
        cs='',
        warp=False,
        trace_type='zscore_day',
        group_by='all3',
        nan_thresh=0.95,
        score_threshold=0.8,
        staging='parsed_10stage',
        bhv_by='day'):
    """
    Function for creating DataFrame of behavioral variables across stages of learning.
    Creates single DataFrame for multiple mice.

    :param mice: list of str, names of mice for analysis
    :param words: list of str, associated parameter hash words
    :param method: str, fit method from tensortools package
        'cp_als', fits CP Decomposition using Alternating
            Least Squares (ALS).
        'ncp_bcd', fits nonnegative CP Decomposition using
            the Block Coordinate Descent (BCD) Method.
        'ncp_hals', fits nonnegative CP Decomposition using
            the Hierarchical Alternating Least Squares
            (HALS) Method.
        'mncp_hals', fits nonnegative CP Decomposition using
            the Hierarchical Alternating Least Squares
            (HALS) Method with missing data.
        'mcp_als', fits CP Decomposition with missing data using
            Alternating Least Squares (ALS).
    :param cs: str, conditioned stimuli, '' defaults to all CSes
    :param warp: boolean, warp trace offset so ensure delivery is a single point in time
    :param trace_type: str, type of calcium imaging trace being used in associated analysis
    :param group_by: str, period of time being analyzed across animal training
    :param nan_thresh: float, fraction of trials that must contain non-NaN entries
    :param score_threshold: float, score threshold for CellReg package alignment score
    :param staging: str, assign predetermined binning method for associated analysis
    :param dp_by: str, 'day' or 'run' TODO add pillow
    :return bhv_df: pandas.DataFrame, DataFrame with behavioral variables grouped by day or run
    """

    # load your metadata
    meta_list = []
    for mouse, word in zip(mice, words):
        load_kwargs = {'mouse': mouse,
                       'method': method,
                       'cs': cs,
                       'warp': warp,
                       'word': word,
                       'trace_type': trace_type,
                       'group_by': group_by,
                       'nan_thresh': nan_thresh,
                       'score_threshold': score_threshold}
        meta = load.groupday_tca_meta(**load_kwargs)

        # add your learning stage column if it doesn't exist
        meta = utils.add_stages_to_meta(meta, staging)

        # add your dprime if it doesn't exist
        if bhv_by.lower() == 'run':
            gb = ['mouse', 'date', 'run']
        elif bhv_by.lower() == 'day':
            gb = ['mouse', 'date']

        meta_list.append(meta)

    meta_all = pd.concat(meta_list, axis=0)

    # add acceleration and change in lick rate to meta
    if 'delta_speed' not in meta_all.columns:
        meta_all['delta_speed'] = meta_all['speed'] - meta_all['pre_speed']
    if 'pre_stim_speed' not in meta_all.columns:
        meta_all['pre_stim_speed'] = meta_all['pre_speed']
    if 'stim_speed' not in meta_all.columns:
        meta_all['stim_speed'] = meta_all['speed']
    if 'pre_lickrate' not in meta_all.columns:
        meta_all['pre_stim_lickrate'] = meta_all['pre_licks']  # /1 because pre is already for one second
    if 'delta_lickrate' not in meta_all.columns:
        meta_all['delta_lickrate'] = np.nan
    if 'stim_lickrate' not in meta_all.columns:
        meta_all['stim_lickrate'] = np.nan

    # loop over and add lickrate to meta_all, accounting for differences in stimulus duration
    for mouse in mice:
        mbool = meta_all.reset_index()['mouse'].isin([mouse]).values
        meta_all['delta_lickrate'].iloc[mbool] = (
                meta_all['anticipatory_licks'].iloc[mbool] / lookups.stim_length[mouse] - meta_all['pre_licks'].iloc[mbool])
        meta_all['stim_lickrate'].iloc[mbool] = (
                meta_all['anticipatory_licks'].iloc[mbool] / lookups.stim_length[mouse])


    # get bhv variables calculated across day or runs, uses .mean() so categorical values are lost
    cat_cols = meta_all.groupby(gb).min()
    bhv_df = meta_all.groupby(gb).mean()
    bhv_df[staging] = cat_cols[staging]

    return bhv_df


def plot_speed(bhv_df, speed_by='run', save_folder='', scale_by=[1, .7, .5], ftype='png', staging='parsed_10stage'):
    """
    Create summary plot of all mice for speed aligned to reversal.

    :param bhv_df: pandas DataFrame, dataframe with single dp value per day or per run for each mouse
    :param dp_by: str, 'run' or 'day' TODO add pillow
    :param save_folder: str, folder to save to
    :param scale_by: list of float, scale factor applied to
    :param ftype: str, file suffix to use for saving 'eps', 'png', 'pdf', etc
    :return: saves a number of plots to the specified folder
    """

    # get mice in DataFrame
    mice = bhv_df.reset_index()['mouse'].unique()
    mice_word = paths.groupmouse_word(mice)

    # plotting parameters that don't change per loop
    cmap = sns.color_palette('tab20', len(mice))[::-1]

    # greate a couple different scales of plots for ease of presentation/figures
    for scale in scale_by:
        fig, ax = plt.subplots(3, 1, figsize=(int(np.ceil(15 * scale)), 4*3), sharex=True)

        for cax, speed_metric in enumerate(['pre_stim_speed', 'stim_speed', 'delta_speed']):

            # baseline and reversal lines
            ax[cax].axhline(0, ls='--', color='#737373', linewidth=1)
            ax[cax].axvline(0, ls='--', color='#737373', linewidth=1)

            for c, mouse in enumerate(mice):
                mboo = bhv_df.reset_index()['mouse'].isin([mouse]).values
                mouse_df = bhv_df.iloc[mboo, :]
                rev_ind = np.where(mouse_df[staging].isin(['early low_dp learning',
                                                           'late low_dp learning',
                                                           'early high_dp learning',
                                                           'late high_dp learning',
                                                           'low_dp learning',
                                                           'high_dp learning']))[0][-1]
                y = mouse_df[speed_metric].values
                x = np.arange(len(y)) - rev_ind - 1  # last learning ind, 0 is your last day of learning
                sns.lineplot(x=x, y=y, label=mouse, color=cmap[c], alpha=0.7, linewidth=2, ax=ax[cax], legend=False)

                # add dashed line for missing running bridging nan period
                if any(np.isnan(y)):
                    first_train_ind = np.where(np.isnan(y))[0][-1] + 1
                    x2 = x[:first_train_ind + 1]
                    scale_by = y[first_train_ind]
                    if scale_by < 0.05 and scale_by > -0.05:
                        scale_by += 0.02
                    y2 = np.ones(len(x2)) * scale_by
                    # removed linestyle but want to to occupy same space as dp (may be other nans, beyond naive)
                    plt.plot(x2, y2, linestyle='', color=cmap[c])

            # despine
            sns.despine()

            # add plot labels and title
            if cax == len(ax) - 1:
                if speed_by.lower() == 'run':
                    if cax == 0:
                        ax[cax].set_title('Running speed across training sessions\n', size=16)
                    ax[cax].set_xlabel('Imaging sessions aligned to task reversal', size=14)
                elif speed_by.lower() == 'day':
                    if cax == 0:
                        ax[cax].set_title('Running speed across training days\n', size=16)
                    ax[cax].set_xlabel('Imaging days aligned to task reversal', size=14)
            if cax == 0:
                ax[cax].legend(title='mice\n', bbox_to_anchor=(1.01, 1.0), loc='upper left', borderaxespad=0)
            ax[cax].set_ylabel(f"running speed (cm/s)\n{speed_metric}", size=14)

            # save your plots
            dp_save_folder = save_folder + ' speed'
            if not os.path.isdir(dp_save_folder):
                os.mkdir(dp_save_folder)
            plt.savefig(
                os.path.join(dp_save_folder, f'{mice_word} speed {speed_by} behavior {scale}sc.{ftype}'),
                bbox_inches='tight')


def plot_speed_across_mice(
        mice,
        words=None,
        method='ncp_hals',
        cs='',
        warp=False,
        trace_type='zscore_day',
        group_by='all3',
        nan_thresh=0.95,
        score_threshold=0.8,
        staging='parsed_10stage',
        speed_by='day',
        scale_by=[1, .7, .5],
        ftype='png'):
    """
    Function for plotting summary of dprime across stages of learning.
    Creates single plot for multiple mice.

    :param mice: list of str, names of mice for analysis
    :param words: list of str, associated parameter hash words
    :param method: str, fit method from tensortools package
        'cp_als', fits CP Decomposition using Alternating
            Least Squares (ALS).
        'ncp_bcd', fits nonnegative CP Decomposition using
            the Block Coordinate Descent (BCD) Method.
        'ncp_hals', fits nonnegative CP Decomposition using
            the Hierarchical Alternating Least Squares
            (HALS) Method.
        'mncp_hals', fits nonnegative CP Decomposition using
            the Hierarchical Alternating Least Squares
            (HALS) Method with missing data.
        'mcp_als', fits CP Decomposition with missing data using
            Alternating Least Squares (ALS).
    :param cs: str, conditioned stimuli, '' defaults to all CSes
    :param warp: boolean, warp trace offset so ensure delivery is a single point in time
    :param trace_type: str, type of calcium imaging trace being used in associated analysis
    :param group_by: str, period of time being analyzed across animal training
    :param nan_thresh: float, fraction of trials that must contain non-NaN entries
    :param score_threshold: float, score threshold for CellReg package alignment score
    :param staging: str, assign predetermined binning method for associated analysis
    :param dp_by: str, 'day' or 'run' TODO add pillow
    """

    # load your metadata
    load_kwargs = {'mice': mice,
                   'method': method,
                   'cs': cs,
                   'warp': warp,
                   'words': words,
                   'trace_type': trace_type,
                   'group_by': group_by,
                   'score_threshold': score_threshold,
                   'nan_thresh': nan_thresh}

    # create a new analysis directory for your mice named 'behavior'
    save_path = paths.groupmouse_analysis_path('behavior', **load_kwargs)

    # load your dprime DataFrame
    speed_df = bhv_df_across_mice(staging=staging, bhv_by=speed_by, **load_kwargs)

    # plot your dprime summary figures and save
    plot_speed(speed_df, speed_by=speed_by, save_folder=save_path, scale_by=scale_by, ftype=ftype, staging=staging)


def plot_licking(bhv_df, lick_by='run', save_folder='', scale_by=[1, .7, .5], ftype='png', staging='parsed_10stage'):
    """
    Create summary plot of all mice for lick aligned to reversal.

    :param bhv_df: pandas DataFrame, dataframe with single dp value per day or per run for each mouse
    :param dp_by: str, 'run' or 'day' TODO add pillow
    :param save_folder: str, folder to save to
    :param scale_by: list of float, scale factor applied to
    :param ftype: str, file suffix to use for saving 'eps', 'png', 'pdf', etc
    :return: saves a number of plots to the specified folder
    """

    # get mice in DataFrame
    mice = bhv_df.reset_index()['mouse'].unique()
    mice_word = paths.groupmouse_word(mice)

    # plotting parameters that don't change per loop
    cmap = sns.color_palette('tab20', len(mice))[::-1]

    # greate a couple different scales of plots for ease of presentation/figures
    for scale in scale_by:
        fig, ax = plt.subplots(3, 1, figsize=(int(np.ceil(15 * scale)), 4*3), sharex=True)

        for cax, lick_metric in enumerate(['pre_stim_lickrate', 'stim_lickrate', 'delta_lickrate']):

            # baseline and reversal lines
            ax[cax].axhline(0, ls='--', color='#737373', linewidth=1)
            ax[cax].axvline(0, ls='--', color='#737373', linewidth=1)

            for c, mouse in enumerate(mice):
                mboo = bhv_df.reset_index()['mouse'].isin([mouse]).values
                mouse_df = bhv_df.iloc[mboo, :]
                rev_ind = np.where(mouse_df[staging].isin(['early low_dp learning',
                                                           'late low_dp learning',
                                                           'early high_dp learning',
                                                           'late high_dp learning',
                                                           'low_dp learning',
                                                           'high_dp learning']))[0][-1]
                y = mouse_df[lick_metric].values
                x = np.arange(len(y)) - rev_ind - 1  # last learning ind, 0 is your last day of learning
                sns.lineplot(x=x, y=y, label=mouse, color=cmap[c], alpha=0.7, linewidth=2, ax=ax[cax], legend=False)

                # add dashed line for missing running bridging nan period
                if any(np.isnan(y)):
                    first_train_ind = np.where(np.isnan(y))[0][-1] + 1
                    x2 = x[:first_train_ind + 1]
                    scale_by = y[first_train_ind]
                    if scale_by < 0.05 and scale_by > -0.05:
                        scale_by += 0.02
                    y2 = np.ones(len(x2)) * scale_by
                    # removed linestyle but want to to occupy same space as dp (may be other nans, beyond naive)
                    plt.plot(x2, y2, linestyle='', color=cmap[c])

            # despine
            sns.despine()

            # add plot labels and title
            if cax == len(ax) - 1:
                if lick_by.lower() == 'run':
                    ax[cax].set_xlabel('Imaging sessions aligned to task reversal', size=14)
                    if cax == 0:
                        ax[cax].set_title('Lick rate across training sessions\n', size=16)
                elif lick_by.lower() == 'day':
                    if cax == 0:
                        ax[cax].set_title('Lick rate across training days\n', size=16)
                    ax[cax].set_xlabel('Imaging days aligned to task reversal', size=14)
            if cax == 0:
                ax[cax].legend(title='mice\n', bbox_to_anchor=(1.01, 1.0), loc='upper left', borderaxespad=0)
            ax[cax].set_ylabel(f"lick rate (licks/s)\n{lick_metric}", size=14)

            # save your plots
            dp_save_folder = save_folder + ' lick'
            if not os.path.isdir(dp_save_folder):
                os.mkdir(dp_save_folder)
            plt.savefig(
                os.path.join(dp_save_folder, f'{mice_word} lick {lick_by} behavior {scale}sc.{ftype}'),
                bbox_inches='tight')

def plot_licking_across_mice(
        mice,
        words=None,
        method='ncp_hals',
        cs='',
        warp=False,
        trace_type='zscore_day',
        group_by='all3',
        nan_thresh=0.95,
        score_threshold=0.8,
        staging='parsed_10stage',
        lick_by='day',
        scale_by=[1, .7, .5],
        ftype='png'):
    """
    Function for plotting summary of dprime across stages of learning.
    Creates single plot for multiple mice.

    :param mice: list of str, names of mice for analysis
    :param words: list of str, associated parameter hash words
    :param method: str, fit method from tensortools package
        'cp_als', fits CP Decomposition using Alternating
            Least Squares (ALS).
        'ncp_bcd', fits nonnegative CP Decomposition using
            the Block Coordinate Descent (BCD) Method.
        'ncp_hals', fits nonnegative CP Decomposition using
            the Hierarchical Alternating Least Squares
            (HALS) Method.
        'mncp_hals', fits nonnegative CP Decomposition using
            the Hierarchical Alternating Least Squares
            (HALS) Method with missing data.
        'mcp_als', fits CP Decomposition with missing data using
            Alternating Least Squares (ALS).
    :param cs: str, conditioned stimuli, '' defaults to all CSes
    :param warp: boolean, warp trace offset so ensure delivery is a single point in time
    :param trace_type: str, type of calcium imaging trace being used in associated analysis
    :param group_by: str, period of time being analyzed across animal training
    :param nan_thresh: float, fraction of trials that must contain non-NaN entries
    :param score_threshold: float, score threshold for CellReg package alignment score
    :param staging: str, assign predetermined binning method for associated analysis
    :param dp_by: str, 'day' or 'run' TODO add pillow
    """

    # load your metadata
    load_kwargs = {'mice': mice,
                   'method': method,
                   'cs': cs,
                   'warp': warp,
                   'words': words,
                   'trace_type': trace_type,
                   'group_by': group_by,
                   'score_threshold': score_threshold,
                   'nan_thresh': nan_thresh}

    # create a new analysis directory for your mice named 'behavior'
    save_path = paths.groupmouse_analysis_path('behavior', **load_kwargs)

    # load your dprime DataFrame
    lick_df = bhv_df_across_mice(staging=staging, bhv_by=lick_by, **load_kwargs)

    # plot your dprime summary figures and save
    plot_licking(lick_df, lick_by=lick_by, save_folder=save_path, scale_by=scale_by, ftype=ftype, staging=staging)
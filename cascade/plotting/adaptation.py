"""Functions for plotting tca decomp."""
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import bottleneck as bt
import os
import flow
from .. import load, paths, utils
from copy import deepcopy


def opt_func(x, a, b):
    return a * np.exp(-b * x)


def opt_func_offset(x, a, b, c):
    return a * np.exp(-b * x) + c


def weighted_avg_first100(
        mouse='OA27',
        trace_type='zscore_day',
        method='ncp_hals',
        cs='',
        warp=False,
        group_by='all2',
        nan_thresh=0.85,
        score_threshold=0.8,
        word_s='determine',
        word_n='directors',
        stim_or_noise='noise',
        color_by='condition',
        rank=15,
        run_threshold=3,
        stim_window='last_sec', # 'first_sec' is the other option
        func_type='no_offset',
        start_time=-1,
        end_time=6,
        ):

    # load TCA models and data
    V, my_sorts_stim = load.groupday_tca_model(
            mouse=mouse,
            trace_type=trace_type,
            method=method,
            cs=cs,
            warp=warp,
            rank=rank,
            word=word_s,
            group_by=group_by,
            nan_thresh=nan_thresh,
            score_threshold=score_threshold,
            full_output=False,
            unsorted=True,
            verbose=False)
    V2, my_sorts_noise = load.groupday_tca_model(
            mouse=mouse,
            trace_type=trace_type,
            method=method,
            cs=cs,
            warp=warp,
            rank=rank,
            word=word_n,
            group_by=group_by,
            nan_thresh=nan_thresh,
            score_threshold=score_threshold,
            full_output=False,
            unsorted=True,
            verbose=False)
    meta_stim = load.groupday_tca_meta(
            mouse=mouse,
            trace_type=trace_type,
            method=method,
            cs=cs,
            warp=warp,
            word=word_s,
            group_by=group_by,
            nan_thresh=nan_thresh,
            score_threshold=score_threshold)
    meta_stim = utils.add_dprime_to_meta(meta_stim)
    input_stim = load.groupday_tca_input_tensor(
            mouse=mouse,
            trace_type=trace_type,
            method=method,
            cs=cs,
            warp=warp,
            word=word_s,
            group_by=group_by,
            nan_thresh=nan_thresh,
            score_threshold=score_threshold)
    # input_bhv = load.groupday_tca_bhv(
    #         mouse=mouse,
    #         trace_type=trace_type,
    #         method=method,
    #         cs=cs,
    #         warp=warp,
    #         word=word_s,
    #         group_by=group_by,
    #         nan_thresh=nan_thresh,
    #         score_threshold=score_threshold)
    
    # pick decay function
    if func_type.lower() == 'no_offset':
        func = opt_func
    else:
        func = opt_func_offset
    func_tag = func_type.lower()

    # set saving path
    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}
    group_pars = {'group_by': group_by}
    if stim_or_noise.lower() == 'stim':
        save_dir = paths.tca_plots(
            mouse, 'group', pars=pars, word=word_s, group_pars=group_pars)
    else:
        save_dir = paths.tca_plots(
            mouse, 'group', pars=pars, word=word_n, group_pars=group_pars)
    save_dir = os.path.join(save_dir, 'adaptation')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, 'trial mean {}'.format(stim_window.lower()))
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, 'rank {}'.format(rank))
    if not os.path.isdir(save_dir): os.mkdir(save_dir)

    # get timestamp info for plotting lines
    run = flow.DateSorter.frommeta(mice=[mouse], exclude_tags=['bad'])[-1].runs(exclude_tags=['bad'])[0]
    t2p = run.trace2p()
    tr = t2p.d['framerate']
    timestep = 1/31
    timestamps = np.arange(start_time, end_time, timestep)[::2][:input_stim.shape[1]]
    times = np.unique(timestamps)
    zero_sec = np.where(times <= 0)[0][-1]
    if tr < 30:
        three_sec = np.where(times <= 2)[0][-1]
    else:
        three_sec = np.where(times <= 3)[0][-1]

    # create weighted average using cell factors
    map_by_comp = {}
    for ci in range(1,rank+1):
        if stim_or_noise == 'stim':  # stim
            weight_vec = V.results[rank][0].factors[0][:, ci-1]
        else:  # noise
            weight_vec = V2.results[rank][0].factors[0][:, ci-1]
        weight_map = np.zeros(input_stim.shape[1:])
        for tri in range(input_stim.shape[2]):
            trialx = weight_vec[:, None] * input_stim[:,:,tri]
            weight_map[:, tri] = bt.nanmean(trialx, axis=0)
        map_by_comp[ci] = weight_map

    # construct boolean of early runs each day
    sessboo = meta_stim.reset_index()['run'].values <= run_threshold

    # construct boolean of first 100 trials per day
    days = meta_stim.reset_index()['date'].unique()
    first100 = np.zeros((len(meta_stim['orientation'].isin([0]).values)))
    for di in days:
        dboo  = meta_stim.reset_index()['date'].isin([di]).values
        first100[np.where(dboo)[0][:100]] = 1
    firstboo = first100 > 0

    # set firstboo to only include early runs/sessions 
    firstboo = firstboo & sessboo

    # create boolean vectors for each trial type
    inds = np.arange(np.sum(firstboo))
    if color_by.lower() == 'orientation':
        # color = ['#6fd174', '#6e8dcc', '#cc6670']
        boo1  = meta_stim['orientation'].isin([0]).values[firstboo]
        boo2  = meta_stim['orientation'].isin([135]).values[firstboo]
        boo3  = meta_stim['orientation'].isin([270]).values[firstboo]
    elif color_by.lower() == 'condition':
        # color = ['#6fd174', '#6e8dcc', '#cc6670']
        boo1  = meta_stim['condition'].isin(['neutral']).values[firstboo]
        boo2  = meta_stim['condition'].isin(['minus']).values[firstboo]
        boo3  = meta_stim['condition'].isin(['plus']).values[firstboo]

    # get useful variables
    pupil = meta_stim['pupil'].values[firstboo]
    dp100 = meta_stim['dprime'].values[firstboo]

    # calculate indices of reversal/learning
    rev_ind = np.where(
        meta_stim['learning_state']
        .isin(['learning'])
        .values[firstboo])[0][-1]
    if np.sum(meta_stim['learning_state'].isin(['naive']).values) > 0:
        lear_ind = np.where(
            meta_stim['learning_state']
            .isin(['naive'])
            .values[firstboo])[0][-1]
    else:
        lear_ind = 0

    # colormap for each day
    cod = sns.color_palette('husl', len(days))

    # choose adapting components
    adapting_comps = range(1, rank+1)

    # plot
    # plt.figure(figsize=(30,6))
    for aci in adapting_comps:
        fig, ax1 = plt.subplots(figsize=(30,6))
        ax2 = ax1.twinx()
        for di, codi in zip(days, cod):

            # get your avg ensemble response vector
            if stim_window.lower() == 'first_sec':
                mean_comp = np.nanmean(map_by_comp[aci][16:32, :], axis=0)[firstboo]
            elif stim_window.lower() == 'last_sec':
                mean_comp = np.nanmean(map_by_comp[aci][(three_sec-16):three_sec, :], axis=0)[firstboo]
            else:
                print('Unrecognized stimulus window for calculating trial averages.')
                return

            # boolean for each day accounting for first 100 trials
            dboo  = meta_stim.reset_index()['date'].isin([di]).values[firstboo]
            
            # add an offset so that the fitting is only calculated on positive values
            offset = np.min(mean_comp)
            
            # get trial values and indices for each trial type
            x1 = inds[dboo & boo1]
            y1 = mean_comp[dboo & boo1]
            x2 = inds[dboo & boo2]
            y2 = mean_comp[dboo & boo2]
            x3 = inds[dboo & boo3]
            y3 = mean_comp[dboo & boo3]
            
            # plot all trials for each day
            ax1.plot(inds[dboo], mean_comp[dboo], 'o', color=codi, alpha=0.3)
            
            # fit trial types with exponential decay and plot
            color = ['#6fd174', '#3b7a3e', '#6e8dcc', '#314773', '#cc6670', '#732931']
            try:
                popt1, pcov1 = curve_fit(func, x1-np.min(x1), y1-offset)
                ax1.plot(x1, func(x1-np.min(x1), *popt1)+offset, color=color[2], linewidth=3)
            except:
                print('Fit failed')
            try:
                popt2, pcov2 = curve_fit(func, x2-np.min(x2), y2-offset)
                ax1.plot(x2, func(x2-np.min(x2), *popt2)+offset, color=color[4], linewidth=3)
            except:
                print('Fit failed')
            try:
                popt3, pcov3 = curve_fit(func, x3-np.min(x3), y3-offset)
                ax1.plot(x3, func(x3-np.min(x3), *popt3)+offset, color=color[0], linewidth=3)
            except:
                print('Fit failed')
        
        # add axis labels, title, and lines for reversal/learning     
        y_min = np.min(mean_comp, axis=0)
        y_max = np.max(mean_comp, axis=0)
        ax1.plot([0, len(mean_comp)], [0, 0], '--k')
        ax1.plot([lear_ind, lear_ind], [y_min, y_max], '--k')
        ax1.plot([rev_ind, rev_ind], [y_min, y_max], '--k')
        ax1.set_title('{}: Component {}: {}: Ensemble average sustained responses (first 100 trials per day)'.format(mouse, aci, stim_or_noise), size=16)
        ax1.set_xlabel('trial number', size=14)
        ax1.set_ylabel('response amplitude (weighted z-score)', size=14)

        # create matching dprime figure 
        ax2.plot(inds, dp100, '-', color='#C880D1')
        ax2.set_ylabel('dprime', color='#C880D1', size=14)

        # save
        file_name = 'Mean Weighted Activity {} Component {} rank {} {}.png'.format(stim_or_noise, aci, rank, func_tag)
        save_path = os.path.join(save_dir, file_name)     
        fig.savefig(save_path, bbox_inches='tight')
        plt.close('all')


def tca_first100(
        mouse='OA27',
        trace_type='zscore_day',
        method='ncp_hals',
        cs='',
        warp=False,
        group_by='all2',
        nan_thresh=0.85,
        score_threshold=0.8,
        word_s='determine',
        word_n='directors',
        stim_or_noise='noise',
        color_by='condition',
        rank=15,
        run_threshold=3,
        stim_window='last_sec', # 'first_sec' is the other option
        func_type='no_offset',
        start_time=-1,
        end_time=6,
        ):
    """
    Plot the first 100 trials of TCA results. Fitting with exponential 
    function.
    """
    # load TCA models and data
    V, my_sorts_stim = load.groupday_tca_model(
            mouse=mouse,
            trace_type=trace_type,
            method=method,
            cs=cs,
            warp=warp,
            rank=rank,
            word=word_s,
            group_by=group_by,
            nan_thresh=nan_thresh,
            score_threshold=score_threshold,
            full_output=False,
            unsorted=True,
            verbose=False)
    V2, my_sorts_noise = load.groupday_tca_model(
            mouse=mouse,
            trace_type=trace_type,
            method=method,
            cs=cs,
            warp=warp,
            rank=rank,
            word=word_n,
            group_by=group_by,
            nan_thresh=nan_thresh,
            score_threshold=score_threshold,
            full_output=False,
            unsorted=True,
            verbose=False)
    meta_stim = load.groupday_tca_meta(
            mouse=mouse,
            trace_type=trace_type,
            method=method,
            cs=cs,
            warp=warp,
            word=word_s,
            group_by=group_by,
            nan_thresh=nan_thresh,
            score_threshold=score_threshold)
    meta_stim = utils.add_dprime_to_meta(meta_stim)
    # input_bhv = load.groupday_tca_bhv(
    #         mouse=mouse,
    #         trace_type=trace_type,
    #         method=method,
    #         cs=cs,
    #         warp=warp,
    #         word=word_s,
    #         group_by=group_by,
    #         nan_thresh=nan_thresh,
    #         score_threshold=score_threshold)
    
    # pick decay function
    if func_type.lower() == 'no_offset':
        func = opt_func
    else:
        func = opt_func_offset
    func_tag = func_type.lower()

    # set saving path
    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}
    group_pars = {'group_by': group_by}
    if stim_or_noise.lower() == 'stim':
        save_dir = paths.tca_plots(
            mouse, 'group', pars=pars, word=word_s, group_pars=group_pars)
    else:
        save_dir = paths.tca_plots(
            mouse, 'group', pars=pars, word=word_n, group_pars=group_pars)
    save_dir = os.path.join(save_dir, 'adaptation')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, 'tca first100 {}'.format(stim_window.lower()))
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, 'rank {}'.format(rank))
    if not os.path.isdir(save_dir): os.mkdir(save_dir)

    # get timestamp info for plotting lines
    run = flow.DateSorter.frommeta(mice=[mouse], exclude_tags=['bad'])[-1].runs(exclude_tags=['bad'])[0]
    t2p = run.trace2p()
    tr = t2p.d['framerate']
    timestep = 1/31
    tpoints = len(V.results[rank][0].factors[0][:, 0])
    timestamps = np.arange(start_time, end_time, timestep)[::2][:tpoints]
    times = np.unique(timestamps)
    zero_sec = np.where(times <= 0)[0][-1]
    if tr < 30:
        three_sec = np.where(times <= 2)[0][-1]
    else:
        three_sec = np.where(times <= 3)[0][-1]

    # construct boolean of early runs each day
    sessboo = meta_stim.reset_index()['run'].values <= run_threshold

    # construct boolean of first 100 trials per day
    days = meta_stim.reset_index()['date'].unique()
    first100 = np.zeros((len(meta_stim['orientation'].isin([0]).values)))
    for di in days:
        dboo  = meta_stim.reset_index()['date'].isin([di]).values
        first100[np.where(dboo)[0][:100]] = 1
    firstboo = first100 > 0

    # set firstboo to only include early runs/sessions 
    firstboo = firstboo & sessboo

    # create boolean vectors for each trial type
    inds = np.arange(np.sum(firstboo))
    if color_by.lower() == 'orientation':
        # color = ['#6fd174', '#6e8dcc', '#cc6670']
        boo1  = meta_stim['orientation'].isin([0]).values[firstboo]
        boo2  = meta_stim['orientation'].isin([135]).values[firstboo]
        boo3  = meta_stim['orientation'].isin([270]).values[firstboo]
    elif color_by.lower() == 'condition':
        # color = ['#6fd174', '#6e8dcc', '#cc6670']
        boo1  = meta_stim['condition'].isin(['neutral']).values[firstboo]
        boo2  = meta_stim['condition'].isin(['minus']).values[firstboo]
        boo3  = meta_stim['condition'].isin(['plus']).values[firstboo]

    # get useful variables
    pupil = meta_stim['pupil'].values[firstboo]
    dp100 = meta_stim['dprime'].values[firstboo]

    # calculate indices of reversal/learning
    rev_ind = np.where(
        meta_stim['learning_state']
        .isin(['learning'])
        .values[firstboo])[0][-1]
    if np.sum(meta_stim['learning_state'].isin(['naive']).values) > 0:
        lear_ind = np.where(
            meta_stim['learning_state']
            .isin(['naive'])
            .values[firstboo])[0][-1]
    else:
        lear_ind = 0

    # colormap for each day
    cod = sns.color_palette('husl', len(days))

    # choose adapting components
    adapting_comps = range(1, rank+1)

    # plot
    # plt.figure(figsize=(30,6))
    for aci in adapting_comps:
        fig, ax1 = plt.subplots(figsize=(30,6))
        ax2 = ax1.twinx()
        for di, codi in zip(days, cod):

            # boolean for each day accounting for first 100 trials
            dboo  = meta_stim.reset_index()['date'].isin([di]).values[firstboo]

            if stim_or_noise == 'stim':  # stim
                comp_vec = V.results[rank][0].factors[2][:, aci-1][firstboo]
            else:  # noise
                comp_vec = V2.results[rank][0].factors[2][:, aci-1][firstboo]
            
            # add an offset so that the fitting is only calculated on positive values
            offset = np.min(comp_vec)

            # get trial values and indices for each trial type
            x1 = inds[dboo & boo1]
            y1 = comp_vec[dboo & boo1]
            x2 = inds[dboo & boo2]
            y2 = comp_vec[dboo & boo2]
            x3 = inds[dboo & boo3]
            y3 = comp_vec[dboo & boo3]
            
            # plot all trials for each day
            ax1.plot(inds[dboo], comp_vec[dboo], 'o', color=codi, alpha=0.3)
            
            # fit trial types with exponential decay and plot
            color = ['#6fd174', '#3b7a3e', '#6e8dcc', '#314773', '#cc6670', '#732931']
            try:
                popt1, pcov1 = curve_fit(func, x1-np.min(x1), y1-offset)
                ax1.plot(x1, func(x1-np.min(x1), *popt1)+offset, color=color[2], linewidth=3)
            except:
                print('Fit failed')
            try:
                popt2, pcov2 = curve_fit(func, x2-np.min(x2), y2-offset)
                ax1.plot(x2, func(x2-np.min(x2), *popt2)+offset, color=color[4], linewidth=3)
            except:
                print('Fit failed')
            try:
                popt3, pcov3 = curve_fit(func, x3-np.min(x3), y3-offset)
                ax1.plot(x3, func(x3-np.min(x3), *popt3)+offset, color=color[0], linewidth=3)
            except:
                print('Fit failed')
        
        # add axis labels, title, and lines for reversal/learning     
        y_min = np.min(comp_vec, axis=0)
        y_max = np.max(comp_vec, axis=0)
        ax1.plot([0, len(comp_vec)], [0, 0], '--k')
        ax1.plot([lear_ind, lear_ind], [y_min, y_max], '--k')
        ax1.plot([rev_ind, rev_ind], [y_min, y_max], '--k')
        ax1.set_title('{}: Component {}: {}: TCA model responses (first 100 trials per day)'.format(mouse, aci, stim_or_noise), size=16)
        ax1.set_xlabel('trial number', size=14)
        ax1.set_ylabel('response amplitude (weighted z-score)', size=14)

        # create matching dprime figure 
        ax2.plot(inds, dp100, '-', color='#C880D1')
        ax2.set_ylabel('dprime', color='#C880D1', size=14)

        # save
        file_name = 'TCA Model Activity {} Component {} rank {} {}.png'.format(stim_or_noise, aci, rank, func_tag)
        save_path = os.path.join(save_dir, file_name)     
        fig.savefig(save_path, bbox_inches='tight')
        plt.close('all')


def projected_heatmap(
        mouse='OA27',
        trace_type='zscore_day',
        method='ncp_hals',
        cs='',
        warp=False,
        group_by='all2',
        nan_thresh=0.85,
        score_threshold=0.8,
        word_s='determine',
        word_n='directors',
        stim_or_noise='noise',
        rank=15,
        run_threshold=3,
        start_time=-1,
        end_time=6,
        cs_bar=True,
        day_bar=True,
        day_line=True,
        run_line=False,
        match_clim=True,
        quinine_ticks=False,
        ensure_ticks=False,
        lick_ticks=False,
        label_cbar=True,
        vmin=None,
        vmax=None):

    """
    Weight your cell activity using cell factors and take mean for each component. 
    Use this to construct heatmaps of the first 100 trials of the day accounting
    for absolute run (session) during the day. 
    """
    # fixed plotting params
    # arthur's predetermined hex colors
    colors = {
        'orange': '#E86E0A',
        'red': '#D61E21',
        'gray': '#7C7C7C',
        'black': '#000000',
        'green': '#75D977',
        'mint': '#47D1A8',
        'purple': '#C880D1',
        'indigo': '#5E5AE6',
        'blue': '#47AEED',  # previously 4087DD
        'yellow': '#F2E205',
    }

    # cs to color mapping
    cs_colors = {
        'plus': 'mint',
        'minus': 'red',
        'neutral': 'blue',
        'pavlovian': 'mint',
        'naive': 'gray'
    }

    # checkerboard overlay or day_bar
    day_colors = {
         'A': '#FDFEFE',
         'B': '#7B7D7D'
    }

#     cmap = sns.diverging_palette(220, 10, sep=30, as_cmap=True)
    cmap = 'seismic'

    # load TCA models and data
    V, my_sorts_stim = load.groupday_tca_model(
            mouse=mouse,
            trace_type=trace_type,
            method=method,
            cs=cs,
            warp=warp,
            rank=rank,
            word=word_s,
            group_by=group_by,
            nan_thresh=nan_thresh,
            score_threshold=score_threshold,
            full_output=False,
            unsorted=True,
            verbose=False)
    V2, my_sorts_noise = load.groupday_tca_model(
            mouse=mouse,
            trace_type=trace_type,
            method=method,
            cs=cs,
            warp=warp,
            rank=rank,
            word=word_n,
            group_by=group_by,
            nan_thresh=nan_thresh,
            score_threshold=score_threshold,
            full_output=False,
            unsorted=True,
            verbose=False)
    meta_stim = load.groupday_tca_meta(
            mouse=mouse,
            trace_type=trace_type,
            method=method,
            cs=cs,
            warp=warp,
            word=word_s,
            group_by=group_by,
            nan_thresh=nan_thresh,
            score_threshold=score_threshold)
    meta_stim = utils.add_dprime_to_meta(meta_stim)
    input_stim = load.groupday_tca_input_tensor(
            mouse=mouse,
            trace_type=trace_type,
            method=method,
            cs=cs,
            warp=warp,
            word=word_s,
            group_by=group_by,
            nan_thresh=nan_thresh,
            score_threshold=score_threshold)
    # input_bhv = load.groupday_tca_bhv(
    #         mouse=mouse,
    #         trace_type=trace_type,
    #         method=method,
    #         cs=cs,
    #         warp=warp,
    #         word=word_s,
    #         group_by=group_by,
    #         nan_thresh=nan_thresh,
    #         score_threshold=score_threshold)

    # get timestamp info for plotting lines
    run = flow.DateSorter.frommeta(mice=[mouse], exclude_tags=['bad'])[-1].runs(exclude_tags=['bad'])[0]
    t2p = run.trace2p()
    tr = t2p.d['framerate']
    timestep = 1/31
    timestamps = np.arange(start_time, end_time, timestep)[::2][:input_stim.shape[1]]
    times = np.unique(timestamps)
    zero_sec = np.where(times <= 0)[0][-1]
    if tr < 30:
        three_sec = np.where(times <= 2)[0][-1]
    else:
        three_sec = np.where(times <= 3)[0][-1]

    # construct boolean of early runs each day
    sessboo = meta_stim.reset_index()['run'].values <= run_threshold

    # construct boolean of first 100 trials per day
    days = meta_stim.reset_index()['date'].unique()
    first100 = np.zeros((len(meta_stim['orientation'].isin([0]).values)))
    for di in days:
        dboo  = meta_stim.reset_index()['date'].isin([di]).values
        first100[np.where(dboo)[0][:100]] = 1
    firstboo = first100 > 0

    # only look at the first 100 trials and 
    meta_stim_sub = meta_stim.iloc[(firstboo & sessboo), :]
    # input_bhv_sub = input_bhv[:,:,(firstboo & sessboo)]
    input_stim_sub = input_stim[:,:,(firstboo & sessboo)]

    # set saving path
    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}
    group_pars = {'group_by': group_by}
    if stim_or_noise.lower() == 'stim':
        save_dir = paths.tca_plots(
            mouse, 'group', pars=pars, word=word_s, group_pars=group_pars)
    else:
        save_dir = paths.tca_plots(
            mouse, 'group', pars=pars, word=word_n, group_pars=group_pars)
    save_dir = os.path.join(save_dir, 'adaptation')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, 'heatmaps')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, 'rank {}'.format(rank))
    if not os.path.isdir(save_dir): os.mkdir(save_dir)

    # created weighted reconstructions of cell activity and plot heatmaps 
    for ci in range(1,rank+1):
        # stim
        if stim_or_noise == 'stim':
            weight_vec = V.results[rank][0].factors[0][:, ci-1]
        # noise
        else:
            weight_vec = V2.results[rank][0].factors[0][:, ci-1]
        weight_map = np.zeros(input_stim_sub.shape[1:])
        for tri in range(input_stim_sub.shape[2]):
            trialx = weight_vec[:, None] * input_stim_sub[:,:,tri]
            weight_map[:, tri] = bt.nanmean(trialx, axis=0)

        # set file and title names
        file_name = 'Heatmap Weighted Activity {} Component {} rank {} vmax.png'.format(stim_or_noise, ci, rank)
        supt = 'Mean Weighted Activity Component {}: {}'.format(ci, stim_or_noise)

        o0 = weight_map.T[meta_stim_sub['orientation'].isin([0]).values, :]
        o135 = weight_map.T[meta_stim_sub['orientation'].isin([135]).values, :]
        o270 = weight_map.T[meta_stim_sub['orientation'].isin([270]).values, :]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30,12))
        fig.suptitle(supt, size=20, ) #x=0.22)
        sns.heatmap(o0, cmap=cmap, center=0, ax=ax1)
        sns.heatmap(o135, cmap=cmap, center=0, ax=ax2)
        sns.heatmap(o270, cmap=cmap, center=0, ax=ax3)

        # loop through axes and plot relevant metadata on top
        oris = [0, 135, 270]
        count = 0
        cmin = []
        cmax = []
        ccmap = []
        for oc, ax in enumerate((ax1, ax2, ax3)):

            # add in a title
            ax.set_title('orientation = {}'.format(oris[oc]), size=18)

            # match vmin and max across plots (first check values)
            if match_clim and vmax is None:
                _vmin, _vmax = ax.collections[0].get_clim()
                cmin.append(_vmin)
                cmax.append(_vmax)
                ccmap.append(ax.collections[0].get_cmap())

            # get metadata for this orientation/set of trials
            meta = meta_stim_sub.loc[meta_stim_sub['orientation'].isin([oris[count]]), ['condition',
                           'ensure', 'quinine', 'firstlick', 'learning_state']]
            meta = meta.reset_index()
            meta = meta.drop_duplicates()
            ensure = np.array(meta['ensure'])
            quinine = np.array(meta['quinine'])
            firstlick = np.array(meta['firstlick'])
            css = deepcopy(meta['condition'])
            learning_state = deepcopy(meta['learning_state'])

            ori_inds = np.array(meta_stim_sub['orientation'].values)
            ori_inds = ori_inds == oris[count]

            # set labels
            if count == 0:
                ax.set_ylabel('Trials', size=18)
            ax.set_xlabel('Time (sec)', size=18)

            # plot cs color bar/line
            if cs_bar:
                css[meta['learning_state'].isin(['naive']).values] = 'naive'
                for cs in np.unique(css):
                    cs_line_color = colors[cs_colors[cs]]
                    cs_y = np.where(css == cs)[0]
                    cs_y = [cs_y[0], cs_y[-1]+1]
                    ax.plot((2, 2), cs_y, color=cs_line_color, ls='-',
                            lw=15, alpha=0.8, solid_capstyle='butt')

            # find days where learning or reversal start
            if day_bar:
                days = np.array(meta_stim_sub.index.get_level_values('date'))
                days = days[ori_inds]
                runs = np.array(meta_stim_sub.index.get_level_values('run'))
                runs = runs[ori_inds]
                count_d = 0
                for day in np.unique(days):
                    day_y = np.where(days == day)[0]
                    day_y = [day_y[0], day_y[-1]+1]
                    day_bar_color = day_colors[sorted(day_colors.keys())[count_d%2]]
                    ax.plot((3.5, 3.5), day_y, color=day_bar_color, ls='-',
                            lw=6, alpha=0.4, solid_capstyle='butt')
                    count_d = count_d + 1

            # get limits for plotting
            y_lim = ax.get_ylim()
            x_lim = ax.get_xlim()

            # plot lines between days
            if day_line:
                days = np.array(meta_stim_sub.index.get_level_values('date'))
                days = days[ori_inds]
                days = np.diff(days)
                day_ind = np.where(days > 0)[0]
                for y in day_ind:
                    day_y = [y+1, y+1]
                    ax.plot(x_lim, day_y, color='#8e8e8e', ls='-', lw=1, alpha=0.8)

            # plot lines between runs
            if run_line:
                runs = np.array(meta_stim_sub.index.get_level_values('run'))
                runs = runs[ori_inds]
                runs = np.diff(runs)
                run_ind = np.where(runs > 0)[0]
                for y in run_ind:
                    run_y = [y+1, y+1]
                    ax.plot(x_lim, run_y, color='#bababa', ls='-', lw=1,  alpha=0.8)

            # plot onset/offest lines
            ax.plot((zero_sec, zero_sec), y_lim, color='#8e8e8e', ls='-', lw=2, alpha=0.8)
            ax.plot((three_sec, three_sec), y_lim, color='#8e8e8e', ls='-', lw=2, alpha=0.8)

            # if you allow caxis to scale automatically, add alpha to ticks
            if not vmin and not vmax:
                tick_alpha = 0.5
            else:
                tick_alpha = 1

            # plot quinine
            if quinine_ticks:
        #         quinine = meta['quinine'].values
                for l in range(len(quinine)):
                    if np.isfinite(quinine[l]):
                        x = [quinine[l], quinine[l]]
                        y = [l+0.5, l+0.5]
                        ax.plot(x, y, color='#0fffc3', ls='',
                                marker='.', markersize=2, alpha=tick_alpha)

            # plot ensure
            if ensure_ticks:
        #         quinine = meta['ensure'].values
                for l in range(len(ensure)):
                    if np.isfinite(ensure[l]):
                        x = [ensure[l], ensure[l]]
                        y = [l+0.5, l+0.5]
                        ax.plot(x, y, color='#ffb30f', ls='',
                                marker='.', markersize=2, alpha=tick_alpha)

            # plot licks
            if lick_ticks:
        #         quinine = meta['firstlick'].values
                for l in range(len(firstlick)):
                    if np.isfinite(firstlick[l]):
                        x = [firstlick[l], firstlick[l]]
                        y = [l+0.5, l+0.5]
                        ax.plot(x, y, color='#7237f2', ls='',
                                marker='.', markersize=2, alpha=tick_alpha)

            # reset yticklabels
            if y_lim[0] < 100:
                step = 10
            elif y_lim[0] < 200:
                step = 20
            elif y_lim[0] < 500:
                step = 50
            elif y_lim[0] < 5000:
                step = 500
            elif y_lim[0] < 10000:
                step = 1000
            elif y_lim[0] >= 10000:
                step = 5000
            base_yticks = range(int(y_lim[-1]), int(y_lim[0]), int(step))
            base_yticks = [s for s in base_yticks]
            base_ylabels = [str(s) for s in base_yticks]

            dates = np.array(meta_stim_sub.index.get_level_values('date'))
            dates = dates[ori_inds]
            date_yticks = []
            date_label = []

            date_rel = meta.reset_index()['date'].unique()

            for day in np.unique(dates):

                # find number of inds needed to shift labels to put in middle of date block
                last_ind = np.where(dates == day)[0][-1]
                first_ind = np.where(dates == day)[0][0]
                shifter = np.round((last_ind - first_ind)/2)
                label_ind = last_ind - shifter

                # get your relative day number
                day_val = np.where(date_rel == day)[0][0] + 1  # add one to make it one-indexed

                # add a pad to keep labels left-justified
                if day_val < 10:
                    pad = '  '
                else:
                    pad = ''

                # if the date label and trial label inds are exactly the same
                # force the label info onto one line of text
                # label days with imaging day number
                if np.isin(label_ind, base_yticks):
                    # remove the existing ind and add a special label to end
                    good_tick = ~np.isin(base_yticks, label_ind)
                    base_yticks = [base_yticks[s] for s in range(len(good_tick))
                                   if good_tick[s]]
                    base_ylabels = [base_ylabels[s] for s in range(len(good_tick))
                                   if good_tick[s]]
                    dpad = '          '
                    dpad = dpad[0:(len('          ') - len(str(label_ind)*2))]
                    base_ylabels.append('Day ' + str(day_val) + dpad + str(label_ind))
                else:
                    base_ylabels.append('Day ' + str(day_val) + '          ' + pad)
                base_yticks.append(label_ind)

            ax.set_yticks(base_yticks)
            ax.set_yticklabels(base_ylabels, size=14)

            # reset xticklabels
            xticklabels = np.array([-1, 0, 1, 2, 3, 4, 5, 6])
            xticklabels = xticklabels[(xticklabels > times[0]) & (xticklabels < times[-1])]
            xticks = [np.where(times <= s)[0][-1] for s in xticklabels]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation='horizontal', size=16)

            # update count through loops
            count = count + 1

        # match vmin and max across plots (using cmax to choose cmap and cmin)
        scale_by = 0.7
        if match_clim and vmax is None:
            max_ind = np.nanargmax(cmax)
            cmin = cmin[max_ind]*scale_by
            cmax = cmax[max_ind]*scale_by
            ccmap = ccmap[max_ind]

            for ax in (ax1, ax2, ax3):
                ax.collections[0].set_clim(vmax=cmax, vmin=cmin)
                ax.collections[0].set_cmap(ccmap)
        save_path = os.path.join(save_dir, file_name)     
        plt.savefig(save_path, bbox_inches='tight')
        plt.close('all')


def bhv_heatmap(
        mouse='OA27',
        trace_type='zscore_day',
        method='ncp_hals',
        cs='',
        warp=False,
        group_by='all2',
        nan_thresh=0.85,
        score_threshold=0.8,
        word='determine',
        run_threshold=3,
        start_time=-1,
        end_time=6,
        cs_bar=True,
        day_bar=True,
        day_line=True,
        run_line=False,
        match_clim=True,
        quinine_ticks=False,
        ensure_ticks=False,
        lick_ticks=False,
        label_cbar=True,
        vmin=None,
        vmax=None):

    """
    Behavioral heatmaps of the first 100 trials of the day accounting for
    absolute run (session) during the day. 
    """

    # fixed plotting params
    # arthur's predetermined hex colors
    colors = {
        'orange': '#E86E0A',
        'red': '#D61E21',
        'gray': '#7C7C7C',
        'black': '#000000',
        'green': '#75D977',
        'mint': '#47D1A8',
        'purple': '#C880D1',
        'indigo': '#5E5AE6',
        'blue': '#47AEED',  # previously 4087DD
        'yellow': '#F2E205',
    }

    # cs to color mapping
    cs_colors = {
        'plus': 'mint',
        'minus': 'red',
        'neutral': 'blue',
        'pavlovian': 'mint',
        'naive': 'gray'
    }

    # checkerboard overlay or day_bar
    day_colors = {
         'A': '#FDFEFE',
         'B': '#7B7D7D'
    }

    # bhv lookup for indexing into 1st dim of input_bhv tensor
    bhv_lookup = {
        'pupil': 0,
        'dpupil': 1,
        'speed': 2,
        'dspeed': 3
    }

#     cmap = sns.diverging_palette(220, 10, sep=30, as_cmap=True)
    # cmap = 'seismic'

    # load TCA models and data
    meta_stim = load.groupday_tca_meta(
            mouse=mouse,
            trace_type=trace_type,
            method=method,
            cs=cs,
            warp=warp,
            word=word,
            group_by=group_by,
            nan_thresh=nan_thresh,
            score_threshold=score_threshold)
    meta_stim = utils.add_dprime_to_meta(meta_stim)
    input_bhv = load.groupday_tca_bhv(
            mouse=mouse,
            trace_type=trace_type,
            method=method,
            cs=cs,
            warp=warp,
            word=word,
            group_by=group_by,
            nan_thresh=nan_thresh,
            score_threshold=score_threshold)

    # get timestamp info for plotting lines
    run = flow.DateSorter.frommeta(mice=[mouse], exclude_tags=['bad'])[-1].runs(exclude_tags=['bad'])[0]
    t2p = run.trace2p()
    tr = t2p.d['framerate']
    timestep = 1/31
    timestamps = np.arange(start_time, end_time, timestep)[::2][:input_bhv.shape[1]]
    times = np.unique(timestamps)
    zero_sec = np.where(times <= 0)[0][-1]
    if tr < 30:
        three_sec = np.where(times <= 2)[0][-1]
    else:
        three_sec = np.where(times <= 3)[0][-1]

    # construct boolean of early runs each day
    sessboo = meta_stim.reset_index()['run'].values <= run_threshold

    # construct boolean of first 100 trials per day
    days = meta_stim.reset_index()['date'].unique()
    first100 = np.zeros((len(meta_stim['orientation'].isin([0]).values)))
    for di in days:
        dboo  = meta_stim.reset_index()['date'].isin([di]).values
        first100[np.where(dboo)[0][:100]] = 1
    firstboo = first100 > 0

    # only look at the first 100 trials and 
    meta_stim_sub = meta_stim.iloc[(firstboo & sessboo), :]
    input_bhv_sub = input_bhv[:,:,(firstboo & sessboo)]

    # set saving path
    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}
    group_pars = {'group_by': group_by}
    save_dir = paths.tca_plots(
        mouse, 'group', pars=pars, word=word, group_pars=group_pars)
    save_dir = os.path.join(save_dir, 'adaptation')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, 'heatmaps')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, 'bhv traces')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)

    # created weighted reconstructions of cell activity and plot heatmaps 
    for bhv_trace_type in bhv_lookup.keys():
        
        # pick your trace
        weight_map = input_bhv_sub[bhv_lookup[bhv_trace_type], :, :]

        # skip behaviors that were not recorded
        if np.sum(np.isnan(weight_map.flatten())) == len(weight_map.flatten()):
            continue

        # set file and title names
        file_name = 'Heatmap behavior {}.png'.format(bhv_trace_type)
        supt = 'Behavioral traces: {}'.format(bhv_trace_type)

        o0 = weight_map.T[meta_stim_sub['orientation'].isin([0]).values, :]
        o135 = weight_map.T[meta_stim_sub['orientation'].isin([135]).values, :]
        o270 = weight_map.T[meta_stim_sub['orientation'].isin([270]).values, :]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30,12))
        fig.suptitle(supt, size=20, ) #x=0.22)
        sns.heatmap(o0, ax=ax1)
        sns.heatmap(o135, ax=ax2)
        sns.heatmap(o270, ax=ax3)

        # loop through axes and plot relevant metadata on top
        oris = [0, 135, 270]
        count = 0
        cmin = []
        cmax = []
        ccmap = []
        for oc, ax in enumerate((ax1, ax2, ax3)):

            # add in a title
            ax.set_title('orientation = {}'.format(oris[oc]), size=18)

            # match vmin and max across plots (first check values)
            if match_clim and vmax is None:
                _vmin, _vmax = ax.collections[0].get_clim()
                cmin.append(_vmin)
                cmax.append(_vmax)
                ccmap.append(ax.collections[0].get_cmap())

            # get metadata for this orientation/set of trials
            meta = meta_stim_sub.loc[meta_stim_sub['orientation'].isin([oris[count]]), ['condition',
                           'ensure', 'quinine', 'firstlick', 'learning_state']]
            meta = meta.reset_index()
            meta = meta.drop_duplicates()
            ensure = np.array(meta['ensure'])
            quinine = np.array(meta['quinine'])
            firstlick = np.array(meta['firstlick'])
            css = deepcopy(meta['condition'])
            learning_state = deepcopy(meta['learning_state'])

            ori_inds = np.array(meta_stim_sub['orientation'].values)
            ori_inds = ori_inds == oris[count]

            # set labels
            if count == 0:
                ax.set_ylabel('Trials', size=18)
            ax.set_xlabel('Time (sec)', size=18)

            # plot cs color bar/line
            if cs_bar:
                css[meta['learning_state'].isin(['naive']).values] = 'naive'
                for cs in np.unique(css):
                    cs_line_color = colors[cs_colors[cs]]
                    cs_y = np.where(css == cs)[0]
                    cs_y = [cs_y[0], cs_y[-1]+1]
                    ax.plot((2, 2), cs_y, color=cs_line_color, ls='-',
                            lw=15, alpha=0.8, solid_capstyle='butt')

            # find days where learning or reversal start
            if day_bar:
                days = np.array(meta_stim_sub.index.get_level_values('date'))
                days = days[ori_inds]
                runs = np.array(meta_stim_sub.index.get_level_values('run'))
                runs = runs[ori_inds]
                count_d = 0
                for day in np.unique(days):
                    day_y = np.where(days == day)[0]
                    day_y = [day_y[0], day_y[-1]+1]
                    day_bar_color = day_colors[sorted(day_colors.keys())[count_d%2]]
                    ax.plot((3.5, 3.5), day_y, color=day_bar_color, ls='-',
                            lw=6, alpha=0.4, solid_capstyle='butt')
                    count_d = count_d + 1

            # get limits for plotting
            y_lim = ax.get_ylim()
            x_lim = ax.get_xlim()

            # plot lines between days
            if day_line:
                days = np.array(meta_stim_sub.index.get_level_values('date'))
                days = days[ori_inds]
                days = np.diff(days)
                day_ind = np.where(days > 0)[0]
                for y in day_ind:
                    day_y = [y+1, y+1]
                    ax.plot(x_lim, day_y, color='#8e8e8e', ls='-', lw=1, alpha=0.8)

            # plot lines between runs
            if run_line:
                runs = np.array(meta_stim_sub.index.get_level_values('run'))
                runs = runs[ori_inds]
                runs = np.diff(runs)
                run_ind = np.where(runs > 0)[0]
                for y in run_ind:
                    run_y = [y+1, y+1]
                    ax.plot(x_lim, run_y, color='#bababa', ls='-', lw=1,  alpha=0.8)

            # plot onset/offest lines
            ax.plot((zero_sec, zero_sec), y_lim, color='#8e8e8e', ls='-', lw=2, alpha=0.8)
            ax.plot((three_sec, three_sec), y_lim, color='#8e8e8e', ls='-', lw=2, alpha=0.8)

            # if you allow caxis to scale automatically, add alpha to ticks
            if not vmin and not vmax:
                tick_alpha = 0.5
            else:
                tick_alpha = 1

            # plot quinine
            if quinine_ticks:
        #         quinine = meta['quinine'].values
                for l in range(len(quinine)):
                    if np.isfinite(quinine[l]):
                        x = [quinine[l], quinine[l]]
                        y = [l+0.5, l+0.5]
                        ax.plot(x, y, color='#0fffc3', ls='',
                                marker='.', markersize=2, alpha=tick_alpha)

            # plot ensure
            if ensure_ticks:
        #         quinine = meta['ensure'].values
                for l in range(len(ensure)):
                    if np.isfinite(ensure[l]):
                        x = [ensure[l], ensure[l]]
                        y = [l+0.5, l+0.5]
                        ax.plot(x, y, color='#ffb30f', ls='',
                                marker='.', markersize=2, alpha=tick_alpha)

            # plot licks
            if lick_ticks:
        #         quinine = meta['firstlick'].values
                for l in range(len(firstlick)):
                    if np.isfinite(firstlick[l]):
                        x = [firstlick[l], firstlick[l]]
                        y = [l+0.5, l+0.5]
                        ax.plot(x, y, color='#7237f2', ls='',
                                marker='.', markersize=2, alpha=tick_alpha)

            # reset yticklabels
            if y_lim[0] < 100:
                step = 10
            elif y_lim[0] < 200:
                step = 20
            elif y_lim[0] < 500:
                step = 50
            elif y_lim[0] < 5000:
                step = 500
            elif y_lim[0] < 10000:
                step = 1000
            elif y_lim[0] >= 10000:
                step = 5000
            base_yticks = range(int(y_lim[-1]), int(y_lim[0]), int(step))
            base_yticks = [s for s in base_yticks]
            base_ylabels = [str(s) for s in base_yticks]

            dates = np.array(meta_stim_sub.index.get_level_values('date'))
            dates = dates[ori_inds]
            date_yticks = []
            date_label = []

            date_rel = meta.reset_index()['date'].unique()

            for day in np.unique(dates):

                # find number of inds needed to shift labels to put in middle of date block
                last_ind = np.where(dates == day)[0][-1]
                first_ind = np.where(dates == day)[0][0]
                shifter = np.round((last_ind - first_ind)/2)
                label_ind = last_ind - shifter

                # get your relative day number
                day_val = np.where(date_rel == day)[0][0] + 1  # add one to make it one-indexed

                # add a pad to keep labels left-justified
                if day_val < 10:
                    pad = '  '
                else:
                    pad = ''

                # if the date label and trial label inds are exactly the same
                # force the label info onto one line of text
                # label days with imaging day number
                if np.isin(label_ind, base_yticks):
                    # remove the existing ind and add a special label to end
                    good_tick = ~np.isin(base_yticks, label_ind)
                    base_yticks = [base_yticks[s] for s in range(len(good_tick))
                                   if good_tick[s]]
                    base_ylabels = [base_ylabels[s] for s in range(len(good_tick))
                                   if good_tick[s]]
                    dpad = '          '
                    dpad = dpad[0:(len('          ') - len(str(label_ind)*2))]
                    base_ylabels.append('Day ' + str(day_val) + dpad + str(label_ind))
                else:
                    base_ylabels.append('Day ' + str(day_val) + '          ' + pad)
                base_yticks.append(label_ind)

            ax.set_yticks(base_yticks)
            ax.set_yticklabels(base_ylabels, size=14)

            # reset xticklabels
            xticklabels = np.array([-1, 0, 1, 2, 3, 4, 5, 6])
            xticklabels = xticklabels[(xticklabels > times[0]) & (xticklabels < times[-1])]
            xticks = [np.where(times <= s)[0][-1] for s in xticklabels]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation='horizontal', size=16)

            # update count through loops
            count = count + 1

        # match vmin and max across plots (using cmax to choose cmap and cmin)
        scale_by = 0.7
        if match_clim and vmax is None:
            max_ind = np.nanargmax(cmax)
            cmin = cmin[max_ind]*scale_by
            cmax = cmax[max_ind]*scale_by
            ccmap = ccmap[max_ind]

            for ax in (ax1, ax2, ax3):
                ax.collections[0].set_clim(vmax=cmax, vmin=cmin)
                ax.collections[0].set_cmap(ccmap)
        save_path = os.path.join(save_dir, file_name)     
        plt.savefig(save_path, bbox_inches='tight')
        plt.close('all')


def fit_linear_template(
        mice=['OA27', 'OA67', 'OA32', 'OA34', 'CC175', 'OA36', 'OA26',
              'VF226'],
        words=['determine', 'pharmacology', 'pharmacology', 'pharmacology',
               'pharmacology', 'pharmacology', 'pharmacology', 'pharmacology'],
        trace_type='zscore_day',
        method='ncp_hals',
        cs='',
        warp=False,
        group_by='all2',
        nan_thresh=0.85,
        score_threshold=0.8
):

    for m, w in zip(mice, words):
        # load TCA models and data
        V, my_sorts_stim = load.groupday_tca_model(
                mouse=mouse,
                trace_type=trace_type,
                method=method,
                cs=cs,
                warp=warp,
                rank=rank,
                word=word,
                group_by=group_by,
                nan_thresh=nan_thresh,
                score_threshold=score_threshold,
                full_output=False,
                unsorted=True,
                verbose=False)
        V2, my_sorts_noise = load.groupday_tca_model(
                mouse=mouse,
                trace_type=trace_type,
                method=method,
                cs=cs,
                warp=warp,
                rank=rank,
                word=word_n,
                group_by=group_by,
                nan_thresh=nan_thresh,
                score_threshold=score_threshold,
                full_output=False,
                unsorted=True,
                verbose=False)
        meta_stim = load.groupday_tca_meta(
                mouse=mouse,
                trace_type=trace_type,
                method=method,
                cs=cs,
                warp=warp,
                word=word_s,
                group_by=group_by,
                nan_thresh=nan_thresh,
                score_threshold=score_threshold)
        meta_stim = utils.add_dprime_to_meta(meta_stim)
        input_stim = load.groupday_tca_input_tensor(
                mouse=mouse,
                trace_type=trace_type,
                method=method,
                cs=cs,
                warp=warp,
                word=word_s,
                group_by=group_by,
                nan_thresh=nan_thresh,
                score_threshold=score_threshold)
        # input_bhv = load.groupday_tca_bhv(
        #         mouse=mouse,
        #         trace_type=trace_type,
        #         method=method,
        #         cs=cs,
        #         warp=warp,
        #         word=word_s,
        #         group_by=group_by,
        #         nan_thresh=nan_thresh,
        #         score_threshold=score_threshold)

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
import scipy as sp


# useful lookup tables
color_dict = {'plus': [0.46, 0.85, 0.47, 1],
     'minus': [0.84, 0.12, 0.13, 1],
     'neutral': [0.28, 0.68, 0.93, 1],
     'learning': [34/255, 110/255, 54/255, 1],
     'reversal': [173/255, 38/255, 26/255, 1],
     'gray': [163/255, 163/255, 163/255, 1]}

lookup = {'OA27': {'plus': 270, 'minus': 135, 'neutral': 0},
     'VF226': {'plus': 0, 'minus': 270, 'neutral': 135},
     'OA67': {'plus': 0, 'minus': 270, 'neutral': 135},
     'OA32': {'plus': 135, 'minus': 0, 'neutral': 270},
     'OA34': {'plus': 270, 'minus': 135, 'neutral': 0},
     'OA36': {'plus': 0, 'minus': 270, 'neutral': 135},
     'OA26': {'plus': 270, 'minus': 135, 'neutral': 0}}

lookup_ori = {'OA27': {270: 'plus', 135: 'minus', 0: 'neutral'},
     'VF226': {0: 'plus', 270: 'minus', 135: 'neutral'},
     'OA67': {0: 'plus', 270: 'minus', 135: 'neutral'},
     'OA32': {135: 'plus', 0: 'minus', 270: 'neutral'},
     'OA34': {270: 'plus', 135: 'minus', 0: 'neutral'},
     'OA36': {0: 'plus', 270: 'minus', 135: 'neutral'},
     'OA26': {270: 'plus', 135: 'minus', 0: 'neutral'}}


# exponential fitting functions 
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


def _get_fitting_template_defaults(mouse, sigma=3, sec=15.5, normalize=True):
    """
    Helper function for convolving Gaussian kernel with onset, sustained 
    stimulus, stimulus offset, and ensure delivery period. These can then be
    used for simple linear regression or GLM. 
    """

    # preallocate Gaussian kernel convolution 
    gker = []

    # pick ranges of relevant time periods for convolution
    onset_r = np.arange(np.floor(sec), np.round(sec + sec/3), 1)
    if mouse in ['OA32', 'OA34', 'OA36']:
        sus_r = np.arange(np.floor(sec), np.round(sec*3), 1)
        off_r = np.arange(np.floor(sec*3), np.round(sec*3 + sec/3), 1)
        ensure_r = np.arange(np.floor(sec*3 + sec/3), np.round(sec*5), 1)
    else:
        sus_r = np.arange(np.floor(sec), np.round(sec*4), 1)
        off_r = np.arange(np.floor(sec*4), np.round(sec*4 + sec/3), 1)
        ensure_r = np.arange(np.floor(sec*4 + sec/3), np.round(sec*6), 1)
    ranges = [onset_r, sus_r, off_r, ensure_r]

    # convolve
    for i in ranges:
        i = [int(s) for s in i]
        starter = np.zeros((108))
        starter[i] = 1
        gker.append(sp.ndimage.gaussian_filter1d(starter, sigma, mode='constant', cval=0))

    # normalize filters
    if normalize:
        gker = [(s - np.min(s))/np.max(s) for s in gker]

    templates = np.vstack(gker).T

    return templates


def sustainedness_daily_mean_comp_plots(
        mice=['OA27', 'OA67', 'OA32', 'OA34', 'OA36', 'OA26',
              'VF226'],
        words=['christina', 'christina', 'christina', 'christina',
               'christina', 'christina', 'christina'],
        trace_type='zscore_day',
        method='ncp_hals',
        ks_thresh=1.31,
        t_or_wa='thresh', # thresh or weighted
        cs='',
        warp=False,
        group_by='all2',
        nan_thresh=0.95,
        score_threshold=0.8):
    """ 
    Giant, gross plotting script to make a zillion summary plots for 
    sustainedness and transientness across mice clustered according to 
    a TCA cell factor chosen by the user. 
    """

    # load in a full size tensor
    model_list = []
    tensor_list = []
    id_list = []
    bhv_list = []
    meta_list = []
    twords=['prints', 'horrible', 'horrible', 'horrible',
               'horrible', 'horrible', 'horrible'],
    tmice=['OA27', 'OA67', 'OA32', 'OA34', 'OA36', 'OA26', 'VF226']
    # return --> model, ids, tensor, meta, bhv
    for mouse, word in zip(tmice, twords):
        out = cas.load.load_all_groupday(
            mouse, word=word, with_model=False, nan_thresh=0.95) 
        model_list.append(out[0])
        tensor_list.append(out[2])
        id_list.append(out[1])
        bhv_list.append(out[4])
        meta_list.append(out[3])

    # load TCA models and data
    rank = 10
    words = ['christina']*len(mice)
    model_list = []
    model_id_list = []
    for mouse, word in zip(mice, words):
        V, my_sorts = cas.load.groupday_tca_model(
                mouse=mouse,
                trace_type='zscore_day',
                method='ncp_hals',
                cs='',
                rank=rank,
                word=word,
                group_by='l_vs_r1_tight',
                nan_thresh=0.95,
                score_threshold=0.8,
                full_output=False,
                unsorted=True,
                verbose=False)
        model_list.append(V)
        ids = cas.load.groupday_tca_ids(
                mouse=mouse,
                trace_type='zscore_day',
                method='ncp_hals',
                cs='',
                word=word,
                group_by='l_vs_r1_tight',
                nan_thresh=0.95,
                score_threshold=0.8)
        model_id_list.append(ids)

    # subset full size tensor to only include cells that were kept in tca model
    subset_tensor_list = []
    subset_model_list = []
    for t1, m1, l1, l2 in zip(tensor_list, model_list, id_list, model_id_list):
        survivor_boo = np.in1d(l1, l2)
        subset_tensor_list.append(t1[survivor_boo,:,:])
        kept_ids1 = l1[survivor_boo]
        survivor_boo = np.in1d(l2, l1)
        subset_model_list.append(m1.results[rank][0].factors[0][survivor_boo, :])
        kept_ids2 = l2[survivor_boo]
        if np.sum((kept_ids1-kept_ids2) != 0) > 0:
            print('Unmatched ids!!!')

    # build cell-weighted/-projected
    projection_list = []
    for c, V in enumerate(subset_model_list):
        map_by_comp = {}
        for ci in range(1,rank+1):
            weight_vec = V[:, ci-1]
            weight_map = np.zeros(subset_tensor_list[c].shape[1:])
            for tri in range(subset_tensor_list[c].shape[2]):
                trialx = weight_vec[:, None] * subset_tensor_list[c][:,:,tri]
                weight_map[:, tri] = bt.nanmean(trialx, axis=0)
            map_by_comp[ci] = weight_map
        projection_list.append(map_by_comp)
        if verbose:
            print('v1: Done mouse {}.'.format(c+1))
        
    # build cell-weighted/-projected
    thresh_list = []
    for c, V in enumerate(subset_model_list):
        map_by_comp = {}
        for ci in range(1,rank+1):
            thresh = np.std(V)*1
            weight_vec = deepcopy(V[:, ci-1])
            weight_vec[weight_vec <= thresh] = 0
            weight_vec[weight_vec > thresh] = 1
            if verbose:
                print('    included {} cells from comp {}'.format(np.sum(weight_vec), ci))
            weight_map = np.zeros(subset_tensor_list[c].shape[1:])
            for tri in range(subset_tensor_list[c].shape[2]):
                trialx = weight_vec[:, None] * subset_tensor_list[c][:,:,tri]
                weight_map[:, tri] = bt.nanmean(trialx, axis=0)
            map_by_comp[ci] = weight_map
        thresh_list.append(map_by_comp)
        if verbose:
            print('v2: Done mouse {}.'.format(c+1))

    # NNLS fitting 
    fits = {}
    daily_avg_dict = {}
    oris = [0, 135, 270]
    for mi, proj, meti in zip(mice, projection_list, meta_list):
        daily_avg_dict[mi] = {}
        A = _get_fitting_template_defaults(mi)
        all_ori = meti['orientation'].values
        all_days = meti.reset_index()['date'].values
        u_days = meti.reset_index()['date'].unique()
        firsty = _first100_bool(meti)
        fits[mi] = {}
        for c, comp_n in enumerate(proj.keys()):
            daily_avg_dict[mi][comp_n] = {}
            fits[mi]['component_{}'.format(c+1)] = {}
            ks_boo = comp_proj_ks_dict[mi][c, :] > ks_thresh 
            for ori in oris:
                cell_mat = np.zeros((proj[comp_n].shape[0], len(u_days)))
                for dc, day_i in enumerate(u_days):
                    day_ori_bool = (all_ori == ori) & (all_days == day_i) & firsty
                    cell_mat[:, dc] = np.nanmean(proj[comp_n][:, day_ori_bool & ks_boo], axis=1)
                all_tr = []
                for tr_n in range(cell_mat.shape[1]):
                    b = deepcopy(cell_mat[:, tr_n])
                    b[b < 0] = 0
                    if np.sum(np.isnan(b)) == len(b):
                        sp_ans = np.zeros(4)
                        sp_ans[:] = np.nan
                        sp_ans = [sp_ans]
                    else:
                        sp_ans = sp.optimize.nnls(A, b)
                    all_tr.append(sp_ans[0])
                fits[mi]['component_{}'.format(c+1)]['ori_{}'.format(ori)] = np.vstack(all_tr)
                fits[mi]['component_{}'.format(c+1)]['trial_inds_{}'.format(ori)] = np.where(all_ori == ori)[0]
                daily_avg_dict[mi][comp_n][ori] = cell_mat
                
        print('Making progress: {} done.'.format(mi))


    fits2 = {}
    daily_avg_dict2 = {}
    oris = [0, 135, 270]
    for mi, proj, meti in zip(mice, thresh_list, meta_list):
        daily_avg_dict2[mi] = {}
        A = _get_fitting_template_defaults(mi)
        all_ori = meti['orientation'].values
        all_days = meti.reset_index()['date'].values
        u_days = meti.reset_index()['date'].unique()
        firsty = _first100_bool(meti)
        fits2[mi] = {}
        for c, comp_n in enumerate(proj.keys()):
            fits2[mi]['component_{}'.format(c+1)] = {}
            ks_boo = comp_proj_ks_dict2[mi][c, :] > ks_thresh 
            for ori in oris:
    #             cell_mat = proj[comp_n][:, all_ori == ori]
                cell_mat = np.zeros((proj[comp_n].shape[0], len(u_days)))
                for dc, day_i in enumerate(u_days):
                    day_ori_bool = (all_ori == ori) & (all_days == day_i) & firsty
                    cell_mat[:, dc] = np.nanmean(proj[comp_n][:, day_ori_bool & ks_boo], axis=1)
                all_tr = []
                for tr_n in range(cell_mat.shape[1]):
                    b = deepcopy(cell_mat[:, tr_n])
                    b[b < 0] = 0
                    if np.sum(np.isnan(b)) == len(b):
                        sp_ans = np.zeros(4)
                        sp_ans[:] = np.nan
                        sp_ans = [sp_ans]
                    else:
                        sp_ans = sp.optimize.nnls(A, b)
                    all_tr.append(sp_ans[0])
                fits2[mi]['component_{}'.format(c+1)]['ori_{}'.format(ori)] = np.vstack(all_tr)
                fits2[mi]['component_{}'.format(c+1)]['trial_inds_{}'.format(ori)] = np.where(all_ori == ori)[0]
                daily_avg_dict2[mi][ori] = cell_mat

        print('Making progress: {} done.'.format(mi))


        alpha = 0.7
    ksthresh = 1.31 #35
    dp_threshold = 2
    t_or_wa = 'thresh'

    folder = 'Sustainedness index plots daily thresh fit on means'
    # folder = 'Sustainedness index plots daily thresh mean of trials fits'
    if not os.path.isdir(folder): os.mkdir(folder)
    save_dir = os.path.join(folder, 'Sustainedness plots')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)

    if t_or_wa == 'thresh':
        which_fits = deepcopy(fits2)
    #     which_ks = deepcopy(proj_ks_dict2)
    elif t_or_wa == 'weighted_avg':
        which_fits = deepcopy(fits)
    #     which_ks = deepcopy(proj_ks_dict)

    for ori in ['plus', 'minus', 'neutral']:
        ti_cont, mi_cont, ci_cont = [], [], []
        ti_norm_cont = []
        for mi, meti, V in zip(mice, meta_list, model_list):
            
            # add dprime to meti
            meti = cas.utils.add_dprime_to_meta(meti)
            
            # get indices for stimulus period
            tstart = int(np.floor(15.5))
            tend = int(np.floor(15.5*1.5))
            if mi in ['OA32', 'OA34', 'OA36']:
                send = int(np.floor(15.5*3))
            else:
                send = int(np.floor(15.5*4))
            sstart = int(np.floor(send-15.5/2))
            
            # calculate change indices for days and reversal/learning
            dates = meti.reset_index()['date']
            udays = {d: c for c, d in enumerate(np.unique(dates))}
            ndays = np.diff([udays[i] for i in np.unique(dates)])
            day_x = np.where(ndays)[0] + 0.5

            # get your learning and reversal start indices 
            rev_ind = np.where(meti['learning_state'].isin(['learning']).values)[0][-1]
            rev_ind = np.where(np.unique(dates) == dates[rev_ind])[0] + 0.5
            if np.sum(meti['learning_state'].isin(['naive']).values) != 0:
                lear_ind = np.where(meti['learning_state'].isin(['naive']).values)[0][-1]
                lear_ind = np.where(np.unique(dates) == dates[lear_ind])[0] + 0.5
            else:
                lear_ind = 0 - 0.5

            fig = plt.figure(figsize=(10, 3))
            gs = fig.add_gridspec(100, 100)
            ax1 = fig.add_subplot(gs[:, :30])
            sns.despine()
            ax2 = fig.add_subplot(gs[:, 40:])
            sns.despine()

            temp_fac = V.results[rank][0].factors[1][:, :]
            ax1.plot(temp_fac, linewidth=2)
            ax2.set_title('temporal factor')

            first_boo = _first100_bool(meti)
            all_means3 = np.zeros((1, 5, len(which_fits[mi].keys())))
            all_means3[:] = np.nan
            for comp_n in range(1, len(which_fits[mi].keys())+1):

                mi_cont.append(mi)
                ci_cont.append(comp_n)
                
                # don't consider offset components
                temp_fac = V.results[rank][0].factors[1][:, comp_n-1]
                if np.argmax(temp_fac) > send:
                    continue
                
                ori_type = 'ori_{}'.format(lookup[mi][ori])
                trial_type = 'trial_inds_{}'.format(lookup[mi][ori])
                fit_mat = which_fits[mi]['component_{}'.format(comp_n)][ori_type]
                y_vec = fit_mat[:,1]/(fit_mat[:,1] + fit_mat[:,0])

                stage_labels = []
                c_day = 0
                for n_day in meta_list[0]['learning_state'].unique(): # forcing to include all stages
                    meti_d = meti.groupby('date').max()
                    day_bool = meti_d['learning_state'].isin([n_day]).values
                    if n_day == 'naive':
                        total_bool = day_bool #& (y_vec < 1)
                        stage_labels.append(n_day)
                            # if there are no trials of a given type skip
                        if np.sum(total_bool) == 0:
                            c_day += 1
                            continue

                        if np.sum(~np.isnan(y_vec)) <= len(y_vec)/3:
                            c_day += 1
                            continue
                        # subtract mean trace from each cell
                        all_means3[0, c_day, comp_n-1] = np.nanmean(y_vec[total_bool])
                        c_day += 1
                    else:
                        for dpi in ['low_dp', 'high_dp']:
                            if dpi == 'high_dp':
                                dp_bool = meti_d['dprime'].values >= dp_threshold
                            elif dpi == 'low_dp':
                                dp_bool = meti_d['dprime'].values < dp_threshold
                            total_bool = day_bool & dp_bool #& (y_vec < 1) #& ori_bool

                            stage_labels.append('{} {}'.format(dpi, n_day))
                            # if there are no trials of a given type skip
                            if np.sum(total_bool) == 0:
                                print('Skipped empty for: {} {}'.format(dpi, n_day))
                                c_day += 1
                                continue
                            if np.sum(~np.isnan(y_vec)) <= len(y_vec)/3:
                                c_day += 1
                                continue

                            # subtract mean trace from each cell
                            all_means3[0, c_day, comp_n-1] = np.nanmean(y_vec[total_bool])
                            c_day += 1

            tuning_index = all_means3.squeeze()
            ti_cont.append(tuning_index)  #for continued analysis 
            tuning_index = tuning_index/tuning_index[2, :]
            ti_norm_cont.append(tuning_index)  #for continued analysis 

            ax2.plot(tuning_index)
            ax2.legend(bbox_to_anchor=(1.05, 1), labels=['component {}'.format(k) for k in thresh_list[0].keys()])
            ax2.set_title('{}: Sustained responses across stages of learning'.format(mi))
            ax1.set_title('{}: TCA component\n thresholded averages'.format(words[0]))
            ax1.set_xlabel('time in trial')
            ax1.set_ylabel('factor weight')
            stim_window_ticks = np.arange(0, 108, 15.5)
            stim_window_labels = np.arange(-1, 7, 1)
            ax1.set_xticks(stim_window_ticks)
            ax1.set_xticklabels(labels=stim_window_labels)
            ax2.set_xlabel('learning stage')
            ax2.set_ylabel(r'sustainedness ($\beta_{sus}$/($\beta_{sus}$+$\beta_{trans}$))')
            stage_ticks = np.arange(0, 5, 1)
            ax2.set_xticks(stage_ticks)
            ax2.set_xticklabels(labels=stage_labels, rotation=45, ha='right')
            plt.savefig('.//{}//Sustainedness plots//{}_{}_{}_{}_sus_avg_stages.pdf'.format(folder, t_or_wa, mi, ori, words[0]), bbox_inches='tight')
    #         plt.close('all')
        
        stacked_nsus_idx = np.vstack([s.T for s in ti_norm_cont])
        stacked_sus_idx = np.vstack([s.T for s in ti_cont])
        index = pd.MultiIndex.from_arrays([mi_cont, ci_cont], names=['mouse', 'component'])
        sus_df = pd.DataFrame(data=stacked_sus_idx, columns=stage_labels, index=index)
        norm_sus_df = pd.DataFrame(data=stacked_nsus_idx, columns=stage_labels, index=index)
        
        plt.figure()
        dsus = ((sus_df['high_dp learning'] - sus_df['low_dp learning'])*-1)/(sus_df['high_dp learning'] + sus_df['low_dp learning'])
        # dsusr = (norm_sus_df['high_dp learning'] - norm_sus_df['low_dp reversal1'])*-1
        tester = pd.DataFrame(data=dsus.T, columns=['delta_sus'], index=sus_df.index)
        sns.regplot(x=sus_df['high_dp learning'], y=tester['delta_sus'], marker='.')
    #     sns.scatterplot(x=sus_df['high_dp learning'], y=tester['delta_sus'], hue=sus_df.reset_index()['mouse'].values)
        sns.scatterplot(x=sus_df['high_dp learning'], y=tester['delta_sus'], s=100,
                        hue=['component {}'.format(s) for s in sus_df.reset_index()['component'].values],
                        style=sus_df.reset_index()['mouse'].values)
        plt.legend(bbox_to_anchor=(1.05, 1))
        # plt.ylim([-0.2, 0.2])
        keep_bool = ~np.isnan(tester['delta_sus'])
        r = sp.stats.pearsonr(sus_df['high_dp learning'].loc[keep_bool], tester['delta_sus'].loc[keep_bool])
        plt.ylabel('$\Delta$SI\n -1*($SI_{high dp}-SI_{low dp})/(SI_{high dp}+SI_{low dp}$)')
        plt.xlabel('sustainedness\n high dprime')
        plt.title('{}: '.format(ori) + 'Sustainedness vs $\Delta$$SI_{lrn-lrn}$' + ': R={}, {}={}'.format(round(r[0], 2), '$-1*log_{10}(p)$', round(-1*np.log10(r[1]), 3)))
        plt.savefig('.//{}//{}_{}_{}_SI_learnhdp_v_learnldp.pdf'.format(folder, t_or_wa, ori, words[0]), bbox_inches='tight')

        plt.figure()
        dsus = ((sus_df['high_dp learning'] - sus_df['naive'])*-1)/(sus_df['high_dp learning'] + sus_df['naive'])
        # dsusr = (norm_sus_df['high_dp learning'] - norm_sus_df['low_dp reversal1'])*-1
        tester = pd.DataFrame(data=dsus.T, columns=['delta_sus'], index=sus_df.index)
        sns.regplot(x=sus_df['high_dp learning'], y=tester['delta_sus'], marker='.')
    #     sns.scatterplot(x=sus_df['high_dp learning'], y=tester['delta_sus'], hue=sus_df.reset_index()['mouse'].values)
        sns.scatterplot(x=sus_df['high_dp learning'], y=tester['delta_sus'], s=100,
                        hue=['component {}'.format(s) for s in sus_df.reset_index()['component'].values],
                        style=sus_df.reset_index()['mouse'].values)
        plt.legend(bbox_to_anchor=(1.05, 1))
        # plt.ylim([-0.2, 0.15])
        keep_bool = ~np.isnan(tester['delta_sus'])
        r = sp.stats.pearsonr(sus_df['high_dp learning'].loc[keep_bool], tester['delta_sus'].loc[keep_bool])
        plt.ylabel('$\Delta$SI\n -1*($SI_{high dp}-SI_{naive})/(SI_{high dp}+SI_{naive}$)')
        plt.xlabel('sustainedness\n high dprime learning')
        plt.title('{}: '.format(ori) + 'Sustainedness vs $\Delta$$SI_{lrn-naive}$' + ': R={}, {}={}'.format(round(r[0], 2), '$-1*log_{10}(p)$', round(-1*np.log10(r[1]), 3)))
        plt.savefig('.//{}//{}_{}_{}_SI_learnhdp_v_naive.pdf'.format(folder, t_or_wa, ori, words[0]), bbox_inches='tight')
        
        plt.figure()
        dsus = ((sus_df['high_dp learning'] - sus_df['low_dp reversal1'])*-1)/(sus_df['high_dp learning'] + sus_df['low_dp reversal1'])
        # dsusr = (norm_sus_df['high_dp learning'] - norm_sus_df['low_dp reversal1'])*-1
        tester = pd.DataFrame(data=dsus.T, columns=['delta_sus'], index=sus_df.index)
        sns.regplot(x=sus_df['high_dp learning'], y=tester['delta_sus'], marker='.')
    #     sns.scatterplot(x=sus_df['high_dp learning'], y=tester['delta_sus'], hue=sus_df.reset_index()['mouse'].values)
        sns.scatterplot(x=sus_df['high_dp learning'], y=tester['delta_sus'], s=100,
                        hue=['component {}'.format(s) for s in sus_df.reset_index()['component'].values],
                        style=sus_df.reset_index()['mouse'].values)
        plt.legend(bbox_to_anchor=(1.05, 1))
        # plt.ylim([-0.2, 0.2])
        keep_bool = ~np.isnan(tester['delta_sus'])
        r = sp.stats.pearsonr(sus_df['high_dp learning'].loc[keep_bool], tester['delta_sus'].loc[keep_bool])
        plt.ylabel('$\Delta$SI\n -1*($SI_{high dp}-SI_{reversal})/(SI_{high dp}+SI_{reversal}$)')
        plt.xlabel('sustainedness\n high dprime learning')
        plt.title('{}: '.format(ori) + 'Sustainedness vs $\Delta$$SI_{lrn-rev}$' + ': R={}, {}={}'.format(round(r[0], 2), '$-1*log_{10}(p)$', round(-1*np.log10(r[1]), 3)))
        plt.savefig('.//{}//{}_{}_{}_SI_learnhdp_v_revldp.pdf'.format(folder, t_or_wa, ori, words[0]), bbox_inches='tight')
        
        plt.figure()
        dsus = ((sus_df['high_dp reversal1'] - sus_df['low_dp reversal1'])*-1)/(sus_df['high_dp reversal1'] + sus_df['low_dp reversal1'])
        # dsusr = (norm_sus_df['high_dp learning'] - norm_sus_df['low_dp reversal1'])*-1
        tester = pd.DataFrame(data=dsus.T, columns=['delta_sus'], index=sus_df.index)
        sns.regplot(x=sus_df['high_dp reversal1'], y=tester['delta_sus'], marker='.')
    #     sns.scatterplot(x=sus_df['high_dp reversal1'], y=tester['delta_sus'], hue=sus_df.reset_index()['mouse'].values)
        sns.scatterplot(x=sus_df['high_dp reversal1'], y=tester['delta_sus'], s=100,
                        hue=['component {}'.format(s) for s in sus_df.reset_index()['component'].values],
                        style=sus_df.reset_index()['mouse'].values)
        plt.legend(bbox_to_anchor=(1.05, 1))
        # plt.ylim([-0.2, 0.2])
        keep_bool = ~np.isnan(tester['delta_sus'])
        r = sp.stats.pearsonr(sus_df['high_dp reversal1'].loc[keep_bool], tester['delta_sus'].loc[keep_bool])
        plt.ylabel('$\Delta$SI\n -1*($SI_{high dp rev}-SI_{low dp rev})/(SI_{high dp rev}+SI_{low dp rev}$)')
        plt.xlabel('sustainedness\n high dprime reversal')
        plt.title('{}: '.format(ori) + 'Sustainedness vs $\Delta$$SI_{rev-rev}$' + ': R={}, {}={}'.format(round(r[0], 2), '$-1*log_{10}(p)$', round(-1*np.log10(r[1]), 3)))
        plt.savefig('.//{}//{}_{}_{}_SI_revhdp_v_revldp.pdf'.format(folder, t_or_wa, ori, words[0]), bbox_inches='tight')


    stages = ['naive', 'low_dp learning', 'high_dp learning', 'low_dp reversal1', 'high_dp reversal1']
    folder = 'Sustainedness plots component thresh scatter'
    tag = 'v1'
    primary_stage = 'high_dp learning'
    secondary_stages = [s for s in stages if s != primary_stage]

    if not os.path.isdir(folder): os.mkdir(folder)
    save_dir = os.path.join(folder, primary_stage)
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
        
    for stages_to_comp in secondary_stages:
        plt.figure()
        # for ori in ['plus', 'minus', 'neutral']:
        #     ori_vec = total_sus.reset_index()['initial CS'].values
        # oriboo = ori_vec == 'initial {}'.format(ori)
        sns.scatterplot(x=total_sus[primary_stage].values, 
                        y=total_sus[stages_to_comp].values,
                        palette=hue_dict,
                        hue=total_sus.reset_index()['initial CS'].values,
                        style=total_sus.reset_index()['mouse'].values)
        xmax = np.nanmax(total_sus[primary_stage].values)
        ymax = np.nanmax(total_sus[stages_to_comp].values)
        maxval = np.nanmax([xmax, ymax])
        plt.plot([0, maxval], [0, maxval], 'k--')
        plt.xlabel(primary_stage)
        plt.ylabel(stages_to_comp)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.ylabel('sustainedness coeff {}'.format(stages_to_comp), size=14)
        plt.xlabel('sustainedness coeff {}'.format(primary_stage), size=14)
        plt.title('Sustainedness coeff {} vs Sustainedness coeff {}\n\n'.format(primary_stage, stages_to_comp), size=16)
        plt.savefig('.//{}//{}_{}_{}_sus_{}_v_{}.pdf'.format(save_dir, t_or_wa, tag, words[0], primary_stage, stages_to_comp),
                    bbox_inches='tight')


        for ori in ['plus', 'minus', 'neutral']:
            plt.figure()
        #     f, ax = plt.subplots(figsize=(7, 7))
        #     ax.set(xscale="log", yscale="log")
            ori_vec = total_sus.reset_index()['initial CS'].values
            oriboo = ori_vec == 'initial {}'.format(ori)
            sns.scatterplot(x=total_sus[primary_stage].iloc[oriboo].values, 
                            y=total_sus[stages_to_comp].iloc[oriboo].values,
                            color=hue_dict['initial {}'.format(ori)],
                            style=total_sus.reset_index()['mouse'].iloc[oriboo].values)
            plt.plot([0, maxval], [0, maxval], 'k--')
        #     plt.yscale('log')
        #     plt.xscale('log')
        #     plt.ylim([-5, None])
        #     plt.xlim([-5, None])
            plt.xlabel(primary_stage)
            plt.ylabel(stages_to_comp)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.ylabel('sustainedness coeff {}'.format(stages_to_comp), size=14)
            plt.xlabel('sustainedness coeff {}'.format(primary_stage), size=14)
            plt.title('Sustainedness coeff {} vs Sustainedness coeff {}\n\n'.format(primary_stage, stages_to_comp), size=16)
            plt.savefig('.//{}//{}_{}_{}_sus_initial_{}_{}_v_{}.pdf'.format(save_dir, t_or_wa, tag, words[0], ori, primary_stage, stages_to_comp),
                        bbox_inches='tight')

    stages = ['naive', 'low_dp learning', 'high_dp learning', 'low_dp reversal1', 'high_dp reversal1']
    folder = 'Transient plots component thresh scatter'
    tag = 'v1'
    primary_stage = 'high_dp learning'
    secondary_stages = [s for s in stages if s != primary_stage]

    if not os.path.isdir(folder): os.mkdir(folder)
    save_dir = os.path.join(folder, primary_stage)
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
        
    for stages_to_comp in secondary_stages:
        plt.figure()
        sns.scatterplot(x=total_sus[primary_stage].values, 
                        y=total_sus[stages_to_comp].values,
                        palette=hue_dict,
                        hue=total_sus.reset_index()['initial CS'].values,
                        style=total_sus.reset_index()['mouse'].values)
        xmax = np.nanmax(total_sus[primary_stage].values)
        ymax = np.nanmax(total_sus[stages_to_comp].values)
        maxval = np.nanmax([xmax, ymax])
        plt.plot([0, maxval], [0, maxval], 'k--')
        plt.xlabel(primary_stage)
        plt.ylabel(stages_to_comp)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.ylabel('Transientness coeff {}'.format(stages_to_comp), size=14)
        plt.xlabel('Transientness coeff {}'.format(primary_stage), size=14)
        plt.title('Transientness coeff {} vs Transientness coeff {}\n\n'.format(primary_stage, stages_to_comp), size=16)
        plt.savefig('.//{}//{}_{}_{}_trans_{}_v_{}.pdf'.format(save_dir, t_or_wa, tag, words[0], primary_stage, stages_to_comp),
                    bbox_inches='tight')


        for ori in ['plus', 'minus', 'neutral']:
            plt.figure()
            ori_vec = total_sus.reset_index()['initial CS'].values
            oriboo = ori_vec == 'initial {}'.format(ori)
            sns.scatterplot(x=total_sus[primary_stage].iloc[oriboo].values, 
                            y=total_sus[stages_to_comp].iloc[oriboo].values,
                            color=hue_dict['initial {}'.format(ori)],
                            style=total_sus.reset_index()['mouse'].iloc[oriboo].values)
            plt.plot([0, maxval], [0, maxval], 'k--')
        #     plt.ylim([-5, None])
        #     plt.xlim([-5, None])
            plt.xlabel(primary_stage)
            plt.ylabel(stages_to_comp)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.ylabel('Transientness coeff {}'.format(stages_to_comp), size=14)
            plt.xlabel('Transientness coeff {}'.format(primary_stage), size=14)
            plt.title('Transientness coeff {} vs Transientness coeff {}\n\n'.format(primary_stage, stages_to_comp), size=16)
            plt.savefig('.//{}//{}_{}_{}_trans_initial_{}_{}_v_{}.pdf'.format(save_dir, t_or_wa, tag, words[0], ori, primary_stage, stages_to_comp),
                        bbox_inches='tight')

    folder = 'Sustainedness index plots daily thresh fit on means grouped PMN'
    # folder = 'Sustainedness index plots daily thresh mean of trials fits'
    if not os.path.isdir(folder): os.mkdir(folder)
    save_dir = os.path.join(folder, 'Sustainedness plots christina ylim')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    tag = 'v2'

    hue_dict = {}
    for ori in ['plus', 'minus', 'neutral']:
        hue_dict['initial {}'.format(ori)] = color_dict[ori]

    plt.figure()
    dsus = total_sus['high_dp learning'] - total_sus['low_dp learning']
    ori_vec = total_sus.reset_index()['initial CS'].values
    tester = pd.DataFrame(data=dsus.T, columns=['delta_sus'], index=total_sus.index)
    for ori in ['plus', 'minus', 'neutral']:
        oriboo = ori_vec == 'initial {}'.format(ori)
        regtester = tester.iloc[oriboo]
        regori = total_sus['high_dp learning'].iloc[oriboo]
        sns.regplot(x=total_sus['high_dp learning'].iloc[oriboo], y=regtester['delta_sus'], marker='.', color=color_dict[ori], dropna=True)
    sns.scatterplot(x=total_sus['high_dp learning'], y=tester['delta_sus'], s=100,
                    hue=total_sus.reset_index()['initial CS'].values, palette=hue_dict,
                    style=total_sus.reset_index()['mouse'].values)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim([0, 1])
    # plt.ylim([-1, 1])
    keep_bool = ~np.isnan(tester['delta_sus'])
    rs, ps = [], []
    for ori in ['plus', 'minus', 'neutral']:
        oriboo = ori_vec == 'initial {}'.format(ori)
        r = sp.stats.pearsonr(total_sus['high_dp learning'].loc[keep_bool & oriboo], tester['delta_sus'].loc[keep_bool & oriboo])
        ps.append(r[1])
        rs.append(r[0])
    plt.ylabel('$\Delta$SI\n $SI_{high dp}-SI_{low dp}$')
    plt.xlabel('sustainedness\n high dprime')
    plt.title('Sustainedness vs $\Delta$$SI_{lrn-lrn}$'
              + ':\n $_i$plus R={}, {}={}'.format(round(rs[0], 2), '$-1*log_{10}(p)$', round(-1*np.log10(ps[0]), 3))
              + '\n $_i$minus R={}, {}={}'.format(round(rs[1], 2), '$-1*log_{10}(p)$', round(-1*np.log10(ps[1]), 3))
              + '\n $_i$neutral R={}, {}={}'.format(round(rs[2], 2), '$-1*log_{10}(p)$', round(-1*np.log10(ps[2]), 3))
             )
    plt.savefig('.//{}//{}_{}_{}_SI_learnhdp_v_learnldp.pdf'.format(folder, t_or_wa, tag, words[0]), bbox_inches='tight')

    plt.figure()
    dsus = total_sus['high_dp learning'] - total_sus['naive']
    ori_vec = total_sus.reset_index()['initial CS'].values
    tester = pd.DataFrame(data=dsus.T, columns=['delta_sus'], index=total_sus.index)
    for ori in ['plus', 'minus', 'neutral']:
        oriboo = ori_vec == 'initial {}'.format(ori)
        regtester = tester.iloc[oriboo]
        regori = total_sus['high_dp learning'].iloc[oriboo]
        sns.regplot(x=total_sus['high_dp learning'].iloc[oriboo], y=regtester['delta_sus'], marker='.', color=color_dict[ori], dropna=True)
    sns.scatterplot(x=total_sus['high_dp learning'], y=tester['delta_sus'], s=100,
                    hue=total_sus.reset_index()['initial CS'].values, palette=hue_dict,
                    style=total_sus.reset_index()['mouse'].values)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim([0, 1])
    # plt.ylim([-1, 1])
    keep_bool = ~np.isnan(tester['delta_sus'])
    rs, ps = [], []
    for ori in ['plus', 'minus', 'neutral']:
        oriboo = ori_vec == 'initial {}'.format(ori)
        r = sp.stats.pearsonr(total_sus['high_dp learning'].loc[keep_bool & oriboo], tester['delta_sus'].loc[keep_bool & oriboo])
        ps.append(r[1])
        rs.append(r[0])
    plt.ylabel('$\Delta$SI\n $SI_{high dp}-SI_{naive}$')
    plt.xlabel('sustainedness\n high dprime learning')
    plt.title('Sustainedness vs $\Delta$$SI_{lrn-naive}$'
              + ':\n $_i$plus R={}, {}={}'.format(round(rs[0], 2), '$-1*log_{10}(p)$', round(-1*np.log10(ps[0]), 3))
              + '\n $_i$minus R={}, {}={}'.format(round(rs[1], 2), '$-1*log_{10}(p)$', round(-1*np.log10(ps[1]), 3))
              + '\n $_i$neutral R={}, {}={}'.format(round(rs[2], 2), '$-1*log_{10}(p)$', round(-1*np.log10(ps[2]), 3))
             )
    plt.savefig('.//{}//{}_{}_{}_SI_learnhdp_v_naive.pdf'.format(folder, t_or_wa, tag, words[0]), bbox_inches='tight')

    plt.figure()
    dsus = total_sus['high_dp learning'] - total_sus['low_dp reversal1']
    ori_vec = total_sus.reset_index()['initial CS'].values
    tester = pd.DataFrame(data=dsus.T, columns=['delta_sus'], index=total_sus.index)
    for ori in ['plus', 'minus', 'neutral']:
        oriboo = ori_vec == 'initial {}'.format(ori)
        regtester = tester.iloc[oriboo]
        regori = total_sus['high_dp learning'].iloc[oriboo]
        sns.regplot(x=total_sus['high_dp learning'].iloc[oriboo], y=regtester['delta_sus'], marker='.', color=color_dict[ori], dropna=True)
    sns.scatterplot(x=total_sus['high_dp learning'], y=tester['delta_sus'], s=100,
                    hue=total_sus.reset_index()['initial CS'].values, palette=hue_dict,
                    style=total_sus.reset_index()['mouse'].values)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim([0, 1])
    # plt.ylim([-1, 1])
    keep_bool = ~np.isnan(tester['delta_sus'])
    rs, ps = [], []
    for ori in ['plus', 'minus', 'neutral']:
        oriboo = ori_vec == 'initial {}'.format(ori)
        r = sp.stats.pearsonr(total_sus['high_dp learning'].loc[keep_bool & oriboo], tester['delta_sus'].loc[keep_bool & oriboo])
        ps.append(r[1])
        rs.append(r[0])
    plt.ylabel('$\Delta$SI\n $SI_{high dp}-SI_{reversal}$')
    plt.xlabel('sustainedness\n high dprime learning')
    plt.title('Sustainedness vs $\Delta$$SI_{lrn-rev}$'
              + ':\n $_i$plus R={}, {}={}'.format(round(rs[0], 2), '$-1*log_{10}(p)$', round(-1*np.log10(ps[0]), 3))
              + '\n $_i$minus R={}, {}={}'.format(round(rs[1], 2), '$-1*log_{10}(p)$', round(-1*np.log10(ps[1]), 3))
              + '\n $_i$neutral R={}, {}={}'.format(round(rs[2], 2), '$-1*log_{10}(p)$', round(-1*np.log10(ps[2]), 3))
             )
    plt.savefig('.//{}//{}_{}_{}_SI_learnhdp_v_revldp.pdf'.format(folder, t_or_wa, tag, words[0]), bbox_inches='tight')

    plt.figure()
    dsus = total_sus['high_dp reversal1'] - total_sus['low_dp reversal1']
    ori_vec = total_sus.reset_index()['initial CS'].values
    tester = pd.DataFrame(data=dsus.T, columns=['delta_sus'], index=total_sus.index)
    for ori in ['plus', 'minus', 'neutral']:
        oriboo = ori_vec == 'initial {}'.format(ori)
        regtester = tester.iloc[oriboo]
        regori = total_sus['high_dp reversal1'].iloc[oriboo]
        sns.regplot(x=total_sus['high_dp reversal1'].iloc[oriboo], y=regtester['delta_sus'], marker='.', color=color_dict[ori], dropna=True)
    sns.scatterplot(x=total_sus['high_dp reversal1'], y=tester['delta_sus'], s=100,
                    hue=total_sus.reset_index()['initial CS'].values, palette=hue_dict,
                    style=total_sus.reset_index()['mouse'].values)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim([0, 1])
    # plt.ylim([-1, 1])
    keep_bool = ~np.isnan(tester['delta_sus'])
    rs, ps = [], []
    for ori in ['plus', 'minus', 'neutral']:
        oriboo = ori_vec == 'initial {}'.format(ori)
        r = sp.stats.pearsonr(total_sus['high_dp reversal1'].loc[keep_bool & oriboo], tester['delta_sus'].loc[keep_bool & oriboo])
        ps.append(r[1])
        rs.append(r[0])
    plt.ylabel('$\Delta$SI\n $SI_{high dp rev}-SI_{low dp rev}$')
    plt.xlabel('sustainedness\n high dprime reversal')
    plt.title('Sustainedness vs $\Delta$$SI_{rev-rev}$'
              + ':\n $_i$plus R={}, {}={}'.format(round(rs[0], 2), '$-1*log_{10}(p)$', round(-1*np.log10(ps[0]), 3))
              + '\n $_i$minus R={}, {}={}'.format(round(rs[1], 2), '$-1*log_{10}(p)$', round(-1*np.log10(ps[1]), 3))
              + '\n $_i$neutral R={}, {}={}'.format(round(rs[2], 2), '$-1*log_{10}(p)$', round(-1*np.log10(ps[2]), 3))
             )
    plt.savefig('.//{}//{}_{}_{}_SI_revhdp_v_revldp.pdf'.format(folder, t_or_wa, tag, words[0]), bbox_inches='tight')

    for ori in ['plus', 'minus', 'neutral']:
        plt.figure()
        dsus = total_sus['high_dp learning'] - total_sus['low_dp reversal1']
        ori_vec = total_sus.reset_index()['initial CS'].values
        tester = pd.DataFrame(data=dsus.T, columns=['delta_sus'], index=total_sus.index)
        # for ori in ['plus', 'minus', 'neutral']:
        oriboo = ori_vec == 'initial {}'.format(ori)
        regtester = tester.iloc[oriboo]
        regori = total_sus['high_dp learning'].iloc[oriboo]
        sns.regplot(x=total_sus['high_dp learning'].iloc[oriboo], y=regtester['delta_sus'], marker='.', color=color_dict[ori], dropna=True)
        sns.scatterplot(x=total_sus['high_dp learning'].iloc[oriboo], y=regtester['delta_sus'], s=100,
                        hue=total_sus.iloc[oriboo, :].reset_index()['initial CS'].values, palette=hue_dict,
                        style=total_sus.iloc[oriboo, :].reset_index()['mouse'].values)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlim([0, 1])
        plt.ylim([-0.67, 0.5])
        keep_bool = ~np.isnan(tester['delta_sus'])
        rs, ps = [], []
        plt.ylabel('$\Delta$SI\n $SI_{high dp}-SI_{reversal}$')
        plt.xlabel('sustainedness\n high dprime learning')
        plt.title('Sustainedness vs $\Delta$$SI_{lrn-rev}$')
#     plt.savefig('.//{}//{}_{}_{}_SI_learnhdp_v_revldp_{}.pdf'.format(folder, t_or_wa, tag, words[0], ori), bbox_inches='tight')






def NNLS_sustainedness_scatter():

    # load in a full size tensor
    model_list = []
    tensor_list = []
    id_list = []
    bhv_list = []
    meta_list = []
    # return --> model, ids, tensor, meta, bhv
    for mouse, word in zip(mice, words):
        out = cas.load.load_all_groupday(
            mouse, word=word, with_model=False, nan_thresh=0.95) 
        model_list.append(out[0])
        tensor_list.append(out[2])
        id_list.append(out[1])
        bhv_list.append(out[4])
        meta_list.append(out[3])

def fit_linear_template(
        mice=['OA27', 'OA67', 'OA32', 'OA34', 'OA36', 'OA26',
              'VF226'],
        words=['determine', 'pharmacology', 'pharmacology', 'pharmacology',
               'pharmacology', 'pharmacology', 'pharmacology'],
        trace_type='zscore_day',
        method='ncp_hals',
        cs='',
        warp=False,
        group_by='all2',
        nan_thresh=0.85,
        score_threshold=0.8
):
    model_list = []
    meta_list = []
    input_list = []
    for mi, wi in zip(mice, words):
        # load TCA models and data
        V, my_sorts = load.groupday_tca_model(
                mouse=mi,
                trace_type=trace_type,
                method=method,
                cs=cs,
                warp=warp,
                rank=rank,
                word=wi,
                group_by=group_by,
                nan_thresh=nan_thresh,
                score_threshold=score_threshold,
                full_output=False,
                unsorted=True,
                verbose=False)
        meta = load.groupday_tca_meta(
                mouse=mi,
                trace_type=trace_type,
                method=method,
                cs=cs,
                warp=warp,
                word=wi,
                group_by=group_by,
                nan_thresh=nan_thresh,
                score_threshold=score_threshold)
        meta = utils.add_dprime_to_meta(meta)
        input_tensor = load.groupday_tca_input_tensor(
                mouse=mi,
                trace_type=trace_type,
                method=method,
                cs=cs,
                warp=warp,
                word=wi,
                group_by=group_by,
                nan_thresh=nan_thresh,
                score_threshold=score_threshold)
        model_list.append(V)
        meta_list.append(meta)
        input_list.append(input_tensor)
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

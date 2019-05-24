"""Functions for plotting tca decomp."""
import os
import flow
import pool
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import tensortools as tt
import seaborn as sns
import pandas as pd
from copy import deepcopy
from .. import df
from .. import tca
from .. import paths
from .. import utils
import warnings


"""
----------------------------- SETS OF PLOTS -----------------------------
"""


def groupday_shortlist(
        mouse,
        trace_type='zscore_day',
        method='ncp_bcd',
        cs='',
        warp=False,
        word=None,
        group_by='all',
        nan_thresh=0.85,
        verbose=False):

    groupday_varex_summary(mouse, trace_type=trace_type, method=method, cs=cs,
                           warp=warp, word=word, group_by=group_by,
                           nan_thresh=nan_thresh, verbose=verbose)
    groupday_factors_annotated(mouse, trace_type=trace_type, method=method,
                               cs=cs, warp=warp, word=word, group_by=group_by,
                               nan_thresh=nan_thresh, verbose=verbose)
    groupday_varex_percell(mouse, method=method, trace_type=trace_type, cs=cs,
                           warp=warp, ve_min=0.05, word=word, group_by=group_by,
                           nan_thresh=nan_thresh, verbose=verbose)


def pairday_shortlist(
        mouse,
        trace_type='zscore_day',
        method='ncp_bcd',
        cs='',
        warp=False,
        word=None,
        verbose=False):

    pairday_varex_summary(mouse, trace_type=trace_type, method=method, cs=cs,
                          warp=warp, word=word, verbose=verbose)
    pairday_factors_annotated(mouse, trace_type=trace_type, method=method,
                              cs=cs, warp=warp, word=word, verbose=verbose)
    pairday_varex_percell(mouse, method=method, trace_type=trace_type, cs=cs,
                          warp=warp, ve_min=0.05, word=word)


def singleday_shortlist(
        mouse,
        trace_type='zscore_day',
        method='ncp_bcd',
        cs='',
        warp=False,
        word=None,
        verbose=False):

    singleday_varex_summary(mouse, trace_type=trace_type, method=method, cs=cs,
                            warp=warp, word=word, verbose=verbose)
    singleday_factors_annotated(mouse, trace_type=trace_type, method=method,
                                cs=cs, warp=warp, word=word, verbose=verbose)
    singleday_varex_percell(mouse, method=method, trace_type=trace_type, cs=cs,
                            warp=warp, ve_min=0.05, word=word)


"""
--------------------------- ACROSS ANIMAL PLOTS ---------------------------
"""


def groupmouse_varex_summary(
        mice,
        trace_type='zscore_day',
        method='ncp_bcd',
        cs='',
        warp=False,
        words=None,
        group_by='all',
        nan_thresh=0.85,
        rectified=False,
        verbose=False):
    """
    Plot reconstruction error as variance explained across all whole groupday
    TCA decomposition ensemble.

    Parameters:
    -----------
    mouse : str; mouse name
    trace_type : str; dff, zscore, deconvolved
    method : str; TCA fit method from tensortools

    Returns:
    --------
    Saves figures to .../analysis folder  .../qc
    """

    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}
    group_pars = {'group_by': group_by}

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

    # save dir
    group_word = paths.groupmouse_word({'mice': mice})
    mouse = 'Group-' + group_word
    save_dir = paths.tca_plots(
        mouse, 'group', pars=pars, word=words[0], group_pars=group_pars)
    save_dir = os.path.join(save_dir, 'qc' + nt_save_tag + r_tag)
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    var_path = os.path.join(
        save_dir, str(mouse) + '_summary_variance_explained' + r_save_tag
        + '_n' + str(len(mice)) + '.pdf')

    # create figure and axes
    buffer = 5
    right_pad = 5
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(
        100, 100, figure=fig, left=0.05, right=.95, top=.95, bottom=0.05)
    ax = fig.add_subplot(gs[10:90-buffer, :90-right_pad])
    cmap = sns.color_palette('hls', len(mice))

    for c, mouse in enumerate(mice):
        # load dir
        load_dir = paths.tca_path(
            mouse, 'group', pars=pars, word=words[c], group_pars=group_pars)
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
        V = ensemble[method]
        X = np.load(input_tensor_path)

        # rectify input tensor (only look at nonnegative variance)
        if rectified:
            X[X < 0] = 0

        # get reconstruction error as variance explained
        var, var_s, x, x_s = [], [], [], []
        for r in V.results:
            bU = V.results[r][0].factors.full()
            var.append((np.nanvar(X) - np.nanvar(X - bU)) / np.nanvar(X))
            x.append(r)
            for it in range(0, len(V.results[r])):
                U = V.results[r][it].factors.full()
                var_s.extend([(np.nanvar(X) - np.nanvar(X - U)) / np.nanvar(X)])
                x_s.extend([r])

        # mean response of neuron across trials
        mU = np.nanmean(X, axis=2, keepdims=True) * np.ones((1, 1, np.shape(X)[2]))
        var_mean = (np.nanvar(X) - np.nanvar(X - mU)) / np.nanvar(X)

        # smoothed response of neuron across time
        smU = np.convolve(
            X.reshape((X.size)),
            np.ones(5, dtype=np.float64)/5, 'same').reshape(np.shape(X))
        var_smooth = (np.nanvar(X) - np.nanvar(X - smU)) / np.nanvar(X)

        # plot
        R = np.max([r for r in V.results.keys()])
        ax.scatter(x_s, var_s, color=cmap[c], alpha=0.5)
        ax.scatter([R+2], var_mean, color=cmap[c], alpha=0.5)
        ax.scatter([R+4], var_smooth, color=cmap[c], alpha=0.5)
        ax.plot(x, var, label=('mouse ' + mouse), color=cmap[c])
        ax.plot([R+1.5, R+2.5], [var_mean, var_mean], color=cmap[c])
        ax.plot([R+3.5, R+4.5], [var_smooth, var_smooth], color=cmap[c])

    # add labels/titles
    x_labels = [str(R) for R in V.results]
    x_labels.extend(
        ['', 'mean\n cell\n response', '', 'smooth\n response\n (0.3s)'])
    ax.set_xticks(range(1, len(V.results) + 5))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('model rank')
    ax.set_ylabel('fractional variance explained')
    ax.set_title(
        'Variance Explained: ' + str(method) + r_tag + ', ' + str(mice))
    ax.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)

    fig.savefig(var_path, bbox_inches='tight')


"""
----------------------------- GROUP DAY PLOTS -----------------------------
"""


def groupday_longform_factors_annotated(
        mouse,
        trace_type='zscore_day',
        method='ncp_bcd',
        cs='',
        warp=False,
        word=None,
        group_by='all',
        nan_thresh=0.85,
        extra_col=1,
        alpha=0.6,
        plot_running=True,
        filetype='pdf',
        verbose=False):

    """
    Plot TCA factors with trial metadata annotations for all days
    and ranks/componenets for TCA decomposition ensembles.

    Parameters:
    -----------
    mouse : str
        Mouse name.
    trace_type : str
        dff, zscore, zscore_iti, zscore_day, deconvolved
    method : str
        TCA fit method from tensortools
    cs : str
        Cs stimuli to include, plus/minus/neutral, 0/135/270, etc. '' empty
        includes all stimuli
    warp : bool
        Use traces with time-warped outcome.
    extra_col : int
        Number of columns to add to the original three factor columns
    alpha : float
        Value between 0 and 1 for transparency of markers
    plot_running : bool
        Include trace of scaled (to plot max) average running speed during trial
    verbose : bool
        Show plots as they are made.

    Returns:
    --------
    Saves figures to .../analysis folder  .../factors annotated
    """

    # use matplotlib plotting defaults
    mpl.rcParams.update(mpl.rcParamsDefault)

    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}
    group_pars = {'group_by': group_by}

    # if cells were removed with too many nan trials
    if nan_thresh:
        nt_tag = '_nantrial' + str(nan_thresh)
        save_tag = ' nantrial ' + str(nan_thresh)
    else:
        nt_tag = ''
        save_tag = ''

    # load dir
    load_dir = paths.tca_path(
        mouse, 'group', pars=pars, word=word, group_pars=group_pars)
    tensor_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_group_decomp_' + str(trace_type) + '.npy')
    meta_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_df_group_meta.pkl')

    # save dir
    save_dir = paths.tca_plots(
        mouse, 'group', pars=pars, word=word, group_pars=group_pars)
    save_dir = os.path.join(save_dir, 'factors annotated long-form' + save_tag)
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    date_dir = os.path.join(save_dir, str(group_by) + ' ' + method)
    if not os.path.isdir(date_dir): os.mkdir(date_dir)

    # load your data
    ensemble = np.load(tensor_path)
    ensemble = ensemble.item()
    meta = pd.read_pickle(meta_path)
    meta = utils.update_naive_cs(meta)
    orientation = meta['orientation']
    trial_num = np.arange(0, len(orientation))
    condition = meta['condition']
    trialerror = meta['trialerror']
    hunger = deepcopy(meta['hunger'])
    speed = meta['speed']
    dates = meta.reset_index()['date']
    learning_state = meta['learning_state']

    # re-balance your factors ()
    print('Re-balancing factors.')
    for r in ensemble[method].results:
        for i in range(len(ensemble[method].results[r])):
            ensemble[method].results[r][i].factors.rebalance()

    # calculate change indices for days and reversal/learning
    udays = {d: c for c, d in enumerate(np.unique(dates))}
    ndays = np.diff([udays[i] for i in dates])
    day_x = np.where(ndays)[0] + 0.5
    ustate = {d: c for c, d in enumerate(np.unique(learning_state))}
    nstate = np.diff([ustate[i] for i in learning_state])
    lstate_x = np.where(nstate)[0] + 0.5

    # merge hunger and tag info for plotting hunger
    tags = meta['tag']
    hunger[tags == 'disengaged'] = 'disengaged'

    # sort neuron factors by component they belong to most
    # if 'mcp_als' has been run make sure the variable is in the correct format
    if isinstance(ensemble[method], dict):
        ensemble2 = {}
        ensemble2[method] = lambda: None
        ensemble[method] = {k: [v] for k, v in ensemble[method].items()}
        ensemble2[method].results = ensemble[method]
        sort_ensemble, my_sorts = tca._sortfactors(ensemble2[method])
    else:
        sort_ensemble, my_sorts = tca._sortfactors(ensemble[method])

    rows = 5
    cols = 3
    for r in sort_ensemble.results:

        U = sort_ensemble.results[r][0].factors

        for comp in range(U.rank):
            fig, axes = plt.subplots(
                rows, cols, figsize=(17, rows),
                gridspec_kw={'width_ratios': [2, 2, 17]})

            # reshape for easier indexing
            ax = np.array(axes).reshape((rows, -1))
            ax[0, 0].set_title('Neuron factors')
            ax[0, 1].set_title('Temporal factors')
            ax[0, 2].set_title('Trial factors')

            # add title to whole figure
            ax[0, 0].text(-1.2, 4, '\n' + mouse + ': \n\nrank: ' + str(int(r))
                          + '\nmethod: ' + method + ' \ngroup_by: '
                          + group_by, fontsize=12,
                          transform=ax[0, 0].transAxes, color='#969696')

            # plot cell factors
            ax[0, 0].bar(
                np.arange(0, len(U.factors[0][:, comp])),
                U.factors[0][:, comp], color='b')
            ax[0, 0].autoscale(enable=True, axis='both', tight=True)

            # plot temporal factors
            ax[0, 1].plot(U.factors[1][:, comp], color='r', linewidth=1.5)
            ax[0, 1].autoscale(enable=True, axis='both', tight=True)

            # add a line for stim onset and offset
            # NOTE: assumes downsample, 1 sec before onset, 3 sec stim
            y_lim = ax[0, 1].get_ylim()
            ons = 15.5*1
            offs = ons+15.5*3
            ax[0, 1].plot([ons, ons], y_lim, ':k')
            ax[0, 1].plot([offs, offs], y_lim, ':k')

            col = cols - 1
            for i in range(rows):

                # get axis values
                y_lim = [0, np.nanmax(U.factors[2][:, comp])]

                # running
                if plot_running:
                    scale_by = np.nanmax(speed)/y_lim[1]
                    if not np.isnan(scale_by):
                        ax[i, col].plot(
                            np.array(speed.tolist())/scale_by,
                            color=[1, 0.1, 0.6, 0.2])
                        # , label='speed')

                # Orientation - main variable to plot
                if i == 0:
                    ori_vals = [0, 135, 270]
                    # color_vals = [[0.28, 0.68, 0.93, alpha],
                    #               [0.84, 0.12, 0.13, alpha],
                    #               [0.46, 0.85, 0.47, alpha]]
                    color_vals = sns.color_palette('husl', 3)
                    for k in range(0, 3):
                        ax[i, col].plot(
                            trial_num[orientation == ori_vals[k]],
                            U.factors[2][orientation == ori_vals[k], comp],
                            'o', label=str(ori_vals[k]), color=color_vals[k],
                            markersize=2, alpha=alpha)

                    ax[i, col].set_title(
                        'Component ' + str(comp + 1) + '\n\n\nTrial factors')
                    ax[i, col].legend(
                        bbox_to_anchor=(1.02, 1), loc='upper left',
                        borderaxespad=0, title='Orientation', markerscale=2,
                        prop={'size': 8})
                    ax[i, col].autoscale(enable=True, axis='both', tight=True)
                    ax[i, col].set_xticklabels([])

                # Condition - main variable to plot
                elif i == 1:
                    cs_vals = ['plus', 'minus', 'neutral']
                    cs_labels = ['plus', 'minus', 'neutral']
                    color_vals = [[0.46, 0.85, 0.47, alpha],
                                  [0.84, 0.12, 0.13, alpha],
                                  [0.28, 0.68, 0.93, alpha]]
                    for k in range(0, 3):
                        ax[i, col].plot(
                            trial_num[condition == cs_vals[k]],
                            U.factors[2][condition == cs_vals[k], comp], 'o',
                            label=str(cs_labels[k]), color=color_vals[k],
                            markersize=2)

                    ax[i, col].legend(
                        bbox_to_anchor=(1.02, 1), loc='upper left',
                        borderaxespad=0, title='Condition', markerscale=2,
                        prop={'size': 8})
                    ax[i, col].autoscale(enable=True, axis='both', tight=True)
                    ax[i, col].set_xticklabels([])

                # Trial error - main variable to plot
                elif i == 2:
                    color_counter = 0
                    error_colors = sns.color_palette(
                        palette='muted', n_colors=10)
                    trialerror_vals = [0, 1]  # 2, 3, 4, 5,] # 6, 7, 8, 9]
                    trialerror_labels = ['hit',
                                         'miss',
                                         'neutral correct reject',
                                         'neutral false alarm',
                                         'minus correct reject',
                                         'minus false alarm',
                                         'blank correct reject',
                                         'blank false alarm',
                                         'pav early licking',
                                         'pav late licking']
                    for k in range(len(trialerror_vals)):
                        ax[i, col].plot(
                            trial_num[trialerror == trialerror_vals[k]],
                            U.factors[2][trialerror == trialerror_vals[k], comp],
                            'o', label=str(trialerror_labels[k]), alpha=alpha,
                            markersize=2, color=error_colors[color_counter])
                        color_counter = color_counter + 1

                    ax[i, col].legend(
                        bbox_to_anchor=(1.02, 1), loc='upper left',
                        borderaxespad=0, title='Trialerror', markerscale=2,
                        prop={'size': 8})
                    ax[i, col].autoscale(enable=True, axis='both', tight=True)
                    ax[i, col].set_xticklabels([])

                # Trial error 2.0 - main variable to plot
                elif i == 3:
                    trialerror_vals = [2, 3]
                    trialerror_labels = ['neutral correct reject',
                                         'neutral false alarm']
                    for k in range(len(trialerror_vals)):
                        ax[i, col].plot(
                            trial_num[trialerror == trialerror_vals[k]],
                            U.factors[2][trialerror == trialerror_vals[k], comp],
                            'o', label=str(trialerror_labels[k]), alpha=alpha,
                            markersize=2, color=error_colors[color_counter])
                        color_counter = color_counter + 1

                    ax[i, col].legend(
                        bbox_to_anchor=(1.02, 1), loc='upper left',
                        borderaxespad=0, title='Trialerror', markerscale=2,
                        prop={'size': 8})
                    ax[i, col].autoscale(enable=True, axis='both', tight=True)
                    ax[i, col].set_xticklabels([])

                # Trial error 3.0 - main variable to plot
                elif i == 4:
                    trialerror_vals = [4, 5]
                    trialerror_labels = ['minus correct reject',
                                         'minus false alarm']
                    for k in range(len(trialerror_vals)):
                        ax[i, col].plot(
                            trial_num[trialerror == trialerror_vals[k]],
                            U.factors[2][trialerror == trialerror_vals[k], comp],
                            'o', label=str(trialerror_labels[k]), alpha=alpha,
                            markersize=2, color=error_colors[color_counter])
                        color_counter = color_counter + 1

                        ax[i, col].legend(
                            bbox_to_anchor=(1.02, 1), loc='upper left',
                            borderaxespad=0, title='Trialerror', markerscale=2,
                            prop={'size': 8})
                    ax[i, col].autoscale(enable=True, axis='both', tight=True)

                # plot days, reversal, or learning lines if there are any
                if col >= 1:
                    y_lim = ax[i, col].get_ylim()
                    if len(day_x) > 0:
                        for k in day_x:
                            ax[i, col].plot(
                                [k, k], y_lim, color='#969696', linewidth=1)
                    if len(lstate_x) > 0:
                        ls_vals = ['naive', 'learning', 'reversal1']
                        ls_colors = ['#66bd63', '#d73027', '#a50026']
                        for k in lstate_x:
                            ls = learning_state[int(k-0.5)]
                            ax[i, col].plot(
                                [k, k], y_lim,
                                color=ls_colors[ls_vals.index(ls)],
                                linewidth=1.5)

                # hide subplots that won't be used
                if i > 0:
                    ax[i, 0].axis('off')
                    ax[i, 1].axis('off')

                # despine plots to look like sns defaults
                sns.despine()

            # save
            if filetype.lower() == 'pdf':
                suffix = '.pdf'
            elif filetype.lower() == 'eps':
                suffix = '.eps'
            else:
                suffix = '.png'
            plt.savefig(os.path.join(date_dir, 'rank_' + str(int(r)) +
                        '_component_' + str(comp + 1) + suffix),
                        bbox_inches='tight')
            if verbose:
                plt.show()
            plt.close('all')


def groupday_factors_annotated(
        mouse,
        trace_type='zscore_day',
        method='ncp_bcd',
        cs='',
        warp=False,
        word=None,
        group_by=None,
        nan_thresh=None,
        extra_col=4,
        alpha=0.6,
        plot_running=True,
        filetype='pdf',
        scale_y=False,
        verbose=False):

    """
    Plot TCA factors with trial metadata annotations for all days
    and ranks/componenets for TCA decomposition ensembles.

    Parameters:
    -----------
    mouse : str
        Mouse name.
    trace_type : str
        dff, zscore, zscore_iti, zscore_day, deconvolved
    method : str
        TCA fit method from tensortools
    cs : str
        Cs stimuli to include, plus/minus/neutral, 0/135/270, etc. '' empty
        includes all stimuli
    warp : bool
        Use traces with time-warped outcome.
    extra_col : int
        Number of columns to add to the original three factor columns
    alpha : float
        Value between 0 and 1 for transparency of markers
    plot_running : bool
        Include trace of scaled (to plot max) average running speed during trial
    verbose : bool
        Show plots as they are made.

    Returns:
    --------
    Saves figures to .../analysis folder  .../factors annotated
    """

    # use matplotlib plotting defaults
    mpl.rcParams.update(mpl.rcParamsDefault)

    # plotting options for the unconstrained and nonnegative models.
    plot_options = {
      'cp_als': {
        'line_kw': {
          'color': 'red',
          'label': 'cp_als',
        },
        'scatter_kw': {
          'color': 'green',
          'alpha': 0.5,
        },
        'bar_kw': {
          'color': 'blue',
          'alpha': 0.5,
        },
      },
      'ncp_hals': {
        'line_kw': {
          'color': 'red',
          'label': 'ncp_hals',
        },
        'scatter_kw': {
          'color': 'green',
          'alpha': 0.5,
        },
        'bar_kw': {
          'color': 'blue',
          'alpha': 0.5,
        },
      },
      'ncp_bcd': {
        'line_kw': {
          'color': 'red',
          'label': 'ncp_bcd',
        },
        'scatter_kw': {
          'color': 'green',
          'alpha': 0.5,
        },
        'bar_kw': {
          'color': 'blue',
          'alpha': 0.5,
        },
      },
      'mcp_als': {
        'line_kw': {
          'color': 'red',
          'label': 'mcp_als',
        },
        'scatter_kw': {
          'color': 'green',
          'alpha': 0.5,
        },
        'bar_kw': {
          'color': 'blue',
          'alpha': 0.5,
        },
      },
      'mncp_hals': {
        'line_kw': {
          'color': 'red',
          'label': 'mcp_als',
        },
        'scatter_kw': {
          'color': 'green',
          'alpha': 0.5,
        },
        'bar_kw': {
          'color': 'blue',
          'alpha': 0.5,
        },
      },
    }

    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}
    group_pars = {'group_by': group_by}

    # if cells were removed with too many nan trials
    if nan_thresh:
        nt_tag = '_nantrial' + str(nan_thresh)
        nt_save_tag = ' nantrial ' + str(nan_thresh)
    else:
        nt_tag = ''
        nt_save_tag = ''

    # load dir
    load_dir = paths.tca_path(
        mouse, 'group', pars=pars, word=word, group_pars=group_pars)
    tensor_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_group_decomp_' + str(trace_type) + '.npy')
    meta_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_df_group_meta.pkl')

    # save dir
    save_dir = paths.tca_plots(
        mouse, 'group', pars=pars, word=word, group_pars=group_pars)
    if scale_y:
        save_tag = nt_save_tag + ' scaled-y'
    else:
        save_tag = nt_save_tag + ''
    save_dir = os.path.join(save_dir, 'factors annotated' + save_tag)
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    date_dir = os.path.join(save_dir, str(group_by) + ' ' + method)
    if not os.path.isdir(date_dir): os.mkdir(date_dir)

    # load your data
    ensemble = np.load(tensor_path)
    ensemble = ensemble.item()
    meta = pd.read_pickle(meta_path)
    orientation = meta['orientation']
    trial_num = np.arange(0, len(orientation))
    condition = meta['condition']
    trialerror = meta['trialerror']
    hunger = deepcopy(meta['hunger'])
    speed = meta['speed']
    dates = meta.reset_index()['date']
    learning_state = meta['learning_state']

    # re-balance your factors ()
    print('Re-balancing factors.')
    for r in ensemble[method].results:
        for i in range(len(ensemble[method].results[r])):
            ensemble[method].results[r][i].factors.rebalance()

    # calculate change indices for days and reversal/learning
    udays = {d: c for c, d in enumerate(np.unique(dates))}
    ndays = np.diff([udays[i] for i in dates])
    day_x = np.where(ndays)[0] + 0.5
    ustate = {d: c for c, d in enumerate(np.unique(learning_state))}
    nstate = np.diff([ustate[i] for i in learning_state])
    lstate_x = np.where(nstate)[0] + 0.5

    # merge hunger and tag info for plotting hunger
    tags = meta['tag']
    hunger[tags == 'disengaged'] = 'disengaged'

    # sort neuron factors by component they belong to most
    # if 'mcp_als' has been run make sure the variable is in the correct format
    sort_ensemble, my_sorts = tca._sortfactors(ensemble[method])

    for r in sort_ensemble.results:

        U = sort_ensemble.results[r][0].factors

        fig, axes = plt.subplots(U.rank, U.ndim + extra_col, figsize=(9 + extra_col, U.rank))
        figt = tt.plot_factors(U, plots=['bar', 'line', 'scatter'],
                        axes=None,
                        fig=fig,
                        scatter_kw=plot_options[method]['scatter_kw'],
                        line_kw=plot_options[method]['line_kw'],
                        bar_kw=plot_options[method]['bar_kw'])
        ax = figt[0].axes
        ax[0].set_title('Neuron factors')
        ax[1].set_title('Temporal factors')
        ax[2].set_title('Trial factors')

        # add title to whole figure
        ax[0].text(-1.2, 4, '\n' + mouse + ': \n\nrank: ' + str(int(r)) +
                   '\nmethod: ' + method + ' \ngroup_by: '
                   + group_by, fontsize=12, transform=ax[0].transAxes,
                   color='#969696')

        # reshape for easier indexing
        ax = np.array(ax).reshape((U.rank, -1))

        # rescale the y-axis for trials
        if scale_y:
            for i in range(U.rank):
                y_lim = np.array(ax[i, 2].get_ylim())*0.8
                y_ticks = ax[i, 2].get_yticks()
                y_ticks[-1] = y_lim[-1]
                y_ticks = np.round(y_ticks, 2)
                # y_tickl = [str(y) for y in y_ticks]
                ax[i, 2].set_ylim(y_lim)
                ax[i, 2].set_yticks(y_ticks)
                ax[i, 2].set_yticklabels(y_ticks)

        # add a line for stim onset and offset
        # NOTE: assumes downsample, 1 sec before onset, 3 sec stim
        for i in range(U.rank):
            y_lim = ax[i, 1].get_ylim()
            ons = 15.5*1
            offs = ons+15.5*3
            ax[i, 1].plot([ons, ons], y_lim, ':k')
            ax[i, 1].plot([offs, offs], y_lim, ':k')

        for col in range(3, 3+extra_col):
            for i in range(U.rank):

                # get axis values
                y_lim = ax[i, 2].get_ylim()
                x_lim = ax[i, 2].get_xlim()
                y_ticks = ax[i, 2].get_yticks()
                y_tickl = ax[i, 2].get_yticklabels()
                x_ticks = ax[i, 2].get_xticks()
                x_tickl = ax[i, 2].get_xticklabels()

                # running
                if plot_running:
                    scale_by = np.nanmax(speed)/y_lim[1]
                    if not np.isnan(scale_by):
                        ax[i, col].plot(np.array(speed.tolist())/scale_by, color=[1, 0.1, 0.6, 0.2])
                        # , label='speed')

                # Orientation - main variable to plot
                if col == 3:
                    ori_vals = [0, 135, 270]
                    color_vals = [[0.28, 0.68, 0.93, alpha], [0.84, 0.12, 0.13, alpha],
                                  [0.46, 0.85, 0.47, alpha]]
                    for k in range(0, 3):
                        ax[i, col].plot(trial_num[orientation == ori_vals[k]],
                                        U.factors[2][orientation == ori_vals[k], i], 'o',
                                        label=str(ori_vals[k]), color=color_vals[k], markersize=2)
                    if i == 0:
                        ax[i, col].set_title('Orientation')
                        ax[i, col].legend(bbox_to_anchor=(0.5,1.02), loc='lower center',
                                          borderaxespad=2.5)
                # Condition - main variable to plot
                elif col == 4:
                    cs_vals = ['plus', 'minus', 'neutral']
                    cs_labels = ['plus', 'minus', 'neutral']
                    color_vals = [[0.46, 0.85, 0.47, alpha], [0.84, 0.12, 0.13, alpha],
                                  [0.28, 0.68, 0.93, alpha]]
                    col = 4
                    for k in range(0, 3):
                        ax[i, col].plot(trial_num[condition == cs_vals[k]],
                                        U.factors[2][condition == cs_vals[k], i], 'o',
                                        label=str(cs_labels[k]), color=color_vals[k], markersize=2)
                    if i == 0:
                        ax[i, col].set_title('Condition')
                        ax[i, col].legend(bbox_to_anchor=(0.5,1.02), loc='lower center',
                                          borderaxespad=2.5)
                # Trial error - main variable to plot
                elif col == 5:
                    trialerror_vals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                    trialerror_labels = ['hit',
                                         'miss',
                                         'neutral correct reject',
                                         'neutral false alarm',
                                         'minus correct reject',
                                         'minus false alarm',
                                         'blank correct reject',
                                         'blank false alarm',
                                         'pav early licking',
                                         'pav late licking']
                    for k in trialerror_vals:
                        ax[i, col].plot(trial_num[trialerror == trialerror_vals[k]],
                                        U.factors[2][trialerror == trialerror_vals[k], i], 'o',
                                        label=str(trialerror_labels[k]), alpha=0.8, markersize=2)
                    if i == 0:
                        ax[i, col].set_title('Trialerror')
                        ax[i, col].legend(bbox_to_anchor=(0.5, 1.02), loc='lower center',
                                          borderaxespad=2.5)
                # State - main variable to plot
                elif col == 6:
                    h_vals = ['hungry', 'sated', 'disengaged']
                    h_labels = ['hungry', 'sated', 'disengaged']
                    color_vals = [[1, 0.6, 0.3, alpha], [0.7, 0.9, 0.4, alpha],
                                  [0.6, 0.5, 0.6, alpha], [0.0, 0.9, 0.4, alpha]]
                    for k in range(0, 3):
                        ax[i, col].plot(trial_num[hunger == h_vals[k]],
                                        U.factors[2][hunger == h_vals[k], i], 'o',
                                        label=str(h_labels[k]), color=color_vals[k], markersize=2)
                    if i == 0:
                        ax[i, col].set_title('State')
                        ax[i, col].legend(bbox_to_anchor=(0.5, 1.02), loc='lower center',
                                          borderaxespad=2.5)

                # plot days, reversal, or learning lines if there are any
                if col >= 2:
                    y_lim = ax[i, col].get_ylim()
                    if len(day_x) > 0:
                        for k in day_x:
                            ax[i, col].plot(
                                [k, k], y_lim, color='#969696', linewidth=1)
                    if len(lstate_x) > 0:
                        ls_vals = ['naive', 'learning', 'reversal1']
                        ls_colors = ['#66bd63', '#d73027', '#a50026']
                        for k in lstate_x:
                            ls = learning_state[int(k-0.5)]
                            ax[i, col].plot(
                                [k, k], y_lim, color=ls_colors[ls_vals.index(ls)],
                                linewidth=1.5)

                # set axes labels
                ax[i, col].set_yticks(y_ticks)
                ax[i, col].set_yticklabels(y_tickl)
                ax[i, col].set_ylim(y_lim)
                ax[i, col].set_xlim(x_lim)

                # format axes
                ax[i, col].locator_params(nbins=4)
                ax[i, col].spines['top'].set_visible(False)
                ax[i, col].spines['right'].set_visible(False)
                ax[i, col].xaxis.set_tick_params(direction='out')
                ax[i, col].yaxis.set_tick_params(direction='out')
                ax[i, col].yaxis.set_ticks_position('left')
                ax[i, col].xaxis.set_ticks_position('bottom')

                # remove xticks on all but bottom row
                if i + 1 != U.rank:
                    plt.setp(ax[i, col].get_xticklabels(), visible=False)

                if col == 3:
                    ax[i, 0].set_ylabel('Component #' + str(i+1), rotation=0,
                                        labelpad=45, verticalalignment='center',
                                        fontstyle='oblique')

        if filetype.lower() == 'pdf':
            suffix = '.pdf'
        elif filetype.lower() == 'eps':
            suffix = '.eps'
        else:
            suffix = '.png'
        plt.savefig(os.path.join(date_dir, 'rank_' + str(int(r)) + suffix),
                                 bbox_inches='tight')
        if verbose:
            plt.show()
        plt.close()


def groupday_varex_summary(
        mouse,
        trace_type='zscore_day',
        method='ncp_bcd',
        cs='',
        warp=False,
        word=None,
        group_by=None,
        nan_thresh=None,
        rectified=False,
        verbose=False):
    """
    Plot reconstruction error as variance explained across all whole groupday
    TCA decomposition ensemble.

    Parameters:
    -----------
    mouse : str; mouse name
    trace_type : str; dff, zscore, deconvolved
    method : str; TCA fit method from tensortools

    Returns:
    --------
    Saves figures to .../analysis folder  .../qc
    """

    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}
    group_pars = {'group_by': group_by}

    # if cells were removed with too many nan trials
    if nan_thresh:
        nt_tag = '_nantrial' + str(nan_thresh)
        nt_save_tag = ' nantrial ' + str(nan_thresh)
    else:
        nt_tag = ''
        nt_save_tag = ''

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

    # save dir
    save_dir = paths.tca_plots(
        mouse, 'group', pars=pars, word=word, group_pars=group_pars)
    save_dir = os.path.join(save_dir, 'qc' + nt_save_tag)
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    var_path = os.path.join(
        save_dir, str(mouse) + '_summary_variance_cubehelix.pdf')

    # load your data
    ensemble = np.load(tensor_path)
    ensemble = ensemble.item()
    V = ensemble[method]
    X = np.load(input_tensor_path)

    # rectify input tensor (only look at nonnegative variance)
    if rectified:
        X[X < 0] = 0

    # get reconstruction error as variance explained
    var, var_s, x, x_s = [], [], [], []
    for r in V.results:
        bU = V.results[r][0].factors.full()
        var.append((np.nanvar(X) - np.nanvar(X - bU)) / np.nanvar(X))
        x.append(r)
        for it in range(0, len(V.results[r])):
            U = V.results[r][it].factors.full()
            var_s.extend([(np.nanvar(X) - np.nanvar(X - U)) / np.nanvar(X)])
            x_s.extend([r])

    # mean response of neuron across trials
    mU = np.nanmean(X, axis=2, keepdims=True) * np.ones((1, 1, np.shape(X)[2]))
    var_mean = (np.nanvar(X) - np.nanvar(X - mU)) / np.nanvar(X)

    # smoothed response of neuron across time
    smU = np.convolve(
        X.reshape((X.size)),
        np.ones(5, dtype=np.float64)/5, 'same').reshape(np.shape(X))
    var_smooth = (np.nanvar(X) - np.nanvar(X - smU)) / np.nanvar(X)

    # create figure and axes
    buffer = 5
    right_pad = 5
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(
        100, 100, figure=fig, left=0.05, right=.95, top=.95, bottom=0.05)
    ax = fig.add_subplot(gs[10:90-buffer, :90-right_pad])
    c = 0
    cmap = sns.color_palette(sns.cubehelix_palette(c+1))

    # plot
    R = np.max([r for r in V.results.keys()])
    ax.scatter(x_s, var_s, color=cmap[c], alpha=0.5)
    ax.scatter([R+2], var_mean, color=cmap[c], alpha=0.5)
    ax.scatter([R+4], var_smooth, color=cmap[c], alpha=0.5)
    ax.plot(x, var, label=('mouse ' + mouse), color=cmap[c])
    ax.plot([R+1.5, R+2.5], [var_mean, var_mean], color=cmap[c])
    ax.plot([R+3.5, R+4.5], [var_smooth, var_smooth], color=cmap[c])

    # add labels/titles
    x_labels = [str(R) for R in V.results]
    x_labels.extend(
        ['', 'mean\n cell\n response', '', 'smooth\n response\n (0.3s)'])
    ax.set_xticks(range(1, len(V.results) + 5))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('model rank')
    ax.set_ylabel('fractional variance explained')
    ax.set_title('Variance Explained: ' + str(method) + ', ' + mouse)
    ax.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)

    fig.savefig(var_path, bbox_inches='tight')


def groupday_varex_percell(  # TODO MAKE THIS WORK FOR GROUPDAY
        mouse,
        method='ncp_bcd',
        trace_type='zscore_day',
        cs='',
        warp=False,
        word=None,
        group_by=None,
        nan_thresh=None,
        ve_min=0.05,
        filetype='pdf'):
    """
    Plot TCA reconstruction error as variance explained per cell
    for TCA decomposition. Create folder of variance explained per cell
    swarm plots. Calculate summary plots of 'fraction of maximum variance
    explained' per cell by rank for all cells given a certain (ve_min) threshold
    for maximum variance explained

    Parameters:
    -----------
    mouse : str; mouse name
    trace_type : str; dff, zscore, deconvolved
    method : str; TCA fit method from tensortools
    ve_min: float; minimum variance explained for best rank per cell
                   to be included in summary of fraction of maximum variance
                   explained

    Returns:
    --------
    Saves figures to .../analysis folder/ .../qc
                                             .../variance explained per cell

    """

    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}

    days = flow.DateSorter.frommeta(mice=[mouse], tags=None)

    # create folder structure if needed
    cs_tag = '' if len(cs) == 0 else ' ' + str(cs)
    warp_tag = '' if warp is False else ' warp'
    folder_name = 'tensors single ' + str(trace_type) + cs_tag + warp_tag

    ve, ve_max, ve_frac, rank_num, day_num, cell_num = [], [], [], [], [], []
    for c, day1 in enumerate(days, 0):

        # get dirs for loading
        load_dir = paths.tca_path(mouse, 'single', pars=pars, word=word)
        if not os.path.isdir(load_dir): os.mkdir(load_dir)
        tensor_path = os.path.join(load_dir, str(day1.mouse) + '_'
                                   + str(day1.date) + '_single_decomp_'
                                   + str(trace_type) + '.npy')
        input_tensor_path = os.path.join(load_dir, str(day1.mouse) + '_'
                                         + str(day1.date) + '_single_tensor_'
                                         + str(trace_type) + '.npy')
        if not os.path.isfile(tensor_path): continue
        if not os.path.isfile(input_tensor_path): continue

        # load your data
        ensemble = np.load(tensor_path)
        ensemble = ensemble.item()
        V = ensemble[method]
        X = np.load(input_tensor_path)

        # get reconstruction error as variance explained per cell
        for cell in range(0, np.shape(X)[0]):
            rank_ve_vec = []
            rank_vec = []
            for r in V.results:
                U = V.results[r][0].factors.full()
                Usub = X - U
                rank_ve = (np.var(X[cell, :, :]) - np.var(Usub[cell, :, :])) / np.var(X[cell, :, :])
                rank_ve_vec.append(rank_ve)
                rank_vec.append(r)
            max_ve = np.max(rank_ve_vec)
            ve.extend(rank_ve_vec)
            ve_max.extend([max_ve for s in rank_ve_vec])
            ve_frac.extend(rank_ve_vec / max_ve)
            rank_num.extend(rank_vec)
            day_num.extend([c+1 for s in rank_ve_vec])
            cell_num.extend([cell for s in rank_ve_vec])

    # build pd dataframe of all variance measures
    index = pd.MultiIndex.from_arrays([
    day_num,
    rank_num,
    ve,
    ve_max,
    ve_frac,
    cell_num,
    ],
    names=['day', 'rank', 'variance_explained', 'max_ve', 'frac_ve', 'cell'])
    df = pd.DataFrame(index=index)
    df = df.reset_index()

    # make a rainbow colormap, HUSL space but does not circle back on itself
    cmap = sns.color_palette('hls', int(np.ceil(1.5*np.unique(df['rank'])[-1])))
    cmap = cmap[0:np.unique(df['rank'])[-1]]

    # Part 1
    # slice df, only look at cells with a max variance >5%
    sliced_df2 = df.loc[(df['day']) & (df['max_ve'] >= ve_min), :]

    # CDF plot
    fig1 = plt.figure(figsize=(15,9))
    for i in np.unique(sliced_df2['rank']):
        input_ve = sliced_df2.loc[(sliced_df2['rank'] == i),'frac_ve']
        ax = sns.distplot(input_ve, kde_kws={'cumulative': True, 'lw': 2, 'color': cmap[i-1], 'label': str(i)}, hist=False)
        lg = ax.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
        lg.set_title('rank')
        ax.set_title(mouse + ', Fraction of maximum variance explained per cell, CDF')
        ax.set_xlabel('Fraction of maximum variance explained')

    # swarm plot
    fig2 =plt.figure(figsize=(18,6))
    ax2 = sns.violinplot(x=sliced_df2['rank'], y=sliced_df2['frac_ve'], size=3, alpha=1, inner=None, palette=cmap)
    ax2.set_title(mouse + ', Fraction of maximum variance explained per cell, violin')
    ax2.set_ylabel('Fraction of maximum variance explained')

    # swarm plot
    fig3 = plt.figure(figsize=(18,6))
    ax3 = sns.swarmplot(x=sliced_df2['rank'], y=sliced_df2['frac_ve'], size=2, alpha=1, palette=cmap)
    ax3.set_title(mouse + ', Fraction of maximum variance explained per cell, swarm')
    ax3.set_ylabel('Fraction of maximum variance explained')

    # set up saving paths/dir
    save_dir = paths.tca_plots(mouse, 'single', pars=pars, word=word)
    save_dir = os.path.join(save_dir, 'qc')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    save_file_base = mouse + '_singleday_frac_max_var_expl_' + trace_type

    # save
    if filetype.lower() == 'pdf':
        suffix = '.pdf'
    elif filetype.lower() == 'eps':
        suffix = '.eps'
    else:
        suffix = '.png'
    fig1.savefig(os.path.join(save_dir, save_file_base + '_CDF' + suffix), bbox_inches='tight')
    fig2.savefig(os.path.join(save_dir, save_file_base + '_violin' + suffix), bbox_inches='tight')
    fig3.savefig(os.path.join(save_dir, save_file_base + '_swarm.png'), bbox_inches='tight')

    # Part 2
    # plot sorted per "cell" varienace explained (approximate, this is by unique max_ve not cells per se)
    # set up saving paths/dir
    save_dir = os.path.join(save_dir, 'variance explained per cell')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    save_file_base = mouse + '_singleday_var_expl_' + trace_type

    for d in np.unique(df['day']):

        sliced_df = df.loc[(df['day'] == d),:]

        # make a rainbow colormap, HUSL space but does not circle back on itself
        cmap = sns.color_palette('hls', int(np.ceil(1.5*np.unique(df['rank'])[-1])))
        cmap = cmap[0:np.unique(df['rank'])[-1]]

        fig0 = plt.figure(figsize=(20, 6))
        ax0 = sns.swarmplot(x=sliced_df['max_ve'], y=sliced_df['variance_explained'],
                            hue=sliced_df['rank'], palette=cmap)
        lg = ax0.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
        lg.set_title('rank')
        ax0.set_xlabel('cell count')
        x_lim = ax0.get_xlim()
        ticks = ax0.get_xticks()
        new_ticks = [t for t in ticks[10::10]]
        ax0.set_xticks(new_ticks)
        ax0.set_xticklabels(np.arange(10, len(ticks), 10))
        ax0.set_title(mouse + ', Variance explained per cell, day ' + str(d))

        fig0.savefig(os.path.join(save_dir, save_file_base + '_day_' + str(d)
                     + suffix), bbox_inches='tight')
        plt.close()


"""
----------------------------- PAIR DAY PLOTS -----------------------------
"""


def pairday_qc(
        mouse,
        trace_type='zscore',
        cs='',
        warp=False,
        word=None,
        verbose=False):
    """
    Plot similarity and error plots for TCA decomposition ensembles.

    Parameters:
    -----------
    mouse : str; mouse name
    trace_type : str; dff, zscore, deconvolved

    Returns:
    --------
    Saves figures to .../analysis folder  .../qc
    """

    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}

    # plotting options for the unconstrained and nonnegative models.
    plot_options = {
      'cp_als': {
        'line_kw': {
          'color': 'green',
          'label': 'cp_als',
        },
        'scatter_kw': {
          'color': 'green',
          'alpha': 0.5,
        },
      },
      'ncp_hals': {
        'line_kw': {
          'color': 'blue',
          'alpha': 0.5,
          'label': 'ncp_hals',
        },
        'scatter_kw': {
          'color': 'blue',
          'alpha': 0.5,
        },
      },
      'ncp_bcd': {
        'line_kw': {
          'color': 'red',
          'alpha': 0.5,
          'label': 'ncp_bcd',
        },
        'scatter_kw': {
          'color': 'red',
          'alpha': 0.5,
        },
      },
    }

    days = flow.DateSorter.frommeta(mice=[mouse], tags=None)

    for c, day1 in enumerate(days, 0):
        try:
            day2 = days[c+1]
        except IndexError:
            print('done.')
            break

        # load
        load_dir = paths.tca_path(mouse, 'pair', pars=pars, word=word)
        tensor_path = os.path.join(load_dir, str(day1.mouse) + '_' + str(day1.date)
                         + '_' + str(day2.date) + '_pair_decomp_' + str(trace_type) + '.npy')
        if not os.path.isfile(tensor_path): continue

        # save
        save_dir = paths.tca_plots(mouse, 'pair', pars=pars, word=word)
        save_dir = os.path.join(save_dir, 'qc')
        if not os.path.isdir(save_dir): os.mkdir(save_dir)
        error_path = os.path.join(save_dir, str(day1.mouse) + '_' + str(day1.date)
                         + '_' + str(day2.date) + '_objective.pdf')
        sim_path = os.path.join(save_dir, str(day1.mouse) + '_' + str(day1.date)
                         + '_' + str(day2.date) + '_similarity.pdf')

        # load your data
        ensemble = np.load(tensor_path)
        ensemble = ensemble.item()

        # plot error and similarity plots across rank number
        plt.figure()
        for m in ensemble:
            tt.plot_objective(ensemble[m], **plot_options[m])  # ax=ax[0])
        plt.legend()
        plt.title('Objective Function')
        plt.savefig(error_path)
        if verbose:
            plt.show()
        plt.clf()

        for m in ensemble:
            tt.plot_similarity(ensemble[m], **plot_options[m])  # ax=ax[1])
        plt.legend()
        plt.title('Iteration Similarity')
        plt.savefig(sim_path)
        if verbose:
            plt.show()
        plt.close()


def pairday_factors(
        mouse,
        trace_type='zscore',
        cs='',
        warp=False,
        word=None,
        verbose=False):
    """
    Plot TCA factors for all days and ranks/components for
    TCA decomposition ensembles.

    Parameters:
    -----------
    mouse : str; mouse name
    trace_type : str; dff, zscore, deconvolved
    method : str; TCA fit method from tensortools

    Returns:
    --------
    Saves figures to .../analysis folder  .../factors
    """

    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}

    # plotting options for the unconstrained and nonnegative models.
    plot_options = {
      'cp_als': {
        'line_kw': {
          'color': 'red',
          'label': 'cp_als',
        },
        'scatter_kw': {
          'color': 'green',
          'alpha': 0.5,
        },
        'bar_kw': {
          'color': 'blue',
          'alpha': 0.5,
        },
      },
      'ncp_hals': {
        'line_kw': {
          'color': 'red',
          'label': 'ncp_hals',
        },
        'scatter_kw': {
          'color': 'green',
          'alpha': 0.5,
        },
        'bar_kw': {
          'color': 'blue',
          'alpha': 0.5,
        },
      },
      'ncp_bcd': {
        'line_kw': {
          'color': 'red',
          'label': 'ncp_bcd',
        },
        'scatter_kw': {
          'color': 'green',
          'alpha': 0.5,
        },
        'bar_kw': {
          'color': 'blue',
          'alpha': 0.5,
        },
      },
    }

    days = flow.DateSorter.frommeta(mice=[mouse], tags=None)

    for c, day1 in enumerate(days, 0):
        try:
            day2 = days[c+1]
        except IndexError:
            print('done.')
            break

        # load
        load_dir = paths.tca_path(mouse, 'pair', pars=pars, word=word)
        tensor_path = os.path.join(load_dir, str(day1.mouse) + '_'
                                   + str(day1.date) + '_' + str(day2.date)
                                   + '_pair_decomp_' + str(trace_type)
                                   + '.npy')
        if not os.path.isfile(tensor_path): continue

        # save
        save_dir = paths.tca_plots(mouse, 'pair', pars=pars, word=word)
        save_dir = os.path.join(save_dir, 'factors')
        if not os.path.isdir(save_dir): os.mkdir(save_dir)

        # load your data
        ensemble = np.load(tensor_path)
        ensemble = ensemble.item()


        # make necessary dirs
        date_dir = os.path.join(
            save_dir, str(day1.date) + '_' + str(day2.date) + ' ' + method)
        if not os.path.isdir(date_dir):
            os.mkdir(date_dir)

        # sort neuron factors by component they belong to most
        sort_ensemble, my_sorts = tca._sortfactors(ensemble[method])

        for r in sort_ensemble.results:

            fig = tt.plot_factors(sort_ensemble.results[r][0].factors,
                            plots=['bar', 'line', 'scatter'],
                            axes=None,
                            scatter_kw=plot_options[method]['scatter_kw'],
                            line_kw=plot_options[method]['line_kw'],
                            bar_kw=plot_options[method]['bar_kw'])

            ax = fig[0].axes
            ax[0].set_title('Neuron factors')
            ax[1].set_title('Temporal factors')
            ax[2].set_title('Trial factors')

            count = 1
            for k in range(0, len(ax)):
                if np.mod(k+1, 3) == 1:
                    ax[k].set_ylabel('Component #' + str(count), rotation=0,
                                     labelpad=45, verticalalignment='center',
                                     fontstyle='oblique')
                    count = count + 1

            # Show plots.
            plt.savefig(
                os.path.join(date_dir, 'rank_' + str(int(r)) + '.png'),
                bbox_inches='tight')
            if verbose:
                plt.show()
            plt.close()


def pairday_factors_annotated(
        mouse,
        trace_type='zscore_day',
        cs='',
        warp=False,
        word=None,
        method='ncp_bcd',
        extra_col=4,
        alpha=0.6,
        plot_running=True,
        scale_y=False,
        filetype='pdf',
        verbose=False):

    """
    Plot TCA factors with trial metadata annotations for all days
    and ranks/components for TCA decomposition ensembles.

    Parameters:
    -----------
    mouse : str
        Mouse name.
    trace_type : str
        dff, zscore, zscore_iti, zscore_day, deconvolved
    method : str
        TCA fit method from tensortools
    cs : str
        Cs stimuli to include, plus/minus/neutral, 0/135/270, etc. '' empty
        includes all stimuli
    warp : bool
        Use traces with time-warped outcome.
    extra_col : int
        Number of columns to add to the original three factor columns
    alpha : float
        Value between 0 and 1 for transparency of markers
    plot_running : bool
        Include trace of scaled (to plot max) average running speed during trial
    verbose : bool
        Show plots as they are made.

    Returns:
    --------
    Saves figures to .../analysis folder  .../factors annotated
    """

    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}

    # use matplotlib plotting defaults
    mpl.rcParams.update(mpl.rcParamsDefault)

    # plotting options for the unconstrained and nonnegative models.
    plot_options = {
      'cp_als': {
        'line_kw': {
          'color': 'red',
          'label': 'cp_als',
        },
        'scatter_kw': {
          'color': 'green',
          'alpha': 0.5,
        },
        'bar_kw': {
          'color': 'blue',
          'alpha': 0.5,
        },
      },
      'ncp_hals': {
        'line_kw': {
          'color': 'red',
          'label': 'ncp_hals',
        },
        'scatter_kw': {
          'color': 'green',
          'alpha': 0.5,
        },
        'bar_kw': {
          'color': 'blue',
          'alpha': 0.5,
        },
      },
      'ncp_bcd': {
        'line_kw': {
          'color': 'red',
          'label': 'ncp_bcd',
        },
        'scatter_kw': {
          'color': 'green',
          'alpha': 0.5,
        },
        'bar_kw': {
          'color': 'blue',
          'alpha': 0.5,
        },
      },
    }

    days = flow.DateSorter.frommeta(mice=[mouse], tags=None)

    for c, day1 in enumerate(days, 0):
        try:
            day2 = days[c+1]
        except IndexError:
            print('done.')
            break

        # load dirs
        load_dir = paths.tca_path(mouse, 'pair', pars=pars, word=word)
        tensor_path = os.path.join(load_dir, str(day1.mouse) + '_'
                                   + str(day1.date) + '_' + str(day2.date)
                                   + '_pair_decomp_' + str(trace_type)
                                   + '.npy')
        meta_path = os.path.join(load_dir, str(day1.mouse) + '_' +
                                 str(day1.date) + '_' + str(day2.date)
                                 + '_df_pair_meta.pkl')
        if not os.path.isfile(tensor_path): continue
        if not os.path.isfile(meta_path): continue

        # save dirs
        save_dir = paths.tca_plots(mouse, 'pair', pars=pars, word=word)
        save_dir = os.path.join(save_dir, 'factors annotated')
        if not os.path.isdir(save_dir): os.mkdir(save_dir)
        date_dir = os.path.join(save_dir, str(day1.date) + '_' + str(day2.date)
                                + ' ' + method)
        if not os.path.isdir(date_dir): os.mkdir(date_dir)

        # load your data
        ensemble = np.load(tensor_path)
        ensemble = ensemble.item()
        meta = pd.read_pickle(meta_path)
        orientation = meta['orientation']
        trial_num = np.arange(0, len(orientation))
        condition = meta['condition']
        trialerror = meta['trialerror']
        hunger = deepcopy(meta['hunger'])
        speed = meta['speed']
        dates = meta.reset_index()['date']
        learning_state = meta['learning_state']

        # calculate change indices for days and reversal/learning
        udays = {d: c for c, d in enumerate(np.unique(dates))}
        ndays = np.diff([udays[i] for i in dates])
        day_x = np.where(ndays)[0] + 0.5
        ustate = {d: c for c, d in enumerate(np.unique(learning_state))}
        nstate = np.diff([ustate[i] for i in learning_state])
        lstate_x = np.where(nstate)[0] + 0.5

        # merge hunger and tag info for plotting hunger
        tags = meta['tag']
        hunger[tags == 'disengaged'] = 'disengaged'

        # sort neuron factors by component they belong to most
        sort_ensemble, my_sorts = tca._sortfactors(ensemble[method])

        for r in sort_ensemble.results:

            U = sort_ensemble.results[r][0].factors

            fig, axes = plt.subplots(U.rank, U.ndim + extra_col,
                                     figsize=(9 + extra_col, U.rank))
            figt = tt.plot_factors(
                U, plots=['bar', 'line', 'scatter'],
                axes=None,
                fig=fig,
                scatter_kw=plot_options[method]['scatter_kw'],
                line_kw=plot_options[method]['line_kw'],
                bar_kw=plot_options[method]['bar_kw'])
            ax = figt[0].axes
            ax[0].set_title('Neuron factors')
            ax[1].set_title('Temporal factors')
            ax[2].set_title('Trial factors')

            # add title to whole figure
            ax[0].text(-1.2, 4, '\n' + mouse + ': \n\nrank: ' + str(int(r))
                       + '\nmethod: ' + method + ' \ndates: '
                       + str(day1.date) + ' - ' + str(day2.date),
                       fontsize=12, transform=ax[0].transAxes,
                       color='#969696')

            # reshape for easier indexing
            ax = np.array(ax).reshape((U.rank, -1))

            # rescale the y-axis for trials
            if scale_y:
                for i in range(U.rank):
                    y_lim = np.array(ax[i, 2].get_ylim())*0.8
                    y_ticks = ax[i, 2].get_yticks()
                    y_ticks[-1] = y_lim[-1]
                    y_ticks = np.round(y_ticks, 2)
                    # y_tickl = [str(y) for y in y_ticks]
                    ax[i, 2].set_ylim(y_lim)
                    ax[i, 2].set_yticks(y_ticks)
                    ax[i, 2].set_yticklabels(y_ticks)

            # add a line for stim onset and offset
            # NOTE: assumes downsample, 1 sec before onset, 3 sec stim
            for i in range(U.rank):
                y_lim = ax[i, 1].get_ylim()
                ons = 15.5*1
                offs = ons+15.5*3
                ax[i, 1].plot([ons, ons], y_lim, ':k')
                ax[i, 1].plot([offs, offs], y_lim, ':k')

            for col in range(3, 3+extra_col):
                for i in range(U.rank):

                    # get axis values
                    y_lim = ax[i, 2].get_ylim()
                    x_lim = ax[i, 2].get_xlim()
                    y_ticks = ax[i, 2].get_yticks()
                    y_tickl = ax[i, 2].get_yticklabels()
                    x_ticks = ax[i, 2].get_xticks()
                    x_tickl = ax[i, 2].get_xticklabels()

                    # running
                    if plot_running:
                        scale_by = np.nanmax(speed)/y_lim[1]
                        if not np.isnan(scale_by):
                            ax[i, col].plot(
                                np.array(speed.tolist())/scale_by,
                                color=[1, 0.1, 0.6, 0.2])
                            # , label='speed')

                    # Orientation - main variable to plot
                    if col == 3:
                        ori_vals = [0, 135, 270]
                        color_vals = [[0.28, 0.68, 0.93, alpha],
                                      [0.84, 0.12, 0.13, alpha],
                                      [0.46, 0.85, 0.47, alpha]]
                        for k in range(0, 3):
                            ax[i, col].plot(
                                trial_num[orientation==ori_vals[k]],
                                U.factors[2][orientation==ori_vals[k], i], 'o',
                                label=str(ori_vals[k]), color=color_vals[k],
                                markersize=2)
                        if i == 0:
                            ax[i, col].set_title('Orientation')
                            ax[i, col].legend(
                                bbox_to_anchor=(0.5,1.02), loc='lower center',
                                borderaxespad=2.5)
                    elif col == 4:
                        cs_vals = ['plus', 'minus', 'neutral']
                        cs_labels = ['plus', 'minus', 'neutral']
                        color_vals = [[0.46, 0.85, 0.47, alpha],
                                      [0.84, 0.12, 0.13, alpha],
                                      [0.28, 0.68, 0.93, alpha]]
                        col = 4
                        for k in range(0,3):
                            ax[i, col].plot(
                                trial_num[condition==cs_vals[k]],
                                U.factors[2][condition==cs_vals[k], i], 'o',
                                label=str(cs_labels[k]), color=color_vals[k],
                                markersize=2)
                        if i == 0:
                            ax[i, col].set_title('Condition')
                            ax[i, col].legend(
                                bbox_to_anchor=(0.5,1.02), loc='lower center',
                                borderaxespad=2.5)
                    elif col == 5:
                        trialerror_vals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                        trialerror_labels = ['hit',
                                             'miss',
                                             'neutral correct reject',
                                             'neutral false alarm',
                                             'minus correct reject',
                                             'minus false alarm',
                                             'blank correct reject',
                                             'blank false alarm',
                                             'pav early licking',
                                             'pav late licking',]
                        for k in trialerror_vals:
                            ax[i, col].plot(
                                trial_num[trialerror==trialerror_vals[k]],
                                U.factors[2][trialerror==trialerror_vals[k], i],
                                'o', label=str(trialerror_labels[k]), alpha=0.8,
                                markersize=2)
                        if i == 0:
                            ax[i, col].set_title('Trialerror')
                            ax[i, col].legend(
                                bbox_to_anchor=(0.5,1.02), loc='lower center',
                                borderaxespad=2.5)

                    elif col == 6:
                        h_vals = ['hungry', 'sated', 'disengaged']
                        h_labels = ['hungry', 'sated', 'disengaged']
                        color_vals = [[1, 0.6, 0.3, alpha],
                                      [0.7, 0.9, 0.4, alpha],
                                      [0.6, 0.5, 0.6, alpha],
                                      [0.0, 0.9, 0.4, alpha]]
                        for k in range(0,3):
                            ax[i, col].plot(
                                trial_num[hunger==h_vals[k]],
                                U.factors[2][hunger==h_vals[k], i],
                                'o', label=str(h_labels[k]),
                                color=color_vals[k], markersize=2)
                        if i == 0:
                            ax[i, col].set_title('State')
                            ax[i, col].legend(
                                bbox_to_anchor=(0.5,1.02), loc='lower center',
                                borderaxespad=2.5)

                    # plot days, reversal, or learning lines if there are any
                    if col >= 2:
                        y_lim = ax[i, col].get_ylim()
                        if len(day_x) > 0:
                            for k in day_x:
                                ax[i, col].plot(
                                    [k, k], y_lim, color='#969696',
                                    linewidth=1)
                        if len(lstate_x) > 0:
                            ls_vals = ['naive', 'learning', 'reversal1']
                            ls_colors = ['#66bd63', '#d73027', '#a50026']
                            for k in lstate_x:
                                ls = learning_state[int(k-0.5)]
                                ax[i, col].plot(
                                    [k, k], y_lim,
                                    color=ls_colors[ls_vals.index(ls)],
                                    linewidth=1.5)

                    # set axes labels
                    ax[i, col].set_yticks(y_ticks)
                    ax[i, col].set_yticklabels(y_tickl)
                    ax[i, col].set_ylim(y_lim)
                    ax[i, col].set_xlim(x_lim)

                    # format axes
                    ax[i, col].locator_params(nbins=4)
                    ax[i, col].spines['top'].set_visible(False)
                    ax[i, col].spines['right'].set_visible(False)
                    ax[i, col].xaxis.set_tick_params(direction='out')
                    ax[i, col].yaxis.set_tick_params(direction='out')
                    ax[i, col].yaxis.set_ticks_position('left')
                    ax[i, col].xaxis.set_ticks_position('bottom')

                    # remove xticks on all but bottom row
                    if i + 1 != U.rank:
                        plt.setp(ax[i, col].get_xticklabels(), visible=False)

                    if col == 3:
                        ax[i, 0].set_ylabel('Component #' + str(i+1), rotation=0,
                          labelpad=45, verticalalignment='center', fontstyle='oblique')

            if filetype.lower() == 'pdf':
                suffix = '.pdf'
            elif filetype.lower() == 'eps':
                suffix = '.eps'
            else:
                suffix = '.png'
            plt.savefig(os.path.join(date_dir, 'rank_' + str(int(r)) + suffix),
                                     bbox_inches='tight')
            if verbose:
                plt.show()
            plt.close()


def pairday_qc_summary(
        mouse,
        trace_type='zscore_day',
        method='ncp_bcd',
        cs='',
        warp=False,
        word=None,
        verbose=False):
    """
    Plot similarity and objective (measure of reconstruction error) plots
    across all days for TCA decomposition ensembles.

    Parameters:
    -----------
    mouse : str; mouse name
    trace_type : str; dff, zscore, deconvolved
    method : str; TCA fit method from tensortools

    Returns:
    --------
    Saves figures to .../analysis folder  .../qc
    """

    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}

    days = flow.DateSorter.frommeta(mice=[mouse], tags=None)

    cmap = sns.color_palette('hls', n_colors=len(days))

    # create figure and axes
    buffer = 5
    right_pad = 5

    fig0 = plt.figure(figsize=(10, 8))
    gs0 = GridSpec(100, 100, figure=fig0, left=0.05, right=.95, top=.95, bottom=0.05)
    ax0 = fig0.add_subplot(gs0[10:90-buffer, :90-right_pad])

    fig1 = plt.figure(figsize=(10, 8))
    gs1 = GridSpec(100, 100, figure=fig1, left=0.05, right=.95, top=.95, bottom=0.05)
    ax1 = fig1.add_subplot(gs1[10:90-buffer, :90-right_pad])

    # plt.figure()
    for c, day1 in enumerate(days, 0):
        try:
            day2 = days[c+1]
        except IndexError:
            print('done.')
            break

        # load dirs
        load_dir = paths.tca_path(mouse, 'pair', pars=pars, word=word)
        tensor_path = os.path.join(load_dir, str(day1.mouse) + '_'
                                   + str(day1.date) + '_' + str(day2.date)
                                   + '_pair_decomp_' + str(trace_type)
                                   + '.npy')
        if not os.path.isfile(tensor_path): continue

        # save dirs
        save_dir = paths.tca_plots(mouse, 'pair', pars=pars, word=word)
        save_dir = os.path.join(save_dir, 'qc')
        if not os.path.isdir(save_dir): os.mkdir(save_dir)
        error_path = os.path.join(save_dir, str(day1.mouse) + '_summary_objective.pdf')
        sim_path = os.path.join(save_dir, str(day1.mouse) + '_summary_similarity.pdf')

        # plotting options for the unconstrained and nonnegative models.
        plot_options = {
          'cp_als': {
            'line_kw': {
              'color': cmap[c],
              'label': 'pair ' + str(c),
            },
            'scatter_kw': {
              'color': cmap[c],
              'alpha': 0.5,
            },
          },
          'ncp_hals': {
            'line_kw': {
              'color': cmap[c],
              'alpha': 0.5,
              'label': 'pair ' + str(c),
            },
            'scatter_kw': {
              'color': cmap[c],
              'alpha': 0.5,
            },
          },
          'ncp_bcd': {
            'line_kw': {
              'color': cmap[c],
              'alpha': 0.5,
              'label': 'pair ' + str(c),
            },
            'scatter_kw': {
              'color': cmap[c],
              'alpha': 0.5,
            },
          },
        }

        # load your data
        ensemble = np.load(tensor_path)
        ensemble = ensemble.item()

        # plot error and similarity plots across rank number
        tt.plot_objective(ensemble[method], **plot_options[method], ax=ax0)
        tt.plot_similarity(ensemble[method], **plot_options[method], ax=ax1)

    # add legend, title
    ax0.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
    ax0.set_title('Objective Function: ' + str(method) + ', ' + mouse)
    ax1.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
    ax1.set_title('Iteration Similarity: ' + str(method) + ', ' + mouse)

    # save figs
    fig0.savefig(error_path, bbox_inches='tight')
    fig1.savefig(sim_path, bbox_inches='tight')

    if verbose:
        fig0.show()
        fig1.show()


def pairday_varex_summary(
        mouse,
        trace_type='zscore_day',
        method='ncp_bcd',
        cs='',
        warp=False,
        verbose=False):
    """
    Plot reconstruction error as variance explained across all days for
    TCA decomposition ensembles.

    Parameters:
    -----------
    mouse : str; mouse name
    trace_type : str; dff, zscore, deconvolved
    method : str; TCA fit method from tensortools

    Returns:
    --------
    Saves figures to .../analysis folder  .../qc
    """

    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}

    days = flow.DateSorter.frommeta(mice=[mouse], tags=None)

    cmap = sns.color_palette(sns.cubehelix_palette(len(days)))

    # create figure and axes
    buffer = 5
    right_pad = 5
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(100, 100, figure=fig, left=0.05, right=.95, top=.95, bottom=0.05)
    ax = fig.add_subplot(gs[10:90-buffer, :90-right_pad])

    for c, day1 in enumerate(days, 0):
        try:
            day2 = days[c+1]
        except IndexError:
            print('done.')
            break

        # load dirs
        load_dir = paths.tca_path(mouse, 'pair', pars=pars, word=word)
        tensor_path = os.path.join(load_dir, str(day1.mouse) + '_'
                                   + str(day1.date) + '_' + str(day2.date)
                                   + '_pair_decomp_' + str(trace_type)
                                   + '.npy')
        input_tensor_path = os.path.join(load_dir, str(day1.mouse) + '_'
                                         + str(day1.date) + '_' + str(day2.date)
                                         + '_pair_tensor_' + str(trace_type)
                                         + '.npy')
        if not os.path.isfile(tensor_path): continue
        if not os.path.isfile(input_tensor_path): continue

        # save dirs
        save_dir = paths.tca_plots(mouse, 'pair', pars=pars, word=word)
        save_dir = os.path.join(save_dir, 'qc')
        if not os.path.isdir(save_dir): os.mkdir(save_dir)
        var_path = os.path.join(save_dir, str(day1.mouse)
                                + '_summary_variance_cubehelix.pdf')

        # load your data
        ensemble = np.load(tensor_path)
        ensemble = ensemble.item()
        V = ensemble[method]
        X = np.load(input_tensor_path)

        # get reconstruction error as variance explained
        var, var_s, x, x_s = [], [], [], []
        for r in V.results:
            bU = V.results[r][0].factors.full()
            var.append((np.var(X) - np.var(X - bU)) / np.var(X))
            x.append(r)
            for it in range(0, len(V.results[r])):
                U = V.results[r][it].factors.full()
                var_s.extend([(np.var(X) - np.var(X - U)) / np.var(X)])
                x_s.extend([r])

        # mean response of neuron across trials
        mU = np.mean(X, axis=2, keepdims=True) * np.ones((1, 1, np.shape(X)[2]))
        var_mean = (np.var(X) - np.var(X - mU)) / np.var(X)

        # smoothed response of neuron across time
        smU = np.convolve(X.reshape((X.size)), np.ones(5, dtype=np.float64)/5, 'same').reshape(np.shape(X))
        var_smooth = (np.var(X) - np.var(X - smU)) / np.var(X)

        # plot
        R = np.max([r for r in V.results.keys()])
        ax.scatter(x_s, var_s, color=cmap[c], alpha=0.5)
        ax.scatter([R+2], var_mean, color=cmap[c], alpha=0.5)
        ax.scatter([R+4], var_smooth, color=cmap[c], alpha=0.5)
        ax.plot(x, var, label=('pair ' + str(c)), color=cmap[c])
        ax.plot([R+1.5, R+2.5], [var_mean, var_mean], color=cmap[c])
        ax.plot([R+3.5, R+4.5], [var_smooth, var_smooth], color=cmap[c])

    # add labels/titles
    x_labels = [str(R) for R in V.results]
    x_labels.extend(['', 'mean\n cell\n response', '', 'smooth\n response\n (0.3s)'])
    ax.set_xticks(range(1, len(V.results) + 5))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('model rank')
    ax.set_ylabel('fractional variance explained')
    ax.set_title('Variance Explained: ' + str(method) + ', ' + mouse)
    ax.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)

    fig.savefig(var_path, bbox_inches='tight')


def pairday_varex_percell(
        mouse,
        method='ncp_bcd',
        trace_type='zscore_day',
        cs='',
        warp=False,
        word=None,
        ve_min=0.05):
    """
    Plot TCA reconstruction error as variance explained per cell
    for TCA decomposition. Create folder of variance explained per cell
    swarm plots. Calculate summary plots of 'fraction of maximum variance
    explained' per cell by rank for all cells given a certain (ve_min) threshold
    for maximum variance explained

    Parameters:
    -----------
    mouse : str; mouse name
    trace_type : str; dff, zscore, deconvolved
    method : str; TCA fit method from tensortools
    ve_min: float; minimum variance explained for best rank per cell
                   to be included in summary of fraction of maximum variance
                   explained

    Returns:
    --------
    Saves figures to .../analysis folder/ .../qc
                                             .../variance explained per cell

    """

    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}

    days = flow.DateSorter.frommeta(mice=[mouse], tags=None)

    ve, ve_max, ve_frac, rank_num, day_num, cell_num = [], [], [], [], [], []
    for c, day1 in enumerate(days, 0):
        try:
            day2 = days[c+1]
        except IndexError:
            print('done.')
            break

        # get dirs for loading
        load_dir = paths.tca_path(mouse, 'pair', pars=pars, word=word)
        tensor_path = os.path.join(load_dir, str(day1.mouse) + '_'
                                   + str(day1.date) + '_' + str(day2.date)
                                   + '_pair_decomp_' + str(trace_type)
                                   + '.npy')
        input_tensor_path = os.path.join(load_dir, str(day1.mouse) + '_'
                                         + str(day1.date) + '_' + str(day2.date)
                                         + '_pair_tensor_' + str(trace_type)
                                         + '.npy')
        if not os.path.isfile(tensor_path): continue
        if not os.path.isfile(input_tensor_path): continue

        # load your data
        ensemble = np.load(tensor_path)
        ensemble = ensemble.item()
        V = ensemble[method]
        X = np.load(input_tensor_path)

        # get reconstruction error as variance explained per cell
        for cell in range(0, np.shape(X)[0]):
            rank_ve_vec = []
            rank_vec = []
            for r in V.results:
                U = V.results[r][0].factors.full()
                Usub = X - U
                rank_ve = (np.var(X[cell, :, :]) - np.var(Usub[cell, :, :])) / np.var(X[cell, :, :])
                rank_ve_vec.append(rank_ve)
                rank_vec.append(r)
            max_ve = np.max(rank_ve_vec)
            ve.extend(rank_ve_vec)
            ve_max.extend([max_ve for s in rank_ve_vec])
            ve_frac.extend(rank_ve_vec / max_ve)
            rank_num.extend(rank_vec)
            day_num.extend([c+1 for s in rank_ve_vec])
            cell_num.extend([cell for s in rank_ve_vec])

    # build pd dataframe of all variance measures
    index = pd.MultiIndex.from_arrays([
    day_num,
    rank_num,
    ve,
    ve_max,
    ve_frac,
    cell_num,
    ],
    names=['day', 'rank', 'variance_explained', 'max_ve', 'frac_ve', 'cell'])
    df = pd.DataFrame(index=index)
    df = df.reset_index()

    # make a rainbow colormap, HUSL space but does not circle back on itself
    cmap = sns.color_palette('hls', int(np.ceil(1.5*np.unique(df['rank'])[-1])))
    cmap = cmap[0:np.unique(df['rank'])[-1]]

    # Part 1
    # slice df, only look at cells with a max variance >5%
    sliced_df2 = df.loc[(df['day']) & (df['max_ve'] >= ve_min), :]

    # CDF plot
    fig1 = plt.figure(figsize=(15,9))
    for i in np.unique(sliced_df2['rank']):
        input_ve = sliced_df2.loc[(sliced_df2['rank'] == i),'frac_ve']
        ax = sns.distplot(
            input_ve, kde_kws={'cumulative': True, 'lw': 2, 'color': cmap[i-1],
            'label': str(i)}, hist=False)
        lg = ax.legend(
            bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
        lg.set_title('rank')
        ax.set_title(
            mouse + ', Fraction of maximum variance explained per cell, CDF')
        ax.set_xlabel('Fraction of maximum variance explained')

    # swarm plot
    fig2 =plt.figure(figsize=(18,6))
    ax2 = sns.violinplot(
        x=sliced_df2['rank'], y=sliced_df2['frac_ve'], size=3, alpha=1,
        inner=None, palette=cmap)
    ax2.set_title(
        mouse + ', Fraction of maximum variance explained per cell, violin')
    ax2.set_ylabel('Fraction of maximum variance explained')

    # swarm plot
    fig3 = plt.figure(figsize=(18,6))
    ax3 = sns.swarmplot(
        x=sliced_df2['rank'], y=sliced_df2['frac_ve'], size=2, alpha=1,
        palette=cmap)
    ax3.set_title(
        mouse + ', Fraction of maximum variance explained per cell, swarm')
    ax3.set_ylabel('Fraction of maximum variance explained')

    # set up saving paths/dir
    save_dir = paths.tca_plots(mouse, 'pair', pars=pars, word=word)
    save_dir = os.path.join(save_dir, 'qc')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    save_file_base = mouse + '_pairday_frac_max_var_expl_' + trace_type

    # save
    fig1.savefig(
        os.path.join(save_dir, save_file_base + '_CDF.pdf'),
        bbox_inches='tight')
    fig2.savefig(
        os.path.join(save_dir, save_file_base + '_violin.pdf'),
        bbox_inches='tight')
    fig3.savefig(
        os.path.join(save_dir, save_file_base + '_swarm.pdf'),
        bbox_inches='tight')

    # Part 2
    # plot sorted per "cell" varienace explained (approximate, this is by
    # unique max_ve not cells per se)
    # set up saving paths/dir
    save_dir = os.path.join(save_dir, 'variance explained per cell')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    save_file_base = mouse + '_pairday_var_expl_' + trace_type

    for d in np.unique(df['day']):

        sliced_df = df.loc[(df['day'] == d),:]

        # make a rainbow colormap, HUSL space but does not circle back on itself
        cmap = sns.color_palette(
            'hls', int(np.ceil(1.5*np.unique(df['rank'])[-1])))
        cmap = cmap[0:np.unique(df['rank'])[-1]]

        fig0 = plt.figure(figsize=(20, 6))
        ax0 = sns.swarmplot(
            x=sliced_df['max_ve'], y=sliced_df['variance_explained'],
            hue=sliced_df['rank'], palette=cmap)
        lg = ax0.legend(
            bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
        lg.set_title('rank')
        ax0.set_xlabel('cell count')
        x_lim = ax0.get_xlim()
        ticks = ax0.get_xticks()
        new_ticks = [t for t in ticks[10::10]]
        ax0.set_xticks(new_ticks)
        ax0.set_xticklabels(np.arange(10, len(ticks), 10))
        ax0.set_title(mouse + ', Variance explained per cell, day ' + str(d))

        fig0.savefig(os.path.join(save_dir, save_file_base + '_day_' + str(d)
                     + '.png'), bbox_inches='tight')
        plt.close()


"""
----------------------------- SINGLE DAY PLOTS -----------------------------
"""

def singleday_qc(
        mouse,
        trace_type='zscore_day',
        cs='',
        warp=False,
        word=None,
        verbose=False):
    """
    Plot similarity and error plots for TCA decomposition ensembles.

    Parameters:
    -----------
    mouse : str; mouse name
    trace_type : str; dff, zscore, deconvolved

    Returns:
    --------
    Saves figures to .../analysis folder  .../qc
    """

    # plotting options for the unconstrained and nonnegative models.
    plot_options = {
      'cp_als': {
        'line_kw': {
          'color': 'green',
          'label': 'cp_als',
        },
        'scatter_kw': {
          'color': 'green',
          'alpha': 0.5,
        },
      },
      'ncp_hals': {
        'line_kw': {
          'color': 'blue',
          'alpha': 0.5,
          'label': 'ncp_hals',
        },
        'scatter_kw': {
          'color': 'blue',
          'alpha': 0.5,
        },
      },
      'ncp_bcd': {
        'line_kw': {
          'color': 'red',
          'alpha': 0.5,
          'label': 'ncp_bcd',
        },
        'scatter_kw': {
          'color': 'red',
          'alpha': 0.5,
        },
      },
    }

    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}

    days = flow.DateSorter.frommeta(mice=[mouse], tags=None)

    for day1 in days:

        # load dir
        load_dir = paths.tca_path(mouse, 'pair', pars=pars, word=word)
        tensor_path = os.path.join(load_dir, str(day1.mouse) + '_' + str(day1.date)
                                   + '_single_decomp_' + str(trace_type) + '.npy')
        if not os.path.isfile(tensor_path): continue

        # save dir
        save_dir = paths.tca_plots(mouse, 'single', pars=pars, word=word)
        save_dir = os.path.join(save_dir, 'qc')
        if not os.path.isdir(save_dir): os.mkdir(save_dir)
        error_path = os.path.join(save_dir, str(day1.mouse) + '_' + str(day1.date)
                                  + '_objective.pdf')
        sim_path = os.path.join(save_dir, str(day1.mouse) + '_' + str(day1.date)
                                + '_similarity.pdf')

        # load your data
        ensemble = np.load(tensor_path)
        ensemble = ensemble.item()

        # plot error and similarity plots across rank number
        plt.figure()
        for m in ensemble:
            tt.plot_objective(ensemble[m], **plot_options[m])  # ax=ax[0])
        plt.legend()
        plt.title('Objective Function')
        plt.savefig(error_path)
        if verbose:
            plt.show()
        plt.clf()

        for m in ensemble:
            tt.plot_similarity(ensemble[m], **plot_options[m])  # ax=ax[1])
        plt.legend()
        plt.title('Iteration Similarity')
        plt.savefig(sim_path)
        if verbose:
            plt.show()
        plt.close()


def singleday_factors(
        mouse,
        trace_type='zscore_day',
        method='ncp_bcd',
        cs='',
        warp=False,
        word=None,
        verbose=False):
    """
    Plot TCA factors for all days and ranks/components for
    TCA decomposition ensembles.

    Parameters:
    -----------
    mouse : str; mouse name
    trace_type : str; dff, zscore, deconvolved
    method : str; TCA fit method from tensortools

    Returns:
    --------
    Saves figures to .../analysis folder  .../factors
    """

    # plotting options for the unconstrained and nonnegative models.
    plot_options = {
      'cp_als': {
        'line_kw': {
          'color': 'red',
          'label': 'cp_als',
        },
        'scatter_kw': {
          'color': 'green',
          'alpha': 0.5,
        },
        'bar_kw': {
          'color': 'blue',
          'alpha': 0.5,
        },
      },
      'ncp_hals': {
        'line_kw': {
          'color': 'red',
          'label': 'ncp_hals',
        },
        'scatter_kw': {
          'color': 'green',
          'alpha': 0.5,
        },
        'bar_kw': {
          'color': 'blue',
          'alpha': 0.5,
        },
      },
      'ncp_bcd': {
        'line_kw': {
          'color': 'red',
          'label': 'ncp_bcd',
        },
        'scatter_kw': {
          'color': 'green',
          'alpha': 0.5,
        },
        'bar_kw': {
          'color': 'blue',
          'alpha': 0.5,
        },
      },
    }

    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}

    days = flow.DateSorter.frommeta(mice=[mouse], tags=None)

    for day1 in days:

        # load dir
        load_dir = paths.tca_path(mouse, 'single', pars=pars, word=word)
        tensor_path = os.path.join(load_dir, str(day1.mouse) + '_' + str(day1.date)
                                   + '_single_decomp_' + str(trace_type) + '.npy')
        if not os.path.isfile(tensor_path): continue

        # save dir
        save_dir = paths.tca_plots(mouse, 'single', pars=pars, word=word)
        save_dir = os.path.join(save_dir, 'factors')
        if not os.path.isdir(save_dir): os.mkdir(save_dir)
        date_dir = os.path.join(save_dir, str(day1.date) + ' ' + method)
        if not os.path.isdir(date_dir): os.mkdir(date_dir)

        # load your data
        ensemble = np.load(tensor_path)
        ensemble = ensemble.item()

        # sort neuron factors by component they belong to most
        sort_ensemble, my_sorts = tca._sortfactors(ensemble[method])

        for r in sort_ensemble.results:

            fig = tt.plot_factors(sort_ensemble.results[r][0].factors, plots=['bar', 'line', 'scatter'],
                            axes=None,
                            scatter_kw=plot_options[method]['scatter_kw'],
                            line_kw=plot_options[method]['line_kw'],
                            bar_kw=plot_options[method]['bar_kw'])

            ax = fig[0].axes
            ax[0].set_title('Neuron factors')
            ax[1].set_title('Temporal factors')
            ax[2].set_title('Trial factors')

            count = 1
            for k in range(0, len(ax)):
                if np.mod(k+1, 3) == 1:
                    ax[k].set_ylabel('Component #' + str(count), rotation=0,
                                     labelpad=45, verticalalignment='center', fontstyle='oblique')
                    count = count + 1

            # Show plots.
            plt.savefig(os.path.join(date_dir, 'rank_' + str(int(r)) + '.png'),bbox_inches='tight')
            if verbose:
                plt.show()
            plt.close()


def singleday_factors_annotated(
        mouse,
        trace_type='zscore_day',
        method='ncp_bcd',
        cs='',
        warp=False,
        word=None,
        extra_col=4,
        alpha=0.6,
        plot_running=True,
        filetype='pdf',
        scale_y=False,
        verbose=False):

    """
    Plot TCA factors with trial metadata annotations for all days
    and ranks/components for TCA decomposition ensembles.

    Parameters:
    -----------
    mouse : str
        Mouse name.
    trace_type : str
        dff, zscore, zscore_iti, zscore_day, deconvolved
    method : str
        TCA fit method from tensortools
    cs : str
        Cs stimuli to include, plus/minus/neutral, 0/135/270, etc. '' empty
        includes all stimuli
    warp : bool
        Use traces with time-warped outcome.
    extra_col : int
        Number of columns to add to the original three factor columns
    alpha : float
        Value between 0 and 1 for transparency of markers
    plot_running : bool
        Include trace of scaled (to plot max) average running speed during trial
    verbose : bool
        Show plots as they are made.

    Returns:
    --------
    Saves figures to .../analysis folder  .../factors annotated
    """

    # plotting options for the unconstrained and nonnegative models.
    plot_options = {
      'cp_als': {
        'line_kw': {
          'color': 'red',
          'label': 'cp_als',
        },
        'scatter_kw': {
          'color': 'green',
          'alpha': 0.5,
        },
        'bar_kw': {
          'color': 'blue',
          'alpha': 0.5,
        },
      },
      'ncp_hals': {
        'line_kw': {
          'color': 'red',
          'label': 'ncp_hals',
        },
        'scatter_kw': {
          'color': 'green',
          'alpha': 0.5,
        },
        'bar_kw': {
          'color': 'blue',
          'alpha': 0.5,
        },
      },
      'ncp_bcd': {
        'line_kw': {
          'color': 'red',
          'label': 'ncp_bcd',
        },
        'scatter_kw': {
          'color': 'green',
          'alpha': 0.5,
        },
        'bar_kw': {
          'color': 'blue',
          'alpha': 0.5,
        },
      },
    }

    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}

    days = flow.DateSorter.frommeta(mice=[mouse], tags=None)

    for day1 in days:

        # load dir
        load_dir = paths.tca_path(mouse, 'single', pars=pars, word=word)
        tensor_path = os.path.join(load_dir, str(day1.mouse) + '_' + str(day1.date)
                                   + '_single_decomp_' + str(trace_type) + '.npy')
        meta_path = os.path.join(load_dir, str(day1.mouse) + '_' + str(day1.date)
                                 + '_df_single_meta.pkl')
        if not os.path.isfile(tensor_path): continue
        if not os.path.isfile(meta_path): continue

        # save dir
        save_dir = paths.tca_plots(mouse, 'single', pars=pars, word=word)
        if scale_y:
            save_tag = ' scaled-y'
        else:
            save_tag = ''
        save_dir = os.path.join(save_dir, 'factors annotated' + save_tag)
        if not os.path.isdir(save_dir): os.mkdir(save_dir)
        date_dir = os.path.join(save_dir, str(day1.date) + ' ' + method)
        if not os.path.isdir(date_dir): os.mkdir(date_dir)

        # load your data
        ensemble = np.load(tensor_path)
        ensemble = ensemble.item()
        meta = pd.read_pickle(meta_path)
        orientation = meta['orientation']
        trial_num = np.arange(0, len(orientation))
        condition = meta['condition']
        trialerror = meta['trialerror']
        hunger = deepcopy(meta['hunger'])
        speed = meta['speed']
        dates = meta.reset_index()['date']
        learning_state = meta['learning_state']

        # calculate change indices for days and reversal/learning
        udays = {d: c for c, d in enumerate(np.unique(dates))}
        ndays = np.diff([udays[i] for i in dates])
        day_x = np.where(ndays)[0] + 0.5
        ustate = {d: c for c, d in enumerate(np.unique(learning_state))}
        nstate = np.diff([ustate[i] for i in learning_state])
        lstate_x = np.where(nstate)[0] + 0.5

        # merge hunger and tag info for plotting hunger
        tags = meta['tag']
        hunger[tags == 'disengaged'] = 'disengaged'

        # sort neuron factors by component they belong to most
        sort_ensemble, my_sorts = tca._sortfactors(ensemble[method])

        for r in sort_ensemble.results:

            U = sort_ensemble.results[r][0].factors

            fig, axes = plt.subplots(U.rank, U.ndim + extra_col, figsize=(9 + extra_col, U.rank))
            figt = tt.plot_factors(U, plots=['bar', 'line', 'scatter'],
                            axes=None,
                            fig=fig,
                            scatter_kw=plot_options[method]['scatter_kw'],
                            line_kw=plot_options[method]['line_kw'],
                            bar_kw=plot_options[method]['bar_kw'])
            ax = figt[0].axes
            ax[0].set_title('Neuron factors')
            ax[1].set_title('Temporal factors')
            ax[2].set_title('Trial factors')

            # add title to whole figure
            ax[0].text(-1.2, 4, '\n' + mouse + ': \n\nrank: ' + str(int(r))
                       + '\nmethod: ' + method + ' \ndate: '
                       + str(day1.date),
                       fontsize=12, transform=ax[0].transAxes,
                       color='#969696')

            # reshape for easier indexing
            ax = np.array(ax).reshape((U.rank, -1))

            # rescale the y-axis for trials
            if scale_y:
                for i in range(U.rank):
                    y_lim = np.array(ax[i, 2].get_ylim())*0.8
                    y_ticks = ax[i, 2].get_yticks()
                    y_ticks[-1] = y_lim[-1]
                    y_ticks = np.round(y_ticks, 2)
                    # y_tickl = [str(y) for y in y_ticks]
                    ax[i, 2].set_ylim(y_lim)
                    ax[i, 2].set_yticks(y_ticks)
                    ax[i, 2].set_yticklabels(y_ticks)

            # add a line for stim onset and offset
            # NOTE: assumes downsample, 1 sec before onset, 3 sec stim
            for i in range(U.rank):
                y_lim = ax[i, 1].get_ylim()
                ons = 15.5*1
                offs = ons+15.5*3
                ax[i, 1].plot([ons, ons], y_lim, ':k')
                ax[i, 1].plot([offs, offs], y_lim, ':k')

            for col in range(3, 3+extra_col):
                for i in range(U.rank):

                    # get axis values
                    y_lim = ax[i, 2].get_ylim()
                    x_lim = ax[i, 2].get_xlim()
                    y_ticks = ax[i, 2].get_yticks()
                    y_tickl = ax[i, 2].get_yticklabels()
                    x_ticks = ax[i, 2].get_xticks()
                    x_tickl = ax[i, 2].get_xticklabels()

                    # running
                    if plot_running:
                        scale_by = np.nanmax(speed)/y_lim[1]
                        if not np.isnan(scale_by):
                            ax[i, col].plot(np.array(speed.tolist())/scale_by, color=[1, 0.1, 0.6, 0.2])
                            # , label='speed')

                    # Orientation - main variable to plot
                    if col == 3:
                        ori_vals = [0, 135, 270]
                        color_vals = [[0.28, 0.68, 0.93, alpha], [0.84, 0.12, 0.13, alpha],
                                      [0.46, 0.85, 0.47, alpha]]
                        for k in range(0, 3):
                            ax[i, col].plot(trial_num[orientation == ori_vals[k]],
                                            U.factors[2][orientation == ori_vals[k], i], 'o',
                                            label=str(ori_vals[k]), color=color_vals[k], markersize=2)
                        if i == 0:
                            ax[i, col].set_title('Orientation')
                            ax[i, col].legend(bbox_to_anchor=(0.5,1.02), loc='lower center',
                                              borderaxespad=2.5)
                    elif col == 4:
                        cs_vals = ['plus', 'minus', 'neutral']
                        cs_labels = ['plus', 'minus', 'neutral']
                        color_vals = [[0.46, 0.85, 0.47, alpha], [0.84, 0.12, 0.13, alpha],
                                      [0.28, 0.68, 0.93, alpha]]
                        col = 4
                        for k in range(0, 3):
                            ax[i, col].plot(trial_num[condition == cs_vals[k]],
                                            U.factors[2][condition == cs_vals[k], i], 'o',
                                            label=str(cs_labels[k]), color=color_vals[k], markersize=2)
                        if i == 0:
                            ax[i, col].set_title('Condition')
                            ax[i, col].legend(bbox_to_anchor=(0.5,1.02), loc='lower center',
                                              borderaxespad=2.5)
                    elif col == 5:
                        trialerror_vals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                        trialerror_labels = ['hit',
                                             'miss',
                                             'neutral correct reject',
                                             'neutral false alarm',
                                             'minus correct reject',
                                             'minus false alarm',
                                             'blank correct reject',
                                             'blank false alarm',
                                             'pav early licking',
                                             'pav late licking',]
                        for k in trialerror_vals:
                            ax[i, col].plot(trial_num[trialerror == trialerror_vals[k]],
                                            U.factors[2][trialerror == trialerror_vals[k], i], 'o',
                                            label=str(trialerror_labels[k]), alpha=0.8, markersize=2)
                        if i == 0:
                            ax[i, col].set_title('Trialerror')
                            ax[i, col].legend(bbox_to_anchor=(0.5, 1.02), loc='lower center',
                                              borderaxespad=2.5)

                    elif col == 6:
                        h_vals = ['hungry', 'sated', 'disengaged']
                        h_labels = ['hungry', 'sated', 'disengaged']
                        color_vals = [[1, 0.6, 0.3, alpha], [0.7, 0.9, 0.4, alpha],
                                      [0.6, 0.5, 0.6, alpha], [0.0, 0.9, 0.4, alpha]]
                        for k in range(0, 3):
                            ax[i, col].plot(trial_num[hunger == h_vals[k]],
                                            U.factors[2][hunger == h_vals[k], i], 'o',
                                            label=str(h_labels[k]), color=color_vals[k], markersize=2)
                        if i == 0:
                            ax[i, col].set_title('State')
                            ax[i, col].legend(bbox_to_anchor=(0.5, 1.02), loc='lower center',
                                              borderaxespad=2.5)

                    # plot days, reversal, or learning lines if there are any
                    if col >= 2:
                        y_lim = ax[i, col].get_ylim()
                        if len(day_x) > 0:
                            for k in day_x:
                                ax[i, col].plot(
                                    [k, k], y_lim, color='#969696', linewidth=1)
                        if len(lstate_x) > 0:
                            ls_vals = ['naive', 'learning', 'reversal1']
                            ls_colors = ['#66bd63', '#d73027', '#a50026']
                            for k in lstate_x:
                                ls = learning_state[int(k-0.5)]
                                ax[i, col].plot(
                                    [k, k], y_lim, color=ls_colors[ls_vals.index(ls)],
                                    linewidth=1.5)

                    # set axes labels
                    ax[i, col].set_yticks(y_ticks)
                    ax[i, col].set_yticklabels(y_tickl)
                    ax[i, col].set_ylim(y_lim)
                    ax[i, col].set_xlim(x_lim)

                    # format axes
                    ax[i, col].locator_params(nbins=4)
                    ax[i, col].spines['top'].set_visible(False)
                    ax[i, col].spines['right'].set_visible(False)
                    ax[i, col].xaxis.set_tick_params(direction='out')
                    ax[i, col].yaxis.set_tick_params(direction='out')
                    ax[i, col].yaxis.set_ticks_position('left')
                    ax[i, col].xaxis.set_ticks_position('bottom')

                    # remove xticks on all but bottom row
                    if i + 1 != U.rank:
                        plt.setp(ax[i, col].get_xticklabels(), visible=False)

                    if col == 3:
                        ax[i, 0].set_ylabel('Component #' + str(i+1), rotation=0,
                                            labelpad=45, verticalalignment='center',
                                            fontstyle='oblique')

            if filetype.lower() == 'pdf':
                suffix = '.pdf'
            elif filetype.lower() == 'eps':
                suffix = '.eps'
            else:
                suffix = '.png'
            plt.savefig(os.path.join(date_dir, 'rank_' + str(int(r)) + suffix),
                                     bbox_inches='tight')
            if verbose:
                plt.show()
            plt.close()


def singleday_qc_summary(
        mouse,
        trace_type='zscore_day',
        method='ncp_bcd',
        cs='',
        warp=False,
        word=None,
        verbose=False):
    """
    Plot similarity and objective (measure of reconstruction error) plots
    across all days for TCA decomposition ensembles.

    Parameters:
    -----------
    mouse : str; mouse name
    trace_type : str; dff, zscore, deconvolved
    method : str; TCA fit method from tensortools

    Returns:
    --------
    Saves figures to .../analysis folder  .../qc
    """
    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}

    days = flow.DateSorter.frommeta(mice=[mouse], tags=None)

    cmap = sns.color_palette('hls', n_colors=len(days))

    # create figure and axes
    buffer = 5
    right_pad = 5

    fig0 = plt.figure(figsize=(10, 8))
    gs0 = GridSpec(100, 100, figure=fig0, left=0.05, right=.95, top=.95, bottom=0.05)
    ax0 = fig0.add_subplot(gs0[10:90-buffer, :90-right_pad])

    fig1 = plt.figure(figsize=(10, 8))
    gs1 = GridSpec(100, 100, figure=fig1, left=0.05, right=.95, top=.95, bottom=0.05)
    ax1 = fig1.add_subplot(gs1[10:90-buffer, :90-right_pad])

    # plt.figure()
    for c, day1 in enumerate(days, 0):

        # load paths
        load_dir = paths.tca_path(mouse, 'single', pars=pars, word=word)
        tensor_path = os.path.join(load_dir, str(day1.mouse) + '_' + str(day1.date)
                                   + '_single_decomp_' + str(trace_type) + '.npy')
        if not os.path.isfile(tensor_path): continue

        # save paths
        save_dir = paths.tca_plots(mouse, 'single', pars=pars, word=word)
        save_dir = os.path.join(save_dir, 'qc')
        if not os.path.isdir(save_dir): os.mkdir(save_dir)
        error_path = os.path.join(save_dir, str(day1.mouse) + '_summary_objective.pdf')
        sim_path = os.path.join(save_dir, str(day1.mouse) + '_summary_similarity.pdf')

        # plotting options for the unconstrained and nonnegative models.
        plot_options = {
          'cp_als': {
            'line_kw': {
              'color': cmap[c],
              'label': 'single ' + str(c),
            },
            'scatter_kw': {
              'color': cmap[c],
              'alpha': 0.5,
            },
          },
          'ncp_hals': {
            'line_kw': {
              'color': cmap[c],
              'alpha': 0.5,
              'label': 'single ' + str(c),
            },
            'scatter_kw': {
              'color': cmap[c],
              'alpha': 0.5,
            },
          },
          'ncp_bcd': {
            'line_kw': {
              'color': cmap[c],
              'alpha': 0.5,
              'label': 'single ' + str(c),
            },
            'scatter_kw': {
              'color': cmap[c],
              'alpha': 0.5,
            },
          },
        }

        # load your data
        ensemble = np.load(tensor_path)
        ensemble = ensemble.item()

        # plot error and similarity plots across rank number
        tt.plot_objective(ensemble[method], **plot_options[method], ax=ax0)
        tt.plot_similarity(ensemble[method], **plot_options[method], ax=ax1)

    # add legend, title
    ax0.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
    ax0.set_title('Objective Function: ' + str(method) + ', ' + mouse)
    ax1.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
    ax1.set_title('Iteration Similarity: ' + str(method) + ', ' + mouse)

    # save figs
    fig0.savefig(error_path, bbox_inches='tight')
    fig1.savefig(sim_path, bbox_inches='tight')

    if verbose:
        fig0.show()
        fig1.show()


def singleday_varex_summary(
        mouse,
        trace_type='zscore_day',
        method='ncp_bcd',
        cs='',
        warp=False,
        word=None,
        verbose=False):
    """
    Plot reconstruction error as variance explained across all days for
    TCA decomposition ensembles.

    Parameters:
    -----------
    mouse : str; mouse name
    trace_type : str; dff, zscore, deconvolved
    method : str; TCA fit method from tensortools

    Returns:
    --------
    Saves figures to .../analysis folder  .../qc
    """

    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}

    days = flow.DateSorter.frommeta(mice=[mouse], tags=None)

    cmap = sns.color_palette(sns.cubehelix_palette(len(days)))

    # create figure and axes
    buffer = 5
    right_pad = 5
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(100, 100, figure=fig, left=0.05, right=.95, top=.95, bottom=0.05)
    ax = fig.add_subplot(gs[10:90-buffer, :90-right_pad])

    for c, day1 in enumerate(days, 0):

        # load dirs
        load_dir = paths.tca_path(mouse, 'single', pars=pars, word=word)
        tensor_path = os.path.join(load_dir, str(day1.mouse) + '_'
                                   + str(day1.date) + '_single_decomp_'
                                   + str(trace_type) + '.npy')
        input_tensor_path = os.path.join(load_dir, str(day1.mouse) + '_'
                                         + str(day1.date) + '_single_tensor_'
                                         + str(trace_type) + '.npy')
        if not os.path.isfile(tensor_path): continue
        if not os.path.isfile(input_tensor_path): continue

        # save dirs
        save_dir = paths.tca_plots(mouse, 'single', pars=pars, word=word)
        save_dir = os.path.join(save_dir, 'qc')
        if not os.path.isdir(save_dir): os.mkdir(save_dir)
        var_path = os.path.join(save_dir, str(day1.mouse) + '_summary_variance_cubehelix.pdf')

        # load your data
        ensemble = np.load(tensor_path)
        ensemble = ensemble.item()
        V = ensemble[method]
        X = np.load(input_tensor_path)

        # get reconstruction error as variance explained
        var, var_s, x, x_s = [], [], [], []
        for r in V.results:
            bU = V.results[r][0].factors.full()
            var.append((np.var(X) - np.var(X - bU)) / np.var(X))
            x.append(r)
            for it in range(0, len(V.results[r])):
                U = V.results[r][it].factors.full()
                var_s.extend([(np.var(X) - np.var(X - U)) / np.var(X)])
                x_s.extend([r])

        # mean response of neuron across trials
        mU = np.mean(X, axis=2, keepdims=True) * np.ones((1, 1, np.shape(X)[2]))
        var_mean = (np.var(X) - np.var(X - mU)) / np.var(X)

        # smoothed response of neuron across time
        smU = np.convolve(X.reshape((X.size)), np.ones(5, dtype=np.float64)/5, 'same').reshape(np.shape(X))
        var_smooth = (np.var(X) - np.var(X - smU)) / np.var(X)

        # plot
        R = np.max([r for r in V.results.keys()])
        ax.scatter(x_s, var_s, color=cmap[c], alpha=0.5)
        ax.scatter([R+2], var_mean, color=cmap[c], alpha=0.5)
        ax.scatter([R+4], var_smooth, color=cmap[c], alpha=0.5)
        ax.plot(x, var, label=('single ' + str(c)), color=cmap[c])
        ax.plot([R+1.5, R+2.5], [var_mean, var_mean], color=cmap[c])
        ax.plot([R+3.5, R+4.5], [var_smooth, var_smooth], color=cmap[c])

    # add labels/titles
    x_labels = [str(R) for R in V.results]
    x_labels.extend(['', 'mean\n cell\n response', '', 'smooth\n response\n (0.3s)'])
    ax.set_xticks(range(1, len(V.results) + 5))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('model rank')
    ax.set_ylabel('fractional variance explained')
    ax.set_title('Variance Explained: ' + str(method) + ', ' + mouse)
    ax.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)

    fig.savefig(var_path, bbox_inches='tight')


def singleday_varex_percell(
        mouse,
        method='ncp_bcd',
        trace_type='zscore_day',
        cs='',
        warp=False,
        word=None,
        ve_min=0.05,
        filetype='pdf'):
    """
    Plot TCA reconstruction error as variance explained per cell
    for TCA decomposition. Create folder of variance explained per cell
    swarm plots. Calculate summary plots of 'fraction of maximum variance
    explained' per cell by rank for all cells given a certain (ve_min)
    threshold for maximum variance explained.

    Parameters:
    -----------
    mouse : str; mouse name
    trace_type : str; dff, zscore, deconvolved
    method : str; TCA fit method from tensortools
    ve_min: float; minimum variance explained for best rank per cell
                   to be included in summary of fraction of maximum variance
                   explained

    Returns:
    --------
    Saves figures to .../analysis folder/ .../qc
                                             .../variance explained per cell

    """

    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}

    days = flow.DateSorter.frommeta(mice=[mouse], tags=None)

    # create folder structure if needed
    cs_tag = '' if len(cs) == 0 else ' ' + str(cs)
    warp_tag = '' if warp is False else ' warp'
    folder_name = 'tensors single ' + str(trace_type) + cs_tag + warp_tag

    ve, ve_max, ve_frac, rank_num, day_num, cell_num = [], [], [], [], [], []
    for c, day1 in enumerate(days, 0):

        # get dirs for loading
        load_dir = paths.tca_path(mouse, 'single', pars=pars, word=word)
        if not os.path.isdir(load_dir): os.mkdir(load_dir)
        tensor_path = os.path.join(load_dir, str(day1.mouse) + '_'
                                   + str(day1.date) + '_single_decomp_'
                                   + str(trace_type) + '.npy')
        input_tensor_path = os.path.join(load_dir, str(day1.mouse) + '_'
                                         + str(day1.date) + '_single_tensor_'
                                         + str(trace_type) + '.npy')
        if not os.path.isfile(tensor_path): continue
        if not os.path.isfile(input_tensor_path): continue

        # load your data
        ensemble = np.load(tensor_path)
        ensemble = ensemble.item()
        V = ensemble[method]
        X = np.load(input_tensor_path)

        # get reconstruction error as variance explained per cell
        for cell in range(0, np.shape(X)[0]):
            rank_ve_vec = []
            rank_vec = []
            for r in V.results:
                U = V.results[r][0].factors.full()
                Usub = X - U
                rank_ve = (np.var(X[cell, :, :]) - np.var(Usub[cell, :, :])) / np.var(X[cell, :, :])
                rank_ve_vec.append(rank_ve)
                rank_vec.append(r)
            max_ve = np.max(rank_ve_vec)
            ve.extend(rank_ve_vec)
            ve_max.extend([max_ve for s in rank_ve_vec])
            ve_frac.extend(rank_ve_vec / max_ve)
            rank_num.extend(rank_vec)
            day_num.extend([c+1 for s in rank_ve_vec])
            cell_num.extend([cell for s in rank_ve_vec])

    # build pd dataframe of all variance measures
    index = pd.MultiIndex.from_arrays([
    day_num,
    rank_num,
    ve,
    ve_max,
    ve_frac,
    cell_num,
    ],
    names=['day', 'rank', 'variance_explained', 'max_ve', 'frac_ve', 'cell'])
    df = pd.DataFrame(index=index)
    df = df.reset_index()

    # make a rainbow colormap, HUSL space but does not circle back on itself
    cmap = sns.color_palette('hls', int(np.ceil(1.5*np.unique(df['rank'])[-1])))
    cmap = cmap[0:np.unique(df['rank'])[-1]]

    # Part 1
    # slice df, only look at cells with a max variance >5%
    sliced_df2 = df.loc[(df['day']) & (df['max_ve'] >= ve_min), :]

    # CDF plot
    fig1 = plt.figure(figsize=(15,9))
    for i in np.unique(sliced_df2['rank']):
        input_ve = sliced_df2.loc[(sliced_df2['rank'] == i),'frac_ve']
        ax = sns.distplot(input_ve, kde_kws={'cumulative': True, 'lw': 2, 'color': cmap[i-1], 'label': str(i)}, hist=False)
        lg = ax.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
        lg.set_title('rank')
        ax.set_title(mouse + ', Fraction of maximum variance explained per cell, CDF')
        ax.set_xlabel('Fraction of maximum variance explained')

    # swarm plot
    fig2 =plt.figure(figsize=(18,6))
    ax2 = sns.violinplot(x=sliced_df2['rank'], y=sliced_df2['frac_ve'], size=3, alpha=1, inner=None, palette=cmap)
    ax2.set_title(mouse + ', Fraction of maximum variance explained per cell, violin')
    ax2.set_ylabel('Fraction of maximum variance explained')

    # swarm plot
    fig3 = plt.figure(figsize=(18,6))
    ax3 = sns.swarmplot(x=sliced_df2['rank'], y=sliced_df2['frac_ve'], size=2, alpha=1, palette=cmap)
    ax3.set_title(mouse + ', Fraction of maximum variance explained per cell, swarm')
    ax3.set_ylabel('Fraction of maximum variance explained')

    # set up saving paths/dir
    save_dir = paths.tca_plots(mouse, 'single', pars=pars, word=word)
    save_dir = os.path.join(save_dir, 'qc')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    save_file_base = mouse + '_singleday_frac_max_var_expl_' + trace_type

    # save
    if filetype.lower() == 'pdf':
        suffix = '.pdf'
    elif filetype.lower() == 'eps':
        suffix = '.eps'
    else:
        suffix = '.png'
    fig1.savefig(os.path.join(save_dir, save_file_base + '_CDF' + suffix), bbox_inches='tight')
    fig2.savefig(os.path.join(save_dir, save_file_base + '_violin' + suffix), bbox_inches='tight')
    fig3.savefig(os.path.join(save_dir, save_file_base + '_swarm.png'), bbox_inches='tight')

    # Part 2
    # plot sorted per "cell" variance explained (approximate, this is by unique
    # max_ve not cells per se)
    # set up saving paths/dir
    save_dir = os.path.join(save_dir, 'variance explained per cell')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    save_file_base = mouse + '_singleday_var_expl_' + trace_type

    for d in np.unique(df['day']):

        sliced_df = df.loc[(df['day'] == d),:]

        # make a rainbow colormap, HUSL space but does not circle back on itself
        cmap = sns.color_palette('hls', int(np.ceil(1.5*np.unique(df['rank'])[-1])))
        cmap = cmap[0:np.unique(df['rank'])[-1]]

        fig0 = plt.figure(figsize=(20, 6))
        ax0 = sns.swarmplot(x=sliced_df['max_ve'], y=sliced_df['variance_explained'],
                            hue=sliced_df['rank'], palette=cmap)
        lg = ax0.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
        lg.set_title('rank')
        ax0.set_xlabel('cell count')
        x_lim = ax0.get_xlim()
        ticks = ax0.get_xticks()
        new_ticks = [t for t in ticks[10::10]]
        ax0.set_xticks(new_ticks)
        ax0.set_xticklabels(np.arange(10, len(ticks), 10))
        ax0.set_title(mouse + ', Variance explained per cell, day ' + str(d))

        fig0.savefig(os.path.join(save_dir, save_file_base + '_day_' + str(d)
                     + suffix), bbox_inches='tight')
        plt.close()

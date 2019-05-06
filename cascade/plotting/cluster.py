"""Functions for plotting clustered factors from tca decomp."""
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
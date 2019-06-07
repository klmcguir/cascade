""" Functions for visualizing TCA model reconstructions """
from .. import paths, tca
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def groupday_cell_trial_recon(
        mouse,
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        word=None,
        group_by='all',
        nan_thresh=0.85,
        rectified=True,
        rank=18,
        verbose=True):
    """
    Plot reconstruction for single cells TCA decomposition ensemble.

    Parameters:
    -----------
    mouse : str; mouse object
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
    ids_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_group_ids_' + str(trace_type) + '.npy')
    meta_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_df_group_meta.pkl')

    # save dir
    save_dir = paths.tca_plots(
        mouse, 'group', pars=pars, word=word, group_pars=group_pars)
    save_dir = os.path.join(save_dir, 'model reconstructions')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    date_dir = os.path.join(save_dir, str(group_by) + ' ' + method)
    if not os.path.isdir(date_dir): os.mkdir(date_dir)

    # load your data
    ensemble = np.load(tensor_path)
    ensemble = ensemble.item()
    V = ensemble[method]
    X = np.load(input_tensor_path)
    ids = np.load(ids_path)
    meta = pd.read_pickle(meta_path)
    orientation = meta['orientation']
    condition = meta['condition']
    dates = meta.reset_index()['date']

    # re-balance your factors
    if verbose:
        print('Re-balancing factors.')
    for r in ensemble[method].results:
        for i in range(len(ensemble[method].results[r])):
            ensemble[method].results[r][i].factors.rebalance()

    # sort cells according to which factor they respond to the most
    sort_ensemble, my_sorts = tca._sortfactors(ensemble[method])
    sorted_ids = ids[my_sorts[rank]]

    # recreate model
    Xhat = V.results[rank][0].factors.full()

    # set subplot row/col
    rows = 1
    cols = 3

    # loop over cells and plot all input trial and reconstructed trials
    for ncell in range(len(my_sorts[18])):
        ident = sorted_ids[ncell]
        fig, axes = plt.subplots(
            rows, cols, figsize=(14, 12),
            gridspec_kw={'width_ratios': [8, 8, 1]})
        data = X[my_sorts[18], :, :]
        data = data[ncell, :, :]
        nan_bool = np.isfinite(data[0, :])
        data = data[:, nan_bool]
        model = Xhat[my_sorts[18], :, :]
        model = model[ncell, :, :]
        model = model[:, nan_bool]

        # reshape for easier indexing
        ax = np.array(axes).reshape((rows, -1))

        # get max from reconstruction to set limits on both plot
        vmax = np.nanmax(model[:])

        # plot
        sns.heatmap(
            data.T, ax=ax[0, 0], vmax=vmax, vmin=0, xticklabels=False,
            yticklabels=False, cbar=False)
        sns.heatmap(
            model.T, ax=ax[0, 1], vmax=vmax, vmin=0, xticklabels=False,
            yticklabels=False, cbar_kws={'label': 'z-score'}, cbar_ax=ax[0, 2])

        # add titles
        ax[0, 0].set_title('Cell ' + str(ident) + ': Input Data')
        ax[0, 1].set_title('Cell ' + str(ident) + ': TCA Model')

        # get limits for plotting
        y_lim = ax[0, 0].get_ylim()
        x_lim = ax[0, 0].get_xlim()

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
        ax[0, 0].set_yticks(base_yticks)
        ax[0, 0].set_yticklabels(base_ylabels)

        # reset xticklabels
        xticklabels = np.array([-1, 0, 1, 2, 3, 4, 5, 6])
        stim_window = np.arange(-1, 7, 1/15.5)[0:108]
        zero_sec = np.where(stim_window <= 0)[0][-1]
        three_sec = np.where(stim_window <= 3)[0][-1]
        xticklabels = xticklabels[(xticklabels > stim_window[0])
                                  & (xticklabels < stim_window[-1])]
        xticks = [np.where(stim_window <= s)[0][-1]
                  for s in xticklabels]
        ax[0, 0].set_xticks(xticks)
        ax[0, 0].set_xticklabels(xticklabels, rotation='horizontal')
        ax[0, 1].set_xticks(xticks)
        ax[0, 1].set_xticklabels(xticklabels, rotation='horizontal')

        # plot onset/offset lines
        ax[0, 0].plot((zero_sec, zero_sec), y_lim, color='#8e8e8e',
                      ls='-', lw=2, alpha=0.8)
        ax[0, 0].plot((three_sec, three_sec), y_lim, color='#8e8e8e',
                      ls='-', lw=2, alpha=0.8)
        ax[0, 1].plot((zero_sec, zero_sec), y_lim, color='#8e8e8e',
                      ls='-', lw=2, alpha=0.8)
        ax[0, 1].plot((three_sec, three_sec), y_lim, color='#8e8e8e',
                      ls='-', lw=2, alpha=0.8)

        # set axis labels
        ax[0, 0].set_xlabel('Time (sec)')
        ax[0, 0].set_ylabel('Trials')
        ax[0, 1].set_xlabel('Time (sec)')

        # day bars
    #     days = np.array(dates.values)
    #     count_d = 0
    #     for day in np.unique(days):
    #         day_y = np.where(days == day)[0]
    #         day_y = [day_y[0], day_y[-1]+1]
    #         day_bar_color = day_colors[sorted(day_colors.keys())[count_d%2]]
    #         ax[0, 0].plot((3.5, 3.5), day_y, color=day_bar_color, ls='-',
    #                 lw=6, alpha=0.4, solid_capstyle='butt')
    #         ax[0, 1].plot((3.5, 3.5), day_y, color=day_bar_color, ls='-',
    #                 lw=6, alpha=0.4, solid_capstyle='butt')
    #         count_d = count_d + 1
    # ax[0, 0].set_colorbar()
    # ax[0, 1].set_colorbar()

        fig.suptitle(mouse + ' sort- ' + str(ncell) + ' cell-' + str(ident)
                     + ' rank-' + str(rank) + ' TCA reconstruction')
        fig.savefig(os.path.join(date_dir, mouse + '_sort ' + str(ncell)
                    + '_cell' + str(ident)
                    + '_rank' + str(rank) + '_TCArecon.png'),
                    bbox_inches='tight')
        plt.close('all')


def groupday_mean_trial_recon(
        mouse,
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        word=None,
        group_by='all',
        nan_thresh=0.85,
        rectified=True,
        sorted=True,
        rank=18,
        verbose=True):
    """
    Plot reconstruction for whole groupday TCA decomposition ensemble.

    Parameters:
    -----------
    mouse : str; mouse object
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
    else:
        nt_tag = ''
    # if sorting using TCA cell factors
    if sorted:
        sort_tag = '_sorted'
    else:
        sort_tag = ''

    # load dir
    load_dir = paths.tca_path(
        mouse, 'group', pars=pars, word=word, group_pars=group_pars)
    tensor_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_group_decomp_' + str(trace_type) + '.npy')
    input_tensor_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_group_tensor_' + str(trace_type) + '.npy')
    ids_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_group_ids_' + str(trace_type) + '.npy')
    meta_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_df_group_meta.pkl')

    # save dir
    save_dir = paths.tca_plots(
        mouse, 'group', pars=pars, word=word, group_pars=group_pars)
    save_dir = os.path.join(save_dir, 'model reconstructions')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    date_dir = os.path.join(save_dir, str(group_by) + ' ' + method)
    if not os.path.isdir(date_dir): os.mkdir(date_dir)

    # load your data
    ensemble = np.load(tensor_path)
    ensemble = ensemble.item()
    V = ensemble[method]
    X = np.load(input_tensor_path)
    ids = np.load(ids_path)
    meta = pd.read_pickle(meta_path)
    orientation = meta['orientation']
    condition = meta['condition']
    dates = meta.reset_index()['date']

    # re-balance your factors
    if verbose:
        print('Re-balancing factors.')
    for r in ensemble[method].results:
        for i in range(len(ensemble[method].results[r])):
            ensemble[method].results[r][i].factors.rebalance()

    # sort cells according to which factor they respond to the most
    sort_ensemble, my_sorts = tca._sortfactors(ensemble[method])
    sorted_ids = ids[my_sorts[rank]]

    # recreate model
    Xhat = V.results[rank][0].factors.full()

    # set subplot row/col
    rows = 1
    cols = 3

    fig, axes = plt.subplots(
        rows, cols, figsize=(14, 12),
        gridspec_kw={'width_ratios': [8, 8, 1]})
    data = X[my_sorts[18], :, :]
    model = Xhat[my_sorts[18], :, :]
    data = np.nanmean(data, axis=2)
    model = np.nanmean(model, axis=2)

    # reshape for easier indexing
    ax = np.array(axes).reshape((rows, -1))

    # get max from reconstruction to set limits on both plot
    vmax = np.nanmax(model[:])

    # plot
    sns.heatmap(
        data, ax=ax[0, 0], vmax=vmax, vmin=0, xticklabels=False,
        yticklabels=False, cbar=False)
    sns.heatmap(
        model, ax=ax[0, 1], vmax=vmax, vmin=0, xticklabels=False,
        yticklabels=False, cbar_kws={'label': 'z-score'}, cbar_ax=ax[0, 2])

    # add titles
    ax[0, 0].set_title(mouse + ' ' + method + ': Input Data')
    ax[0, 1].set_title(mouse + ' ' + method + ': TCA Model')

    # get limits for plotting
    y_lim = ax[0, 0].get_ylim()
    x_lim = ax[0, 0].get_xlim()

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
    ax[0, 0].set_yticks(base_yticks)
    ax[0, 0].set_yticklabels(base_ylabels)

    # reset xticklabels
    xticklabels = np.array([-1, 0, 1, 2, 3, 4, 5, 6])
    stim_window = np.arange(-1, 7, 1/15.5)[0:108]
    zero_sec = np.where(stim_window <= 0)[0][-1]
    three_sec = np.where(stim_window <= 3)[0][-1]
    xticklabels = xticklabels[(xticklabels > stim_window[0])
                              & (xticklabels < stim_window[-1])]
    xticks = [np.where(stim_window <= s)[0][-1]
              for s in xticklabels]
    ax[0, 0].set_xticks(xticks)
    ax[0, 0].set_xticklabels(xticklabels, rotation='horizontal')
    ax[0, 1].set_xticks(xticks)
    ax[0, 1].set_xticklabels(xticklabels, rotation='horizontal')

    # plot onset/offset lines
    ax[0, 0].plot((zero_sec, zero_sec), y_lim, color='#8e8e8e',
                  ls='-', lw=2, alpha=0.8)
    ax[0, 0].plot((three_sec, three_sec), y_lim, color='#8e8e8e',
                  ls='-', lw=2, alpha=0.8)
    ax[0, 1].plot((zero_sec, zero_sec), y_lim, color='#8e8e8e',
                  ls='-', lw=2, alpha=0.8)
    ax[0, 1].plot((three_sec, three_sec), y_lim, color='#8e8e8e',
                  ls='-', lw=2, alpha=0.8)

    # set axis labels
    ax[0, 0].set_xlabel('Time (sec)')
    ax[0, 0].set_ylabel('Cells')
    ax[0, 1].set_xlabel('Time (sec)')

    # day bars
#     days = np.array(dates.values)
#     count_d = 0
#     for day in np.unique(days):
#         day_y = np.where(days == day)[0]
#         day_y = [day_y[0], day_y[-1]+1]
#         day_bar_color = day_colors[sorted(day_colors.keys())[count_d%2]]
#         ax[0, 0].plot((3.5, 3.5), day_y, color=day_bar_color, ls='-',
#                 lw=6, alpha=0.4, solid_capstyle='butt')
#         ax[0, 1].plot((3.5, 3.5), day_y, color=day_bar_color, ls='-',
#                 lw=6, alpha=0.4, solid_capstyle='butt')
#         count_d = count_d + 1
# ax[0, 0].set_colorbar()
# ax[0, 1].set_colorbar()

    fig.suptitle(mouse + ' mean across trials rank- ' + str(rank) +
                 ' TCA reconstruction')
    fig.savefig(os.path.join(date_dir, mouse + '_mean_rank' + str(rank)
                + sort_tag + '_TCArecon.png'),
                bbox_inches='tight')
    plt.close('all')

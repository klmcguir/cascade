"""Functions for plotting tca decomp pairwise correlations."""
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
----------------------------- SINGLEDAY PLOTS -----------------------------
"""


def singleday_noisecorr(
        # TCA params
        mouse,
        trace_type='zscore_day',
        cs='',
        warp=False,
        word=None,
        # Noise corr params
        corr_trace_type='zscore_day',
        remove_licking=True,
        running_lowspeed=False,
        running_highspeed=False,
        speed_thresh=10,
        randomizations=500,
        vis_drive_thresh=0,
        sort_rank=10,
        # clustering params
        cluster=False,

        verbose=False):

    """
    Plot noise correlations per day sorted by TCA neuron factors.

    Parameters:
    -----------
    mouse : str
        mouse name
    trace_type : str
        trace type of TCA decomposition to load (i.e., dff, zscore_day,
        deconvolved)
    cs : str
        stimulus type
    warp : bool
        warp ensure to be at same point in time
    word : str
        hash word for TCA parameters, used for loading
    corr_trace_type : str
        trace type for running noise correlations
    remove_licking : bool
        nan traces after first lick in trial or median latency of lowest
        lick-latency cs
    running_lowspeed : bool
        use only low running speed (<= speed_thresh) trials
    running_highspeed : bool
        use only high running speed (> speed_thresh) trials
    speed_thresh : int/float
        threshold for running speed (cm/s)
    randomizations : int
        number of bootstrap iterations for removing signal correlations
    vis_drive_thresh : int/float
        additional visual drive filtering, cells will already have been
        filtered according to "word" TCA parameters using "ids" variable
    sort_rank : int (1 indexed)
        rank of TCA decomposition to use for clustering noise correlation
        results
    verbose : bool
        optionally plot figures as you loop through days

    Returns:
    --------
    Saves figures to .../analysis folder  .../noise corr
    """

    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}

    days = flow.DateSorter.frommeta(mice=[mouse], tags=None)

    clus_daily = []
    acorr_daily = []
    acorr_daily_sort = []
    aclus_daily_sort = []

    for day1 in days:

        # load dir
        load_dir = paths.tca_path(mouse, 'single',
                                  pars=pars, word=word)
        tensor_path = os.path.join(load_dir, str(day1.mouse)
                                   + '_' + str(day1.date)
                                   + '_single_decomp_'
                                   + str(trace_type) + '.npy')
        input_tensor_path = os.path.join(load_dir, str(day1.mouse)
                                         + '_' + str(day1.date)
                                         + '_single_tensor_'
                                         + str(trace_type) + '.npy')
        input_ids_path = os.path.join(load_dir, str(day1.mouse)
                                      + '_' + str(day1.date)
                                      + '_single_ids_'
                                      + str(trace_type) + '.npy')
        meta_path = os.path.join(load_dir, str(day1.mouse)
                                 + '_' + str(day1.date)
                                 + '_df_single_meta.pkl')

        # save dir
        save_dir = paths.tca_plots(mouse, 'single', pars=pars, word=word)
        if remove_licking:
            lick_tag = '-no-lick'
        else:
            lick_tag = ''
        if running_lowspeed:
            run_tag = '-lowspeed' + str(speed_thresh)
        elif running_highspeed:
            run_tag = '-highspeed' + str(speed_thresh)
        else:
            run_tag = ''
        if cluster:
            clus_save, clus_tag = '-hierclus', ' -hierclus'
        else:
            clus_save, clus_tag = '', ''
        save_dir = os.path.join(save_dir, 'noise corr ' + corr_trace_type + ' '
                                + lick_tag + run_tag + clus_tag)
        if not os.path.isdir(save_dir): os.mkdir(save_dir)

        # load your data
        ensemble = np.load(tensor_path)
        ensemble = ensemble.item()
        ids = np.load(input_ids_path)

        # all ids for the day
        d1_ids = flow.xday._read_crossday_ids(day1.mouse, day1.date)
        d1_ids = np.array([int(s) for s in d1_ids])

        # get new tensor and trial metadata
        # tensors and thus clusters are already sorted by id number
        # make sure to apply this to your data before clusters are applied
        meta = utils.getdailymeta(day1, tags='hungry', run_types='training')
        orientation = meta['orientation']
        speed = meta['speed']
        intensor = utils.getdailycstraces(
            day1, cs='', trace_type=corr_trace_type, start_time=-1, end_time=6,
            smooth=False)
        intensor = intensor[np.isin(d1_ids, ids), :, :]
        id_sorter = np.argsort(d1_ids[np.isin(d1_ids, ids)])
        intensor = intensor[id_sorter, :, :]

        # nan area after first lick in trial
        # if no first lick, us lowest latency median first lick for three stimuli
        if remove_licking:
            firstlick = meta['firstlick']
            if np.isfinite(firstlick).sum() == 0:
                # skip days where there are no licks at all
                print('No licking on '
                      + str(day1.date) + ', skipping for -no-lick.')
                continue
            median_lick = []
            for cs in [0, 135, 270]:
                median_lick.append(meta.loc[(orientation == cs), :]['firstlick'].median())
            median_lick = np.nanmin(median_lick)
            for tri in range(np.shape(intensor)[2]):
                if np.isfinite(firstlick[tri]):
                    intensor[:, int(np.floor(15.5 + firstlick[tri] - 1)):, tri] = np.nan
                else:
                    # (if there were licks recorded)
                    if np.isfinite(median_lick):
                        intensor[:, int(np.floor(15.5 + median_lick)):, tri] = np.nan

        # split running into high and low groups
        if running_highspeed and running_lowspeed:
            print('WTF. High and low speed boolean?')
        elif running_highspeed:
            speed_bool = speed > speed_thresh
            intensor = intensor[:, :, speed_bool]
        elif running_lowspeed:
            speed_bool = speed <= speed_thresh
            intensor = intensor[:, :, speed_bool]

        # sort neuron factors by component they belong to most
        # assumes only one method was used
        method = [i for i in ensemble.keys()][0]
        sort_ensemble, my_sorts = tca._sortfactors(ensemble[method])

        # run noise corr for each cs
        cs_clusters = {}
        cscorr = {}
        sscorr = {}
        for cs in ['0', '135', '270']:

            # boolean index for cs accounting for running boolean
            if running_highspeed or running_lowspeed:
                csbool = np.isin(orientation, int(cs))[speed_bool]
            else:
                csbool = np.isin(orientation, int(cs))

            # select trials
            trs = np.nanmean(intensor[:, 16:48, csbool], axis=1)
            ncells = np.shape(trs)[0]

            # additional visual drive thresholding
            drive = pool.calc.driven.visually(day1, cs)
            drive_bool = drive[np.isin(d1_ids, ids)] > vis_drive_thresh

            # get TCA clusters
            # factors are already sorted, so these will define
            # clusters, no need to sort again
            cell_clusters = {}
            for k in sort_ensemble.results.keys():
                factors = sort_ensemble.results[k][0].factors[0]
                max_fac = np.argmax(factors, axis=1)
                max_fac_val = np.max(factors, axis=1)
                max_fac[max_fac_val == 0] = -1
                cell_clusters[k] = max_fac[drive_bool]
            cs_clusters[cs] = cell_clusters

            # Catch cases when there aren't enough trials
            if np.shape(trs)[1] < 10:
                print('Not enough trials (' + str(np.shape(trs)[1]) + ') in '
                      + str(day1.date) + ' ' + cs +
                      ' trs to calculate noise corr.')
                continue

            stimorder = np.arange(np.shape(trs)[1])
            corrs = np.zeros((ncells, ncells))
            corrs[:, :] = np.nan
            if np.sum(np.invert(np.isfinite(trs))) == 0:
                corrs = np.corrcoef(trs)

                for i in range(randomizations):
                    for c in range(ncells):
                        np.random.shuffle(stimorder)
                        trs[c, :] = trs[c, stimorder]

                    corrs -= np.corrcoef(trs)/float(randomizations)
            sscorr[cs] = corrs[:, my_sorts[sort_rank-1]][my_sorts[sort_rank-1], :][drive_bool, :][:, drive_bool]
            cscorr[cs] = corrs[drive_bool, :][:, drive_bool]
        acorr_daily.append(cscorr)
        acorr_daily_sort.append(sscorr)
        aclus_daily_sort.append(cs_clusters)

    # plot
    cmap = sns.color_palette('muted', sort_rank)
    whitergb = (1, 1, 1)
    for c, corr in enumerate(acorr_daily_sort):
        for cs in ['0', '135', '270']:
            if ~np.isin(cs, list(corr.keys())):
                continue
            if np.sum(np.isnan(corr[cs]).flatten()) > 0:
                continue
            if len(corr[cs]) == 0:
                continue
            color_vec = []
            for k in aclus_daily_sort[c][cs][sort_rank]:
                if k >= 0:
                    color_vec.append(cmap[int(k)])
                else:
                    color_vec.append(whitergb)
            fig = plt.figure()
            a = sns.clustermap(
                corr[cs], row_colors=color_vec, col_colors=color_vec,
                row_cluster=cluster, col_cluster=cluster, method='ward')
            a.cax.set_ylabel('correlation\ncoefficient (R)')
            a.cax.set_title(
                'Noise Corr: ' +
                str(days[c].date) + ': ' + corr_trace_type + ': rank '
                + str(sort_rank) + ': ' + cs +
                ' ' + lick_tag + run_tag + clus_save,
                horizontalalignment='left')
            a.ax_heatmap.set_xlabel('Cell Number')
            a.ax_heatmap.set_ylabel('Cell Number')

            # save, show figs as you go if verbose
            save_path = os.path.join(
                save_dir, 'cs' + cs + '-' +
                str(days[c].date) + '-rank' +
                str(int(sort_rank)) + lick_tag + run_tag + clus_save + '.png')
            plt.savefig(save_path, bbox_inches='tight')
            if verbose:
                plt.show()
            plt.close('all')


def singleday_signalcorr(
        # TCA params
        mouse,
        trace_type='zscore_day',
        cs='',
        warp=False,
        word=None,
        # Signal corr params
        corr_trace_type='zscore_day',
        remove_licking=True,
        running_lowspeed=False,
        running_highspeed=False,
        speed_thresh=10,
        randomizations=500,
        vis_drive_thresh=0,
        sort_rank=10,
        # clustering params
        cluster=False,

        verbose=False):

    """
    Plot signal correlations per day sorted by TCA neuron factors.

    Parameters:
    -----------
    mouse : str
        mouse name
    trace_type : str
        trace type of TCA decomposition to load (i.e., dff, zscore_day,
        deconvolved)
    cs : str
        stimulus type
    warp : bool
        warp ensure to be at same point in time
    word : str
        hash word for TCA parameters, used for loading
    corr_trace_type : str
        trace type for running noise correlations
    remove_licking : bool
        nan traces after first lick in trial or median latency of lowest
        lick-latency cs
    running_lowspeed : bool
        use only low running speed (<= speed_thresh) trials
    running_highspeed : bool
        use only high running speed (> speed_thresh) trials
    speed_thresh : int/float
        threshold for running speed (cm/s)
    randomizations : int
        number of bootstrap iterations for calculating signal correlations
    vis_drive_thresh : int/float
        additional visual drive filtering, cells will already have been
        filtered according to "word" TCA parameters using "ids" variable
    sort_rank : int (1 indexed)
        rank of TCA decomposition to use for clustering noise correlation
        results
    verbose : bool
        optionally plot figures as you loop through days

    Returns:
    --------
    Saves figures to .../analysis folder  .../noise corr
    """

    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}

    days = flow.DateSorter.frommeta(mice=[mouse], tags=None)

    clus_daily = []
    acorr_daily = []
    acorr_daily_sort = []
    aclus_daily_sort = []

    for day1 in days:

        # load dir
        load_dir = paths.tca_path(mouse, 'single',
                                  pars=pars, word=word)
        tensor_path = os.path.join(load_dir, str(day1.mouse)
                                   + '_' + str(day1.date)
                                   + '_single_decomp_'
                                   + str(trace_type) + '.npy')
        input_tensor_path = os.path.join(load_dir, str(day1.mouse)
                                         + '_' + str(day1.date)
                                         + '_single_tensor_'
                                         + str(trace_type) + '.npy')
        input_ids_path = os.path.join(load_dir, str(day1.mouse)
                                      + '_' + str(day1.date)
                                      + '_single_ids_'
                                      + str(trace_type) + '.npy')
        meta_path = os.path.join(load_dir, str(day1.mouse)
                                 + '_' + str(day1.date)
                                 + '_df_single_meta.pkl')

        # save dir
        save_dir = paths.tca_plots(mouse, 'single', pars=pars, word=word)
        if remove_licking:
            lick_tag = '-no-lick'
        else:
            lick_tag = ''
        if running_lowspeed:
            run_tag = '-lowspeed' + str(speed_thresh)
        elif running_highspeed:
            run_tag = '-highspeed' + str(speed_thresh)
        else:
            run_tag = ''
        if cluster:
            clus_save, clus_tag = '-hierclus', ' -hierclus'
        else:
            clus_save, clus_tag = '', ''
        save_dir = os.path.join(save_dir, 'signal corr ' + corr_trace_type +
                                ' ' + lick_tag + run_tag + clus_tag)
        if not os.path.isdir(save_dir): os.mkdir(save_dir)

        # load your data
        ensemble = np.load(tensor_path)
        ensemble = ensemble.item()
        ids = np.load(input_ids_path)

        # all ids for the day
        d1_ids = flow.xday._read_crossday_ids(day1.mouse, day1.date)
        d1_ids = np.array([int(s) for s in d1_ids])

        # get new tensor and trial metadata
        # tensors and thus clusters are already sorted by id number
        # make sure to apply this to your data before clusters are applied
        meta = utils.getdailymeta(day1, tags='hungry', run_types='training')
        orientation = meta['orientation']
        speed = meta['speed']
        intensor = utils.getdailycstraces(
            day1, cs='', trace_type=corr_trace_type, start_time=-1, end_time=6,
            smooth=False)
        intensor = intensor[np.isin(d1_ids, ids), :, :]
        id_sorter = np.argsort(d1_ids[np.isin(d1_ids, ids)])
        intensor = intensor[id_sorter, :, :]

        # nan area after first lick in trial
        # if no first lick, us lowest latency median first
        # lick for three stimuli
        if remove_licking:
            firstlick = meta['firstlick']
            if np.isfinite(firstlick).sum() == 0:
                # skip days where there are no licks at all
                print('No licking on '
                      + str(day1.date) + ', skipping for -no-lick.')
                continue
            median_lick = []
            for cs in [0, 135, 270]:
                median_lick.append(meta.loc[(orientation == cs), :]['firstlick'].median())
            median_lick = np.nanmin(median_lick)
            for tri in range(np.shape(intensor)[2]):
                if np.isfinite(firstlick[tri]):
                    intensor[:, int(np.floor(15.5 + firstlick[tri] - 1)):, tri] = np.nan
                else:
                    # (if there were licks recorded)
                    if np.isfinite(median_lick):
                        intensor[:, int(np.floor(15.5 + median_lick)):, tri] = np.nan

        # split running into high and low groups
        if running_highspeed and running_lowspeed:
            print('WTF. High and low speed boolean?')
        elif running_highspeed:
            speed_bool = speed > speed_thresh
            intensor = intensor[:, :, speed_bool]
        elif running_lowspeed:
            speed_bool = speed <= speed_thresh
            intensor = intensor[:, :, speed_bool]

        # sort neuron factors by component they belong to most
        # assumes only one method was used
        method = [i for i in ensemble.keys()][0]
        sort_ensemble, my_sorts = tca._sortfactors(ensemble[method])

        # run noise corr for each cs
        cs_clusters = {}
        cscorr = {}
        sscorr = {}
        for cs in ['0', '135', '270']:

            # boolean index for cs accounting for running boolean
            if running_highspeed or running_lowspeed:
                csbool = np.isin(orientation, int(cs))[speed_bool]
            else:
                csbool = np.isin(orientation, int(cs))

            # select trials
            trs = np.nanmean(intensor[:, 16:48, csbool], axis=1)
            ncells = np.shape(trs)[0]

            # additional visual drive thresholding
            drive = pool.calc.driven.visually(day1, cs)
            drive_bool = drive[np.isin(d1_ids, ids)] > vis_drive_thresh

            # get TCA clusters
            # factors are already sorted, so these will define
            # clusters, no need to sort again
            cell_clusters = {}
            for k in sort_ensemble.results.keys():
                factors = sort_ensemble.results[k][0].factors[0]
                max_fac = np.argmax(factors, axis=1)
                max_fac_val = np.max(factors, axis=1)
                max_fac[max_fac_val == 0] = -1
                cell_clusters[k] = max_fac[drive_bool]
            cs_clusters[cs] = cell_clusters

            # Catch cases when there aren't enough trials
            if np.shape(trs)[1] < 10:
                print('Not enough trials (' + str(np.shape(trs)[1]) + ') in '
                      + str(day1.date) + ' ' + cs +
                      ' trs to calculate signal corr.')
                continue

            stimorder = np.arange(np.shape(trs)[1])
            corrs = np.zeros((ncells, ncells))
            if np.sum(np.invert(np.isfinite(trs))) == 0:

                for i in range(randomizations):
                    for c in range(ncells):
                        np.random.shuffle(stimorder)
                        trs[c, :] = trs[c, stimorder]

                    corrs += np.corrcoef(trs)/float(randomizations)

                # zero diagonal so you can see things
                for i in range(ncells):
                    corrs[i, i] = 0

            sscorr[cs] = corrs[:, my_sorts[sort_rank-1]][my_sorts[sort_rank-1], :][drive_bool, :][:, drive_bool]
            cscorr[cs] = corrs[drive_bool, :][:, drive_bool]
        acorr_daily.append(cscorr)
        acorr_daily_sort.append(sscorr)
        aclus_daily_sort.append(cs_clusters)

    # plot
    cmap = sns.color_palette('muted', sort_rank)
    whitergb = (1, 1, 1)
    for c, corr in enumerate(acorr_daily_sort):
        for cs in ['0', '135', '270']:
            if ~np.isin(cs, list(corr.keys())):
                continue
            if np.sum(np.isnan(corr[cs]).flatten()) > 0:
                continue
            if len(corr[cs]) == 0:
                continue
            color_vec = []
            for k in aclus_daily_sort[c][cs][sort_rank]:
                if k >= 0:
                    color_vec.append(cmap[int(k)])
                else:
                    color_vec.append(whitergb)
            fig = plt.figure()
            a = sns.clustermap(
                corr[cs], row_colors=color_vec, col_colors=color_vec,
                row_cluster=cluster, col_cluster=cluster, method='ward')
            a.cax.set_ylabel('correlation\ncoefficient (R)')
            a.cax.set_title(
                'Signal Corr :' +
                str(days[c].date) + ': ' + corr_trace_type + ': rank '
                + str(sort_rank) + ': ' + cs +
                ' ' + lick_tag + run_tag + clus_save,
                horizontalalignment='left')
            a.ax_heatmap.set_xlabel('Cell Number')
            a.ax_heatmap.set_ylabel('Cell Number')

            # save, show figs as you go if verbose
            save_path = os.path.join(
                save_dir, 'cs' + cs + '-' +
                str(days[c].date) + '-rank' +
                str(int(sort_rank)) + lick_tag + run_tag + clus_save + '.png')
            plt.savefig(save_path, bbox_inches='tight')
            if verbose:
                plt.show()
            plt.close('all')

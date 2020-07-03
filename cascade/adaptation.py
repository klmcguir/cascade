"""Functions for fitting analyzing different timescales of adaptation."""
from scipy.optimize import curve_fit
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import numpy as np
import os
from . import load, paths, utils, lookups, tuning
from copy import deepcopy
import scipy as sp


# TODO
#  def calc_daily_adaptation_of_sustainedness()
#  - this is a testing the hypothesis that daily adaptation comes from
#  - take mean of first 30 trials of each cue type per cell. Then fit sustainedness
#  def calc_daily_sustainedness()
#  - take mean per day for each cell. Then fit sustainedness


def calc_daily_transientness(
        mice,
        words=None,
        method='ncp_hals',
        cs='',
        warp=False,
        trace_type='zscore_day',
        group_by='all3',
        nan_thresh=0.95,
        score_threshold=0.8,
        rank=15,
        annotate=True,
        over_components=False,
        project=False):
    """
    Function for plotting and saving NNLS fitting per day across stages of learning for transientness.
    Fits TCA results for individual results and saves a plot for each mouse. Saves DataFrame of results for all mice.

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
    :param rank: int, rank of TCA model to use for fitting
    :param annotate: boolean, add text annotations per day of fitting results
    :param over_components: boolean, fit over components or over individual cells
    :param project: boolean, temporally project you data (matching cells across time) to a larger dataset
    :return exp_fit_df: pandas.DataFrame, parameters an error for fitting exponential decay to TCA trial factors
    """

    # load your data
    sus_list = []
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
        model, ids, tensor, meta, _, sorts = load.load_all_groupday(**load_kwargs)

        # temporal projection to allow you to use TCA data tp classify larger time period
        if project:
            root_word = 'trans_projection'
            meta, model, tensor, ids, sorts = temporal_projection(meta, model, sorts, ids)
        else:
            root_word = 'transient'

        # create a new analysis directory for your mouse named 'sustained'
        save_path = paths.groupmouse_analysis_path(root_word, mice=mice, words=words, **load_kwargs)

        # create new DataFrame for adaptation
        if over_components:
            df = transientness_from_components(
                meta, model, tensor, sorts, rank=rank, save_folder=save_path, annotate=annotate)
        else:
            df = transientness_from_cells(
                meta, model, tensor, sorts, ids, rank=rank, save_folder=save_path, annotate=annotate)
        sus_list.append(df)

    transientness_df = pd.concat(sus_list, axis=0)

    # save
    comp_or_cell = 'comp' if over_components else 'cell'
    save_folder = save_path + f' day transient {comp_or_cell} rank {rank}'
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    transientness_df.to_pickle(
        os.path.join(save_folder, f'TCA_daily_transientness_r{rank}_{comp_or_cell}.pkl'))

    return transientness_df


def transientness_from_components(meta, model, tensor, sorts, rank=15, save_folder='', annotate=True):
    """
    Function for fitting daily adaptation of TCA trial factors across learning.
    Saves plot of fit performance as well as returning a DataFrame of fit parameters.

    :param meta: pandas.DataFrame, metadata DataFrame where each index is a unique trial
    :param model: tensortools.ensemble, TCA results
    :param tensor: numpy.ndarray (3D), input data for TCA, trials triggered on stimulus onset
    :param sorts: list of numpy.ndarray, sorting schemes for each rank of decomposition based on cell factors
    :param rank: int, rank of TCA model to use for fitting
    :param save_folder: str, directory to save plots into
    :param annotate: boolean, add text annotations per day of fitting results
    :return: exp_fit_df: pandas.DataFrame, fit parameters and errors
    """

    # get mouse from metadata index, must have only one mouse
    assert len(meta.reset_index()['mouse'].unique()) == 1
    mouse = meta.reset_index()['mouse'].unique()[0]

    # make sure that the date vec allows for 0.5 days at reversal and learning
    meta = utils.update_meta_date_vec(meta)

    # add all stage parsing to meta for use adding to adaptation metadata
    meta = utils.add_5stages_to_meta(meta)
    meta = utils.add_10stages_to_meta(meta)
    meta = utils.add_11stages_to_meta(meta)

    # create dprime and day vector that has 108 points per day to match timepoints in a trial
    day_meta = meta.groupby('date').max()
    dp_single_day = day_meta['dprime']
    dp100 = np.zeros(len(dp_single_day) * 108)
    adjusted_date_vec = np.zeros(len(dp_single_day) * 108)
    adjusted_learning_state = np.empty(len(dp_single_day) * 108, dtype='<U8')
    for c, dps in enumerate(dp_single_day):
        x = np.arange(108) + (108 * c)
        dp100[x] = np.ones(108) * dps
        adjusted_date_vec[x] = np.ones(108) * day_meta.reset_index()['date'].values[c]
        adjusted_learning_state[x] = np.array([day_meta['learning_state'].values[c]] * 108)

    # calculate change indices for days and reversal/learning
    ndays = np.diff(adjusted_date_vec)
    day_x = np.where(ndays)[0] + 0.5
    rev_ind = np.where(np.isin(adjusted_learning_state, 'learning'))[0][-1]
    lear_ind = np.where(np.isin(adjusted_learning_state, 'learning'))[0][0]

    # create a stereotypical time vector for stimulus box
    time_vec = np.arange(-1, 6, 1 / 15.5)[:108]
    stim_bool = (time_vec > 0) & (time_vec < lookups.stim_length[mouse])

    # sort your tensor and model
    rank_sorter = sorts[rank - 1]
    cells_sorted = model.results[rank][0].factors[0][rank_sorter, :]
    tensor_sorted = tensor[rank_sorter, :, :]

    # get cell groupings for cells above 1 std deviation weight per cell factor
    thresh = np.std(cells_sorted, axis=0) * 1
    weights = deepcopy(cells_sorted)
    for i in range(cells_sorted.shape[1]):
        weights[weights[:, i] < thresh[i], i] = np.nan
    above_thresh = ~np.isnan(np.nanmax(weights, axis=1))
    best_cluster = np.argmax(cells_sorted, axis=1)[above_thresh]
    tensor_sorted = tensor_sorted[above_thresh, :, :]

    # calculate mean cell response per day for tensor
    tensor_cell_dict = {}
    for cue_n in ['plus', 'minus', 'neutral']:
        tensor_cell_dict[cue_n] = utils.tensor_mean_per_day(
            meta, tensor_sorted, initial_cue=True, cue=cue_n, nan_licking=False)

    # create component-averaged tensors. You will now have a dict for each cue, containing a tensor.
    # This tensor is the average response dynamics per day for a "component" (an average across cells)
    # [n, :, :] is components, [:, n, :] is timepoints, [:, :, n] is days.
    tensor_comp_dict = {}
    for cue_n in ['plus', 'minus', 'neutral']:
        tensor_comp_dict[cue_n] = np.zeros((rank, tensor_cell_dict[cue_n].shape[1], tensor_cell_dict[cue_n].shape[2]))
        tensor_comp_dict[cue_n][:] = np.nan
        for clus_n in np.unique(best_cluster):
            clus_bool = best_cluster == clus_n
            tensor_comp_dict[cue_n][clus_n, :, :] = np.nanmean(tensor_cell_dict[cue_n][clus_bool, :, :], axis=0)

    # create component-averaged trial matrices
    mean_trial_mat_cells = utils.tensor_mean_per_trial(meta, tensor_sorted, nan_licking=False)
    mat_df = pd.DataFrame(data=mean_trial_mat_cells)
    mat_df['groups'] = best_cluster
    mean_trial_mat_comps = mat_df.groupby('groups').mean().values
    assert mean_trial_mat_comps.shape[0] == rank  # matrix should now have one row per component

    # get tuning per cell and per component-average
    tuning_vec_comps = []
    for comp_n in range(mean_trial_mat_comps.shape[0]):
        tuning_vec_comps.append(tuning.calc_tuning_from_meta_and_vec(meta, mean_trial_mat_comps[comp_n, :]))

    # plot and create DataFrame
    df_list = []
    df_columns = ['onset beta', 'sustained beta', 'offset beta', 'response window beta', 'transientness', 'ramp index']
    date_vec = meta.reset_index()['date'].unique()
    for cue_n in ['plus', 'minus', 'neutral']:

        # drop broadly tuned cells and components (permissive, allows tuning == 'plus-neutral' for plus)
        # drop components with an offset response as well
        # comp_bool = np.array([cue_n in s for s in tuning_vec_cells])  # permissive
        comp_bool = np.array([s in cue_n for s in tuning_vec_comps])  # strict
        off_bool = np.argmax(model.results[rank][0].factors[1][:, :], axis=0) > 15.5 * (1 + lookups.stim_length[mouse])

        cue_tensor = tensor_comp_dict[cue_n][comp_bool & ~off_bool, :, :]
        tuning_vec = np.array(tuning_vec_comps)[comp_bool & ~off_bool]
        cue_clusters = np.unique(best_cluster)[comp_bool & ~off_bool] + 1  # +1 so that cluster matches component #

        fig, ax = plt.subplots(cue_tensor.shape[0], 1, figsize=(30, 4 * cue_tensor.shape[0]), sharey=True, sharex=True)
        lbl_done = False  # only add to legend once
        # is only one component is tuned to a cue make sure indexing doesn't break
        if cue_tensor.shape[0] == 1:
            ax = [ax]
        elif cue_tensor.shape[0] == 0:
            continue
        for comp_n in range(cue_tensor.shape[0]):

            # preallocate ramp fit vector, same length as meta
            all_fits = np.zeros((len(date_vec), len(df_columns)))
            all_fits[:] = np.nan

            # plot dprime on second y axis
            dp_ax = ax[comp_n].twinx()
            dp_ax.plot(dp100, '-', color='#C880D1', linewidth=2)
            dp_ax.set_ylabel('dprime', color='#C880D1', size=14)

            # plot date and revesal/learning vertical lines
            first_day = True
            if len(day_x) > 0:
                for k in day_x:
                    if first_day:
                        ax[comp_n].axvline(k, color=lookups.color_dict['gray'], linewidth=2,
                                           label='day transitions')
                        first_day = False
                    else:
                        ax[comp_n].axvline(k, color=lookups.color_dict['gray'], linewidth=2)
            ax[comp_n].axvline(lear_ind, linestyle='--', color=lookups.color_dict['learning'],
                               linewidth=3, label='learning starts')
            ax[comp_n].axvline(rev_ind, linestyle='--', color=lookups.color_dict['reversal'],
                               linewidth=3, label='reversal starts')

            # get y lim upper bound for plotting text annotations
            y_txt = np.nanmax(cue_tensor[comp_n, :, :])

            # get tuning of cue as string
            pref_tuning = tuning_vec[comp_n]

            for day_n in range(len(date_vec)):

                # select indices and cell data for a single day and cell
                inds = np.where(adjusted_date_vec == date_vec[day_n])[0]
                fit_comp_vec = deepcopy(cue_tensor[comp_n, :, day_n])

                # skip if day has no values
                if np.sum(np.isnan(fit_comp_vec)) == len(fit_comp_vec):
                    continue

                # plot mean response for each day
                if comp_n == 0 and not lbl_done:
                    ax[comp_n].plot(inds, fit_comp_vec, linewidth=3, color=lookups.color_dict[cue_n],
                                    label='stimulus response')
                else:
                    ax[comp_n].plot(inds, fit_comp_vec, linewidth=3, color=lookups.color_dict[cue_n])

                # for ramp index calc
                epsilon = 0.00001  # tiny value to prevent divide by zero
                mid_ind = int(np.floor((lookups.stim_length[mouse] / 2 + 1) * 15.5))
                last_ind = int(np.floor((lookups.stim_length[mouse] + 1) * 15.5))
                y1 = fit_comp_vec[16:mid_ind]  # first half of response
                y2 = fit_comp_vec[mid_ind:last_ind]  # second half of response
                my1 = np.mean(y1) + epsilon
                my2 = np.mean(y2) + epsilon

                # calculate log2 ramp index
                day_ramp_index = np.log2(my1 / my2)
                all_fits[day_n, -1] = day_ramp_index

                # add text for fits
                txt_results = []
                txt_results.append(f'tuned: {pref_tuning}\n')
                txt_results.append(
                    f'{cue_n} ramp: {round(day_ramp_index, 2)}\n')

                # set ramp to 0 if it is being calculated on tiny values
                if my1 < 0.1 and my2 < 0.1:
                    all_fits[day_n, -1] = 0

                # set ramp to 0 it is being calculated on a non-preferred or broadly tuned cell/component
                if cue_n not in pref_tuning:
                    all_fits[day_n, -1] = 0

                # NNLS fitting
                filters = _get_gaussian_fitting_template(mouse)
                fit_comp_vec[fit_comp_vec < 0] = 0  # rectify for non-negative fitting
                nnls_fits = sp.optimize.nnls(filters, fit_comp_vec)[0]
                all_fits[day_n, :-2] = nnls_fits
                trans = nnls_fits[0] / (nnls_fits[0] + nnls_fits[1])
                all_fits[day_n, -2] = trans

                # add text for nnls fits
                txt_results.append(
                    f'{cue_n} trans: {round(trans, 2)}\n')

                # plot NNLS fit
                reconstruction = filters @ nnls_fits[:, None]
                if comp_n == 0 and not lbl_done:
                    ax[comp_n].plot(inds, reconstruction, linewidth=3, color=lookups.color_dict[cue_n + '2'],
                                    label='NNLS fit')
                    lbl_done = True
                else:
                    ax[comp_n].plot(inds, reconstruction, linewidth=3, color=lookups.color_dict[cue_n + '2'])

                # add text summary of fits per day
                if annotate:
                    txt_label = ''
                    for s in txt_results:
                        txt_label = txt_label + s
                    ax[comp_n].text(inds[int(np.floor(len(inds) / 2))], y_txt, txt_label, ha='center', va='top')

            # set labels for subplots
            ax[comp_n].set_ylabel(f'component {cue_clusters[comp_n]}\n\nresponse amplitude\n\u0394F/F\u2080 (z-score)',
                                  size=14)
            if comp_n == 0:
                ax[comp_n].set_title(f'{mouse}: cue {cue_n}:\nNNLS fits and ramp index for within trial dynamics',
                                     size=16)
                ax[comp_n].legend(bbox_to_anchor=(1.05, 1.05), loc='upper left')
            elif comp_n == cue_tensor.shape[0] - 1:
                ax[comp_n].set_xlabel('average response per day', size=14)

            # plot Rectangle for stimulus period
            boxes = []
            for day_n in range(len(date_vec)):
                inds = np.where(adjusted_date_vec == date_vec[day_n])[0]
                box_inds = inds[stim_bool]
                if comp_n == 0:
                    y = ax[comp_n].get_ylim()[0]
                boxes.append(Rectangle((box_inds[0], y * 1.1), np.sum(stim_bool), np.abs(y)))
            pc = PatchCollection(boxes, facecolor=lookups.color_dict['gray'], edgecolor=lookups.color_dict['gray'],
                                 alpha=0.7)
            ax[comp_n].add_collection(pc)

            # create a dataframe for the results from each cell, rows are days
            comp_fit_df = pd.DataFrame(data=all_fits, columns=df_columns)
            comp_fit_df['date'] = date_vec
            comp_fit_df['mouse'] = mouse
            comp_fit_df['tuning'] = pref_tuning
            comp_fit_df['initial cue'] = cue_n
            comp_fit_df['best component'] = cue_clusters[comp_n]

            # share some useful columns from meta
            for col in ['parsed_stage', 'parsed_10stage', 'parsed_11stage', 'dprime']:
                comp_fit_df[col] = day_meta[col].values

            # collect fit results for each component
            df_list.append(comp_fit_df)

        # save
        annotxt = '_annot' if annotate else ''
        dp_save_folder = save_folder + f' day transient comp rank {rank}'
        if not os.path.isdir(dp_save_folder):
            os.mkdir(dp_save_folder)
        plt.savefig(
            os.path.join(
                dp_save_folder,
                f'{mouse}_r{rank}_{cue_n}_comp_daily_nnls_ramp{annotxt}.png'),
            bbox_inches='tight')
        plt.close('all')

    exp_fit_df = pd.concat(df_list, axis=0)

    return exp_fit_df


def transientness_from_cells(meta, model, tensor, sorts, ids, rank=15, save_folder='', annotate=True):
    """
    Function for fitting daily adaptation of TCA trial factors across learning.
    Saves plot of fit performance as well as returning a DataFrame of fit parameters.

    :param meta: pandas.DataFrame, metadata DataFrame where each index is a unique trial
    :param model: tensortools.ensemble, TCA results
    :param tensor: numpy.ndarray (3D), input data for TCA, trials triggered on stimulus onset
    :param sorts: list of numpy.ndarray, sorting schemes for each rank of decomposition based on cell factors
    :param ids: list of numpy.ndarray, unique cell ids
    :param rank: int, rank of TCA model to use for fitting
    :param save_folder: str, directory to save plots into
    :param annotate: boolean, add text annotations per day of fitting results
    :return: exp_fit_df: pandas.DataFrame, fit parameters and errors
    """

    # get mouse from metadata index, must have only one mouse
    assert len(meta.reset_index()['mouse'].unique()) == 1
    mouse = meta.reset_index()['mouse'].unique()[0]

    # make sure that the date vec allows for 0.5 days at reversal and learning
    meta = utils.update_meta_date_vec(meta)

    # add all stage parsing to meta for use adding to adaptation metadata
    meta = utils.add_5stages_to_meta(meta)
    meta = utils.add_10stages_to_meta(meta)
    meta = utils.add_11stages_to_meta(meta)

    # create dprime and day vector that has 108 points per day to match timepoints in a trial
    day_meta = meta.groupby('date').max()
    dp_single_day = day_meta['dprime']
    dp100 = np.zeros(len(dp_single_day) * 108)
    adjusted_date_vec = np.zeros(len(dp_single_day) * 108)
    adjusted_learning_state = np.empty(len(dp_single_day) * 108, dtype='<U8')
    for c, dps in enumerate(dp_single_day):
        x = np.arange(108) + (108 * c)
        dp100[x] = np.ones(108) * dps
        adjusted_date_vec[x] = np.ones(108) * day_meta.reset_index()['date'].values[c]
        adjusted_learning_state[x] = np.array([day_meta['learning_state'].values[c]] * 108)

    # calculate change indices for days and reversal/learning
    ndays = np.diff(adjusted_date_vec)
    day_x = np.where(ndays)[0] + 0.5
    rev_ind = np.where(np.isin(adjusted_learning_state, 'learning'))[0][-1]
    lear_ind = np.where(np.isin(adjusted_learning_state, 'learning'))[0][0]

    # create a stereotypical time vector for stimulus box
    time_vec = np.arange(-1, 6, 1 / 15.5)[:108]
    stim_bool = (time_vec > 0) & (time_vec < lookups.stim_length[mouse])

    # sort your tensor and model
    rank_sorter = sorts[rank - 1]
    cells_sorted = model.results[rank][0].factors[0][rank_sorter, :]
    tensor_sorted = tensor[rank_sorter, :, :]
    ids_sorted = ids[rank_sorter]

    # get cell groupings for cells above 1 std deviation weight per cell factor
    thresh = np.std(cells_sorted, axis=0) * 1
    weights = deepcopy(cells_sorted)
    for i in range(cells_sorted.shape[1]):
        weights[weights[:, i] < thresh[i], i] = np.nan
    above_thresh = ~np.isnan(np.nanmax(weights, axis=1))
    best_cluster = np.argmax(cells_sorted, axis=1)[above_thresh] + 1  # to make clusters match component number
    tensor_sorted = tensor_sorted[above_thresh, :, :]
    ids_sorted = ids_sorted[above_thresh]

    # calculate mean cell response per day for tensor
    tensor_cell_dict = {}
    for cue_n in ['plus', 'minus', 'neutral']:
        tensor_cell_dict[cue_n] = utils.tensor_mean_per_day(
            meta, tensor_sorted, initial_cue=True, cue=cue_n, nan_licking=False)

    # calculate mean cell response per trial
    mean_trial_mat_cells = utils.tensor_mean_per_trial(meta, tensor_sorted, nan_licking=False)

    # get tuning per cell
    tuning_vec_cells = []
    for cell_n in range(mean_trial_mat_cells.shape[0]):
        tuning_vec_cells.append(tuning.calc_tuning_from_meta_and_vec(meta, mean_trial_mat_cells[cell_n, :]))

    # plot and create DataFrame
    df_list = []
    df_columns = ['onset beta', 'sustained beta', 'offset beta', 'response window beta', 'transientness', 'ramp index']
    date_vec = meta.reset_index()['date'].unique()
    for cue_n in ['plus', 'minus', 'neutral']:

        # drop broadly tuned cells (permissive, allows tuning == 'plus-neutral' for cue_n 'plus')
        # since we only measured tuning on cell stimulus period, cells
        # with an offset response should be dropped unless they have a stimulus component
        # cell_bool = np.array([cue_n in s for s in tuning_vec_cells])  # permissive
        cell_bool = np.array([s in cue_n for s in tuning_vec_cells])  # strict

        cue_tensor = tensor_cell_dict[cue_n][cell_bool, :, :]
        tuning_vec = np.array(tuning_vec_cells)[cell_bool]
        cue_clusters = best_cluster[cell_bool] + 1  # +1 so that cluster matches component #
        cue_ids = ids_sorted[cell_bool]

        # if you have no cells skip this plot
        if cue_tensor.shape[0] == 0:
            continue

        for cell_n in range(cue_tensor.shape[0]):
            fig, ax = plt.subplots(1, 1, figsize=(30, 4 * 1), sharey=True, sharex=True)
            lbl_done = False  # only add to legend once

            # preallocate ramp fit vector, same length as meta
            all_fits = np.zeros((len(date_vec), len(df_columns)))
            all_fits[:] = np.nan

            # plot dprime on second y axis
            dp_ax = ax.twinx()
            dp_ax.plot(dp100, '-', color='#C880D1', linewidth=2)
            dp_ax.set_ylabel('dprime', color='#C880D1', size=14)

            # plot date and revesal/learning vertical lines
            first_day = True
            if len(day_x) > 0:
                for k in day_x:
                    if first_day:
                        ax.axvline(k, color=lookups.color_dict['gray'], linewidth=2,
                                   label='day transitions')
                        first_day = False
                    else:
                        ax.axvline(k, color=lookups.color_dict['gray'], linewidth=2)
            ax.axvline(lear_ind, linestyle='--', color=lookups.color_dict['learning'],
                       linewidth=3, label='learning starts')
            ax.axvline(rev_ind, linestyle='--', color=lookups.color_dict['reversal'],
                       linewidth=3, label='reversal starts')

            # get y lim upper bound for plotting text annotations
            y_txt = np.nanmax(cue_tensor[cell_n, :, :])

            # get tuning of cue as string
            pref_tuning = tuning_vec[cell_n]

            for day_n in range(len(date_vec)):

                # select indices and cell data for a single day and cell
                inds = np.where(adjusted_date_vec == date_vec[day_n])[0]
                fit_comp_vec = deepcopy(cue_tensor[cell_n, :, day_n])

                # skip if day has no values
                if np.sum(np.isnan(fit_comp_vec)) == len(fit_comp_vec):
                    continue

                # plot mean response for each day
                if not lbl_done:
                    ax.plot(inds, fit_comp_vec, linewidth=3, color=lookups.color_dict[cue_n],
                            label='stimulus response')
                else:
                    ax.plot(inds, fit_comp_vec, linewidth=3, color=lookups.color_dict[cue_n])

                # for ramp index calc
                epsilon = 0.00001  # tiny value to prevent divide by zero
                mid_ind = int(np.floor((lookups.stim_length[mouse] / 2 + 1) * 15.5))
                last_ind = int(np.floor((lookups.stim_length[mouse] + 1) * 15.5))
                y1 = fit_comp_vec[16:mid_ind]  # first half of response
                y2 = fit_comp_vec[mid_ind:last_ind]  # second half of response
                my1 = np.mean(y1) + epsilon
                my2 = np.mean(y2) + epsilon

                # calculate log2 ramp index
                day_ramp_index = np.log2(my1 / my2)
                all_fits[day_n, -1] = day_ramp_index

                # add text for fits
                txt_results = []
                txt_results.append(f'tuned: {pref_tuning}\n')
                txt_results.append(
                    f'{cue_n} ramp: {round(day_ramp_index, 2)}\n')

                # set ramp to 0 if it is being calculated on tiny values
                if my1 < 0.1 and my2 < 0.1:
                    all_fits[day_n, -1] = 0

                # set ramp to 0 it is being calculated on a non-preferred or broadly tuned cell/component
                if cue_n not in pref_tuning:
                    all_fits[day_n, -1] = 0

                # NNLS fitting
                filters = _get_gaussian_fitting_template(mouse)
                fit_comp_vec[fit_comp_vec < 0] = 0  # rectify for non-negative fitting
                nnls_fits = sp.optimize.nnls(filters, fit_comp_vec)[0]
                all_fits[day_n, :-2] = nnls_fits
                trans = nnls_fits[0] / (nnls_fits[0] + nnls_fits[1])
                all_fits[day_n, -2] = trans

                # add text for nnls fits
                txt_results.append(
                    f'{cue_n} trans: {round(trans, 2)}\n')

                # plot NNLS fit
                reconstruction = filters @ nnls_fits[:, None]
                if not lbl_done:
                    ax.plot(inds, reconstruction, linewidth=3, color=lookups.color_dict[cue_n + '2'],
                            label='NNLS fit')
                    lbl_done = True
                else:
                    ax.plot(inds, reconstruction, linewidth=3, color=lookups.color_dict[cue_n + '2'])

                # add text summary of fits per day
                if annotate:
                    txt_label = ''
                    for s in txt_results:
                        txt_label = txt_label + s
                    ax.text(inds[int(np.floor(len(inds) / 2))], y_txt, txt_label, ha='center', va='top')

            # set labels for subplots
            ax.set_ylabel(f'cell {cell_n + 1}\n\nresponse amplitude\n\u0394F/F\u2080 (z-score)', size=14)
            ax.set_title(
                f'{mouse}: cell {cell_n}: cue {cue_n}: best comp {cue_clusters[cell_n]}:\nNNLS fits and ramp index for within trial dynamics',
                size=16)
            ax.legend(bbox_to_anchor=(1.05, 1.05), loc='upper left')
            ax.set_xlabel('average response per day', size=14)

            # plot Rectangle for stimulus period
            boxes = []
            for day_n in range(len(date_vec)):
                inds = np.where(adjusted_date_vec == date_vec[day_n])[0]
                box_inds = inds[stim_bool]
                y = ax.get_ylim()[0]
                boxes.append(Rectangle((box_inds[0], y * 1.1), np.sum(stim_bool), np.abs(y)))
            pc = PatchCollection(boxes, facecolor=lookups.color_dict['gray'], edgecolor=lookups.color_dict['gray'],
                                 alpha=0.7)
            ax.add_collection(pc)

            # create a dataframe for the results from each cell, rows are days
            comp_fit_df = pd.DataFrame(data=all_fits, columns=df_columns)
            comp_fit_df['cell'] = cue_ids[cell_n]
            comp_fit_df['date'] = date_vec
            comp_fit_df['mouse'] = mouse
            comp_fit_df['tuning'] = pref_tuning
            comp_fit_df['initial cue'] = cue_n
            comp_fit_df['best component'] = cue_clusters[cell_n]

            # share some useful columns from meta
            for col in ['parsed_stage', 'parsed_10stage', 'parsed_11stage', 'dprime']:
                comp_fit_df[col] = day_meta[col].values

            # collect fit results for each component
            df_list.append(comp_fit_df)

            # save
            annotxt = '_annot' if annotate else ''
            dp_save_folder = save_folder + f' day transient cell rank {rank}'
            if not os.path.isdir(dp_save_folder):
                os.mkdir(dp_save_folder)
            dp_save_folder = os.path.join(dp_save_folder, mouse)
            if not os.path.isdir(dp_save_folder):
                os.mkdir(dp_save_folder)
            plt.savefig(
                os.path.join(
                    dp_save_folder,
                    f'{mouse}_r{rank}_{cue_n}_{cell_n}_TCA_daily_nnls_ramp{annotxt}.png'),
                bbox_inches='tight')
            plt.close('all')

    exp_fit_df = pd.concat(df_list, axis=0)

    return exp_fit_df


def calc_daily_ramp(
        mice,
        words=None,
        method='ncp_hals',
        cs='',
        warp=False,
        trace_type='zscore_day',
        group_by='all3',
        nan_thresh=0.95,
        score_threshold=0.8,
        rank=15,
        annotate=True,
        over_components=False):
    """
    Function for plotting and saving ramp index (not exponential decay) fitting per day across stages of learning.
    Fits trial factors and saves a plot for each mouse. Saves DataFrame of results for all mice.

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
    :param rank: int, rank of TCA model to use for fitting
    :param annotate: boolean, add text annotations per day of fitting results
    :return exp_fit_df: pandas.DataFrame, parameters an error for fitting exponential decay to TCA trial factors
    """

    # load your metadata and TCA models
    adapt_list = []
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
        model, ids, tensor, meta, _, sorts = load.load_all_groupday(**load_kwargs)

        # create a new analysis directory for your mouse named 'adaptation'
        save_path = paths.groupmouse_analysis_path('ramp', mice=mice, words=words, **load_kwargs)

        # create new DataFrame for adaptation
        if over_components:
            df = ramp_from_meta_model(meta, model, rank=rank, save_folder=save_path, annotate=annotate)
        else:
            df = ramp_from_cells(meta, model, tensor, sorts, ids, rank=rank, save_folder=save_path, annotate=annotate)
        adapt_list.append(df)

    ramp_fit_df = pd.concat(adapt_list, axis=0)

    # save
    comp_or_cell = 'comp' if over_components else 'cell'
    save_folder = save_path + f' day ramp {comp_or_cell} rank {rank}'
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    ramp_fit_df.to_pickle(
        os.path.join(save_folder, f'TCA_daily_ramp_r{rank}_{comp_or_cell}.pkl'))

    return ramp_fit_df


def ramp_from_meta_model(meta, model, rank=15, save_folder='', annotate=True):
    """
    Function for fitting daily adaptation of TCA trial factors across learning.
    Saves plot of fit performance as well as returning a DataFrame of fit parameters.

    :param meta: pandas.DataFrame, metadata DataFrame where each index is a unique trial
    :param model: tensortools.ensemble, TCA results
    :param rank: int, rank of TCA model to use for fitting
    :param save_folder: str, directory to save plots into
    :param annotate: boolean, add text annotations per day of fitting results
    :return: exp_fit_df: pandas.DataFrame, fit parameters and errors
    """

    # get mouse from metadata index, must have only one mouse
    assert len(meta.reset_index()['mouse'].unique()) == 1
    mouse = meta.reset_index()['mouse'].unique()[0]

    # make sure that the date vec allows for 0.5 days at reversal and learning
    meta = utils.update_meta_date_vec(meta)

    # add all stage parsing to meta for use adding to adaptation metadata
    meta = utils.add_5stages_to_meta(meta)
    meta = utils.add_10stages_to_meta(meta)
    meta = utils.add_11stages_to_meta(meta)

    # control for running (either choose all running or all stationary)
    loco_boo = stable_locomotor_behavior(meta, select_on_early_trials=True)

    # construct boolean of early runs across all training
    # run number has to be three or less to be considered for fitting
    session_boo = meta.reset_index()['run'].values <= 3

    # only consider the first 100 trials from each day
    first_boo = _first100_bool(meta)

    lick_boo = meta.reset_index()['anticipatory_licks'].values/lookups.stim_length[mouse] <= 3
    hmm_boo = meta.reset_index()['hmm_engaged'].values

    # set firstboo to only include early runs/sessions
    first_boo = first_boo & session_boo & loco_boo & hmm_boo

    # create dprime vector
    if 'dprime_run' not in meta.columns:
        meta = utils.add_dprime_run_to_meta(meta)
    dp100 = meta['dprime_run'].values[first_boo]

    # calculate change indices for days and reversal/learning
    dates = meta.reset_index()['date'].iloc[first_boo]
    ndays = np.diff(dates.values)
    day_x = np.where(ndays)[0] + 0.5
    rev_ind = np.where(meta['learning_state'].iloc[first_boo].isin(['learning']).values)[0][-1]
    lear_ind = np.where(meta['learning_state'].iloc[first_boo].isin(['learning']).values)[0][0]

    # preallocate list for collecting fit parameters
    df_list = []

    # plot
    fig, ax = plt.subplots(rank, 1, figsize=(30, 4 * rank), sharey=True, sharex=True)
    for comp_n in range(model.results[rank][0].factors[2].shape[1]):

        # preallocate ramp fit vector, same length as meta
        ramp_fits = np.zeros(len(meta))
        ramp_fits[:] = np.nan

        # plot dprime on second y axis
        dp_ax = ax[comp_n].twinx()
        dp_ax.plot(dp100, '-', color='#C880D1', linewidth=2)
        dp_ax.set_ylabel('dprime', color='#C880D1', size=14)

        # plot date and revesal/learning vertical lines
        first_day = True
        if len(day_x) > 0:
            for k in day_x:
                if first_day:
                    ax[comp_n].axvline(k, color=lookups.color_dict['gray'], linewidth=2,
                                       label='day transitions')
                    first_day = False
                else:
                    ax[comp_n].axvline(k, color=lookups.color_dict['gray'], linewidth=2)
        ax[comp_n].axvline(lear_ind, linestyle='--', color=lookups.color_dict['learning'],
                           linewidth=3, label='learning starts')
        ax[comp_n].axvline(rev_ind, linestyle='--', color=lookups.color_dict['reversal'],
                           linewidth=3, label='reversal starts')

        # select component for fitting
        comp_vec = model.results[rank][0].factors[2][:, comp_n]
        y_txt = np.max(comp_vec[first_boo])  # get y lim upper bound for plotting text annotations

        # get tuning of cue as string
        pref_tuning = tuning.calc_tuning_from_meta_and_vec(meta, comp_vec)

        skipped_days = {'plus': 0, 'minus': 0, 'neutral': 0}
        for di in meta.reset_index()['date'].unique():

            # boolean for each day accounting for first 100 trials
            day_boo = meta.reset_index()['date'].isin([di]).values

            # skip if too few values are returned for fitting
            trial_count = np.sum(first_boo & day_boo)
            if trial_count <= 10:
                print(f'{mouse}: {trial_count} < required trial number. day:{di}, comp: {comp_n + 1}')
                continue

            txt_results = [f'tuned: {pref_tuning}\n']
            for cue in meta['initial_condition'].unique():

                # boolean for each cue type
                cue_boo = meta['initial_condition'].isin([cue])

                # skip if too few values are returned for fitting
                trial_count = np.sum(first_boo & day_boo & cue_boo)
                if trial_count < 10:
                    # only print skips when they are on the cue that matters
                    if cue in pref_tuning:
                        print(f'{mouse}: {trial_count} < required trial number. cue: {cue},  day: {di}, comp: {comp_n + 1}')
                    skipped_days[cue] += 1
                    continue

                # create consecutive indices for plotting from full length meta
                inds_pseudo = np.zeros(len(meta))
                inds_pseudo[first_boo] = np.arange(np.sum(first_boo))

                # select indices and TCA data for a single day and cue
                inds = inds_pseudo[first_boo & day_boo & cue_boo]
                fit_comp_vec = comp_vec[first_boo & day_boo & cue_boo]

                # plot all trials for each day
                ax[comp_n].plot(inds, fit_comp_vec, 'o', color=lookups.color_dict[cue], alpha=0.3)

                # rename for compactness in fitting function
                epsilon = 0.00001  # tiny value to prevent divide by zero
                mid_ind = int(np.floor(len(fit_comp_vec) / 2))
                y1 = fit_comp_vec[:mid_ind]  # first half of first 100 trials
                y2 = fit_comp_vec[-mid_ind:]  # second half of first 100 trials
                my1 = np.mean(y1) + epsilon
                my2 = np.mean(y2) + epsilon

                # calculate log2 ramp index
                day_ramp_index = np.log2(my1 / my2)
                ramp_fits[first_boo & day_boo & cue_boo] = day_ramp_index

                # add text for fits
                txt_results.append(
                    f'{cue}: {round(day_ramp_index, 2)}\n')

                # set ramp to 0 if it is being calculated on tiny values
                if my1 < 0.1 and my2 < 0.1:
                    ramp_fits[first_boo & day_boo & cue_boo] = 0

                # set ramp to 0 if it is being calculated on a non-preferred or broadly tuned cell/component
                if cue not in pref_tuning:
                    ramp_fits[first_boo & day_boo & cue_boo] = np.nan
                # TODO considering making this nan, was a zero

            # add text summary of fits per day
            if annotate and len(txt_results) > 1:
                txt_label = ''
                for s in txt_results:
                    txt_label = txt_label + s
                ax[comp_n].text(inds[int(np.floor(len(inds) / 2))], y_txt, txt_label, ha='center', va='top')

        # set labels for subplots
        ax[comp_n].set_ylabel(f'component {comp_n + 1}\n\ntrial factor amplitude\n(weighted z-score)', size=14)
        if comp_n == 0:
            ax[comp_n].set_title(f'{mouse}: Daily ramp index', size=16)
            ax[comp_n].legend(bbox_to_anchor=(1.05, 1.05), loc='upper left')
        elif comp_n == rank - 1:
            ax[comp_n].set_xlabel('trial number', size=14)

        # if most days couldn't be fit skip component, trying to avoid comps that are inconsistently nonzero
        day_count = len(meta.reset_index()['date'].unique())
        if pref_tuning not in ['plus', 'minus', 'neutral']:
            ramp_fits[:] = np.nan
        elif skipped_days[pref_tuning] > day_count/2:
            print(f'{mouse}: comp {comp_n + 1} set to NaN. Skipped {skipped_days[pref_tuning]}/{day_count} days.')
            ramp_fits[:] = np.nan

        # check that ramp fits
        # create a dataframe with matching indices to meta
        data = {'ramp index': ramp_fits}
        comp_fit_df = pd.DataFrame(data=data, index=meta.index)
        comp_fit_df['component'] = comp_n + 1
        comp_fit_df['preferred tuning'] = pref_tuning

        # share some useful columns from meta
        for col in ['parsed_stage', 'parsed_10stage', 'parsed_11stage', 'initial_condition']:
            comp_fit_df[col] = meta[col]

        # collect fit results for each component
        df_list.append(comp_fit_df)

    # save
    annotxt = '_annot' if annotate else ''
    dp_save_folder = save_folder + f' day ramp rank {rank}'
    if not os.path.isdir(dp_save_folder):
        os.mkdir(dp_save_folder)
    plt.savefig(
        os.path.join(
            dp_save_folder,
            f'{mouse}_r{rank}_TCA_daily_ramp{annotxt}.png'),
        bbox_inches='tight')

    exp_fit_df = pd.concat(df_list, axis=0)

    return exp_fit_df


def ramp_from_cells(meta, model, tensor, sorts, ids, rank=15, save_folder='', annotate=True):
    # TODO add in ramp details from above
    """
    Function for fitting daily adaptation of TCA trial factors across learning.
    Saves plot of fit performance as well as returning a DataFrame of fit parameters.

    :param meta: pandas.DataFrame, metadata DataFrame where each index is a unique trial
    :param model: tensortools.ensemble, TCA results
    :param tensor: numpy.ndarray (3D), input data for TCA, trials triggered on stimulus onset
    :param sorts: list of numpy.ndarray, sorting schemes for each rank of decomposition based on cell factors
    :param ids: list of numpy.ndarray, unique cell ids
    :param rank: int, rank of TCA model to use for fitting
    :param save_folder: str, directory to save plots into
    :param annotate: boolean, add text annotations per day of fitting results
    :return: exp_fit_df: pandas.DataFrame, fit parameters and errors
    """

    # get mouse from metadata index, must have only one mouse
    assert len(meta.reset_index()['mouse'].unique()) == 1
    mouse = meta.reset_index()['mouse'].unique()[0]

    # make sure that the date vec allows for 0.5 days at reversal and learning
    meta = utils.update_meta_date_vec(meta)

    # add all stage parsing to meta for use adding to adaptation metadata
    meta = utils.add_5stages_to_meta(meta)
    meta = utils.add_10stages_to_meta(meta)
    meta = utils.add_11stages_to_meta(meta)

    # create dprime and day vector that has 108 points per day to match timepoints in a trial
    day_meta = meta.groupby('date').max()
    dp_single_day = day_meta['dprime']
    dp100 = np.zeros(len(dp_single_day) * 108)
    adjusted_date_vec = np.zeros(len(dp_single_day) * 108)
    adjusted_learning_state = np.empty(len(dp_single_day) * 108, dtype='<U8')
    for c, dps in enumerate(dp_single_day):
        x = np.arange(108) + (108 * c)
        dp100[x] = np.ones(108) * dps
        adjusted_date_vec[x] = np.ones(108) * day_meta.reset_index()['date'].values[c]
        adjusted_learning_state[x] = np.array([day_meta['learning_state'].values[c]] * 108)

    # calculate change indices for days and reversal/learning
    ndays = np.diff(adjusted_date_vec)
    day_x = np.where(ndays)[0] + 0.5
    rev_ind = np.where(np.isin(adjusted_learning_state, 'learning'))[0][-1]
    lear_ind = np.where(np.isin(adjusted_learning_state, 'learning'))[0][0]

    # create a stereotypical time vector for stimulus box
    time_vec = np.arange(-1, 6, 1 / 15.5)[:108]
    stim_bool = (time_vec > 0) & (time_vec < lookups.stim_length[mouse])

    # sort your tensor and model
    rank_sorter = sorts[rank - 1]
    cells_sorted = model.results[rank][0].factors[0][rank_sorter, :]
    tensor_sorted = tensor[rank_sorter, :, :]
    ids_sorted = ids[rank_sorter]

    # get cell groupings for cells above 1 std deviation weight per cell factor
    thresh = np.std(cells_sorted, axis=0) * 1
    weights = deepcopy(cells_sorted)
    for i in range(cells_sorted.shape[1]):
        weights[weights[:, i] < thresh[i], i] = np.nan
    above_thresh = ~np.isnan(np.nanmax(weights, axis=1))
    best_cluster = np.argmax(cells_sorted, axis=1)[above_thresh] + 1  # to make clusters match component number
    tensor_sorted = tensor_sorted[above_thresh, :, :]
    ids_sorted = ids_sorted[above_thresh]

    # calculate mean cell response per day for tensor
    tensor_cell_dict = {}
    for cue_n in ['plus', 'minus', 'neutral']:
        tensor_cell_dict[cue_n] = utils.tensor_mean_per_day(
            meta, tensor_sorted, initial_cue=True, cue=cue_n, nan_licking=False)

    # calculate mean cell response per trial
    mean_trial_mat_cells = utils.tensor_mean_per_trial(meta, tensor_sorted, nan_licking=False)

    # get tuning per cell
    tuning_vec_cells = []
    for cell_n in range(mean_trial_mat_cells.shape[0]):
        tuning_vec_cells.append(tuning.calc_tuning_from_meta_and_vec(meta, mean_trial_mat_cells[cell_n, :]))

    # plot and create DataFrame
    df_list = []
    df_columns = ['onset beta', 'sustained beta', 'offset beta', 'response window beta', 'transientness', 'ramp index']
    date_vec = meta.reset_index()['date'].unique()
    for cue_n in ['plus', 'minus', 'neutral']:

        # drop broadly tuned cells (permissive, allows tuning == 'plus-neutral' for cue_n 'plus')
        # since we only measured tuning on cell stimulus period, cells
        # with an offset response should be dropped unless they have a stimulus component
        # cell_bool = np.array([cue_n in s for s in tuning_vec_cells])  # permissive
        cell_bool = np.array([s in cue_n for s in tuning_vec_cells])  # strict

        cue_tensor = tensor_cell_dict[cue_n][cell_bool, :, :]
        tuning_vec = np.array(tuning_vec_cells)[cell_bool]
        cue_clusters = best_cluster[cell_bool] + 1  # +1 so that cluster matches component #
        cue_ids = ids_sorted[cell_bool]

        # if you have no cells skip this plot
        if cue_tensor.shape[0] == 0:
            continue

        for cell_n in range(cue_tensor.shape[0]):
            fig, ax = plt.subplots(1, 1, figsize=(30, 4 * 1), sharey=True, sharex=True)
            lbl_done = False  # only add to legend once

            # preallocate ramp fit vector, same length as meta
            all_fits = np.zeros((len(date_vec), len(df_columns)))
            all_fits[:] = np.nan

            # plot dprime on second y axis
            dp_ax = ax.twinx()
            dp_ax.plot(dp100, '-', color='#C880D1', linewidth=2)
            dp_ax.set_ylabel('dprime', color='#C880D1', size=14)

            # plot date and revesal/learning vertical lines
            first_day = True
            if len(day_x) > 0:
                for k in day_x:
                    if first_day:
                        ax.axvline(k, color=lookups.color_dict['gray'], linewidth=2,
                                   label='day transitions')
                        first_day = False
                    else:
                        ax.axvline(k, color=lookups.color_dict['gray'], linewidth=2)
            ax.axvline(lear_ind, linestyle='--', color=lookups.color_dict['learning'],
                       linewidth=3, label='learning starts')
            ax.axvline(rev_ind, linestyle='--', color=lookups.color_dict['reversal'],
                       linewidth=3, label='reversal starts')

            # get y lim upper bound for plotting text annotations
            y_txt = np.nanmax(cue_tensor[cell_n, :, :])

            # get tuning of cue as string
            pref_tuning = tuning_vec[cell_n]

            for day_n in range(len(date_vec)):

                # select indices and cell data for a single day and cell
                inds = np.where(adjusted_date_vec == date_vec[day_n])[0]
                fit_comp_vec = deepcopy(cue_tensor[cell_n, :, day_n])

                # skip if day has no values
                if np.sum(np.isnan(fit_comp_vec)) == len(fit_comp_vec):
                    continue

                # plot mean response for each day
                if not lbl_done:
                    ax.plot(inds, fit_comp_vec, linewidth=3, color=lookups.color_dict[cue_n],
                            label='stimulus response')
                else:
                    ax.plot(inds, fit_comp_vec, linewidth=3, color=lookups.color_dict[cue_n])

                # for ramp index calc
                epsilon = 0.00001  # tiny value to prevent divide by zero
                mid_ind = int(np.floor((lookups.stim_length[mouse] / 2 + 1) * 15.5))
                last_ind = int(np.floor((lookups.stim_length[mouse] + 1) * 15.5))
                y1 = fit_comp_vec[16:mid_ind]  # first half of response
                y2 = fit_comp_vec[mid_ind:last_ind]  # second half of response
                my1 = np.mean(y1) + epsilon
                my2 = np.mean(y2) + epsilon

                # calculate log2 ramp index
                day_ramp_index = np.log2(my1 / my2)
                all_fits[day_n, -1] = day_ramp_index

                # add text for fits
                txt_results = []
                txt_results.append(f'tuned: {pref_tuning}\n')
                txt_results.append(
                    f'{cue_n} ramp: {round(day_ramp_index, 2)}\n')

                # set ramp to 0 if it is being calculated on tiny values
                if my1 < 0.1 and my2 < 0.1:
                    all_fits[day_n, -1] = 0

                # set ramp to 0 it is being calculated on a non-preferred or broadly tuned cell/component
                if cue_n not in pref_tuning:
                    all_fits[day_n, -1] = 0

                # NNLS fitting
                filters = _get_gaussian_fitting_template(mouse)
                fit_comp_vec[fit_comp_vec < 0] = 0  # rectify for non-negative fitting
                nnls_fits = sp.optimize.nnls(filters, fit_comp_vec)[0]
                all_fits[day_n, :-2] = nnls_fits
                trans = nnls_fits[0] / (nnls_fits[0] + nnls_fits[1])
                all_fits[day_n, -2] = trans

                # add text for nnls fits
                txt_results.append(
                    f'{cue_n} trans: {round(trans, 2)}\n')

                # plot NNLS fit
                reconstruction = filters @ nnls_fits[:, None]
                if not lbl_done:
                    ax.plot(inds, reconstruction, linewidth=3, color=lookups.color_dict[cue_n + '2'],
                            label='NNLS fit')
                    lbl_done = True
                else:
                    ax.plot(inds, reconstruction, linewidth=3, color=lookups.color_dict[cue_n + '2'])

                # add text summary of fits per day
                if annotate:
                    txt_label = ''
                    for s in txt_results:
                        txt_label = txt_label + s
                    ax.text(inds[int(np.floor(len(inds) / 2))], y_txt, txt_label, ha='center', va='top')

            # set labels for subplots
            ax.set_ylabel(f'cell {cell_n + 1}\n\nresponse amplitude\n\u0394F/F\u2080 (z-score)', size=14)
            ax.set_title(
                f'{mouse}: cell {cell_n}: cue {cue_n}: best comp {cue_clusters[cell_n]}:\nNNLS fits and ramp index for within trial dynamics',
                size=16)
            ax.legend(bbox_to_anchor=(1.05, 1.05), loc='upper left')
            ax.set_xlabel('average response per day', size=14)

            # plot Rectangle for stimulus period
            boxes = []
            for day_n in range(len(date_vec)):
                inds = np.where(adjusted_date_vec == date_vec[day_n])[0]
                box_inds = inds[stim_bool]
                y = ax.get_ylim()[0]
                boxes.append(Rectangle((box_inds[0], y * 1.1), np.sum(stim_bool), np.abs(y)))
            pc = PatchCollection(boxes, facecolor=lookups.color_dict['gray'], edgecolor=lookups.color_dict['gray'],
                                 alpha=0.7)
            ax.add_collection(pc)

            # create a dataframe for the results from each cell, rows are days
            comp_fit_df = pd.DataFrame(data=all_fits, columns=df_columns)
            comp_fit_df['cell'] = cue_ids[cell_n]
            comp_fit_df['date'] = date_vec
            comp_fit_df['mouse'] = mouse
            comp_fit_df['tuning'] = pref_tuning
            comp_fit_df['initial cue'] = cue_n
            comp_fit_df['best component'] = cue_clusters[cell_n]

            # share some useful columns from meta
            for col in ['parsed_stage', 'parsed_10stage', 'parsed_11stage', 'dprime']:
                comp_fit_df[col] = day_meta[col].values

            # collect fit results for each component
            df_list.append(comp_fit_df)

            # save
            annotxt = '_annot' if annotate else ''
            dp_save_folder = save_folder + f' day transient cell rank {rank}'
            if not os.path.isdir(dp_save_folder):
                os.mkdir(dp_save_folder)
            dp_save_folder = os.path.join(dp_save_folder, mouse)
            if not os.path.isdir(dp_save_folder):
                os.mkdir(dp_save_folder)
            plt.savefig(
                os.path.join(
                    dp_save_folder,
                    f'{mouse}_r{rank}_{cue_n}_{cell_n}_TCA_daily_nnls_ramp{annotxt}.png'),
                bbox_inches='tight')
            plt.close('all')

    exp_fit_df = pd.concat(df_list, axis=0)

    return exp_fit_df


def calc_daily_adaptation(
        mice,
        words=None,
        method='ncp_hals',
        cs='',
        warp=False,
        trace_type='zscore_day',
        group_by='all3',
        nan_thresh=0.95,
        score_threshold=0.8,
        rank=15,
        fit_offset=False,
        annotate=True,
        norm=True):
    """
    Function for plotting and saving exponential decay fitting per day across stages of learning.
    Fits TCA trial factors and saves a plot for each mouse. Saves DataFrame of results for all mice.

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
    :param rank: int, rank of TCA model to use for fitting
    :param fit_offset: boolean, fit exponential decay with or without an offset parameter
    :param annotate: boolean, add text annotations per day of fitting results
    :param norm: boolean, normalize each day before fitting
    :return exp_fit_df: pandas.DataFrame, parameters an error for fitting exponential decay to TCA trial factors
    """

    # load your metadata and TCA models
    adapt_list = []
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

        # load TCA models and data
        model, my_sorts = load.groupday_tca_model(full_output=False, unsorted=True, verbose=False, **load_kwargs)

        # create a new analysis directory for your mouse named 'adaptation'
        save_path = paths.groupmouse_analysis_path('adaptation', mice=mice, words=words, **load_kwargs)

        # create new DataFrame for adaptation
        adapt_list.append(
            adaptation_from_meta_model(
                meta, model, rank=rank, save_folder=save_path, fit_offset=fit_offset, annotate=annotate, norm=norm))

    exp_fit_df = pd.concat(adapt_list, axis=0)

    # save
    save_folder = save_path + f' day adapt rank {rank}'
    normtxt = '_norm' if norm else ''
    offtxt = 'offset_incl_' if fit_offset else ''
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    exp_fit_df.to_pickle(
        os.path.join(save_folder, f'{offtxt}TCA_daily_adaptation_r{rank}{normtxt}.pkl'))

    return exp_fit_df


def adaptation_from_meta_model(meta, model, rank=15, save_folder='', fit_offset=True, annotate=True, norm=True):
    """
    Function for fitting daily adaptation of TCA trial factors across learning.
    Saves plot of fit performance as well as returning a DataFrame of fit parameters.

    :param meta: pandas.DataFrame, metadata DataFrame where each index is a unique trial
    :param model: tensortools.ensemble, TCA results
    :param rank: int, rank of TCA model to use for fitting
    :param save_folder: str, directory to save plots into
    :param fit_offset: boolean, fit exponential decay with or without an offset parameter
    :param annotate: boolean, add text annotations per day of fitting results
    :param norm: boolean, normalize each day before fitting
    :return: exp_fit_df: pandas.DataFrame, fit parameters and errors
    """

    # get mouse from metadata index, must have only one mouse
    assert len(meta.reset_index()['mouse'].unique()) == 1
    mouse = meta.reset_index()['mouse'].unique()[0]

    # make sure that the date vec allows for 0.5 days at reversal and learning
    meta = utils.update_meta_date_vec(meta)

    # add all stage parsing to meta for use adding to adaptation metadata
    meta = utils.add_5stages_to_meta(meta)
    meta = utils.add_10stages_to_meta(meta)
    meta = utils.add_11stages_to_meta(meta)

    # construct boolean of early runs across all training
    # run number has to be three or less to be considered for fitting
    session_boo = meta.reset_index()['run'].values <= 3

    # only consider the first 100 trials from each day
    first_boo = _first100_bool(meta)

    # set firstboo to only include early runs/sessions
    first_boo = first_boo & session_boo

    # pick decay function
    if fit_offset:
        func = decay_func_offset
        offset_tag = 'offset_incl_'
        bounds = ([0, -10., 0.], [10., 10., 10.])  # constrain fitting
    else:
        func = decay_func
        offset_tag = ''
        bounds = ([0, -10.], [10., 10.])  # constrain fitting

    # create dprime vector
    if 'dprime_run' not in meta.columns:
        meta = utils.add_dprime_run_to_meta(meta)
    dp100 = meta['dprime_run'].values[first_boo]

    # calculate change indices for days and reversal/learning
    dates = meta.reset_index()['date'].iloc[first_boo]
    ndays = np.diff(dates.values)
    day_x = np.where(ndays)[0] + 0.5
    rev_ind = np.where(meta['learning_state'].iloc[first_boo].isin(['learning']).values)[0][-1]
    lear_ind = np.where(meta['learning_state'].iloc[first_boo].isin(['learning']).values)[0][0]

    # preallocate list for collecting fit parameters
    df_list = []

    # plot
    fig, ax = plt.subplots(rank, 1, figsize=(30, 4 * rank), sharey=True, sharex=True)
    for comp_n in range(model.results[rank][0].factors[2].shape[1]):

        # preallocate fitting parameters
        initial_state_pars = np.zeros(len(meta))
        initial_state_pars[:] = np.nan
        decay_pars = np.zeros(len(meta))
        decay_pars[:] = np.nan
        if fit_offset:
            offset_pars = np.zeros(len(meta))
            offset_pars[:] = np.nan
        fit_err = np.zeros(len(meta))
        fit_err[:] = np.nan

        # plot dprime on second y axis
        dp_ax = ax[comp_n].twinx()
        dp_ax.plot(dp100, '-', color='#C880D1', linewidth=2)
        dp_ax.set_ylabel('dprime', color='#C880D1', size=14)

        # plot date and revesal/learning vertical lines
        first_day = True
        if len(day_x) > 0:
            for k in day_x:
                if first_day:
                    ax[comp_n].axvline(k, color=lookups.color_dict['gray'], linewidth=2,
                                       label='day transitions')
                    first_day = False
                else:
                    ax[comp_n].axvline(k, color=lookups.color_dict['gray'], linewidth=2)
        ax[comp_n].axvline(lear_ind, linestyle='--', color=lookups.color_dict['learning'],
                           linewidth=3, label='learning starts')
        ax[comp_n].axvline(rev_ind, linestyle='--', color=lookups.color_dict['reversal'],
                           linewidth=3, label='reversal starts')

        # select component for fitting
        comp_vec = model.results[rank][0].factors[2][:, comp_n]
        y_txt = np.max(comp_vec[first_boo])  # get y lim upper bound for plotting text annotations

        for di in meta.reset_index()['date'].unique():

            # boolean for each day accounting for first 100 trials
            day_boo = meta.reset_index()['date'].isin([di]).values

            # skip if too few values are returned for fitting
            trial_count = np.sum(first_boo & day_boo)
            if trial_count <= 10:
                print(f'{mouse}: {trial_count} < required trial number. day:{di}, comp: {comp_n + 1}')
                continue

            txt_results = []
            for cue in meta['initial_condition'].unique():

                # boolean for each cue type
                cue_boo = meta['initial_condition'].isin([cue])

                # skip if too few values are returned for fitting
                trial_count = np.sum(first_boo & day_boo & cue_boo)
                if trial_count <= 10:
                    print(f'{mouse}: {trial_count} < required trial number. cue: {cue},  day: {di}, comp: {comp_n + 1}')
                    continue

                # create consecutive indices for plotting from full length meta
                inds_pseudo = np.zeros(len(meta))
                inds_pseudo[first_boo] = np.arange(np.sum(first_boo))

                # select indices and TCA data for a single day and cue
                inds = inds_pseudo[first_boo & day_boo & cue_boo]
                fit_comp_vec = comp_vec[first_boo & day_boo & cue_boo]
                if norm:
                    fit_comp_vec = fit_comp_vec / np.nanmax(fit_comp_vec)

                # plot all trials for each day
                ax[comp_n].plot(inds, fit_comp_vec, 'o', color=lookups.color_dict[cue], alpha=0.3)

                # rename for compactness in fitting function
                x1 = inds
                y1 = fit_comp_vec

                # initial parameter guess
                if fit_offset:
                    p0 = [np.nanmean(y1[:5]), .5, np.nanmean(y1[-5:])]
                else:
                    p0 = [np.nanmean(y1[:5]), .5]

                # fit trial types with exponential decay and plot
                try:
                    popt1, pcov1 = curve_fit(func, x1 - np.min(x1), y1, bounds=bounds, p0=p0)
                    perr = np.sqrt(np.diag(pcov1))
                    ax[comp_n].plot(x1, func(x1 - np.min(x1), *popt1),
                                    color=lookups.color_dict[cue + '2'], linewidth=3)
                    fit_err[first_boo & day_boo & cue_boo] = np.max(perr)
                    initial_state_pars[first_boo & day_boo & cue_boo] = popt1[0]
                    decay_pars[first_boo & day_boo & cue_boo] = popt1[1]
                    if fit_offset:
                        offset_pars[first_boo & day_boo & cue_boo] = popt1[2]

                    # add text for fits
                    txt_results.append(
                        f'{cue}: {[round(s, 2) for s in popt1]}\nerror: {round(np.max(perr), 2)}\n')
                except:
                    print(f'{mouse}: Fit failed. cue: {cue}, day: {di}, comp: {comp_n + 1}')

            # add text summary of fits per day
            if annotate:
                txt_label = ''
                for s in txt_results:
                    txt_label = txt_label + s
                ax[comp_n].text(inds[int(np.floor(len(inds) / 2))], y_txt, txt_label, ha='center', va='top')

            # set labels for subplots
        ax[comp_n].set_ylabel(f'component {comp_n + 1}\n\ntrial factor amplitude\n(weighted z-score)', size=14)
        if comp_n == 0:
            ax[comp_n].set_title(f'{mouse}: Daily adaptation', size=16)
            ax[comp_n].legend(bbox_to_anchor=(1.05, 1.05), loc='upper left')
        elif comp_n == rank - 1:
            ax[comp_n].set_xlabel('trial number', size=14)

        # create a dataframe with matching indices to meta
        data = {'a': initial_state_pars, 'b': decay_pars, 'error': fit_err}
        if fit_offset:
            data['c'] = offset_pars
        comp_fit_df = pd.DataFrame(data=data, index=meta.index)
        comp_fit_df['component'] = comp_n + 1

        # share some useful columns from meta
        for col in ['parsed_stage', 'parsed_10stage', 'parsed_11stage', 'initial_condition']:
            comp_fit_df[col] = meta[col]

        # collect fit results for each component
        df_list.append(comp_fit_df)

    # save
    annotxt = '_annot' if annotate else ''
    normtxt = '_norm' if norm else ''
    dp_save_folder = save_folder + f' day adapt rank {rank}'
    if not os.path.isdir(dp_save_folder):
        os.mkdir(dp_save_folder)
    plt.savefig(
        os.path.join(
            dp_save_folder,
            f'{offset_tag}{mouse}_r{rank}_TCA_daily_adaptation{annotxt}{normtxt}.png'),
        bbox_inches='tight')

    exp_fit_df = pd.concat(df_list, axis=0)

    return exp_fit_df


def temporal_projection(meta, model, sorts, ids):
    """
    Function for taking a smaller tensor (i.e. from 'learning') and using this to subset a larger dataset,
    including the same cells from the smaller dataset.
    Returns updated TCA data, variables, metadata, etc. but now these have been updated to reflect even more time.

    You want to keep the clustering from your original "smaller" model so sorts and model should be identical to your
    input.

    :param meta: pandas.DataFrame, metadata DataFrame where each index is a unique trial
    :param model: tensortools.ensemble, TCA results
    :param sorts: list of numpy.ndarray, sorting schemes for each rank of decomposition based on cell factors
    :param ids: list of cell ids
    :return: the same values, subset for cells that were included in ids, but now over group_by='all3' tensor
    """

    # check your mouse name and make sure there is only one in metadata
    mouse = meta.reset_index()['mouse'].unique()
    assert len(mouse) == 1
    mouse = mouse[0]

    # define largest known model
    word = 'signals' if mouse in 'OA27' else 'answered'
    load_kwargs = {'mouse': mouse,
                   'method': 'ncp_hals',
                   'cs': '',
                   'warp': False,
                   'word': word,
                   'trace_type': 'zscore_day',
                   'group_by': 'all3',
                   'nan_thresh': 0.95,
                   'score_threshold': 0.8}
    _, l_ids, l_tensor, l_meta, _, l_sorts = load.load_all_groupday(**load_kwargs, with_model=False)

    # subset full size tensor to only include cells that were kept in tca model
    survivor_boo = np.in1d(l_ids, ids)
    updated_tensor = l_tensor[survivor_boo, :, :]
    updated_ids = l_ids[survivor_boo]
    # updated_model = deepcopy(l_model)
    # for rank in updated_model.results.keys():
    #     updated_model.results[rank][0].factors[0] = updated_model.results[rank][0].factors[0][survivor_boo, :]
    updated_meta = l_meta
    # _, updated_sorts = utils.sortfactors(updated_model)

    # ensure your old model has the same cells
    # in most cases the bigger model should have all the same cells plus extra, so this won't actually chang anything
    survivor_boo_checkup = np.in1d(ids, l_ids)

    # warn user if the bigger dataset was missing cells in the smaller dataset
    if np.sum(survivor_boo_checkup) != len(survivor_boo_checkup):
        updated_ids_checkup = ids[survivor_boo_checkup]
        print(f'!!!!! {mouse} temporal projection signals/answered missing cells. {np.sum(survivor_boo_checkup) }/{len(survivor_boo_checkup)} will be kept.')
        old_model = deepcopy(model)
        for rank in old_model.results.keys():
            for it in range(len(old_model.results[rank])):
                # update the expected shape of the factor for tensortools.KTensor
                sz = [f.shape[0] for f in old_model.results[rank][it].factors]
                sz[0] = np.sum(survivor_boo_checkup)
                old_model.results[rank][it].factors.shape = tuple(sz)
                old_model.results[rank][it].factors[0] = old_model.results[rank][it].factors[0][survivor_boo_checkup, :]
        _, updated_sorts = utils.sortfactors(old_model)
    else:
        old_model = model
        updated_sorts = sorts
        updated_ids_checkup = ids

    # make sure your ids are now matched before return. If the ids are matched your indices are matched
    assert np.sum((updated_ids - updated_ids_checkup) != 0) == 0

    return updated_meta, old_model, updated_tensor, updated_ids, updated_sorts


def decay_func(x, a, b):
    """
    Helper function defining exponential decay function without offset term.

    :param x: vector of x values
    :param a: initial parameter, defines
    :param b: decay constant lambda, defines rate of decay
    :return: a * np.exp(-b * x), our decay function
    """
    return a * np.exp(-b * x)


def decay_func_offset(x, a, b, c):
    """
    Helper function defining exponential decay function with offset term.

    :param x: vector of x values
    :param a: initial parameter, defines
    :param b: decay constant lambda, defines rate of decay
    :param c: offset parameter, lower bound/y-asymptote for decay
    :return: a * np.exp(-b * x) + c, our decay function
    """
    return a * np.exp(-b * x) + c


def _first100_bool(meta):
    """
    Helper function to get a boolean vector of the first 100 trials for each day.
    If a day is shorter than 100 trials use the whole day.
    """

    days = meta.reset_index()['date'].unique()

    first100 = np.zeros((len(meta)))
    for di in days:
        dboo = meta.reset_index()['date'].isin([di]).values
        daylength = np.sum(dboo)
        if daylength > 100:
            first100[np.where(dboo)[0][:100]] = 1
        else:
            first100[dboo] = 1
    firstboo = first100 > 0

    return firstboo


def _first21_bool(meta):
    """
    Helper function to get a boolean vector of the first and last 21 trials for each day.
    If a day is shorter than 42 trials skip the whole day.
    """

    days = meta.reset_index()['date'].unique()
    first100 = _first100_bool(meta)

    first21 = np.zeros((len(meta)))
    last21 = np.zeros((len(meta)))
    for di in days:
        dboo = meta.reset_index()['date'].isin([di]).values & first100
        day_inds = np.where(dboo)[0]
        if len(day_inds) < 60:
            print(f'adaptation._first21_bool: {di} skipped: only {len(day_inds)} trials in day')
            continue
        first21[day_inds[:30]] = 1
        last21[day_inds[-30:]] = 1

    firstboo = first21 > 0
    lastboo = last21 > 0

    return firstboo, lastboo


def _get_gaussian_fitting_template(mouse, sigma=3.5, shift=0, sec=15.5, normalize=True):
    """
        Helper function for convolving Gaussian kernel with onset, sustained
    stimulus, stimulus offset, and ensure delivery period. These can then be
    used for simple linear regression or GLM.

    :param mouse: str, name of mouse, used for checking length of stimulus shown
    :param sigma: float, kwarg used for scipy.ndimage.gaussian_filter1d() to define sigma (width) of gaussian
    :param shift: float, temporal shift in frames to be applied to stimulus onset filter
    :param sec: float, how many frames are in a second
    :param normalize: boolean, ensure that amplitude of all gaussian kernels is the same
    :return: templates: numpy.ndarray, rows are timepoints, columns are each convolution for different epochs of trial
    """

    # dumb params
    blur = sec / 2
    lag = sec / 2
    elag = sec / 1.5

    # pick ranges of relevant time periods for convolution
    onset_r = [18 + shift]

    # stimulus length is 2 seconds
    if lookups.stim_length[mouse] == 2:
        sus_r = np.arange(np.floor(sec + lag), np.round(sec * 3 + blur), 1)
        off_r = np.arange(np.floor(sec * 3 + sec / 4 + lag), np.round(sec * 3 + sec / 3 + blur), 1)
        ensure_r = np.arange(np.floor(sec * 3 + sec / 3 + elag), np.round(sec * 5 + blur), 1)

    # stimulus length is 3 seconds
    else:
        sus_r = np.arange(np.floor(sec + lag), np.round(sec * 4 + blur), 1)
        off_r = np.arange(np.floor(sec * 4 + sec / 4 + lag), np.round(sec * 4 + sec / 3 + blur), 1)
        ensure_r = np.arange(np.floor(sec * 4 + sec / 3 + elag), np.round(sec * 6 + blur), 1)
    ranges = [onset_r, sus_r, off_r, ensure_r]

    # convolve
    gker = []  # preallocate Gaussian kernel convolution
    for i in ranges:
        i = [int(s) for s in i]
        starter = np.zeros(108)
        starter[i] = 1
        gker.append(sp.ndimage.gaussian_filter1d(starter, sigma, mode='constant', cval=0))

    # normalize filters
    if normalize:
        gker = [(s - np.min(s)) / np.max(s) for s in gker]

    templates = np.vstack(gker).T

    return templates


def plot_test_of_template(mouse):
    """
    Helper function to plot your templates used for fitting NNLS.

    :param mouse: str, name of mouse. Used to define templates.
    :return:
    """

    # pick colormaps
    colors = sns.color_palette('RdPu', len(np.arange(0, 2, .1)))
    colors2 = sns.color_palette('GnBu', len(np.arange(0, 2, .1)))

    # for plot readability, change the color scheme of lines every 5th line
    mod_by = 5

    # create timepoints assuming 15.5 Hz for 7 seconds
    times = np.arange(-1, 7, 1)
    timepts = np.arange(0, 108, 15.5)

    A = _get_gaussian_fitting_template(mouse, sigma=4, shift=2)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for c, i in enumerate(np.arange(0, 2, .1)):
        if c % mod_by == 0:
            ax[0].plot(A[:, 0] + A[:, 1] * i, label=round(1 / (1 + i), 2), color=colors2[c])
        else:
            ax[0].plot(A[:, 0] + A[:, 1] * i, label=round(1 / (1 + i), 2), color=colors[c])
    ax[0].legend(title='transientness', bbox_to_anchor=(1.05, 1.03), loc='upper left')
    ax[0].set_xticks(timepts)
    ax[0].set_xticklabels(labels=times, size=14)
    ax[0].set_xlabel('time from stimulus onset')
    ax[0].set_ylabel('weight (AU)')
    ax[0].set_title(f'{mouse} NNLS templates:\nChanging sustained filter')

    for c, i in enumerate(np.arange(0, 2, .1)):
        if c % mod_by == 0:
            ax[1].plot(A[:, 0] * i + A[:, 1], label=round(i / (1 + i), 2), color=colors2[c])
        else:
            ax[1].plot(A[:, 0] * i + A[:, 1], label=round(i / (1 + i), 2), color=colors[c])
    ax[1].legend(title='transientness', bbox_to_anchor=(1.05, 1.03), loc='upper left')
    ax[1].set_xticks(timepts)
    ax[1].set_xticklabels(labels=times, size=14)
    ax[1].set_xlabel('time from stimulus onset')
    ax[1].set_ylabel('weight (AU)')
    ax[1].set_title(f'{mouse} NNLS templates:\nChanging onset filter')
    plt.savefig(
        paths.default_dir(foldername='NNLS templates', filename=f'{mouse}_test_sustainedess_template.png'),
        bbox_inches='tight')

def stable_locomotor_behavior(meta, select_on_early_trials=False):
    """
    Helper function to decide if the animal tends to run or remain stationary during stimulus presentations.
    Returns a boolean for stable running or stationary behavior based on mouses behavioral trends.
    :param meta: pandas.DataFrame, trial metadata
    :return: motor_bool: boolean, trials with matched locomotor behavior
    """

    # check your mouse name and make sure there is only one in metadata
    mouse = meta.reset_index()['mouse'].unique()
    assert len(mouse) == 1
    mouse = mouse[0]

    # get your running speed during the stimulus presentation
    speed_vec = meta['speed'].values

    # if you want to assign behavior based on earliest trials
    # i.e., if you want to make a comparison to something early in a day, but match for behavior
    if select_on_early_trials:
        first21, last21 = _first21_bool(meta)
        speed_vec = speed_vec[first21]

    low_speed = np.sum(speed_vec < 5)
    high_speed = np.sum(speed_vec > 10)

    if low_speed >= high_speed:
        motor_bool = meta['speed'].values < 5
        print(f'{mouse}: motor behavior set to "LOW speed" <5 cm/s, {low_speed} > {high_speed} trials.')
        print(f'    "LOW speed", n = {np.sum(motor_bool)} trials total.')
    else:
        motor_bool = meta['speed'].values > 10
        print(f'{mouse}: motor behavior set to "HIGH speed" >10 cm/s, {high_speed} > {low_speed} trials.')
        print(f'    "HIGH speed", n = {np.sum(motor_bool)} trials total.')

    return motor_bool


def get_sensory_history_mod_from_betas(beta_df):

    # TODO add in component tuning from cascade.tuning to select columns per component

    # useful lookups
    cols = ['prev_same_init_plus', 'prev_diff_init_plus'
            'prev_same_init_minus', 'prev_diff_init_minus'
            'prev_same_init_neutral', 'prev_diff_init_neutral']
    cols_by_cue = {'plus': ['prev_same_init_plus', 'prev_diff_init_plus'],
                   'minus': ['prev_same_init_minus', 'prev_diff_init_minus'],
                   'neutral': ['prev_same_init_neutral', 'prev_diff_init_neutral']}

    # calculate a sensory history modulation index per cue
    all_cue_indices = np.zeros((beta_df.shape[0], 3))
    for c, cue in enumerate(['plus', 'minus', 'neutral']):
        cue_df = beta_df.loc[:, cols_by_cue[cue]]
        following_same = cue_df.iloc[:, 0]
        following_diff = cue_df.iloc[:, 1]
        # index_vec = following_same / (following_same + following_diff)
        index_vec = (following_diff - following_same) / (following_same + following_diff)
        all_cue_indices[:, c] = index_vec

    # create single vector that only accounts for preferred cue
    highest_beta = np.zeros((beta_df.shape[0], 3))
    for c, cue in enumerate(['plus', 'minus', 'neutral']):
        cue_df = beta_df.loc[:, cols_by_cue[cue]]
        highest_beta[:, c] = cue_df.iloc[:, 1]
    best_col = np.argmax(highest_beta, axis=1)

    # only calculate sensory history if one of the cues is the highest beta
    sensory_betas = np.argmax(beta_df.values, axis=1) < 6

    # account for preferred cue and sensory drive
    preferred_sensory_history_mod = np.array([all_cue_indices[c, s] for c, s in enumerate(best_col)])
    preferred_sensory_history_mod[~sensory_betas] = np.nan

    return preferred_sensory_history_mod





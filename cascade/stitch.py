"""Functions for stitching tca factors together across days."""
import random
import numpy as np
import os
import flow
import pandas as pd
from copy import deepcopy
from . import tca
from . import paths


def tri_factor_similarity(
        mouse,
        rank_num=10,
        match_by='tri_sim',
        trace_type='zscore_day',
        method='ncp_bcd',
        cs='',
        warp=False,
        word=None):
    """
    Use TCA neuron factor weights and xday alignment
    to match TCA components across days.

    Uses the best correlation coefficient of weight matrices for matching.

    Parameters
    ----------
    mouse, str
        mouse name
    rank_num, int
        rank of TCA (number of components) you will align
    match_by , str
        type of similarity matrix combination used for matching
        options: tri_sim, tri_sim_prob
    trace_type, str
        type of trace used in TCA, used for loading
    method, str
        fit method used in TCA, used for loading
        options: 'cp_als', 'ncp_bcd', 'ncp_hals', 'mcp_als'
    cs, str
        cses used in TCA, used for loading
        default: '', includes all trials
    warp, bool
        warped offsets used in TCA? used for loading
    word, str
        hash word for TCA parameters, used for loading

    Returns
    -------
    transition_weights, numpy ndarray
        components x days
    temporal_factors, list of numpy ndarray
        [days x time , ...]
    sim_mat_by_day, list of numpy ndarray
        [components day 2 x components day 1 , ...]
    """

    # pars for loading tca data
    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}

    # create datesorter
    days = flow.metadata.DateSorter.frommeta(mice=[mouse], tags=None)

    conds_by_day = []
    oris_by_day = []
    trialerr_by_day = []
    neuron_ids_by_day = []
    neuron_clusters_by_day = []
    factors_by_day = []
    # loop through days in REVERSE order
    for day1 in days[::-1]:

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

        # load your metadata, skip post reversal days
        meta = pd.read_pickle(meta_path)
        condition = meta['condition']
        orientation = meta['orientation']
        trialerror = meta['trialerror']
        if 'reversal1' in meta['learning_state'].unique() \
        or 'reversal2' in meta['learning_state'].unique():
            continue

        # skip days that do not have minus, AND neutral
        if 'minus' not in meta['condition'].unique() \
        or 'neutral' not in meta['condition'].unique() \
        or 'plus' not in meta['condition'].unique():
            continue

        # load your data
        ensemble = np.load(tensor_path)
        ensemble = ensemble.item()
        ids = np.load(input_ids_path)

        # sort neuron factors by component they belong to most
        sort_ensemble, my_sorts = tca._sortfactors(ensemble[method])

        cell_ids = {}  # keys are rank
        cell_clusters = {}
        itr_num = 0  # use only best iteration of TCA, index 0
        for k in sort_ensemble.results.keys():
            # factors are already sorted, so these will define
            # clusters, no need to sort again
            factors = sort_ensemble.results[k][itr_num].factors[0]
            max_fac = np.argmax(factors, axis=1)
            cell_clusters[k] = max_fac
            cell_ids[k] = ids[my_sorts[k-1]]

        neuron_ids_by_day.append(cell_ids[rank_num])
        neuron_clusters_by_day.append(cell_clusters[rank_num])
        factors_by_day.append(sort_ensemble.results[rank_num][0])
        conds_by_day.append(condition)
        oris_by_day.append(orientation)
        trialerr_by_day.append(trialerror)

    # ------------------------------------------------------------

    # create similarity matrices for comparison of all neuron
    # factors weights between a pair of days
    sim_mat_neuro_by_day = []
    sim_mat_tempo_by_day = []
    sim_mat_trial_tuning_by_day = []
    prob_mat_by_day = []
    boot_num = 300  # number of iterations for bootstrapping
    for i in range(len(factors_by_day)-1):

        # always compare in the time-forward direction
        # (remember that factors_by_day is in reverse order)
        ids1 = neuron_ids_by_day[i+1]
        ids2 = neuron_ids_by_day[i]
        ids1_bool = np.isin(ids1, ids2)
        ids2_bool = np.isin(ids2, ids1)

        # get sort order to match ids between days
        ids1_sort = np.argsort(ids1[ids1_bool])
        ids2_sort = np.argsort(ids2[ids2_bool])

        # get neuron factor weight matrices for ids matched
        # between days
        ids1_weights = factors_by_day[i+1].factors[0][ids1_bool, :]
        ids2_weights = factors_by_day[i].factors[0][ids2_bool, :]

        # get temporal factor weight matrices
        tempo1_weights = factors_by_day[i+1].factors[1][:, :]
        tempo2_weights = factors_by_day[i].factors[1][:, :]

        # get trial factor weight matrices
        trial1_weights = factors_by_day[i+1].factors[2][:, :]
        trial2_weights = factors_by_day[i].factors[2][:, :]
        # get trial factor orientations
        trial1_oris = oris_by_day[i+1]
        trial2_oris = oris_by_day[i]

        # get tuning of trials to 0, 135, 270
        # (mean response per-factor per-ori) / sum(mean responses)
        tuning1_weights = np.zeros((3, rank_num))
        tuning2_weights = np.zeros((3, rank_num))
        oris_to_check = [0, 135, 270]
        for c, ori in enumerate(oris_to_check):
            tuning1_weights[c, :] = np.nanmean(trial1_weights[trial1_oris == ori, :], axis=0)
            tuning2_weights[c, :] = np.nanmean(trial2_weights[trial2_oris == ori, :], axis=0)
        # normalize using summed mean response to all three
        tuning1_total = np.nansum(tuning1_weights, axis=0)
        tuning2_total = np.nansum(tuning2_weights, axis=0)
        for c in range(len(oris_to_check)):
            tuning1_weights[c, :] = np.divide(tuning1_weights[c, :], tuning1_total)
            tuning2_weights[c, :] = np.divide(tuning2_weights[c, :], tuning2_total)

        # get the correlation matrix for different days
        # do the full comparison with both days then select
        # only the off-diagonal quadrant of the correlation matrix
        # so you are only comparing day1-day2 factors rather than
        # day2-day1
        ids_corr = np.corrcoef(ids1_weights[ids1_sort, :].T,
                               y=ids2_weights[ids2_sort, :].T)
        sim_mat_neuro_by_day.append(deepcopy(ids_corr[-10:, 0:10]))

        tempo_corr = np.corrcoef(tempo1_weights[:, :].T,
                                 y=tempo2_weights[:, :].T)
        sim_mat_tempo_by_day.append(deepcopy(tempo_corr[-10:, 0:10]))

        tuning_corr = np.corrcoef(tuning1_weights[:, :].T,
                                 y=tuning2_weights[:, :].T)
        sim_mat_trial_tuning_by_day.append(deepcopy(tuning_corr[-10:, 0:10]))

        # find transiiton probabilities by asking which factors
        # are matched (without replacement) in random shuffled order.
        # this will allow for balancing of components with
        # similar weights
        sz = ids_corr[-10:, 0:10]
        my_sort = np.zeros(np.shape(sz)[0], dtype=np.int64)
        bins = np.empty(np.shape(sz)[0])
        bins = np.empty((boot_num, np.shape(sz)[0]))
        bins[:] = np.nan
        # iterate multiple times, saving results into "bins"
        for it in range(boot_num):
            corner = deepcopy(ids_corr[-10:, 0:10])
            for k in random.sample(list(range(np.shape(corner)[0])),
                                   np.shape(corner)[0]):
                pos = np.argmax(corner[:, k])
                bins[it, k] = pos
                corner[pos, :] = 0
        # take mean of each occurance of pos by the number of
        # iterations to get the probabilty of matching to
        # that cluster
        prob_mat = np.zeros((np.shape(corner)))
        for k in range(np.shape(corner)[0]):
            prob_mat[k, :] = np.mean(bins == k, axis=0)
        prob_mat_by_day.append(prob_mat)

    # ------------------------------------------------------------

    # align: get factor index, stepping through similarity
    # matrices for pairs of days
    if match_by == 'tri_sim':
        # elementwise multiplication of the prob & sim mat
        # match_mat = [a*b*c for a, b, c in zip(sim_mat_neuro_by_day, sim_mat_neuro_by_day, sim_mat_trial_tuning_by_day)]
        match_mat = [(a+b*0.8+c*0.2)/2 for a, b, c in zip(sim_mat_neuro_by_day, sim_mat_tempo_by_day, sim_mat_trial_tuning_by_day)]
    elif match_by == 'tri_sim_prob':
        # elementwise multiplication of the prob & sim mat
        match_mat = [a*b*c*d for a, b, c, d in zip(sim_mat_neuro_by_day, sim_mat_tempo_by_day, sim_mat_trial_tuning_by_day, prob_mat_by_day)]
    else:
        print('Unregognized matching method in match_by.')
        return


    temporal_factors_list = []
    transition_weights = np.zeros((np.shape(match_mat[0])[0], len(match_mat)))
    # factors_by_day_ = factors_by_day[::-1]
    for comp_num in range(np.shape(match_mat[0])[0]):
        temporal_factors = []
        for i in range(len(match_mat)):
            # get similarity matrices. (i.e., [day1-->day2])
            # reverse order matching, forward in time.
            sim_mat = match_mat[i]
            if i == 0:
                starting_comp = np.argmax(sim_mat[comp_num, :])
                transition_weights[comp_num, i] = sim_mat[comp_num, starting_comp]
                fac_1 = factors_by_day[i+1].factors[1][:, starting_comp]
                fac_2 = factors_by_day[i].factors[1][:, comp_num]
                temporal_factors.append(fac_2)
                temporal_factors.append(fac_1)
            else:
                new_starting_comp = np.argmax(sim_mat[starting_comp, :])
                transition_weights[comp_num, i] = sim_mat[starting_comp, new_starting_comp]
                fac_n = factors_by_day[i+1].factors[1][:, new_starting_comp]
                temporal_factors.append(fac_n)
                starting_comp = new_starting_comp

        temporal_factors_list.append(temporal_factors[::-1])

    # reverse order of output variables so they are correct
    # [day1, day2, day3, etc]
    transition_weights = transition_weights[:, ::-1]
    temporal_factors_list = temporal_factors_list[::-1]
    sim_mat_neuro_by_day = sim_mat_neuro_by_day[::-1]
    sim_mat_tempo_by_day = sim_mat_tempo_by_day[::-1]
    sim_mat_trial_tuning_by_day = sim_mat_trial_tuning_by_day[::-1]
    match_mat = match_mat[::-1]

    return {'trans': transition_weights,
            'tempo_fac': temporal_factors_list,
            'neuro_sim': sim_mat_neuro_by_day,
            'tempo_sim': sim_mat_tempo_by_day,
            'ttuning_sim': sim_mat_trial_tuning_by_day,
            'tri_sim': match_mat
            }


def neuron_similarity(
        mouse,
        rank_num=10,
        match_by='similarity',
        trace_type='zscore_day',
        method='ncp_bcd',
        cs='',
        warp=False,
        word=None):
    """
    Use TCA neuron factor weights and xday alignment
    to match TCA components across days.

    Uses the best correlation coefficient of weight matrices for matching.

    Parameters
    ----------
    mouse, str
        mouse name
    rank_num, int
        rank of TCA (number of components) you will align
    match_by , str
        type of similarity matrix used for matching
        options: similarity, probability, combo
    trace_type, str
        type of trace used in TCA, used for loading
    method, str
        fit method used in TCA, used for loading
        options: 'cp_als', 'ncp_bcd', 'ncp_hals', 'mcp_als'
    cs, str
        cses used in TCA, used for loading
        default: '', includes all trials
    warp, bool
        warped offsets used in TCA? used for loading
    word, str
        hash word for TCA parameters, used for loading

    Returns
    -------
    transition_weights, numpy ndarray
        components x days
    temporal_factors, list of numpy ndarray
        [days x time , ...]
    sim_mat_by_day, list of numpy ndarray
        [components day 2 x components day 1 , ...]
    """

    # pars for loading tca data
    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}

    # create datesorter
    days = flow.metadata.DateSorter.frommeta(mice=[mouse], tags=None)

    conds_by_day = []
    trialerr_by_day = []
    neuron_ids_by_day = []
    neuron_clusters_by_day = []
    factors_by_day = []
    for day1 in days[::-1]:

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

        # load your metadata, skip post reversal days
        meta = pd.read_pickle(meta_path)
        condition = meta['condition']
        trialerror = meta['trialerror']
        if 'reversal1' in meta['learning_state'].unique() \
        or 'reversal2' in meta['learning_state'].unique():
            continue

        # skip days that do not have minus, AND neutral
        if 'minus' not in meta['condition'].unique() \
        or 'neutral' not in meta['condition'].unique() \
        or 'plus' not in meta['condition'].unique():
            continue

        # load your data
        ensemble = np.load(tensor_path)
        ensemble = ensemble.item()
        ids = np.load(input_ids_path)

        # sort neuron factors by component they belong to most
        sort_ensemble, my_sorts = tca._sortfactors(ensemble[method])

        cell_ids = {}  # keys are rank
        cell_clusters = {}
        itr_num = 0  # use only best iteration of TCA, index 0
        for k in sort_ensemble.results.keys():
            # factors are already sorted, so these will define
            # clusters, no need to sort again
            factors = sort_ensemble.results[k][itr_num].factors[0]
            max_fac = np.argmax(factors, axis=1)
            cell_clusters[k] = max_fac
            cell_ids[k] = ids[my_sorts[k-1]]

        neuron_ids_by_day.append(cell_ids[rank_num])
        neuron_clusters_by_day.append(cell_clusters[rank_num])
        factors_by_day.append(sort_ensemble.results[rank_num][0])
        conds_by_day.append(condition)
        trialerr_by_day.append(trialerror)

    # ------------------------------------------------------------

    # create similarity matrices for comparison of all neuron
    # factors weights between a pair of days
    sim_mat_by_day = []
    sim_mat_tempo_by_day = []
    prob_mat_by_day = []
    boot_num = 300  # number of iterations for bootstrapping
    for i in range(len(factors_by_day)-1):

        # always compare in the time-forward direction
        # (remember that factors_by_day is in reverse order)
        ids1 = neuron_ids_by_day[i+1]
        ids2 = neuron_ids_by_day[i]
        ids1_bool = np.isin(ids1, ids2)
        ids2_bool = np.isin(ids2, ids1)

        # get sort order to match ids between days
        ids1_sort = np.argsort(ids1[ids1_bool])
        ids2_sort = np.argsort(ids2[ids2_bool])

        # get neuron factor weight matrices for ids matched
        # between days
        ids1_weights = factors_by_day[i+1].factors[0][ids1_bool, :]
        ids2_weights = factors_by_day[i].factors[0][ids2_bool, :]

        # get temporal factor weight matrices
        tempo1_weights = factors_by_day[i+1].factors[1][:, :]
        tempo2_weights = factors_by_day[i].factors[1][:, :]

        # get the correlation matrix for different days
        # do the full comparison with both days then select
        # only the off-diagonal quadrant of the correlation matrix
        # so you are only comparing day1-day2 factors rather than
        # day2-day1
        ids_corr = np.corrcoef(ids1_weights[ids1_sort, :].T,
                               y=ids2_weights[ids2_sort, :].T)
        sim_mat_by_day.append(deepcopy(ids_corr[-10:, 0:10]))

        tempo_corr = np.corrcoef(tempo1_weights[:, :].T,
                                 y=tempo2_weights[:, :].T)
        sim_mat_tempo_by_day.append(deepcopy(tempo_corr[-10:, 0:10]))

        # find transiiton probabilities by asking which factors
        # are matched (without replacement) in random shuffled order.
        # this will allow for balancing of components with
        # similar weights
        sz = ids_corr[-10:, 0:10]
        my_sort = np.zeros(np.shape(sz)[0], dtype=np.int64)
        bins = np.empty(np.shape(sz)[0])
        bins = np.empty((boot_num, np.shape(sz)[0]))
        bins[:] = np.nan
        # iterate multiple times, saving results into "bins"
        for it in range(boot_num):
            corner = deepcopy(ids_corr[-10:, 0:10])
            for k in random.sample(list(range(np.shape(corner)[0])),
                                   np.shape(corner)[0]):
                pos = np.argmax(corner[:, k])
                bins[it, k] = pos
                corner[pos, :] = 0
        # take mean of each occurance of pos by the number of
        # iterations to get the probabilty of matching to
        # that cluster
        prob_mat = np.zeros((np.shape(corner)))
        for k in range(np.shape(corner)[0]):
            prob_mat[k, :] = np.mean(bins == k, axis=0)
        prob_mat_by_day.append(prob_mat)

    # ------------------------------------------------------------

    # align: get factor index, stepping through similarity
    # matrices for pairs of days

    # choose list of simialrity matrices to use for matching
    # components across days
    if match_by == 'similarity':
        match_mat = sim_mat_by_day
    elif match_by == 'probability':
        match_mat = prob_mat_by_day
    elif match_by == 'combination':
        # elementwise multiplication of the prob & sim mat
        match_mat = [np.multiply(a, b) for a, b in zip(sim_mat_by_day, prob_mat_by_day)]
    elif match_by == 'sim_combination':
        # elementwise multiplication of the prob & sim mat
        match_mat = [np.multiply(a, b) for a, b in zip(sim_mat_by_day, sim_mat_tempo_by_day)]
    else:
        print('Unregognized matching method in match_by.')
        return

    temporal_factors_list = []
    transition_weights = np.zeros((np.shape(match_mat[0])[0], len(match_mat)))
    # factors_by_day_ = factors_by_day[::-1]
    for comp_num in range(np.shape(match_mat[0])[0]):
        temporal_factors = []
        for i in range(len(match_mat)):
            # get similarity matrices. (i.e., [day1-->day2])
            # reverse order matching, forward in time.
            sim_mat = match_mat[i]
            if i == 0:
                starting_comp = np.argmax(sim_mat[comp_num, :])
                transition_weights[comp_num, i] = sim_mat[comp_num, starting_comp]
                fac_1 = factors_by_day[i+1].factors[1][:, starting_comp]
                fac_2 = factors_by_day[i].factors[1][:, comp_num]
                temporal_factors.append(fac_2)
                temporal_factors.append(fac_1)
            else:
                new_starting_comp = np.argmax(sim_mat[starting_comp, :])
                transition_weights[comp_num, i] = sim_mat[starting_comp, new_starting_comp]
                fac_n = factors_by_day[i+1].factors[1][:, new_starting_comp]
                temporal_factors.append(fac_n)
                starting_comp = new_starting_comp

        temporal_factors_list.append(temporal_factors[::-1])

    # reverse order of output variables so they are correct
    # [day1, day2, day3, etc]
    transition_weights = transition_weights[:, ::-1]
    temporal_factors_list = temporal_factors_list[::-1]
    sim_mat_by_day = sim_mat_by_day[::-1]
    sim_mat_tempo_by_day = sim_mat_tempo_by_day[::-1]

    return {'trans': transition_weights,
            'tempo_fac': temporal_factors_list,
            'neuro_sim': sim_mat_by_day,
            'tempo_sim': sim_mat_tempo_by_day}

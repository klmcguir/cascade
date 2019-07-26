"""Functions for stitching tca factors together across days."""
import random
import numpy as np
import os
import flow
import pandas as pd
from copy import deepcopy
from munkres import Munkres
from . import tca
from . import paths


def match_singleday_to_groupday(
        mouse,
        rank_num=10,
        trace_type='zscore_day',
        method='ncp_bcd',
        cs='',
        warp=False,
        word='convinced',
        group_word='supply',
        group_by='high_dprime_learning',
        nan_thresh=0.85,
        exclude_reversal=True,
        sim_thresh=0.2,
        ratio_thresh=0.6):

    # ------------------- GROUPDAY LOADING -------------------

    # groupday loader
    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}
    group_pars = {'group_by': group_by}

    # if cells were removed with too many nan trials
    if nan_thresh:
        nt_tag = '_nantrial' + str(nan_thresh)
    else:
        nt_tag = ''

    # load dir
    load_dir = paths.tca_path(
        mouse, 'group', pars=pars, word=group_word, group_pars=group_pars)
    template_tensor_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_group_decomp_' + str(trace_type) + '.npy')
    template_meta_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_df_group_meta.pkl')
    template_ids_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_group_ids_' + str(trace_type) + '.npy')

    # load your data
    template_ensemble = np.load(template_tensor_path)
    template_ensemble = template_ensemble.item()
    ids = np.load(template_ids_path)
    meta = pd.read_pickle(template_meta_path)

    # get trial metadata
    template_orientation = meta['orientation']
    template_trial_num = np.arange(0, len(template_orientation))
    template_condition = meta['condition']
    template_trialerror = meta['trialerror']
    template_hunger = deepcopy(meta['hunger'])
    template_speed = meta['speed']
    template_dates = meta.reset_index()['date']
    template_learning_state = meta['learning_state']

    # merge hunger and tag info for plotting hunger
    template_tags = meta['tag']
    template_hunger[template_tags == 'disengaged'] = 'disengaged'

    # calculate change indices for days and reversal/learning
    udays = {d: c for c, d in enumerate(np.unique(template_dates))}
    ndays = np.diff([udays[i] for i in template_dates])
    day_x = np.where(ndays)[0] + 0.5
    ustate = {d: c for c, d in enumerate(np.unique(template_learning_state))}
    nstate = np.diff([ustate[i] for i in template_learning_state])
    lstate_x = np.where(nstate)[0] + 0.5

    # sort neuron factors by component they belong to most
    template_sort_ensemble, template_my_sorts = tca._sortfactors(
        template_ensemble[method])

    # put ids in correct order (k is 1 indexed)
    template_cell_ids = {}  # keys are rank
    for k in template_sort_ensemble.results.keys():
        template_cell_ids[k] = ids[template_my_sorts[k-1]]

    # get template factors of interest
    template_factors = template_sort_ensemble.results[rank_num][0]
    template_cell_ids = template_cell_ids[rank_num]

    # ------------------- SINGLEDAY LOADING -------------------

    # pars for loading tca data
    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}

    # create datesorter
    days = flow.DateSorter.frommeta(
        mice=[mouse], tags=None, exclude_tags=['bad'])

    conds_by_day = []
    oris_by_day = []
    trialerr_by_day = []
    neuron_ids_by_day = []
    neuron_clusters_by_day = []
    factors_by_day = []
    ndate = []
    # noise_corr_by_day = []

    # loop through days in FORWARD order
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

        # load your metadata, skip post reversal days
        meta = pd.read_pickle(meta_path)
        condition = meta['condition']
        orientation = meta['orientation']
        trialerror = meta['trialerror']
        if exclude_reversal:
            if 'reversal1' in meta['learning_state'].unique() \
            or 'reversal2' in meta['learning_state'].unique():
                continue

        # skip days that do not have minus, AND neutral
        if 'minus' not in meta['condition'].unique() \
        or 'neutral' not in meta['condition'].unique() \
        or 'plus' not in meta['condition'].unique():
            continue

        # print
        ndate.append(day1.date)

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

        # get noise corr for each
        # cs_noise = []
        # for cs in ['0', '135', '270']:
        #     cs_noise.append(pool.calc.correlations.noise(day1, cs))

        # noise_corr_by_day.append(cs_noise)
        neuron_ids_by_day.append(cell_ids[rank_num])
        neuron_clusters_by_day.append(cell_clusters[rank_num])
        factors_by_day.append(sort_ensemble.results[rank_num][0])
        conds_by_day.append(condition)
        oris_by_day.append(orientation)
        trialerr_by_day.append(trialerror)

    # ------------------- CORRELATE -------------------

    # create similarity matrices for comparison of all neuron
    # factors weights between a pair of days
    noise_mat_by_day = []
    on_sim_mat_by_day = []
    off_sim_mat_by_day = []
    tuning_sim_mat_by_day = []
    noise_id = []
    cells_compared = []
    for i in range(len(factors_by_day)):

        # always compare in the time-forward direction
        ids1 = template_cell_ids
        ids2 = neuron_ids_by_day[i]
        ids1_bool = np.isin(ids1, ids2)
        ids2_bool = np.isin(ids2, ids1)

        # get sort order to match ids between days
        ids1_sort = np.argsort(ids1[ids1_bool])
        ids2_sort = np.argsort(ids2[ids2_bool])

        # get neuron factor weight matrices for ids matched between days
        ids1_weights = template_factors.factors[0][ids1_bool, :]
        ids2_weights = factors_by_day[i].factors[0][ids2_bool, :]
        cells_compared.append(np.shape(ids1_weights)[0])

        # get noise corr only for cells included in comparison
        # cs_noise = noise_corr_by_day[i]
        # id_cs_noise = []
        # cs_noise_bool = np.isin(np.arange(len(cs_noise[0]))+1, ids2)
        # for cs in range(len(['0', '135', '270'])):
        #     cs_bool = cs_noise[cs][cs_noise_bool, :][:, cs_noise_bool]
        #     id_cs_noise.append(cs_bool)

        on_corr = np.corrcoef(
            ids1_weights[ids1_sort, :].T, y=ids2_weights[ids2_sort, :].T)
        on_corr[np.isnan(on_corr)] = 0
        on_sim_mat_by_day.append(deepcopy(on_corr[-rank_num:, 0:rank_num]))
        # noise_mat_by_day.append(id_cs_noise)

    # ------------------- STITCH TOGETHER CORRELATION MAPS -------------------

    # find the best correlated factor and second-best correlated factor in the
    # groupday template for each singleday factor
    best_arg = np.zeros((len(on_sim_mat_by_day), rank_num))
    best_val = np.zeros((len(on_sim_mat_by_day), rank_num))
    second_best_arg = np.zeros((len(on_sim_mat_by_day), rank_num))
    second_best_val = np.zeros((len(on_sim_mat_by_day), rank_num))
    test_temp = []
    for c, sim_mat in enumerate(on_sim_mat_by_day):
        y = np.argmax(sim_mat, axis=0)
        y2 = np.argsort(sim_mat, axis=0)[-2]  # second best value
        yv = np.max(sim_mat, axis=0)
        yv2 = np.sort(sim_mat, axis=0)[-2]
        best_arg[c, :] = y
        best_val[c, :] = yv
        second_best_arg[c, :] = y2
        second_best_val[c, :] = yv2

    # ------------------- FILTER CORRELATION MAPS -------------------

    # create boolean matrices to remove bad matches from your best_arg factor
    # matches
    numer = second_best_val
    numer[numer < 0] = 0
    denom = best_val
    denom[denom < 0] = 0
    ratio = numer/denom  # 1 is bad, means the best corr and 2nd best are same
    bad_sim_bool = best_val < sim_thresh
    bad_ratio_bool = ratio > ratio_thresh

    # set matches that are poorly correlated or too similar to the next-best
    # match to -1
    factor_map = best_arg
    factor_map[bad_sim_bool] = -1

    similarity_maps = {
        'best_val': best_val, 'best_arg': best_arg,
        'second_best_val': second_best_val, 'second_best_arg': second_best_arg}

    return factor_map, ndate, similarity_maps


def factor_squid(
        mouse,
        rank_num=10,
        match_by='polr',
        trace_type='zscore_day',
        method='ncp_bcd',
        cs='',
        warp=False,
        word=None):
    """
    Use TCA neuron factor weights and xday alignment
    to match TCA components across days. Uses the best correlation
    coefficient of neuron factor weight matrices for matching
    using a custom matching algorithm ("Squid matching").

    Parameters
    ----------
    mouse, str
        mouse name
    rank_num, int
        rank of TCA (number of components) you will align
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
    days = flow.DateSorter.frommeta(
        mice=[mouse], tags=None, exclude_tags=['bad'])

    conds_by_day = []
    oris_by_day = []
    trialerr_by_day = []
    neuron_ids_by_day = []
    neuron_clusters_by_day = []
    factors_by_day = []
    # loop through days in FORWARD order
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

    # get all pairwise correlation coeffs for matched neuron factors
    # this should be done on a time forward vector and can
    # include reversal
    sim_mats_by_day = []
    for k in range(len(factors_by_day)):

        # sim mats, day k comps are columns, day i comps are rows
        sim_mats = np.zeros((rank_num, rank_num, len(factors_by_day)))

        for i in range(len(factors_by_day)):

            # always compare in the time-forward direction
            ids1 = neuron_ids_by_day[k]
            ids2 = neuron_ids_by_day[i]
            ids1_bool = np.isin(ids1, ids2)
            ids2_bool = np.isin(ids2, ids1)

            ids1_sort = np.argsort(ids1[ids1_bool])
            ids2_sort = np.argsort(ids2[ids2_bool])

            ids1_weights = factors_by_day[k].factors[0][ids1_bool,:]
            ids2_weights = factors_by_day[i].factors[0][ids2_bool,:]

            # compare correlations of vectors that have been sorted
            # to match neuron orders
            ids_corr = np.corrcoef(ids1_weights[ids1_sort, :].T,
                                   y=ids2_weights[ids2_sort, :].T)
            sim_mats[:, :, i] = deepcopy(ids_corr[-rank_num:, 0:rank_num])

        sim_mats_by_day.append(sim_mats)

    # ------------------------------------------------------------

    # "Squid matching"

    # get a transition matrix for consecutive days in the forward
    # direction; i.e. for rank 10 tca: 10x10 matrix where each
    # A[:,i] are the counts that a factor was matched to output is a
    # set of 10x10 matrices for each pair of days (so length is 1
    # less than day number)

    trans_mats = np.zeros((rank_num, rank_num,
                          len(factors_by_day)-1))
    # trans_mats2 = np.zeros((rank_num, rank_num,
    #                        len(factors_by_day)-1))

    for i in range(len(sim_mats_by_day)-1):

        # FORWARD day[i]-->day[i+1]

        # best match of day i components to day n
        step_out_lookup = np.argmax(sim_mats_by_day[i], axis=0)
        # best match of day n components to day i
        step_in_lookup = np.argmax(sim_mats_by_day[i+1], axis=1)
        # get transition matrix A[:,k], where k is comp number on
        # day i and A[j,k] is probability of compenent k being
        # matched to comp j on SUBSEQUENT day
        counter_mat = np.zeros((rank_num, rank_num))
        for k in range(rank_num):
            vec_out = step_out_lookup[k, :]
            for j in range(len(vec_out)):
                best_step_in = step_in_lookup[vec_out[j], j]
                counter_mat[best_step_in, k] += 1
        # aggregate results for all paired-day comparisons
        trans_mats[:, :, i] = counter_mat

        # --------------------------------------------------------

        # REVERSE day[i]<--day[i+1]

        # best match of day n components to day i (the opposite
        # comparison as FORWARD block)
        step_out_lookup2 = np.argmax(sim_mats_by_day[i+1], axis=0)
        # best match of day i components to day n
        step_in_lookup2 = np.argmax(sim_mats_by_day[i], axis=1)

        # get transition matrix A[:,k], where k is comp number on
        # day n (i+1) and A[j,k] is probability of compenent k being
        # matched to comp j on PREVIOUS day
        counter_mat2 = np.zeros((rank_num, rank_num))
        for k in range(rank_num):
            vec_out2 = step_out_lookup2[k, :]
            for j in range(len(vec_out2)):
                best_step_in2 = step_in_lookup2[vec_out2[j], j]
                counter_mat2[best_step_in2, k] += 1

        # transpose and add to counter_mat results, this should now
        # be in the same reference frame
    #     trans_mats2[:,:,i] = counter_mat2.T
        trans_mats[:, :, i] += counter_mat2.T

    # if you now divide this by day number *2 if added += above
    # (number of comparisons) you should get a "probability" of a
    # given transition
    trans_mats = trans_mats/(len(sim_mats_by_day)*2)
    # trans_mats2 = trans_mats2/(len(sim_mats_by_day))

    # ------------------------------------------------------------

    # "Path of least resistance"
    if match_by.lower() == 'polr':

        # step through transition matrix and match factors taking
        # the best step at each node. Build a single transition matrix
        # for each factor/component.

        # FORWARD
        transform_matrix = trans_mats
        paths1 = np.zeros((np.shape(transform_matrix)[2]+1,rank_num))
        probs1 = np.zeros((np.shape(transform_matrix)[2],rank_num))
        for k in range(rank_num):
            comp_idx = k
            best_path = np.zeros((np.shape(transform_matrix)[2])+1)
            best_path[0] = comp_idx
            best_prob = np.zeros((np.shape(transform_matrix)[2]))
            for i in range(np.shape(transform_matrix)[2]):
                comp_idx_new = np.argmax(transform_matrix[:, comp_idx, i])
                best_path[i+1] = comp_idx_new
                best_prob[i] = transform_matrix[comp_idx_new, comp_idx, i]
                comp_idx = comp_idx_new
            paths1[:, k] = best_path
            probs1[:, k] = best_prob

        # REVERSE
        transform_matrix = trans_mats
        paths2 = np.zeros((np.shape(transform_matrix)[2]+1,rank_num))
        probs2 = np.zeros((np.shape(transform_matrix)[2],rank_num))
        for k in range(rank_num):
            comp_idx = k
            best_path = np.zeros((np.shape(transform_matrix)[2])+1)
            best_path[-1] = comp_idx
            best_prob = np.zeros((np.shape(transform_matrix)[2]))
            for i in range(np.shape(transform_matrix)[2])[::-1]:
                comp_idx_new = np.argmax(transform_matrix[comp_idx, :, i])
                best_path[i] = comp_idx_new
                best_prob[i] = transform_matrix[comp_idx, comp_idx_new, i]
                comp_idx = comp_idx_new
            paths2[:, k] = best_path
            probs2[:, k] = best_prob

    elif match_by.lower() == 'munkres':

        # step through transition matrix and match factors using
        # the Hungarian algorithm. Build a single transition matrix
        # for each factor/component.

        # FORWARD
        transform_matrix = trans_mats
        paths1 = np.zeros((np.shape(transform_matrix)[2]+1,rank_num))
        probs1 = np.zeros((np.shape(transform_matrix)[2],rank_num))
        matrix = deepcopy(transform_matrix[:, :, 0])*-1
        indexes = np.array(Munkres().compute(matrix))
        sort_order = np.argsort([s[1] for s in indexes])
        for c, k in enumerate(indexes[sort_order]):
            paths1[c, 0] = k[1]  # day 1 ind
            paths1[c, 1] = k[0]  # day 2 ind
            probs1[c, 0] = transform_matrix[k[0], k[1]]
        for i in range(1, np.shape(transform_matrix)[2]):
            matrix = deepcopy(transform_matrix[:, :, 0])*-1
            indexes = np.array(Munkres().compute(matrix))
            for k in range(rank_num):
                add_ind = indexes[k][0]  # day 2 ind
                match_ind = indexes[k][1]  # day 1 ind
                add_weight = transform_matrix[add_ind, match_ind]
                paths1[paths1[:, i] == match_ind, i+1] = add_ind
                probs1[paths1[:, i] == match_ind, i] = add_weight

    # get matched temporal factors
    temporal_factors_list = []
    for comp_num in range(np.shape(paths1)[1]):
        temporal_factors = []
        for i in range(np.shape(paths1)[0]):
                matched_comp = int(paths1[i, comp_num])
                fac_n = factors_by_day[i].factors[1][:, matched_comp]
                temporal_factors.append(fac_n)
        temporal_factors_list.append(temporal_factors)

    # Set outputs
    transition_weights = probs1.T
    temporal_factors_list = temporal_factors_list
    sim_mat_neuro_by_day = []
    # [sim_mats_by_day[i][:, :, i] for i in range(np.shape(sim_mats_by_day)[2])]
    sim_mat_tempo_by_day = []
    sim_mat_trial_tuning_by_day = []
    match_mat = []

    return {'trans': transition_weights,
            'tempo_fac': temporal_factors_list,
            'neuro_sim': sim_mat_neuro_by_day,
            'tempo_sim': sim_mat_tempo_by_day,
            'ttuning_sim': sim_mat_trial_tuning_by_day,
            'tri_sim': match_mat
            }


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
    days = flow.DateSorter.frommeta(
        mice=[mouse], tags=None, exclude_tags=['bad'])

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
        sim_mat_neuro_by_day.append(deepcopy(ids_corr[-rank_num:, 0:rank_num]))

        tempo_corr = np.corrcoef(tempo1_weights[:, :].T,
                                 y=tempo2_weights[:, :].T)
        sim_mat_tempo_by_day.append(deepcopy(tempo_corr[-rank_num:, 0:rank_num]))

        tuning_corr = np.corrcoef(tuning1_weights[:, :].T,
                                 y=tuning2_weights[:, :].T)
        sim_mat_trial_tuning_by_day.append(deepcopy(tuning_corr[-rank_num:, 0:rank_num]))

        # find transiiton probabilities by asking which factors
        # are matched (without replacement) in random shuffled order.
        # this will allow for balancing of components with
        # similar weights
        sz = ids_corr[-rank_num:, 0:rank_num]
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
    days = flow.DateSorter.frommeta(
        mice=[mouse], tags=None, exclude_tags=['bad'])

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
        sim_mat_by_day.append(deepcopy(ids_corr[-rank_num:, 0:rank_num]))

        tempo_corr = np.corrcoef(tempo1_weights[:, :].T,
                                 y=tempo2_weights[:, :].T)
        sim_mat_tempo_by_day.append(deepcopy(tempo_corr[-rank_num:, 0:rank_num]))

        # find transiiton probabilities by asking which factors
        # are matched (without replacement) in random shuffled order.
        # this will allow for balancing of components with
        # similar weights
        sz = ids_corr[-rank_num:, 0:rank_num]
        my_sort = np.zeros(np.shape(sz)[0], dtype=np.int64)
        bins = np.empty(np.shape(sz)[0])
        bins = np.empty((boot_num, np.shape(sz)[0]))
        bins[:] = np.nan
        # iterate multiple times, saving results into "bins"
        for it in range(boot_num):
            corner = deepcopy(ids_corr[-rank_num:, 0:rank_num])
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

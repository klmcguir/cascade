import numpy as np
import pandas as pd
from . import drive, utils, paths, lookups, load, categorize
from sklearn.metrics import roc_auc_score
from copy import deepcopy

import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from .trialanalysis import build_any_mat, build_cue_mat, build_speed_mat_shuffle
import os

"""
TODO: function to get preferred tuning from component category vector 
"""

def ab_index_df(match_to='onsets', index_on='pre_speed', return_matrix=True):
    """Generate a set of index calculations for a given index_on trial category.
    NOTE: preferred index type as of now is simple 'a-b' (.../[max_over_all_stages])

    Parameters
    ----------
    match_to : str, optional
        Use offsets or onsets (groups of cells), by default 'onsets'
    index_on : str, optional
        Index category to use, by default 'speed'
    return_matrix : bool, optional
        Optionally return your unwrapped tensor matrix split across trial categories and tuning. 

    Returns
    -------
    pandas.DataFrame
        Indices for each cell. 
    numpy.ndarray, optional
        Matrix of all of your cell averages for each trial category and tuning condition. 

    Raises
    ------
    NotImplementedError
        When you ask for an index type that has not been set up yet.
    """

    # if the index file exists return the file 
    save_path = os.path.join(lookups.coreroot, f'{match_to}_{index_on}_index_2s_df.pkl')
    matrix_path = os.path.join(lookups.coreroot, f'{match_to}_{index_on}_mat.npy')
    if os.path.isfile(save_path):
        index_df = pd.read_pickle(save_path)
        if return_matrix:
            mat2ds = np.load(matrix_path)
            return index_df, mat2ds
        return index_df

    # load tca data
    on_ens = load.core_tca_data(match_to=match_to)

    # laod drivenness metrics
    pref_ind = drive.preferred_cue_ind(match_to=match_to)
    driven = drive.preferred_drive_mat(match_to=match_to)

    # parse the index you are going to calculate
    if index_on == 'pre_speed':
        meta_col = 'pre_speed'
        meta_col_vals = ['placeholder']
    elif index_on == 'th':
        meta_col = 'th'
        meta_col_vals = ['placeholder']
    elif index_on == 'gonogo':
        meta_col = 'trialerror'
        meta_col_vals = [0, 3, 5]
    elif index_on == 'disengaged': # engaged vs disengaged
        meta_col = 'hmm_engaged'
        meta_col_vals = [True]
    else:
        raise NotImplementedError # add in other possibilites above

    # build a tensor of all your tuning conditions and user defined split on some condition
    mat2ds = build_any_mat(on_ens['mouse_vec'],
                           on_ens['cell_vec'],
                           meta_col=meta_col,
                           meta_col_vals=meta_col_vals,
                           limit_tag=match_to,
                           allstage=True,
                           norm_please=False,  # will normalize post load
                           no_disengaged=False if index_on.lower() == 'disengaged' else True,
                           add_broad_joint_tuning=True,
                           n_in_a_row=2,
                           staging='parsed_11stage')
    np.save(matrix_path, mat2ds)

    # take mean across 2s stim or response window
    mat_list = []
    for i in range(mat2ds.shape[2]):
        wrap = utils.wrap_tensor(mat2ds[:,:,i])
        mean_per_stage = np.nanmean(wrap[:,17:,:], axis=1)
        mat_list.append(mean_per_stage[:, :, None])
    mean_stack = np.dstack(mat_list)
    mean_stack.shape

    # normalize to max
    max_vec = np.nanmax(np.nanmax(mean_stack, axis=2), axis=1)
    max_vec[max_vec <= 0] = np.nan
    norm_stack = mean_stack/max_vec[:, None, None]
    # norm_stack = mean_stack

    # rectify data to make index more interpretable
    rect_stack = deepcopy(norm_stack)
    rect_stack[rect_stack < 0] = 0

    # blank unpreferred stimuli
    pref_stack = deepcopy(rect_stack)
    total_pre = np.nansum(pref_ind, axis=1)
    # no pref
    pref_stack[total_pre == 0, :, :] = np.nan
    # single tuning
    for i in range(3):
        slice_not_preferred = ~(pref_ind[:, i] == 1) & (total_pre == 1)
        slice_preferred = (pref_ind[:, i] == 1) & (total_pre == 1)
        pref_stack[slice_not_preferred, :, i] = np.nan
        pref_stack[slice_not_preferred, :, i+6] = np.nan
        pref_stack[slice_preferred, :, 3:6] = np.nan
        pref_stack[slice_preferred, :, 9:12] = np.nan
    # broad tuning
    slice_not_preferred = ~(total_pre == 3)
    pref_stack[slice_not_preferred, :, 3] = np.nan
    pref_stack[slice_not_preferred, :, 3+6] = np.nan
    for i in range(12):
        if i == 3 or i == 3 + 6:
            continue
        pref_stack[~slice_not_preferred, :, i] = np.nan
    # joint tuning --> 'joint-becomes_unrewarded-remains_unrewarded'
    slice_not_preferred = ~(total_pre == 2)  & ((pref_ind[:, 0] == 1) & (pref_ind[:, 1] == 1))
    slice_preferred = (total_pre == 2)  & ((pref_ind[:, 0] == 1) & (pref_ind[:, 1] == 1))
    pref_stack[slice_not_preferred, :, 4] = np.nan
    pref_stack[slice_not_preferred, :, 4+6] = np.nan
    for i in range(12):
        if i == 4 or i == 4 + 6:
            continue
        pref_stack[slice_preferred, :, i] = np.nan
    # joint tuning --> 'joint-becomes_rewarded-remains_unrewarded'
    slice_not_preferred = ~(total_pre == 2)  & ((pref_ind[:, 1] == 1) & (pref_ind[:, 2] == 1))
    slice_preferred = (total_pre == 2)  & ((pref_ind[:, 1] == 1) & (pref_ind[:, 2] == 1))
    pref_stack[slice_not_preferred, :, 5] = np.nan
    pref_stack[slice_not_preferred, :, 5+6] = np.nan
    for i in range(12):
        if i == 5 or i == 5 + 6:
            continue
        pref_stack[slice_preferred, :, i] = np.nan
    # max across depth can't be more than 2 for any cell for any stage (one per condition)
    assert np.all(np.nanmax(np.sum(np.isfinite(pref_stack), axis=2), axis=1) <= 2)

    # get peak stage index per cell
    no_na = deepcopy(pref_stack)
    no_na[np.isnan(no_na)] = -10
    best_stage = np.nanargmax(np.nanmax(no_na[:,:,:], axis=2)[:, 1:], axis=1) + 1 # don't allow naive to be best

    # smoosh cues together for fast and slow
    pref_fast = np.nanmean(pref_stack[:,:,:6], axis=2)
    pref_slow = np.nanmean(pref_stack[:,:,6:], axis=2)

    # nan undriven stages for each cell
    pref_fast[driven == 0] = np.nan
    pref_slow[driven == 0] = np.nan

    # take index for each driven stage tehn average index over stages
    index_each_stage = np.nanmean(pref_fast - pref_slow / (pref_fast + pref_slow), axis=1)

    # take mean across stages then index
    a = np.nanmean(pref_fast[:,1:], axis=1) # don't include naive
    b = np.nanmean(pref_slow[:,1:], axis=1)
    index_avg_stage = a - b / (a + b)

    # take diference since everything is normalized to peak
    index_diff_stage = a - b
    avg_a = deepcopy(a)
    avg_b = deepcopy(b)

    # take index using only the maximum driven stage per cell
    a = np.zeros(len(best_stage)) + np.nan
    b = np.zeros(len(best_stage)) + np.nan
    for celli in range(len(best_stage)):
        a[celli] = pref_fast[celli, best_stage[celli]]
        b[celli] = pref_slow[celli, best_stage[celli]]
    index_diff_best_stage = a - b
    index_best_stage = a - b / (a + b)

    # take index using only the maximum driven pre/post reversal per cell
    a = np.zeros(len(best_stage)) + np.nan
    b = np.zeros(len(best_stage)) + np.nan
    for celli in range(len(best_stage)):
        if best_stage[celli] <= 5:
            a[celli] = np.nanmean(pref_fast[celli, :6])
            b[celli] = np.nanmean(pref_slow[celli, :6])
        elif best_stage[celli] > 5:
            a[celli] = np.nanmean(pref_fast[celli, 6:])
            b[celli] = np.nanmean(pref_slow[celli, 6:])
    index_diff_best_half = a - b
    index_best_half = a - b / (a + b)

    data = {
        'mouse': on_ens['mouse_vec'],
        'cell_id': on_ens['cell_vec'],
        'cell_cats': on_ens['cell_cats'],
        'index_on': [index_on] * len(on_ens['cell_cats']),
        'onsets_or_offsets': [match_to] * len(on_ens['cell_cats']),
        f'a-b': index_diff_stage,
        f'a-b/a+b': index_avg_stage,
        f'a-b/a+b_eachstageavg': index_each_stage,
        f'a-b_maxstage': index_diff_best_stage,
        f'a-b/a+b_maxstage': index_best_stage,
        f'a-b_maxhalf': index_diff_best_half,
        f'a-b/a+b_maxhalf': index_best_half,
        f'a': avg_a,
        f'b': avg_b,
    }
    index_df = pd.DataFrame(data=data).set_index(['mouse', 'cell_id'])
    index_df.to_pickle(save_path)

    if return_matrix:
        return index_df, mat2ds
    return index_df


def ab_index_df_shuffle(match_to='onsets', index_on='pre_speed', return_matrix=True, boot_n=10):
    """Generate a set of index calculations for a given index_on trial category.
    This is a SHUFFLE without replacement. To test how likely it would be to get a similar
    a-b index given totaly random trial inputs taken from the same pool of trials. 
    NOTE: preferred index type as of now is simple 'a-b' (.../[max_over_all_stages])

    Parameters
    ----------
    match_to : str, optional
        Use offsets or onsets (groups of cells), by default 'onsets'
    index_on : str, optional
        Index category to use, by default 'speed'
    return_matrix : bool, optional
        Optionally return your unwrapped tensor matrix split across trial categories and tuning. 
    boot_n : int, optional
        Number of shuffles to perform. 

    Returns
    -------
    pandas.DataFrame
        Indices for each cell. 
    numpy.ndarray, optional
        Matrix of all of your cell averages for each trial category and tuning condition. 

    Raises
    ------
    NotImplementedError
        When you ask for an index type that has not been set up yet.
    """

    # if the index file exists return the file 
    save_path = os.path.join(lookups.coreroot, f'{match_to}_{index_on}_index_2s_df_shuffle{boot_n}.pkl')
    matrix_path = os.path.join(lookups.coreroot, f'{match_to}_{index_on}_mat_shuffle{boot_n}.npy')
    if os.path.isfile(save_path):
        index_df = pd.read_pickle(save_path)
        if return_matrix:
            mat2ds = np.load(matrix_path, allow_pickle=True)
            return index_df, mat2ds
        return index_df

    # load tca data
    on_ens = load.core_tca_data(match_to=match_to)

    # laod drivenness metrics
    pref_ind = drive.preferred_cue_ind(match_to=match_to)
    driven = drive.preferred_drive_mat(match_to=match_to)

    # parse the index you are going to calculate
    if index_on == 'pre_speed':
        meta_col = 'pre_speed'
        meta_col_vals = ['placeholder']
    # elif index_on == 'th':
    #     meta_col = 'th'
    #     meta_col_vals = ['placeholder']
    # elif index_on == 'gonogo':
    #     meta_col = 'trialerror'
    #     meta_col_vals = [0, 3, 5]
    # elif index_on == 'disengaged': # engaged vs disengaged
    #     meta_col = 'hmm_engaged'
    #     meta_col_vals = [True]
    else:
        raise NotImplementedError # add in other possibilites above

    # build a tensor of all your tuning conditions and user defined split on some condition
    mat2ds_list = build_speed_mat_shuffle(on_ens['mouse_vec'],
                           on_ens['cell_vec'],
                           limit_tag=match_to,
                           allstage=True,
                           norm_please=False,  # will normalize post load
                           no_disengaged=False if index_on.lower() == 'disengaged' else True,
                           add_broad_joint_tuning=True,
                           staging='parsed_11stage',
                           boot_n=boot_n)
    np.save(matrix_path, mat2ds_list, allow_pickle=True)

    shuffle_list = []
    for booti, mat2ds in enumerate(mat2ds_list):
        # take mean across 2s stim or response window
        mat_list = []
        for i in range(mat2ds.shape[2]):
            wrap = utils.wrap_tensor(mat2ds[:,:,i])
            mean_per_stage = np.nanmean(wrap[:,17:,:], axis=1)
            mat_list.append(mean_per_stage[:, :, None])
        mean_stack = np.dstack(mat_list)
        mean_stack.shape

        # normalize to max
        max_vec = np.nanmax(np.nanmax(mean_stack, axis=2), axis=1)
        max_vec[max_vec <= 0] = np.nan
        norm_stack = mean_stack/max_vec[:, None, None]
        # norm_stack = mean_stack

        # rectify data to make index more interpretable
        rect_stack = deepcopy(norm_stack)
        rect_stack[rect_stack < 0] = 0

        # blank unpreferred stimuli
        pref_stack = deepcopy(rect_stack)
        total_pre = np.nansum(pref_ind, axis=1)
        # no pref
        pref_stack[total_pre == 0, :, :] = np.nan
        # single tuning
        for i in range(3):
            slice_not_preferred = ~(pref_ind[:, i] == 1) & (total_pre == 1)
            slice_preferred = (pref_ind[:, i] == 1) & (total_pre == 1)
            pref_stack[slice_not_preferred, :, i] = np.nan
            pref_stack[slice_not_preferred, :, i+6] = np.nan
            pref_stack[slice_preferred, :, 3:6] = np.nan
            pref_stack[slice_preferred, :, 9:12] = np.nan
        # broad tuning
        slice_not_preferred = ~(total_pre == 3)
        pref_stack[slice_not_preferred, :, 3] = np.nan
        pref_stack[slice_not_preferred, :, 3+6] = np.nan
        for i in range(12):
            if i == 3 or i == 3 + 6:
                continue
            pref_stack[~slice_not_preferred, :, i] = np.nan
        # joint tuning --> 'joint-becomes_unrewarded-remains_unrewarded'
        slice_not_preferred = ~(total_pre == 2)  & ((pref_ind[:, 0] == 1) & (pref_ind[:, 1] == 1))
        slice_preferred = (total_pre == 2)  & ((pref_ind[:, 0] == 1) & (pref_ind[:, 1] == 1))
        pref_stack[slice_not_preferred, :, 4] = np.nan
        pref_stack[slice_not_preferred, :, 4+6] = np.nan
        for i in range(12):
            if i == 4 or i == 4 + 6:
                continue
            pref_stack[slice_preferred, :, i] = np.nan
        # joint tuning --> 'joint-becomes_rewarded-remains_unrewarded'
        slice_not_preferred = ~(total_pre == 2)  & ((pref_ind[:, 1] == 1) & (pref_ind[:, 2] == 1))
        slice_preferred = (total_pre == 2)  & ((pref_ind[:, 1] == 1) & (pref_ind[:, 2] == 1))
        pref_stack[slice_not_preferred, :, 5] = np.nan
        pref_stack[slice_not_preferred, :, 5+6] = np.nan
        for i in range(12):
            if i == 5 or i == 5 + 6:
                continue
            pref_stack[slice_preferred, :, i] = np.nan
        # max across depth can't be more than 2 for any cell for any stage (one per condition)
        assert np.all(np.nanmax(np.sum(np.isfinite(pref_stack), axis=2), axis=1) <= 2)

        # get peak stage index per cell
        no_na = deepcopy(pref_stack)
        no_na[np.isnan(no_na)] = -10
        best_stage = np.nanargmax(np.nanmax(no_na[:,:,:], axis=2)[:, 1:], axis=1) + 1 # don't allow naive to be best

        # smoosh cues together for fast and slow
        pref_fast = np.nanmean(pref_stack[:,:,:6], axis=2)
        pref_slow = np.nanmean(pref_stack[:,:,6:], axis=2)

        # nan undriven stages for each cell
        pref_fast[driven == 0] = np.nan
        pref_slow[driven == 0] = np.nan

        # take index for each driven stage tehn average index over stages
        index_each_stage = np.nanmean(pref_fast - pref_slow / (pref_fast + pref_slow), axis=1)

        # take mean across stages then index
        a = np.nanmean(pref_fast[:,1:], axis=1) # don't include naive
        b = np.nanmean(pref_slow[:,1:], axis=1)
        index_avg_stage = a - b / (a + b)

        # take diference since everything is normalized to peak
        index_diff_stage = a - b
        avg_a = deepcopy(a)
        avg_b = deepcopy(b)

        # take index using only the maximum driven stage per cell
        a = np.zeros(len(best_stage)) + np.nan
        b = np.zeros(len(best_stage)) + np.nan
        for celli in range(len(best_stage)):
            a[celli] = pref_fast[celli, best_stage[celli]]
            b[celli] = pref_slow[celli, best_stage[celli]]
        index_diff_best_stage = a - b
        index_best_stage = a - b / (a + b)

        # take index using only the maximum driven pre/post reversal per cell
        a = np.zeros(len(best_stage)) + np.nan
        b = np.zeros(len(best_stage)) + np.nan
        for celli in range(len(best_stage)):
            if best_stage[celli] <= 5:
                a[celli] = np.nanmean(pref_fast[celli, :6])
                b[celli] = np.nanmean(pref_slow[celli, :6])
            elif best_stage[celli] > 5:
                a[celli] = np.nanmean(pref_fast[celli, 6:])
                b[celli] = np.nanmean(pref_slow[celli, 6:])
        index_diff_best_half = a - b
        index_best_half = a - b / (a + b)

        data = {
            'mouse': on_ens['mouse_vec'],
            'cell_id': on_ens['cell_vec'],
            'cell_cats': on_ens['cell_cats'],
            'boot_n': booti,
            'index_on': [index_on] * len(on_ens['cell_cats']),
            'onsets_or_offsets': [match_to] * len(on_ens['cell_cats']),
            f'a-b': index_diff_stage,
            f'a-b/a+b': index_avg_stage,
            f'a-b/a+b_eachstageavg': index_each_stage,
            f'a-b_maxstage': index_diff_best_stage,
            f'a-b/a+b_maxstage': index_best_stage,
            f'a-b_maxhalf': index_diff_best_half,
            f'a-b/a+b_maxhalf': index_best_half,
            f'a': avg_a,
            f'b': avg_b,
        }
        index_df = pd.DataFrame(data=data).set_index(['mouse', 'cell_id'])
        shuffle_list.append(index_df)
    index_df = pd.concat(shuffle_list, axis=0)
    index_df.to_pickle(save_path)

    if return_matrix:
        return index_df, mat2ds_list
    return index_df


def ab_index_df_by_stage(match_to='onsets', index_on='pre_speed', return_matrix=True, staging='parsed_11stage'):
    """Generate a set of index calculations for a given index_on trial category.
    NOTE: preferred index type as of now is simple 'a-b' (.../[max_over_all_stages])

    TODO: Need to implement it so staging is passed down through drivenness calcs. For now only parsed_11stage will work

    Parameters
    ----------
    match_to : str, optional
        Use offsets or onsets (groups of cells), by default 'onsets'
    index_on : str, optional
        Index category to use, by default 'speed'
    return_matrix : bool, optional
        Optionally return your unwrapped tensor matrix split across trial categories and tuning. 

    Returns
    -------
    pandas.DataFrame
        Indices for each cell. 
    numpy.ndarray, optional
        Matrix of all of your cell averages for each trial category and tuning condition. 

    Raises
    ------
    NotImplementedError
        When you ask for an index type that has not been set up yet.
    """

    # if the index file exists return the file 
    save_path = os.path.join(lookups.coreroot, f'{match_to}_{index_on}_by_{staging}_index_2s_df.pkl')
    matrix_path = os.path.join(lookups.coreroot, f'{match_to}_{index_on}_by_{staging}_mat.npy')
    if os.path.isfile(save_path):
        index_df = pd.read_pickle(save_path)
        if return_matrix:
            mat2ds = np.load(matrix_path)
            return index_df, mat2ds
        return index_df

    # load tca data
    on_ens = load.core_tca_data(match_to=match_to)

    # laod drivenness metrics
    pref_ind = drive.preferred_cue_ind(match_to=match_to)
    driven = drive.preferred_drive_mat(match_to=match_to, staging=staging)

    # parse the index you are going to calculate
    if index_on == 'pre_speed':
        meta_col = 'pre_speed'
        meta_col_vals = ['placeholder']
    elif index_on == 'th':
        meta_col = 'th'
        meta_col_vals = ['placeholder']
    elif index_on == 'gonogo':
        meta_col = 'trialerror'
        meta_col_vals = [0, 3, 5]
    elif index_on == 'disengaged': # engaged vs disengaged
        meta_col = 'hmm_engaged'
        meta_col_vals = [True]
    else:
        raise NotImplementedError # add in other possibilites above

    # build a tensor of all your tuning conditions and user defined split on some condition
    mat2ds = build_any_mat(on_ens['mouse_vec'],
                           on_ens['cell_vec'],
                           meta_col=meta_col,
                           meta_col_vals=meta_col_vals,
                           limit_tag=match_to,
                           allstage=True,
                           norm_please=False,  # will normalize post load
                           no_disengaged=False if index_on.lower() == 'disengaged' else True,
                           add_broad_joint_tuning=True,
                           n_in_a_row=2,
                           staging=staging)
    np.save(matrix_path, mat2ds)

    # take mean across 2s stim or response window
    mat_list = []
    for i in range(mat2ds.shape[2]):
        wrap = utils.wrap_tensor(mat2ds[:,:,i])
        mean_per_stage = np.nanmean(wrap[:,17:,:], axis=1)
        mat_list.append(mean_per_stage[:, :, None])
    mean_stack = np.dstack(mat_list)
    mean_stack.shape

    # normalize to max
    max_vec = np.nanmax(np.nanmax(mean_stack, axis=2), axis=1)
    max_vec[max_vec <= 0] = np.nan
    norm_stack = mean_stack/max_vec[:, None, None]
    # norm_stack = mean_stack

    # rectify data to make index more interpretable
    rect_stack = deepcopy(norm_stack)
    rect_stack[rect_stack < 0] = 0

    # blank unpreferred stimuli
    pref_stack = deepcopy(rect_stack)
    total_pre = np.nansum(pref_ind, axis=1)
    # no pref
    pref_stack[total_pre == 0, :, :] = np.nan
    # single tuning
    for i in range(3):
        slice_not_preferred = ~(pref_ind[:, i] == 1) & (total_pre == 1)
        slice_preferred = (pref_ind[:, i] == 1) & (total_pre == 1)
        pref_stack[slice_not_preferred, :, i] = np.nan
        pref_stack[slice_not_preferred, :, i+6] = np.nan
        pref_stack[slice_preferred, :, 3:6] = np.nan
        pref_stack[slice_preferred, :, 9:12] = np.nan
    # broad tuning
    slice_not_preferred = ~(total_pre == 3)
    pref_stack[slice_not_preferred, :, 3] = np.nan
    pref_stack[slice_not_preferred, :, 3+6] = np.nan
    for i in range(12):
        if i == 3 or i == 3 + 6:
            continue
        pref_stack[~slice_not_preferred, :, i] = np.nan
    # joint tuning --> 'joint-becomes_unrewarded-remains_unrewarded'
    slice_not_preferred = ~(total_pre == 2)  & ((pref_ind[:, 0] == 1) & (pref_ind[:, 1] == 1))
    slice_preferred = (total_pre == 2)  & ((pref_ind[:, 0] == 1) & (pref_ind[:, 1] == 1))
    pref_stack[slice_not_preferred, :, 4] = np.nan
    pref_stack[slice_not_preferred, :, 4+6] = np.nan
    for i in range(12):
        if i == 4 or i == 4 + 6:
            continue
        pref_stack[slice_preferred, :, i] = np.nan
    # joint tuning --> 'joint-becomes_rewarded-remains_unrewarded'
    slice_not_preferred = ~(total_pre == 2)  & ((pref_ind[:, 1] == 1) & (pref_ind[:, 2] == 1))
    slice_preferred = (total_pre == 2)  & ((pref_ind[:, 1] == 1) & (pref_ind[:, 2] == 1))
    pref_stack[slice_not_preferred, :, 5] = np.nan
    pref_stack[slice_not_preferred, :, 5+6] = np.nan
    for i in range(12):
        if i == 5 or i == 5 + 6:
            continue
        pref_stack[slice_preferred, :, i] = np.nan
    # max across depth can't be more than 2 for any cell for any stage (one per condition)
    assert np.all(np.nanmax(np.sum(np.isfinite(pref_stack), axis=2), axis=1) <= 2)

    # get peak stage index per cell
    no_na = deepcopy(pref_stack)
    no_na[np.isnan(no_na)] = -10
    best_stage = np.nanargmax(np.nanmax(no_na[:,:,:], axis=2)[:, 1:], axis=1) + 1 # don't allow naive to be best

    df_list = []
    for sc, stagi in enumerate(lookups.staging[staging]):

        # smoosh cues together for fast and slow
        pref_fast = np.nanmean(pref_stack[:,:,:6], axis=2)
        pref_slow = np.nanmean(pref_stack[:,:,6:], axis=2)

        # nan undriven stages for each cell
        pref_fast[driven == 0] = np.nan
        pref_slow[driven == 0] = np.nan

        # take index for each driven stage tehn average index over stages
        index_each_stage = np.nanmean(pref_fast - pref_slow / (pref_fast + pref_slow), axis=1)

        # take mean across stages then index
        a = pref_fast[:, sc]
        b = pref_slow[:, sc]
        index_avg_stage = a - b / (a + b)

        # take diference since everything is normalized to peak
        index_diff_stage = a - b
        avg_a = deepcopy(a)
        avg_b = deepcopy(b)


        data = {
            'mouse': on_ens['mouse_vec'],
            staging: [stagi] * len(on_ens['mouse_vec']),
            'cell_id': on_ens['cell_vec'],
            'cell_cats': on_ens['cell_cats'],
            'index_on': [index_on] * len(on_ens['cell_cats']),
            'onsets_or_offsets': [match_to] * len(on_ens['cell_cats']),
            f'a-b': index_diff_stage,
            f'a-b/a+b': index_avg_stage,
            f'a-b/a+b_eachstageavg': index_each_stage,
            f'a': avg_a,
            f'b': avg_b,
        }
        index_df = pd.DataFrame(data=data).set_index(['mouse', 'cell_id'])
        df_list.append(index_df)
    index_df = pd.concat(df_list, axis=0)
    index_df.to_pickle(save_path)

    if return_matrix:
        return index_df, mat2ds
    return index_df


def calculate_index_core_reversal(match_to='onsets', index_type='ab_ab', on='go', staging='parsed_4stage'):

    # calculate drive
    drive_mat_list = drive.drive_day_mat_from_core_reversal_data(match_to='onsets')

    # load all of your raw reversal n=7 data
    load_dict = load.core_reversal_data(limit_to=None, match_to=match_to)

    # load models for cell categorization
    ensemble = np.load(paths.analysis_file('tca_ensemble_v4i10_noT0_20210307.npy', 'tca_dfs'), allow_pickle=True).item()
    tca_data_dict = np.load(paths.analysis_file('input_data_v4i10_noT0_20210307.npy', 'tca_dfs'), allow_pickle=True).item()
    mod = 'v4i10_norm_on_noT0' if match_to == 'onsets' else 'v4i10_norm_off_noT0'
    mouse_mod = 'v4i10_on_mouse_noT0' if match_to == 'onsets' else 'v4i10_off_mouse_noT0'
    rank = 9 if match_to == 'onsets' else 8
    factors = ensemble[mod].results[rank][0].factors
    cell_cats = categorize.best_comp_cats(factors)
    mouse_vec = tca_data_dict[mouse_mod]

    # calculate drivenness on each cell looping over mice
    si_list = []
    for meta, ids, tensor, dmat in zip(load_dict['meta_list'], load_dict['id_list'], load_dict['tensor_list'],
                                       drive_mat_list):

        # make sure binning method is in meta
        meta = utils.add_stages_to_meta(meta, staging)

        # get cetegory vector for individual mice
        mouse_boo = mouse_vec == utils.meta_mouse(meta)
        mouse_cats = cell_cats[mouse_boo]

        # calculate index of choice
        # ignore python and numpy divide by zero warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with np.errstate(invalid='ignore', divide='ignore'):
                si_mat = selectivity_index(meta,
                                           tensor,
                                           drive_mat=dmat,
                                           index_type=index_type,
                                           on=on,
                                           staging=staging,
                                           cats=mouse_cats,
                                           plot_please=False)
        si_list.append(si_mat)
    si_mat = np.vstack(si_list)

    return si_mat


def selectivity_index(meta,
                      tensor,
                      drive_mat=None,
                      index_type='ab_ab',
                      on='go',
                      staging='parsed_4stage',
                      driven_only=True,
                      cats=None,
                      plot_please=False):

    # for plotting cell tuning
    # trial_means = np.nanmean(tensor[:, 17:, :], axis=1)
    # cues = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']
    # cue_means = np.zeros((trial_means.shape[0], len(cues))) + np.nan
    # for cc, cue in enumerate(cues):
    #     cue_boo = meta.mismatch_condition.isin([cue])
    #     cue_means[:, cc] = np.nanmean(trial_means[:, cue_boo], axis=1)
    # pref_cue = np.argmax(cue_means, axis=1)
    # pref_list = [cues[s] for s in pref_cue]

    # add reversal condition to metadata dataframe.
    if 'mismatch_condition' not in meta.columns:
        meta = utils.add_reversal_mismatch_condition_to_meta(meta)

    # nan trials for undriven days/cues
    if driven_only:
        cues = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']
        if drive_mat is not None:
            tensor = deepcopy(tensor)
            for ci, cue in enumerate(cues):
                cue_boo = meta.mismatch_condition.isin([cue]).values
                trial_boo = drive_mat[:, :, ci] == 0
                for celli in range(tensor.shape[0]):
                    tensor[celli, :, trial_boo[celli, :] & cue_boo] = np.nan

    stages = lookups.staging[staging]
    m_si_index = np.zeros((tensor.shape[0], len(stages))) + np.nan
    for c, di in enumerate(stages):
        stage_boo = meta[staging].isin([di]).values
        day_stage_boo = stage_boo
        # stage_days = meta.loc[stage_boo, :].reset_index()['date'].unique()
        # day_means = np.zeros((tensor.shape[0], tensor.shape[1], len(stage_days)))
        # day_means[:] = np.nan
        #         for c2, di2 in tqdm(enumerate(stage_days),
        #                             desc=f'{utils.meta_mouse(meta)}: {di}, cue SVM',
        #                             total=len(stage_days)):
        #             day_boo = meta.reset_index()['date'].isin([di2]).values

        # define trials you calc index on
        if on == 'go':
            hit_boo = meta.trialerror.isin([0, 3, 5]) #3,5
            cr_boo = meta.trialerror.isin([1, 2, 4]) #2, 4
        elif on == 'correct':
            hit_boo = meta.trialerror.isin([0, 2, 4])
            cr_boo = meta.trialerror.isin([1, 3, 5])
        elif on == 'th':
            prev_same = (meta.prev_same_plus | meta.prev_same_neutral | meta.prev_same_minus)
            hit_boo = ~prev_same
            cr_boo = prev_same
        elif on == 'speed':
            hit_boo = meta.pre_speed.ge(10) #3,5
            cr_boo = meta.pre_speed.le(5) #2, 4
        elif on == 'rh' or on == 'reward':
            # reward history, focusing only on FC-FC trials
            hit_boo = meta.prev_reward & (meta.prev_same_plus | meta.prev_same_neutral | meta.prev_same_minus)
            cr_boo = ~meta.prev_reward & (meta.prev_same_plus | meta.prev_same_neutral | meta.prev_same_minus)
        elif on == 'ph' or on == 'punishment':
            # punishment history, focusing only on QC-QC trials
            hit_boo = meta.prev_punish & (meta.prev_same_plus | meta.prev_same_neutral | meta.prev_same_minus)
            cr_boo = ~meta.prev_punish & (meta.prev_same_plus | meta.prev_same_neutral | meta.prev_same_minus)
        elif on == 'rh2' or on == 'reward2':
            # reward history, focusing only on FC-FC trials
            hit_boo = meta.prev_reward #& meta.trialerror.isin([0])
            cr_boo = ~meta.prev_reward #& meta.trialerror.isin([0])
        elif on == 'ph2' or on == 'punishment2':
            # punishment history, focusing only on QC-QC trials
            hit_boo = meta.prev_punish #& (meta.prev_same_plus | meta.prev_same_neutral | meta.prev_same_minus)
            cr_boo = ~meta.prev_punish #& (meta.prev_same_plus | meta.prev_same_neutral | meta.prev_same_minus)

        cr_tensor = tensor[:, 17:, day_stage_boo & cr_boo]
        hit_tensor = tensor[:, 17:, day_stage_boo & hit_boo]
        # plt_cr_tensor =  tensor[:, :, day_stage_boo & cr_boo]
        # plt_hit_tensor = tensor[:, :, day_stage_boo & hit_boo]

        # nan the unpreferred cue types for each cell
        if cats is not None:
            hit_meta = meta.loc[day_stage_boo & hit_boo]
            cr_meta = meta.loc[day_stage_boo & cr_boo]
            cue_groups = [[ 0,  5,  6], [4,  8], [1,  2,  3], [7], [-1]]
            cue_groups_labels =[
                ['becomes_unrewarded'],
                ['remains_unrewarded'],
                ['becomes_rewarded'],
                ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded'],
                ['supressed']  # will nan out
            ]
            # nan unpreffered cue for cells based on TCA preferred tuning
            for cg, lbl in zip(cue_groups, cue_groups_labels):
                cell_cg_boo = np.isin(cats, cg)
                hit_lbl_boo = ~hit_meta.mismatch_condition.isin(lbl)
                cr_lbl_boo = ~cr_meta.mismatch_condition.isin(lbl)
                for celli in np.where(cell_cg_boo)[0]:
                    cr_tensor[celli, :, cr_lbl_boo] = np.nan
                    hit_tensor[celli, :, hit_lbl_boo] = np.nan
                    # plt_cr_tensor[celli, :, cr_lbl_boo] = np.nan
                    # plt_hit_tensor[celli, :, hit_lbl_boo] = np.nan
        cr_mean = np.nanmean(cr_tensor, axis=2)
        hit_mean = np.nanmean(hit_tensor, axis=2)

        # calculate discriminability variable for hit and CR trials
        if index_type == 'selectivity':
            # SI = 2*(AUC([prevdiff_discrim_dot, prevsame_discrim_dot])-0.5)
            # Value-guided remapping of sensory cortex by lateral orbitofrontal cortex, 2020
            # Abhishek Banerjee1,3, Giuseppe Parente1,4, Jasper Teutsch1,3,4, Christopher Lewis1,
            # Fabian F. Voigt1,2 & Fritjof Helmchen1,2

            # calculate descrimination score
            dot_mat_cr = np.zeros((cr_tensor.shape[0], cr_tensor.shape[2]))
            for i in range(cr_tensor.shape[0]):
                dot_mat_cr[i, :] = cr_mean[i,:] @ cr_tensor[i, :, :] - hit_mean[i,:] @ cr_tensor[i, :, :]
            dot_mat_hit = np.zeros((hit_tensor.shape[0], hit_tensor.shape[2]))
            for i in range(hit_tensor.shape[0]):
                dot_mat_hit[i, :] = hit_mean[i,:] @ hit_tensor[i, :, :] - cr_mean[i,:] @ hit_tensor[i, :, :]

                # calculate si index for stage, skip if all trials are nan
            y = np.array(['hit'] * np.sum(day_stage_boo & hit_boo) + ['CR'] * np.sum(day_stage_boo & cr_boo))
            y_hat = np.concatenate([dot_mat_hit, dot_mat_cr], axis=1)
            si_vec = np.zeros(y_hat.shape[0]) + np.nan
            for i in range(y_hat.shape[0]):
                if np.isnan(y_hat[i, :]).all():
                    continue
                notna = ~np.isnan(y_hat[i, :])
                gcount = np.sum(np.isin(y[notna], 'hit'))
                ngcount = np.sum(np.isin(y[notna], 'CR'))
                if gcount < 10 and ngcount < 10:
                    print(f'Go: {gcount} NoGO: {ngcount}')
                    continue
                try:
                    si = 2*(roc_auc_score(y[notna], y_hat[i, notna]) - 0.5)
                except:
                    si = np.nan
                si_vec[i] = si

            m_si_index[:, c] = si_vec

        elif index_type == 'ab_ab':

            # a - b / a + b
            dot_mat_hit = np.nanmean(np.nanmean(hit_tensor, axis=1), axis=1)
            dot_mat_cr = np.nanmean(np.nanmean(cr_tensor, axis=1), axis=1)
            dot_mat_hit[dot_mat_hit < 0] = 0.0000001
            dot_mat_cr[dot_mat_cr < 0] = 0.0000001
            m_si_index[:, c] = (dot_mat_hit - dot_mat_cr) / (dot_mat_hit + dot_mat_cr)

        elif index_type == 'ab_mean':

            # a - b / mean(all a and b trials)
            all_trial_tensor = np.nanmean(np.nanmean(np.concatenate([hit_tensor, cr_tensor], axis=2), axis=1), axis=1)
            dot_mat_hit = np.nanmean(np.nanmean(hit_tensor, axis=1), axis=1)
            dot_mat_cr = np.nanmean(np.nanmean(cr_tensor, axis=1), axis=1)
            dot_mat_hit[dot_mat_hit < 0] = np.nan
            dot_mat_cr[dot_mat_cr < 0] = np.nan
            all_trial_tensor[all_trial_tensor < 0] = 0.0000001
            m_si_index[:, c] = (dot_mat_hit - dot_mat_cr) / all_trial_tensor

        elif index_type == 'auc':

            dot_mat_hit = np.nanmean(hit_tensor, axis=1)
            dot_mat_cr = np.nanmean(cr_tensor, axis=1)
            y = np.array(['hit'] * np.sum(day_stage_boo & hit_boo) + ['CR'] * np.sum(day_stage_boo & cr_boo))
            y_hat = np.concatenate([dot_mat_hit, dot_mat_cr], axis=1)
            si_vec = np.zeros(y_hat.shape[0]) + np.nan
            for i in range(y_hat.shape[0]):
                if np.isnan(y_hat[i, :]).all():
                    continue
                notna = ~np.isnan(y_hat[i, :])
                gcount = np.sum(np.isin(y[notna], 'hit'))
                ngcount = np.sum(np.isin(y[notna], 'CR'))
                if gcount < 10 and ngcount < 10:
                    print(f'Go: {gcount} NoGO: {ngcount}')
                    continue
                try:
                    si = 2*(roc_auc_score(y[notna], y_hat[i, notna]) - 0.5)
                except:
                    si = np.nan
                si_vec[i] = si

            m_si_index[:, c] = si_vec

#         # save for plotting
#         if c == 1:
#             plt_cr_tensor1 =  np.nanmean(plt_cr_tensor, axis=2)
#             plt_hit_tensor1 = np.nanmean(plt_hit_tensor, axis=2)
#             yhat1 = deepcopy(y_hat)
#             y1 = deepcopy(y)
#         elif c == 3:
#             plt_cr_tensor3 =  np.nanmean(plt_cr_tensor, axis=2)
#             plt_hit_tensor3 = np.nanmean(plt_hit_tensor, axis=2)
#             yhat3 = deepcopy(y_hat)
#             y3 = deepcopy(y)

#     if plot_please:
#         counter = 0
#         for i in range(m_si_index.shape[0]):

#             cell_si = m_si_index[i, :]
#             tuning = pref_list[i]

#             if not np.isnan(cell_si[1]) and not np.isnan(cell_si[3]):

#                 if counter > 10:
#                     continue

#                 # looking at upper left quadrant
#                 if cell_si[1] > -0.2 or  cell_si[3] < 0.2:
#                     continue
#                 fig, ax = plt.subplots(2, 2, sharex='col', sharey='col')
#                 ax[0, 0].plot(plt_cr_tensor1[i,:], label='nogo', color='black')
#                 ax[0, 0].plot(plt_hit_tensor1[i,:], label='go', color=lookups.color_dict[tuning])
#                 ax[0, 0].legend()
#                 ax[0, 0].axvline(15.5, color='red', linestyle='--')
#                 ax[1, 0].plot(plt_cr_tensor3[i,:], label='nogo', color='black')
#                 ax[1, 0].plot(plt_hit_tensor3[i,:], label='go', color=lookups.color_dict[tuning])
#                 ax[1, 0].legend()
#                 ax[1, 0].axvline(15.5, color='red', linestyle='--')
#                 c1 = [lookups.color_dict[tuning] if s == 'hit' else 'black' for s in y1]
#                 ax[0, 1].scatter(np.arange(yhat1.shape[1]), yhat1[i, :], c=c1)
#                 ax[0, 1].legend(labels=['go', 'nogo'])
#                 c3 = [lookups.color_dict[tuning] if s == 'hit' else 'black' for s in y3]
#                 ax[1, 1].scatter(np.arange(yhat3.shape[1]), yhat3[i, :], c=c3)
#                 ax[1, 1].legend(labels=['go', 'nogo'])
#                 ax[0, 0].set_ylabel(f'late_learning responses\nSI={round(cell_si[1], 3)}', ha='right', rotation=0)
#                 ax[1, 0].set_ylabel(f'late_reversal responses\nSI={round(cell_si[3], 3)}', ha='right', rotation=0)
#                 ax[1, 0].set_xlabel('time from onset', ha='right', rotation=0)
#                 ax[1, 1].set_xlabel('trials', ha='right', rotation=0)
#                 ax[0, 0].set_title('average responses')
#                 ax[0, 1].set_title('trial dot product')

#                 plt.suptitle(f'{utils.meta_mouse(meta)} celln{i} {tuning}')
#                 save_file = paths.analysis_file(f'{utils.meta_mouse(meta)}_celln{i}_{tuning}.png', 'selectivity_testing/')
#                 plt.savefig(save_file, bbox_inches='tight', facecolor='white')
#                 counter += 1
#         plt.close('all')

    return m_si_index


def onset_histogram(index, cell_cats=None, cell_sorter=None, rr=9):

    # get param to define caegories in usual sort order (by TCA comp)
    cues = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']
    cats_in_order = pd.unique(cell_cats[cell_sorter])
    new_order = cats_in_order[:-1]  # last cat is -1 is cells with no cattegory (0 weight)

    _, ax = plt.subplots(rr, 4, figsize=(15,25), sharex=True, sharey='row')
    for stage_n in range(4):
        for ni, cati in enumerate(new_order):
            cvboo = cell_cats == cati
            for ci, cue in enumerate(cues):
                if ci != 0 and ni <= 2:
                    continue
                if ci != 1 and (ni > 2 and ni <= 4):
                    continue
                if ci != 2 and (ni > 4 and ni <= 7):
                    continue

                r_index = index[cvboo, stage_n]
                sns.histplot(r_index, bins=np.arange(-1, 1.1, 0.1),
                            ax=ax[ni, stage_n], color=lookups.color_dict[cue])
                ax[ni, stage_n].axvline(0, linestyle='--', color='black')
                # ignore python and numpy divide by zero warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    with np.errstate(invalid='ignore', divide='ignore'):
                        r_mean = np.nanmean(r_index)
                ax[ni, stage_n].axvline(r_mean, linestyle='--', color='red')

                if ni == rr-1:
                    ax[ni, stage_n].set_xlabel('RH index')

    # plt.suptitle(f'ONSET running modulation index', position=(0.5, 0.9), size=20)
    # plt.savefig(os.path.join(save_folder, f'on_RUNmodindex_balancestage.png'))

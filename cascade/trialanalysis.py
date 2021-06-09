import numpy as np
from numpy.core.fromnumeric import sort
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from copy import deepcopy
import os
import warnings
from tqdm import tqdm
from cascade import load, utils, lookups, sorters, paths, categorize


def build_any_mat(mouse_vec,
                  cell_vec,
                  meta_col='trialerror',
                  meta_col_vals=[0, 3, 5],
                  limit_tag=None,
                  allstage=False,
                  norm_please=True,
                  no_disengaged=False,
                  n_in_a_row=2,
                  add_broad_joint_tuning=False,
                  staging='parsed_11stage'):
    """Build an unwrapped stage matrix based on a metadata boolean vector and the ~ of that vector.

    Parameters
    ----------
    mouse_vec : numpy.ndarray or list
        Vector as long as cell numbers, that specifies mouse identity. 
    cell_vec : numpy.ndarray or list
        Vector as long as cell numbers, that specifies cell_id.
    meta_col : str
        Column name to condition on.
    meta_col_vals : list
        Values withing metadata column to filter on. Will go into pandas.DataFrame.isin().
    limit_tag : str, optional
        'onsets' or 'offsets', optionally limit to 1s baseline and 2s post-baseline, by default None
    allstage : bool, optional
        Include naive data (or limit to learning and reversal if False), by default False
    no_disengaged : bool, optional
        remove disengaged trials, sparing naive, by default False
    add_broad_joint_tuning : bool, optional
        calculated matrices across broad and joint tuning cases as well

    Returns
    -------
    numpy.ndarray 
        Array that is cells x times-&-stages x cues-&-conditions, taking a balanced average across each
        cue and condition.
    """

    # if trial history modulation is asked for by setting 'th'
    if meta_col == 'th':
        return build_th_mat(
            mouse_vec,
            cell_vec,
            limit_tag=limit_tag,
            allstage=allstage,
            norm_please=norm_please,
            no_disengaged=no_disengaged,
            add_broad_joint_tuning=add_broad_joint_tuning,
            n_in_a_row=n_in_a_row,
            staging=staging,
        )
    elif meta_col == 'pre_speed':
        return build_speed_mat(
            mouse_vec,
            cell_vec,
            limit_tag=limit_tag,
            allstage=allstage,
            norm_please=norm_please,
            no_disengaged=no_disengaged,
            add_broad_joint_tuning=add_broad_joint_tuning,
            staging=staging,
        )

    # specifiy cue order
    cues = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']
    if add_broad_joint_tuning:
        cues = [
            'becomes_unrewarded',
            'remains_unrewarded',
            'becomes_rewarded',
            'broad',
            'joint-becomes_unrewarded-remains_unrewarded',
            'joint-becomes_rewarded-remains_unrewarded',
        ]
        three_cues = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']

    # load data only for your cells of interest
    mice = np.unique(mouse_vec)
    cell_id_list = [cell_vec[mouse_vec == mi] for mi in mice]
    load_dict = load.data_filtered(mice=mice, keep_ids=cell_id_list, limit_to=limit_tag, no_disengaged=no_disengaged)

    # create unwrapped tensor stack for first condition
    full_tensor_stack = []
    for meta, tensor in zip(load_dict['meta_list'], load_dict['tensor_list']):
        meta = utils.add_reversal_mismatch_condition_to_meta(meta)
        cue_stack = []
        for cue in cues:
            # lookup_cue = lookups.lookup_mm_inv[utils.meta_mouse(meta)]
            # meta_bool = meta.initial_condition.isin([lookup_cue[cue]]).values
            if 'broad' in cue:
                meta_bool = meta.mismatch_condition.isin(cues).values
                # meta_bool = meta.initial_condition.isin([lookup_cue[s] for s in three_cues]).values
            elif 'joint' in cue:
                meta_bool = meta.mismatch_condition.isin([s for s in three_cues if s in cue]).values
                # meta_bool = meta.initial_condition.isin([lookup_cue[s] for s in [s for s in three_cues if s in cue]]).values
            else:
                meta_bool = meta.mismatch_condition.isin([cue]).values
                # meta_bool = meta.initial_condition.isin([lookup_cue[cue]]).values

            # any additional meta filtering
            go_meta_bool = meta_bool & meta[meta_col].isin(meta_col_vals).values

            stage_mean_tensor = utils.balanced_mean_per_stage(meta, tensor, meta_bool=go_meta_bool, staging=staging)
            if allstage:
                flat_cue_tensor = utils.unwrap_tensor(stage_mean_tensor[:, :, :])
            else:
                # remove naive
                flat_cue_tensor = utils.unwrap_tensor(stage_mean_tensor[:, :, 1:])
            cue_stack.append(flat_cue_tensor)
        cue_stack = np.dstack(cue_stack)
        full_tensor_stack.append(cue_stack)
    full_tensor_stack = np.vstack(full_tensor_stack)
    go_stack = deepcopy(full_tensor_stack)

    # create unwrapped tensor stack for ~ of first condition
    full_tensor_stack = []
    for meta, tensor in zip(load_dict['meta_list'], load_dict['tensor_list']):
        meta = utils.add_reversal_mismatch_condition_to_meta(meta)
        cue_stack = []
        for cue in cues:
            # lookup_cue = lookups.lookup_mm_inv[utils.meta_mouse(meta)]
            # meta_bool = meta.initial_condition.isin([lookup_cue[cue]]).values
            if 'broad' in cue:
                meta_bool = meta.mismatch_condition.isin(cues).values
                # meta_bool = meta.initial_condition.isin([lookup_cue[s] for s in three_cues]).values
            elif 'joint' in cue:
                meta_bool = meta.mismatch_condition.isin([s for s in three_cues if s in cue]).values
                # meta_bool = meta.initial_condition.isin([lookup_cue[s] for s in [s for s in three_cues if s in cue]]).values
            else:
                meta_bool = meta.mismatch_condition.isin([cue]).values
                # meta_bool = meta.initial_condition.isin([lookup_cue[cue]]).values

            # any additional meta filtering
            nogo_meta_bool = meta_bool & ~meta[meta_col].isin(meta_col_vals).values

            stage_mean_tensor = utils.balanced_mean_per_stage(meta, tensor, meta_bool=nogo_meta_bool, staging=staging)
            if allstage:
                flat_cue_tensor = utils.unwrap_tensor(stage_mean_tensor[:, :, :])
            else:
                # remove naive
                flat_cue_tensor = utils.unwrap_tensor(stage_mean_tensor[:, :, 1:])
            cue_stack.append(flat_cue_tensor)
        cue_stack = np.dstack(cue_stack)
        full_tensor_stack.append(cue_stack)
    full_tensor_stack = np.vstack(full_tensor_stack)
    nogo_stack = deepcopy(full_tensor_stack)

    full_tensor_stack = np.dstack([go_stack, nogo_stack])
    # Optionally normalize each cell to its max across conditions
    if norm_please:
        full_tensor_stack, _ = _row_norm(full_tensor_stack)

    return full_tensor_stack


def build_th_mat(mouse_vec,
                 cell_vec,
                 limit_tag=None,
                 allstage=False,
                 norm_please=True,
                 no_disengaged=False,
                 add_broad_joint_tuning=False,
                 n_in_a_row=2,
                 staging='parsed_11stage'):

    # specifiy cue order
    cues = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']
    if add_broad_joint_tuning:
        cues = [
            'becomes_unrewarded',
            'remains_unrewarded',
            'becomes_rewarded',
            'broad',
            'joint-becomes_unrewarded-remains_unrewarded',
            'joint-becomes_rewarded-remains_unrewarded',
        ]
        three_cues = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']

    # load data and filter
    mice = np.unique(mouse_vec)
    cell_id_list = [cell_vec[mouse_vec == mi] for mi in mice]
    load_dict = load.data_filtered(mice=mice, keep_ids=cell_id_list, limit_to=limit_tag, no_disengaged=no_disengaged)

    full_tensor_stack = []
    for meta, tensor in zip(load_dict['meta_list'], load_dict['tensor_list']):
        # add mismatch condition to meta to meta
        meta = utils.add_reversal_mismatch_condition_to_meta(meta)
        cue_stack = []
        for cue in cues:
            # lookup_cue = lookups.lookup_mm_inv[utils.meta_mouse(meta)]
            # meta_bool = meta.initial_condition.isin([lookup_cue[cue]]).values
            if 'broad' in cue:
                meta_bool = meta.mismatch_condition.isin(cues).values
                # meta_bool = meta.initial_condition.isin([lookup_cue[s] for s in three_cues]).values
            elif 'joint' in cue:
                meta_bool = meta.mismatch_condition.isin([s for s in three_cues if s in cue]).values
                # meta_bool = meta.initial_condition.isin([lookup_cue[s] for s in [s for s in three_cues if s in cue]]).values
            else:
                meta_bool = meta.mismatch_condition.isin([cue]).values
                # meta_bool = meta.initial_condition.isin([lookup_cue[cue]]).values

            # additional meta filtering (PREV DIFFERENT)
            go_meta_bool = meta_bool & ~(meta.prev_same_plus | meta.prev_same_minus | meta.prev_same_neutral)

            stage_mean_tensor = utils.balanced_mean_per_stage(meta, tensor, meta_bool=go_meta_bool, staging=staging)
            if allstage:
                flat_cue_tensor = utils.unwrap_tensor(stage_mean_tensor[:, :, :])
            else:
                # remove naive
                flat_cue_tensor = utils.unwrap_tensor(stage_mean_tensor[:, :, 1:])
            cue_stack.append(flat_cue_tensor)
        cue_stack = np.dstack(cue_stack)
        full_tensor_stack.append(cue_stack)
    full_tensor_stack = np.vstack(full_tensor_stack)
    go_stack = deepcopy(full_tensor_stack)

    full_tensor_stack = []
    for meta, tensor in zip(load_dict['meta_list'], load_dict['tensor_list']):

        # get vector counting how many previous trials were the same
        prev_same = (meta.prev_same_plus | meta.prev_same_minus | meta.prev_same_neutral).values
        same_in_a_row = np.zeros(len(meta))
        same_in_a_row[prev_same] = 2
        possible_double = np.diff(same_in_a_row, prepend=0) == 0
        possible_triple = np.diff(np.diff(same_in_a_row, prepend=0), prepend=0) == 0
        possible_quad = np.diff(np.diff(np.diff(same_in_a_row, prepend=0), prepend=0), prepend=0) == 0
        same_in_a_row[prev_same & possible_double] = 3
        same_in_a_row[prev_same & possible_double & possible_triple] = 4
        same_in_a_row[prev_same & possible_double & possible_triple & possible_quad] = 5

        meta = utils.add_reversal_mismatch_condition_to_meta(meta)
        cue_stack = []
        for cue in cues:
            # lookup_cue = lookups.lookup_mm_inv[utils.meta_mouse(meta)]
            # meta_bool = meta.initial_condition.isin([lookup_cue[cue]]).values
            if 'broad' in cue:
                meta_bool = meta.mismatch_condition.isin(cues).values
                # meta_bool = meta.initial_condition.isin([lookup_cue[s] for s in three_cues]).values
            elif 'joint' in cue:
                meta_bool = meta.mismatch_condition.isin([s for s in three_cues if s in cue]).values
                # meta_bool = meta.initial_condition.isin([lookup_cue[s] for s in [s for s in three_cues if s in cue]]).values
            else:
                meta_bool = meta.mismatch_condition.isin([cue]).values
                # meta_bool = meta.initial_condition.isin([lookup_cue[cue]]).values

            # additional meta filtering
            if n_in_a_row == 2:
                nogo_meta_bool = meta_bool & prev_same
            elif n_in_a_row > 2:
                nogo_meta_bool = meta_bool & (same_in_a_row >= n_in_a_row)
            else:
                raise ValueError

            stage_mean_tensor = utils.balanced_mean_per_stage(meta, tensor, meta_bool=nogo_meta_bool, staging=staging)
            if allstage:
                flat_cue_tensor = utils.unwrap_tensor(stage_mean_tensor[:, :, :])
            else:
                # remove naive
                flat_cue_tensor = utils.unwrap_tensor(stage_mean_tensor[:, :, 1:])
            cue_stack.append(flat_cue_tensor)
        cue_stack = np.dstack(cue_stack)
        full_tensor_stack.append(cue_stack)
    full_tensor_stack = np.vstack(full_tensor_stack)
    nogo_stack = deepcopy(full_tensor_stack)

    full_tensor_stack = np.dstack([go_stack, nogo_stack])
    # Optionally normalize each cell to its max across conditions
    if norm_please:
        full_tensor_stack, _ = _row_norm(full_tensor_stack)

    return full_tensor_stack


def build_speed_mat(mouse_vec,
                 cell_vec,
                 limit_tag=None,
                 allstage=False,
                 norm_please=True,
                 no_disengaged=False,
                 add_broad_joint_tuning=False,
                 staging='parsed_11stage'):

    # specifiy cue order
    cues = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']
    if add_broad_joint_tuning:
        cues = [
            'becomes_unrewarded',
            'remains_unrewarded',
            'becomes_rewarded',
            'broad',
            'joint-becomes_unrewarded-remains_unrewarded',
            'joint-becomes_rewarded-remains_unrewarded',
        ]
        three_cues = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']

    # load data and filter
    mice = np.unique(mouse_vec)
    cell_id_list = [cell_vec[mouse_vec == mi] for mi in mice]
    load_dict = load.data_filtered(mice=mice, keep_ids=cell_id_list, limit_to=limit_tag, no_disengaged=no_disengaged)

    full_tensor_stack = []
    for meta, tensor in zip(load_dict['meta_list'], load_dict['tensor_list']):
        meta = utils.add_reversal_mismatch_condition_to_meta(meta)
        cue_stack = []
        for cue in cues:
            # lookup_cue = lookups.lookup_mm_inv[utils.meta_mouse(meta)]
            # meta_bool = meta.initial_condition.isin([lookup_cue[cue]]).values
            if 'broad' in cue:
                meta_bool = meta.mismatch_condition.isin(cues).values
            elif 'joint' in cue:
                meta_bool = meta.mismatch_condition.isin([s for s in three_cues if s in cue]).values
            else:
                meta_bool = meta.mismatch_condition.isin([cue]).values

            # additional meta filtering
            go_meta_bool = meta_bool #& meta.pre_speed.gt(10)

            stage_mean_tensor = utils.balanced_mean_per_stage(
                meta, tensor, meta_bool=go_meta_bool, staging=staging,
                filter_running='high_pre_speed_only'
            )
            if allstage:
                flat_cue_tensor = utils.unwrap_tensor(stage_mean_tensor[:, :, :])
            else:
                # remove naive
                flat_cue_tensor = utils.unwrap_tensor(stage_mean_tensor[:, :, 1:])
            cue_stack.append(flat_cue_tensor)
        cue_stack = np.dstack(cue_stack)
        full_tensor_stack.append(cue_stack)
    full_tensor_stack = np.vstack(full_tensor_stack)
    go_stack = deepcopy(full_tensor_stack)

    full_tensor_stack = []
    for meta, tensor in zip(load_dict['meta_list'], load_dict['tensor_list']):
        meta = utils.add_reversal_mismatch_condition_to_meta(meta)
        cue_stack = []
        for cue in cues:
            # lookup_cue = lookups.lookup_mm_inv[utils.meta_mouse(meta)]
            # meta_bool = meta.initial_condition.isin([lookup_cue[cue]]).values
            if 'broad' in cue:
                meta_bool = meta.mismatch_condition.isin(cues).values
            elif 'joint' in cue:
                meta_bool = meta.mismatch_condition.isin([s for s in three_cues if s in cue]).values
            else:
                meta_bool = meta.mismatch_condition.isin([cue]).values

            # additional meta filtering
            nogo_meta_bool = meta_bool #& meta.pre_speed.le(4)

            stage_mean_tensor = utils.balanced_mean_per_stage(
                meta, tensor, meta_bool=nogo_meta_bool, staging=staging,
                filter_running='low_pre_speed_only'
                )
            if allstage:
                flat_cue_tensor = utils.unwrap_tensor(stage_mean_tensor[:, :, :])
            else:
                # remove naive
                flat_cue_tensor = utils.unwrap_tensor(stage_mean_tensor[:, :, 1:])
            cue_stack.append(flat_cue_tensor)
        cue_stack = np.dstack(cue_stack)
        full_tensor_stack.append(cue_stack)
    full_tensor_stack = np.vstack(full_tensor_stack)
    nogo_stack = deepcopy(full_tensor_stack)

    full_tensor_stack = np.dstack([go_stack, nogo_stack])
    # Optionally normalize each cell to its max across conditions
    if norm_please:
        full_tensor_stack, _ = _row_norm(full_tensor_stack)

    return full_tensor_stack


def build_speed_mat_shuffle(mouse_vec,
                 cell_vec,
                 limit_tag=None,
                 allstage=False,
                 norm_please=True,
                 no_disengaged=False,
                 add_broad_joint_tuning=False,
                 staging='parsed_11stage',
                 boot_n=500):

    # specifiy cue order
    cues = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']
    if add_broad_joint_tuning:
        cues = [
            'becomes_unrewarded',
            'remains_unrewarded',
            'becomes_rewarded',
            'broad',
            'joint-becomes_unrewarded-remains_unrewarded',
            'joint-becomes_rewarded-remains_unrewarded',
        ]
        three_cues = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']

    # load data and filter
    mice = np.unique(mouse_vec)
    cell_id_list = [cell_vec[mouse_vec == mi] for mi in mice]
    load_dict = load.data_filtered(mice=mice, keep_ids=cell_id_list, limit_to=limit_tag, no_disengaged=no_disengaged)

    boot_tensor_stack_list = []
    rand_rng = np.random.default_rng()
    for booti in range(boot_n):
        full_tensor_stack = []
        full_tensor_stack2 = []
        for meta, tensor in zip(load_dict['meta_list'], load_dict['tensor_list']):
            meta = utils.add_reversal_mismatch_condition_to_meta(meta)
            cue_stack = []
            cue_stack2 = []
            for cue in cues:
                # lookup_cue = lookups.lookup_mm_inv[utils.meta_mouse(meta)]
                # meta_bool = meta.initial_condition.isin([lookup_cue[cue]]).values
                if 'broad' in cue:
                    meta_bool = meta.mismatch_condition.isin(cues).values
                elif 'joint' in cue:
                    meta_bool = meta.mismatch_condition.isin([s for s in three_cues if s in cue]).values
                else:
                    meta_bool = meta.mismatch_condition.isin([cue]).values

                # additional meta filtering
                any_qualified_run = meta_bool & (meta.pre_speed.gt(10).values | meta.pre_speed.le(4).values)
                number_high_speed = np.sum(meta_bool & meta.pre_speed.gt(10))
                possible_inds = np.where(any_qualified_run)[0]
                run_inds = rand_rng.choice(possible_inds, size=number_high_speed, replace=False)
                run_boo = np.isin(np.arange(len(meta_bool)), run_inds)
                norun_inds = possible_inds[~np.isin(possible_inds, run_inds)]
                norun_boo = np.isin(np.arange(len(meta_bool)), norun_inds)
                go_meta_bool = meta_bool & run_boo
                nogo_meta_bool = meta_bool & norun_boo

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    stage_mean_tensor = utils.balanced_mean_per_stage(
                        meta, tensor, meta_bool=go_meta_bool, staging=staging,
                        # filter_running='high_pre_speed_only'
                    )
                    stage_mean_tensor2 = utils.balanced_mean_per_stage(
                        meta, tensor, meta_bool=nogo_meta_bool, staging=staging,
                        # filter_running='high_pre_speed_only'
                    )
                if allstage:
                    flat_cue_tensor = utils.unwrap_tensor(stage_mean_tensor[:, :, :])
                    flat_cue_tensor2 = utils.unwrap_tensor(stage_mean_tensor2[:, :, :])
                else:
                    # remove naive
                    flat_cue_tensor = utils.unwrap_tensor(stage_mean_tensor[:, :, 1:])
                    flat_cue_tensor2 = utils.unwrap_tensor(stage_mean_tensor2[:, :, 1:])
                cue_stack.append(flat_cue_tensor)
                cue_stack2.append(flat_cue_tensor2)
            cue_stack = np.dstack(cue_stack)
            cue_stack2 = np.dstack(cue_stack2)
            full_tensor_stack.append(cue_stack)
            full_tensor_stack2.append(cue_stack2)
        full_tensor_stack = np.vstack(full_tensor_stack)
        full_tensor_stack2 = np.vstack(full_tensor_stack2)
        full_tensor_stack = np.dstack([full_tensor_stack, full_tensor_stack2])

        # Optionally normalize each cell to its max across conditions
        if norm_please:
            full_tensor_stack, _ = _row_norm(full_tensor_stack)
        boot_tensor_stack_list.append(full_tensor_stack)

    return boot_tensor_stack_list


def build_th_mat_flat(mouse_vec,
                      cell_vec,
                      limit_tag=None,
                      allstage=False,
                      norm_please=True,
                      staging='parsed_11sage',
                      no_disengaged=False):

    # specifiy cue order
    cues = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']

    # raise NotImplementedError

    # load data and filter
    mice = np.unique(mouse_vec)
    cell_id_list = [cell_vec[mouse_vec == mi] for mi in mice]
    load_dict = load.data_filtered(mice=mice, keep_ids=cell_id_list, limit_to=limit_tag, no_disengaged=no_disengaged)

    full_tensor_stack = []
    for meta, tensor in zip(load_dict['meta_list'], load_dict['tensor_list']):
        cue_stack = []
        for cue in cues:
            lookup_cue = lookups.lookup_mm_inv[utils.meta_mouse(meta)]
            meta_bool = meta.initial_condition.isin([lookup_cue[cue]]).values

            # additional meta filtering
            go_meta_bool = meta_bool & (meta.prev_same_plus | meta.prev_same_minus | meta.prev_same_neutral)

            # flat_cue_tensor = utils.flat_balanced_mean_per_stage(meta, tensor, meta_bool=go_meta_bool, staging=staging)
            flat_cue_tensor = utils.simple_mean_per_stage(meta, tensor, meta_bool=go_meta_bool, staging=staging)
            # flat_cue_tensor = utils.balanced_mean_per_stage(meta, tensor, meta_bool=go_meta_bool, staging=staging)
            # flat_cue_tensor = np.nanmean(_row_norm(flat_cue_tensor)[0][:, 17:, :], axis=1)
            flat_cue_tensor = np.nanmean(flat_cue_tensor[:, 17:, :], axis=1)
            if not allstage and staging == 'parsed_11stage':
                # remove naive
                flat_cue_tensor = flat_cue_tensor[:, 1:]
            cue_stack.append(flat_cue_tensor)
        cue_stack = np.dstack(cue_stack)
        full_tensor_stack.append(cue_stack)
    full_tensor_stack = np.vstack(full_tensor_stack)
    go_stack = deepcopy(full_tensor_stack)

    full_tensor_stack = []
    for meta, tensor in zip(load_dict['meta_list'], load_dict['tensor_list']):
        cue_stack = []
        for cue in cues:
            lookup_cue = lookups.lookup_mm_inv[utils.meta_mouse(meta)]
            meta_bool = meta.initial_condition.isin([lookup_cue[cue]]).values

            # additional meta filtering
            nogo_meta_bool = meta_bool & ~(meta.prev_same_plus | meta.prev_same_minus | meta.prev_same_neutral)

            # flat_cue_tensor = utils.flat_balanced_mean_per_stage(meta, tensor, meta_bool=nogo_meta_bool, staging=staging)
            flat_cue_tensor = utils.simple_mean_per_stage(meta, tensor, meta_bool=nogo_meta_bool, staging=staging)
            # flat_cue_tensor = utils.balanced_mean_per_stage(meta, tensor, meta_bool=nogo_meta_bool, staging=staging)
            # flat_cue_tensor = np.nanmean(_row_norm(flat_cue_tensor)[0][:, 17:, :], axis=1)
            flat_cue_tensor = np.nanmean(flat_cue_tensor[:, 17:, :], axis=1)
            if not allstage and staging == 'parsed_11stage':
                # remove naive
                flat_cue_tensor = flat_cue_tensor[:, 1:]
            cue_stack.append(flat_cue_tensor)
        cue_stack = np.dstack(cue_stack)
        full_tensor_stack.append(cue_stack)
    full_tensor_stack = np.vstack(full_tensor_stack)
    nogo_stack = deepcopy(full_tensor_stack)

    full_tensor_stack = np.dstack([go_stack, nogo_stack])
    # Optionally normalize each cell to its max across conditions
    if norm_please:
        full_tensor_stack, _ = _row_norm(full_tensor_stack)

    return full_tensor_stack


def build_speed_mat_flat(mouse_vec,
                      cell_vec,
                      limit_tag=None,
                      allstage=False,
                      norm_please=True,
                      staging='parsed_11sage',
                      no_disengaged=False):

    # specifiy cue order
    cues = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']

    # raise NotImplementedError

    # load data and filter
    mice = np.unique(mouse_vec)
    cell_id_list = [cell_vec[mouse_vec == mi] for mi in mice]
    load_dict = load.data_filtered(mice=mice, keep_ids=cell_id_list, limit_to=limit_tag, no_disengaged=no_disengaged)

    full_tensor_stack = []
    for meta, tensor in zip(load_dict['meta_list'], load_dict['tensor_list']):
        cue_stack = []
        for cue in cues:
            lookup_cue = lookups.lookup_mm_inv[utils.meta_mouse(meta)]
            meta_bool = meta.initial_condition.isin([lookup_cue[cue]]).values

            # additional meta filtering
            go_meta_bool = meta_bool & (meta.pre_speed.ge(10))

            # flat_cue_tensor = utils.flat_balanced_mean_per_stage(meta, tensor, meta_bool=go_meta_bool, staging=staging)
            # flat_cue_tensor = utils.simple_mean_per_stage(meta, tensor, meta_bool=go_meta_bool, staging=staging)
            flat_cue_tensor = utils.balanced_mean_per_stage(meta, tensor, meta_bool=go_meta_bool, staging=staging)
            flat_cue_tensor = np.nanmean(flat_cue_tensor[:, 17:, :], axis=1)
            if not allstage and staging == 'parsed_11stage':
                # remove naive
                flat_cue_tensor = flat_cue_tensor[:, 1:]
            cue_stack.append(flat_cue_tensor)
        cue_stack = np.dstack(cue_stack)
        full_tensor_stack.append(cue_stack)
    full_tensor_stack = np.vstack(full_tensor_stack)
    go_stack = deepcopy(full_tensor_stack)

    full_tensor_stack = []
    for meta, tensor in zip(load_dict['meta_list'], load_dict['tensor_list']):
        cue_stack = []
        for cue in cues:
            lookup_cue = lookups.lookup_mm_inv[utils.meta_mouse(meta)]
            meta_bool = meta.initial_condition.isin([lookup_cue[cue]]).values

            # additional meta filtering
            nogo_meta_bool = meta_bool & meta.pre_speed.le(5)

            # flat_cue_tensor = utils.flat_balanced_mean_per_stage(meta, tensor, meta_bool=nogo_meta_bool, staging=staging)
            # flat_cue_tensor = utils.simple_mean_per_stage(meta, tensor, meta_bool=nogo_meta_bool, staging=staging)
            flat_cue_tensor = utils.balanced_mean_per_stage(meta, tensor, meta_bool=nogo_meta_bool, staging=staging)
            flat_cue_tensor = np.nanmean(flat_cue_tensor[:, 17:, :], axis=1)
            if not allstage and staging == 'parsed_11stage':
                # remove naive
                flat_cue_tensor = flat_cue_tensor[:, 1:]
            cue_stack.append(flat_cue_tensor)
        cue_stack = np.dstack(cue_stack)
        full_tensor_stack.append(cue_stack)
    full_tensor_stack = np.vstack(full_tensor_stack)
    nogo_stack = deepcopy(full_tensor_stack)

    full_tensor_stack = np.dstack([go_stack, nogo_stack])
    # Optionally normalize each cell to its max across conditions
    # if norm_please:
    #     full_tensor_stack, _ = _row_norm(full_tensor_stack)

    return full_tensor_stack


def build_cue_mat(mouse_vec, cell_vec, limit_tag=None, allstage=False, norm_please=True, no_disengaged=False, load_kws={}):
    """Build an unwrapped stage matrix by cue.

    Parameters
    ----------
    mouse_vec : numpy.ndarray or list
        Vector as long as cell numbers, that specifies mouse identity. 
    cell_vec : numpy.ndarray or list
        Vector as long as cell numbers, that specifies cell_id.
    limit_tag : str, optional
        'onsets' or 'offsets', optionally limit to 1s baseline and 2s post-baseline, by default None
    allstage : bool, optional
        Include naive data (or limit to learning and reversal if False), by default False
    no_disengaged : bool, optional
        remove disengaged trials, sparing naive, by default False

    Returns
    -------
    numpy.ndarray 
        Array that is cells x times-&-stages x cues, taking a balanced average across each
        cue.
    """

    # specifiy cue order
    cues = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']

    # load data only for your cells of interest
    mice = np.unique(mouse_vec)
    cell_id_list = [cell_vec[mouse_vec == mi] for mi in mice]
    load_dict = load.data_filtered(mice=mice,
                                   keep_ids=cell_id_list,
                                   limit_to=limit_tag,
                                   no_disengaged=no_disengaged,
                                   **load_kws)

    # create unwrapped tensor stack by cue
    full_tensor_stack = []
    for meta, tensor in zip(load_dict['meta_list'], load_dict['tensor_list']):
        cue_stack = []
        for cue in cues:
            lookup_cue = lookups.lookup_mm_inv[utils.meta_mouse(meta)]
            meta_bool = meta.initial_condition.isin([lookup_cue[cue]]).values
            stage_mean_tensor = utils.balanced_mean_per_stage(meta, tensor, meta_bool=meta_bool)
            if allstage:
                flat_cue_tensor = utils.unwrap_tensor(stage_mean_tensor[:, :, :])
            else:
                # remove naive
                flat_cue_tensor = utils.unwrap_tensor(stage_mean_tensor[:, :, 1:])
            cue_stack.append(flat_cue_tensor)
        cue_stack = np.dstack(cue_stack)
        full_tensor_stack.append(cue_stack)
    full_tensor_stack = np.vstack(full_tensor_stack)

    if norm_please:
        full_tensor_stack, _ = _row_norm(full_tensor_stack)

    return full_tensor_stack


def build_pav_mat(mouse_vec, cell_vec, limit_tag=None, allstage=False, norm_please=True, no_disengaged=False, load_kws={}):
    """Build an unwrapped stage matrix by cue.

    Parameters
    ----------
    mouse_vec : numpy.ndarray or list
        Vector as long as cell numbers, that specifies mouse identity. 
    cell_vec : numpy.ndarray or list
        Vector as long as cell numbers, that specifies cell_id.
    limit_tag : str, optional
        'onsets' or 'offsets', optionally limit to 1s baseline and 2s post-baseline, by default None
    allstage : bool, optional
        Include naive data (or limit to learning and reversal if False), by default False
    no_disengaged : bool, optional
        remove disengaged trials, sparing naive, by default False

    Returns
    -------
    numpy.ndarray 
        Array that is cells x times-&-stages x cues, taking a balanced average across each
        cue.
    """

    # specifiy cue order
    cues = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded',
            'pav_becomes_unrewarded', 'pav_becomes_rewarded']
    main_cues = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']

    # load data only for your cells of interest
    mice = np.unique(mouse_vec)
    cell_id_list = [cell_vec[mouse_vec == mi] for mi in mice]
    load_dict = load.data_filtered(mice=mice,
                                   keep_ids=cell_id_list,
                                   limit_to=limit_tag,
                                   no_disengaged=no_disengaged,
                                   **load_kws)

    # create unwrapped tensor stack by cue
    full_tensor_stack = []
    for meta, tensor in zip(load_dict['meta_list'], load_dict['tensor_list']):
        cue_stack = []
        for cue in cues:
            lookup_cue = lookups.lookup_mm_inv[utils.meta_mouse(meta)]
            if 'pav_' in cue:
                jcue = [s for s in main_cues if s in cue][0]
                meta_bool = meta.initial_condition.isin([lookup_cue[jcue]]).values
                meta_bool = meta_bool & meta.condition.isin(['pavlovian']) & meta.trialerror.isin([9])
            else:
                meta_bool = meta.initial_condition.isin([lookup_cue[cue]]).values
                meta_bool = meta_bool & ~meta.condition.isin(['pavlovian'])
            stage_mean_tensor = utils.balanced_mean_per_stage(meta, tensor, meta_bool=meta_bool)
            if allstage:
                flat_cue_tensor = utils.unwrap_tensor(stage_mean_tensor[:, :, :])
            else:
                # remove naive
                flat_cue_tensor = utils.unwrap_tensor(stage_mean_tensor[:, :, 1:])
            cue_stack.append(flat_cue_tensor)
        cue_stack = np.dstack(cue_stack)
        full_tensor_stack.append(cue_stack)
    full_tensor_stack = np.vstack(full_tensor_stack)

    if norm_please:
        full_tensor_stack, _ = _row_norm(full_tensor_stack)

    return full_tensor_stack


def build_correct_mat(mouse_vec, cell_vec, limit_tag=None, allstage=False, norm_please=True, no_disengaged=False):
    """Build an unwrapped stage matrix by cue.

    NOTE: Only uses correct trials. 

    Parameters
    ----------
    mouse_vec : numpy.ndarray or list
        Vector as long as cell numbers, that specifies mouse identity. 
    cell_vec : numpy.ndarray or list
        Vector as long as cell numbers, that specifies cell_id.
    limit_tag : str, optional
        'onsets' or 'offsets', optionally limit to 1s baseline and 2s post-baseline, by default None
    allstage : bool, optional
        Include naive data (or limit to learning and reversal if False), by default False
    no_disengaged : bool, optional
        remove disengaged trials, sparing naive, by default False

    Returns
    -------
    numpy.ndarray 
        Array that is cells x times-&-stages x cues, taking a balanced average across each
        cue.
    """

    # specifiy cue order
    cues = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']

    # load data only for your cells of interest
    mice = np.unique(mouse_vec)
    cell_id_list = [cell_vec[mouse_vec == mi] for mi in mice]
    load_dict = load.data_filtered(mice=mice, keep_ids=cell_id_list, limit_to=limit_tag, no_disengaged=no_disengaged)

    # create unwrapped tensor stack by cue
    full_tensor_stack = []
    for meta, tensor in zip(load_dict['meta_list'], load_dict['tensor_list']):
        cue_stack = []
        for cue in cues:
            lookup_cue = lookups.lookup_mm_inv[utils.meta_mouse(meta)]
            meta_bool = meta.trialerror.isin([0, 2, 4]).values | meta.learning_state.isin(['naive']).values
            meta_bool = meta_bool & meta.initial_condition.isin([lookup_cue[cue]]).values
            stage_mean_tensor = utils.balanced_mean_per_stage(meta, tensor, meta_bool=meta_bool)
            if allstage:
                flat_cue_tensor = utils.unwrap_tensor(stage_mean_tensor[:, :, :])
            else:
                # remove naive
                flat_cue_tensor = utils.unwrap_tensor(stage_mean_tensor[:, :, 1:])
            cue_stack.append(flat_cue_tensor)
        cue_stack = np.dstack(cue_stack)
        full_tensor_stack.append(cue_stack)
    full_tensor_stack = np.vstack(full_tensor_stack)

    if norm_please:
        full_tensor_stack, _ = _row_norm(full_tensor_stack)

    return full_tensor_stack


def build_any_trace(mouse_vec,
                    cell_vec,
                    cell_cats,
                    meta_col='trialerror',
                    meta_col_vals=[0, 3, 5],
                    limit_tag=None,
                    allstage=False,
                    norm_please=True,
                    sem_over='cells',
                    no_disengaged=False,
                    staging='parsed_11stage'):
    """Build average unwrapped traces and SEM for groups of cells.

    Parameters
    ----------
    mouse_vec : numpy.ndarray or list
        Vector as long as cell numbers, that specifies mouse identity. 
    cell_vec : numpy.ndarray or list
        Vector as long as cell numbers, that specifies cell_id.
    cell_cats : numpy.ndarray or list
        Categories as integers for each cell. 
    meta_col : str
        Column name to condition on.
    meta_col_vals : list
        Values withing metadata column to filter on. Will go into pandas.DataFrame.isin().
    limit_tag : str, optional
        'onsets' or 'offsets', optionally limit to 1s baseline and 2s post-baseline, by default None
    allstage : bool, optional
        Include naive data (or limit to learning and reversal if False), by default False
    sem_over : str, optional
        Take sem over 'cells' or over 'mice', by default 'cells'
    no_disengaged : bool, optional
        remove disengaged trials, sparing naive, by default False

    Returns
    -------
    numpy.ndarray 
        Array that is cells x times-&-stages x cues-&-conditions, taking a balanced average across each
        cue and condition.
    """

    if meta_col == 'th':
        full_tensor_stack = build_th_mat(mouse_vec,
                                         cell_vec,
                                         limit_tag=limit_tag,
                                         allstage=allstage,
                                         norm_please=norm_please,
                                         no_disengaged=no_disengaged,
                                         staging=staging)
    else:
        full_tensor_stack = build_any_mat(mouse_vec,
                                          cell_vec,
                                          meta_col=meta_col,
                                          meta_col_vals=meta_col_vals,
                                          limit_tag=limit_tag,
                                          allstage=allstage,
                                          norm_please=norm_please,
                                          no_disengaged=no_disengaged,
                                          staging=staging)
    if sem_over == 'cells':
        avg_stack, sem_stack = _average_sem_over_cell_cats(full_tensor_stack, cell_cats)
    elif sem_over == 'mice':
        avg_stack, sem_stack = _average_sem_over_mice_cats(full_tensor_stack, cell_cats, mouse_vec)
    else:
        raise ValueError

    return avg_stack, sem_stack


def unwrapped_heatmaps_2cond(folder_name,
                             name1='go',
                             name2='nogo',
                             meta_col='trialerror',
                             meta_col_vals=[0, 3, 5],
                             no_disengaged=False,
                             all_stages=False,
                             save_please=False,
                             forced_order=True):
    """Generate mutliple heatmaps with multiple sorts for cells averaged across stages for subsets of trials.

    Parameters
    ----------
    folder_name : str
        Name of subfolder for save.
    name1 : str, optional
        Name of first condition, the first 3 slices of the unwrapped matrix (depth), by default 'go'
    name2 : str, optional
        Name of second condition, the last 3 slices of the unwrapped matrix (depth), ~name1, by default 'nogo'
    meta_col : str, optional
        Columns in metadata pandas.DataFrame to use for filtering trials, by default 'trialerror'
    meta_col_vals : list, optional
        Values in meta_col to use to define name1 condition, name2 values are set by taking ~ of name1, by default [0, 3, 5]
    no_disengaged : bool, optional
        remove disengaged trials, sparing naive, by default False

    Raises
    ------
    ValueError
        If an unwrapped TCA model is called for that is not '_on' or '_off' containing.
    ValueError
        If a cascade.sorter is called using a str that does not exist. 
    """

    # load data
    ensemble = np.load(paths.analysis_file('tca_ensemble_v4i10_noT0_20210307.npy', 'tca_dfs'), allow_pickle=True).item()
    data_dict = np.load(paths.analysis_file('input_data_v4i10_noT0_20210307.npy', 'tca_dfs'), allow_pickle=True).item()

    # get sort order for data
    sort_ensembles = {}
    cell_orders = {}
    tune_orders = {}
    for k, v in ensemble.items():
        sort_ensembles[k], cell_orders[k], tune_orders[k] = utils.sort_and_rescale_factors(v)

    # get all versions of models
    models = ['v4i10_norm_on_noT0', 'v4i10_norm_off_noT0']
    if all_stages:
        mod_w_naive = [s + '_allstages' for s in models]
        models = models + mod_w_naive

    # cues to use in order
    cues = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']

    # perform calculations
    # ------------------------------------------------------------------------------------
    for mod in tqdm(models, total=len(models), desc='Plotting unwrapped heatmaps'):
        for sort_by in ['comprun']: # 'complick', 'comppupil'
            # for sort_by in ['unwrappedTCAsort', 'mouseunsort', 'runcorrsortnobd', 'runcorrsort', 'mousecuesort']:

            # set up save location
            if no_disengaged:
                save_folder = paths.analysis_dir(f'figures/figure_3/{folder_name}_nodis')
            else:
                save_folder = paths.analysis_dir(f'figures/figure_3/{folder_name}')

            # set rank and TCA model to always be norm models for sorting
            if '_on' in mod:
                rr = 9
                factors = ensemble['v4i10_norm_on_noT0'].results[rr][0].factors
                cell_cats = categorize.best_comp_cats(factors)
                cell_sorter = cell_orders['v4i10_norm_on_noT0'][rr - 1]
                mouse_vec = data_dict['v4i10_on_mouse_noT0']
                cell_vec = data_dict['v4i10_on_cell_noT0']
                limit_tag = 'onsets'
            elif '_off' in mod:
                rr = 8
                factors = ensemble['v4i10_norm_off_noT0'].results[rr][0].factors
                cell_cats = categorize.best_comp_cats(factors)
                cell_sorter = cell_orders['v4i10_norm_off_noT0'][rr - 1]
                mouse_vec = data_dict['v4i10_off_mouse_noT0']
                cell_vec = data_dict['v4i10_off_cell_noT0']
                limit_tag = 'offsets'
                if 'runcorr' in sort_by:
                    continue
            else:
                raise ValueError

            # build your stack of heatmaps
            if meta_col == 'pre_speed' or sort_by == 'comprun': # Speed in precalculated matrices is based on pre_speed
                name1 = 'fast'
                name2 = 'slow'
                if '_on' in mod:
                    mat2ds = data_dict['v4i10_speed_norm_on_noT0']
                elif '_off' in mod:
                    mat2ds = data_dict['v4i10_speed_norm_off_noT0']
                else:
                    raise ValueError
            # elif sort_by == 'complick':
            #     mat2ds = build_any_mat(mouse_vec,
            #                            cell_vec,
            #                            limit_tag=limit_tag,
            #                            allstage=True if '_allstages' in mod else False,
            #                            meta_col=meta_col,
            #                            meta_col_vals=meta_col_vals,
            #                            no_disengaged=no_disengaged)
            # elif sort_by == 'comppupil':
            #     mat2ds = build_any_mat(mouse_vec,
            #                            cell_vec,
            #                            limit_tag=limit_tag,
            #                            allstage=True if '_allstages' in mod else False,
            #                            meta_col=meta_col,
            #                            meta_col_vals=meta_col_vals,
            #                            no_disengaged=no_disengaged)
            else:
                mat2ds = build_any_mat(mouse_vec,
                                       cell_vec,
                                       limit_tag=limit_tag,
                                       allstage=True if '_allstages' in mod else False,
                                       meta_col=meta_col,
                                       meta_col_vals=meta_col_vals,
                                       no_disengaged=no_disengaged)
            # force order of columns to be bec_un, bec_rew, remains_un
            if forced_order:
                mat2ds = mat2ds[:,:, [0, 2, 1, 3, 5, 4]]

            if sort_by == 'mouseunsort':
                cell_sorter = np.arange(len(cell_sorter), dtype=int)  # keep in order
            elif sort_by == 'mousecuesort':
                cell_sorter = sorters.sort_by_cue_mouse(mat2ds, mouse_vec)
            elif sort_by == 'cuesort':
                cell_sorter = sorters.sort_by_cue_peak(mat2ds, mouse_vec)
            elif sort_by == 'runcorrsort':
                cell_sorter = sorters.run_corr_sort(mouse_vec, cell_vec, data_dict, mod, stim_or_baseline_corr='stim')
            elif sort_by == 'unwrappedTCAsort':
                cell_sorter = sorters.pick_comp_order(cell_cats, cell_sorter)
            elif sort_by == 'comprun':
                cell_sorter = sorters.pick_comp_order_plus_bhv_mod(
                    cell_cats, cell_sorter, bhv_type='speed', bhv_baseline_or_stim='baseline'
                    )
            elif sort_by == 'complick':
                cell_sorter = sorters.pick_comp_order_plus_bhv_mod(
                    cell_cats, cell_sorter, bhv_type='lick', bhv_baseline_or_stim='baseline'
                    ) #fix_onset_bhv=True
            elif sort_by == 'comppupil':
                cell_sorter = sorters.pick_comp_order_plus_bhv_mod(
                    cell_cats, cell_sorter, bhv_type='pupil', bhv_baseline_or_stim='baseline'
                    ) # fix_onset_bhv=True
            elif sort_by == 'runcorrsortnobd':
                cell_sorter = sorters.run_corr_sort_nobroad(mouse_vec,
                                                            cell_vec,
                                                            cell_cats,
                                                            cell_sorter,
                                                            data_dict,
                                                            mod,
                                                            stim_or_baseline_corr='stim')
            else:
                raise ValueError

            # remap mouse vector for color axis
            mouse_mapper = {k: c for c, k in enumerate(np.unique(mouse_vec))}
            number_mouse_mat = np.array([mouse_mapper[s] for s in mouse_vec])
            number_comp_mat = np.array([s + len(np.unique(mouse_vec)) for s in cell_cats])
            cmap1 = sns.color_palette('muted', len(np.unique(mouse_vec)))
            cmap2 = sns.color_palette('Set3', rr)
            cmap = cmap1 + cmap2

            # keep track of units for plotting
            if '_norm' in mod:
                clabel = 'normalized \u0394F/F'
            elif '_scale' in mod:
                clabel = '\u0394F/F (scaled z-score)'
            else:
                clabel = '\u0394F/F (z-score)'

            # pick your training stage
            if '_allstages' in mod:
                stages = lookups.staging['parsed_11stage_T']
            else:
                stages = lookups.staging['parsed_11stage_T'][1:]

            # plot heatmap
            speed_fast_or_slow = name1
            ax = []
            fig = plt.figure(figsize=(30, 15))
            gs = fig.add_gridspec(100, 110)
            ax.append(fig.add_subplot(gs[:, 2:5]))
            ax.append(fig.add_subplot(gs[:, 10:38]))
            ax.append(fig.add_subplot(gs[:, 40:68]))
            ax.append(fig.add_subplot(gs[:, 70:98]))
            ax.append(fig.add_subplot(gs[:30, 105:108]))

            # plot "categorical" heatmap using defined color mappings
            if sort_by in ['unwrappedTCAsort', 'comprun', 'comppupil', 'complick'] or 'runcorr' in sort_by:
                color_vecs = np.concatenate([number_mouse_mat[cell_sorter, None], number_comp_mat[cell_sorter, None]], axis=1)
                if '_on' in mod or '_off' in mod:
                    fix_cmap = sns.color_palette('Set3', 9) #cas.lookups.cmap_fixed_sort_rank9_onset
                    # fix_cmap + [(0,0,0,0)]
                    just_comps = np.array(cell_cats[cell_sorter, None], dtype=float)
                    just_comps[just_comps == -1] = np.nan
                    sns.heatmap(just_comps, cmap=fix_cmap, ax=ax[0], cbar=False)
                    ax[0].set_xticks([0.5])
                    ax[0].set_xticklabels(['Component'], rotation=45, ha='right', size=18)
                    ax[0].set_yticks([])
                else:
                    sns.heatmap(color_vecs, cmap=cmap, ax=ax[0], cbar=False)
                    ax[0].set_xticks([0.5, 1.5])
                    ax[0].set_xticklabels(['Mouse', 'Component'], rotation=45, ha='right', size=18)
            elif 'mouse' in sort_by:
                color_vecs = np.concatenate([number_mouse_mat[cell_sorter, None], number_comp_mat[cell_sorter, None]],
                                            axis=1)
                sns.heatmap(color_vecs, cmap=cmap, ax=ax[0], cbar=False)
                ax[0].set_xticks([0.5, 1.5])
                ax[0].set_xticklabels(['mouse', 'component'], rotation=45, ha='right', size=18)
            else:
                sns.heatmap(number_mouse_mat[cell_sorter, None], cmap=cmap1, ax=ax[0], cbar=False)
                ax[0].set_xticklabels(['mouse'], rotation=45, ha='right', size=18)
            ax[0].set_yticklabels([])

            if '_norm' in mod:
                vmax = 1
            else:
                vmax = None

            for i in range(1, 4):
                ii = deepcopy(i)
                if speed_fast_or_slow == name2:
                    ii = i + 3
                if i == 3:
                    g = sns.heatmap(mat2ds[cell_sorter, :, ii - 1],
                                    ax=ax[i],
                                    center=0,
                                    vmax=vmax,
                                    vmin=-0.5,
                                    cmap='vlag',
                                    cbar_ax=ax[4],
                                    cbar_kws={'label': clabel})
                    cbar = g.collections[0].colorbar
                    cbar.set_label(clabel, size=16)
                else:
                    g = sns.heatmap(mat2ds[cell_sorter, :, ii - 1],
                                    ax=ax[i],
                                    center=0,
                                    vmax=vmax,
                                    vmin=-0.5,
                                    cmap='vlag',
                                    cbar=False)
                g.set_facecolor('#c5c5c5')
                ax[i].set_title(f'initial cue: {cues[i-1]}\n', size=20)
                stim_starts = [15.5 + 47 * s for s in np.arange(len(stages))]
                stim_labels = [f'0\n\n{s}' if c % 2 == 0 else f'0\n{s}' for c, s in enumerate(stages)]
                ax[i].set_xticks(stim_starts)
                ax[i].set_xticklabels(stim_labels, rotation=0)
                if i == 1:
                    ax[i].set_ylabel('cell number', size=18)
                ax[i].set_xlabel('\ntime from stimulus onset (sec)', size=18)
                if i > 1:
                    ax[i].set_yticks([])
            plt.suptitle(f'trialerror: {speed_fast_or_slow} {sort_by} {mod}', position=(0.5, 0.98), size=20)
            if save_please:
                plt.savefig(os.path.join(save_folder, f'{mod}_{sort_by}_{speed_fast_or_slow}_rank{rr}_heatmap.png'),
                        bbox_inches='tight')
            #         plt.savefig(os.path.join(save_folder, f'{mod}_{sort_by}_rank{rr}_heatmap.png'), bbox_inches='tight', dpi=300)

            # plot heatmap
            speed_fast_or_slow = name2
            ax = []
            fig = plt.figure(figsize=(30, 15))
            gs = fig.add_gridspec(100, 110)
            ax.append(fig.add_subplot(gs[:, 2:5]))
            ax.append(fig.add_subplot(gs[:, 10:38]))
            ax.append(fig.add_subplot(gs[:, 40:68]))
            ax.append(fig.add_subplot(gs[:, 70:98]))
            ax.append(fig.add_subplot(gs[:30, 105:108]))

            # plot "categorical" heatmap using defined color mappings
            if sort_by in ['unwrappedTCAsort', 'comprun', 'comppupil', 'complick'] or 'runcorr' in sort_by:
                color_vecs = np.concatenate([number_mouse_mat[cell_sorter, None], number_comp_mat[cell_sorter, None]], axis=1)
                if '_on' in mod or '_off' in mod:
                    fix_cmap = sns.color_palette('Set3', 9) #cas.lookups.cmap_fixed_sort_rank9_onset
                    # fix_cmap + [(0,0,0,0)]
                    just_comps = np.array(cell_cats[cell_sorter, None], dtype=float)
                    just_comps[just_comps == -1] = np.nan
                    sns.heatmap(just_comps, cmap=fix_cmap, ax=ax[0], cbar=False)
                    ax[0].set_xticks([0.5])
                    ax[0].set_xticklabels(['Component'], rotation=45, ha='right', size=18)
                    ax[0].set_yticks([])
                else:
                    sns.heatmap(color_vecs, cmap=cmap, ax=ax[0], cbar=False)
                    ax[0].set_xticks([0.5, 1.5])
                    ax[0].set_xticklabels(['Mouse', 'Component'], rotation=45, ha='right', size=18)
            elif 'mouse' in sort_by:
                color_vecs = np.concatenate([number_mouse_mat[cell_sorter, None], number_comp_mat[cell_sorter, None]],
                                            axis=1)
                sns.heatmap(color_vecs, cmap=cmap, ax=ax[0], cbar=False)
                ax[0].set_xticks([0.5, 1.5])
                ax[0].set_xticklabels(['mouse', 'component'], rotation=45, ha='right', size=18)
            else:
                sns.heatmap(number_mouse_mat[cell_sorter, None], cmap=cmap1, ax=ax[0], cbar=False)
                ax[0].set_xticklabels(['mouse'], rotation=45, ha='right', size=18)
            ax[0].set_yticklabels([])
            # ax[0].set_ylabel('cell number', size=14)

            if '_norm' in mod:
                vmax = 1
            else:
                vmax = None

            for i in range(1, 4):
                ii = deepcopy(i)
                if speed_fast_or_slow == name2:
                    ii = i + 3
                if i == 3:
                    g = sns.heatmap(mat2ds[cell_sorter, :, ii - 1],
                                    ax=ax[i],
                                    center=0,
                                    vmax=vmax,
                                    vmin=-0.5,
                                    cmap='vlag',
                                    cbar_ax=ax[4],
                                    cbar_kws={'label': clabel})
                    cbar = g.collections[0].colorbar
                    cbar.set_label(clabel, size=16)
                else:
                    g = sns.heatmap(mat2ds[cell_sorter, :, ii - 1],
                                    ax=ax[i],
                                    center=0,
                                    vmax=vmax,
                                    vmin=-0.5,
                                    cmap='vlag',
                                    cbar=False)
                g.set_facecolor('#c5c5c5')
                ax[i].set_title(f'initial cue: {cues[i-1]}\n', size=20)
                stim_starts = [15.5 + 47 * s for s in np.arange(len(stages))]
                stim_labels = [f'0\n\n{s}' if c % 2 == 0 else f'0\n{s}' for c, s in enumerate(stages)]
                ax[i].set_xticks(stim_starts)
                ax[i].set_xticklabels(stim_labels, rotation=0)
                if i == 1:
                    ax[i].set_ylabel('cell number', size=18)
                ax[i].set_xlabel('\ntime from stimulus onset (sec)', size=18)
                if i > 1:
                    ax[i].set_yticks([])
            plt.suptitle(f'trialerror: {speed_fast_or_slow} {sort_by} {mod}', position=(0.5, 0.98), size=20)
            if save_please:
                plt.savefig(os.path.join(save_folder, f'{mod}_{sort_by}_{speed_fast_or_slow}_rank{rr}_heatmap.png'),
                        bbox_inches='tight')


def unwrapped_heatmaps_1cond(folder_name='cue_unwrapped', name1='all', no_disengaged=False, forced_order=True):
    """Generate multiple sort ordered heatmaps of the standard set of cells and reveral (n=7) mice.

    Parameters
    ----------
    folder_name : str, optional
        Name of folder in analysis folder, under tca_dfs/modulation_heatmaps/, by default 'cue_unwrapped'
    name1 : str, optional
        Name of condition of plotting for consistency of naming with other plots, by default 'all'
    no_disengaged : bool, optional
        remove disengaged trials, sparing naive, by default False

    Raises
    ------
    ValueError
        If an unwrapped TCA model is called for that is not '_on' or '_off' containing.
    ValueError
        If a cascade.sorter is called using a str that does not exist. 
    """

    # load data
    ensemble = np.load(paths.analysis_file('tca_ensemble_v4i10_noT0_20210307.npy', 'tca_dfs'), allow_pickle=True).item()
    data_dict = np.load(paths.analysis_file('input_data_v4i10_noT0_20210307.npy', 'tca_dfs'), allow_pickle=True).item()

    # get sort order for data
    sort_ensembles = {}
    cell_orders = {}
    tune_orders = {}
    for k, v in ensemble.items():
        sort_ensembles[k], cell_orders[k], tune_orders[k] = utils.sort_and_rescale_factors(v)

    # get all versions of models
    models = ['v4i10_norm_on_noT0', 'v4i10_norm_off_noT0']
    mod_w_naive = [s + '_allstages' for s in models]
    models = models + mod_w_naive

    # cues to use in order
    cues = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']

    # perform calculations
    # ------------------------------------------------------------------------------------
    for mod in tqdm(models, total=len(models), desc='Plotting unwrapped heatmaps'):
        for sort_by in ['runcorrsortnobd', 'unwrappedTCAsort', 'mouseunsort', 'runcorrsort', 'mousecuesort']:

            # set up save location
            if no_disengaged:
                save_folder = paths.analysis_dir(f'figures/figure_3/{folder_name}_nodis')
            else:
                save_folder = paths.analysis_dir(f'figures/figure_3/{folder_name}')

            # set rank and TCA model to always be norm models for sorting
            if '_on' in mod:
                rr = 9
                factors = ensemble['v4i10_norm_on_noT0'].results[rr][0].factors
                cell_cats = categorize.best_comp_cats(factors)
                cell_sorter = cell_orders['v4i10_norm_on_noT0'][rr - 1]
                mouse_vec = data_dict['v4i10_on_mouse_noT0']
                cell_vec = data_dict['v4i10_on_cell_noT0']
                limit_tag = 'onsets'
            elif '_off' in mod:
                rr = 8
                factors = ensemble['v4i10_norm_off_noT0'].results[rr][0].factors
                cell_cats = categorize.best_comp_cats(factors)
                cell_sorter = cell_orders['v4i10_norm_off_noT0'][rr - 1]
                mouse_vec = data_dict['v4i10_off_mouse_noT0']
                cell_vec = data_dict['v4i10_off_cell_noT0']
                limit_tag = 'offsets'
                if 'runcorr' in sort_by:
                    continue
            else:
                raise ValueError

            # build your stack of heatmaps
            mat2ds = build_cue_mat(mouse_vec,
                                   cell_vec,
                                   limit_tag=limit_tag,
                                   allstage=True if '_allstages' in mod else False,
                                   no_disengaged=no_disengaged)
            # force order of columns to be bec_un, bec_rew, remains_un
            if forced_order:
                mat2ds = mat2ds[:,:, [0, 2, 1]]

            if sort_by == 'mouseunsort':
                cell_sorter = np.arange(len(cell_sorter), dtype=int)  # keep in order
            elif sort_by == 'mousecuesort':
                cell_sorter = sorters.sort_by_cue_mouse(mat2ds, mouse_vec)
            elif sort_by == 'cuesort':
                cell_sorter = sorters.sort_by_cue_peak(mat2ds, mouse_vec)
            elif sort_by == 'runcorrsort':
                cell_sorter = sorters.run_corr_sort(mouse_vec, cell_vec, data_dict, mod, stim_or_baseline_corr='stim')
            elif sort_by == 'unwrappedTCAsort':
                cell_sorter = sorters.pick_comp_order(cell_cats, cell_sorter)
            elif sort_by == 'comprun':
                cell_sorter = sorters.pick_comp_order_plus_bhv_mod(
                    cell_cats, cell_sorter, bhv_type='speed', bhv_baseline_or_stim='baseline'
                    )
            elif sort_by == 'runcorrsortnobd':
                cell_sorter = sorters.run_corr_sort_nobroad(mouse_vec,
                                                            cell_vec,
                                                            cell_cats,
                                                            cell_sorter,
                                                            data_dict,
                                                            mod,
                                                            stim_or_baseline_corr='stim')
            else:
                raise ValueError

            # remap mouse vector for color axis
            mouse_mapper = {k: c for c, k in enumerate(np.unique(mouse_vec))}
            number_mouse_mat = np.array([mouse_mapper[s] for s in mouse_vec])
            number_comp_mat = np.array([s + len(np.unique(mouse_vec)) for s in cell_cats])
            cmap1 = sns.color_palette('muted', len(np.unique(mouse_vec)))
            cmap2 = sns.color_palette('Set3', rr)
            cmap = cmap1 + cmap2

            # keep track of units for plotting
            if '_norm' in mod:
                clabel = 'normalized \u0394F/F'
            elif '_scale' in mod:
                clabel = '\u0394F/F (scaled z-score)'
            else:
                clabel = '\u0394F/F (z-score)'

            # pick your training stage
            if '_allstages' in mod:
                stages = lookups.staging['parsed_11stage_T']
            else:
                stages = lookups.staging['parsed_11stage_T'][1:]

            # plot heatmap
            ax = []
            fig = plt.figure(figsize=(30, 15))
            gs = fig.add_gridspec(100, 110)
            ax.append(fig.add_subplot(gs[:, 2:5]))
            ax.append(fig.add_subplot(gs[:, 10:38]))
            ax.append(fig.add_subplot(gs[:, 40:68]))
            ax.append(fig.add_subplot(gs[:, 70:98]))
            ax.append(fig.add_subplot(gs[:30, 105:108]))

            # plot "categorical" heatmap using defined color mappings
            if sort_by in ['unwrappedTCAsort', 'comprun', 'comppupil', 'complick'] or 'runcorr' in sort_by:
                color_vecs = np.concatenate([number_mouse_mat[cell_sorter, None], number_comp_mat[cell_sorter, None]], axis=1)
                if '_on' in mod:
                    fix_cmap = sns.color_palette('Set3', 9) #cas.lookups.cmap_fixed_sort_rank9_onset
                    # fix_cmap + [(0,0,0,0)]
                    just_comps = np.array(cell_cats[cell_sorter, None], dtype=float)
                    just_comps[just_comps == -1] = np.nan
                    sns.heatmap(just_comps, cmap=fix_cmap, ax=ax[0], cbar=False)
                    ax[0].set_xticks([0.5])
                    ax[0].set_xticklabels(['Component'], rotation=45, ha='right', size=18)
                    ax[0].set_yticks([])
                else:
                    sns.heatmap(color_vecs, cmap=cmap, ax=ax[0], cbar=False)
                    ax[0].set_xticks([0.5, 1.5])
                    ax[0].set_xticklabels(['Mouse', 'Component'], rotation=45, ha='right', size=18)
            elif 'mouse' in sort_by:
                color_vecs = np.concatenate([number_mouse_mat[cell_sorter, None], number_comp_mat[cell_sorter, None]],
                                            axis=1)
                sns.heatmap(color_vecs, cmap=cmap, ax=ax[0], cbar=False)
                ax[0].set_xticks([0.5, 1.5])
                ax[0].set_xticklabels(['mouse', 'component'], rotation=45, ha='right', size=18)
            else:
                sns.heatmap(number_mouse_mat[cell_sorter, None], cmap=cmap1, ax=ax[0], cbar=False)
                ax[0].set_xticklabels(['mouse'], rotation=45, ha='right', size=18)
            ax[0].set_yticklabels([])

            if '_norm' in mod:
                vmax = 1
            else:
                vmax = None

            for i in range(1, 4):
                ii = deepcopy(i)
                if i == 3:
                    g = sns.heatmap(mat2ds[cell_sorter, :, ii - 1],
                                    ax=ax[i],
                                    center=0,
                                    vmax=vmax,
                                    vmin=-0.5,
                                    cmap='vlag',
                                    cbar_ax=ax[4],
                                    cbar_kws={'label': clabel})
                    cbar = g.collections[0].colorbar
                    cbar.set_label(clabel, size=16)
                else:
                    g = sns.heatmap(mat2ds[cell_sorter, :, ii - 1],
                                    ax=ax[i],
                                    center=0,
                                    vmax=vmax,
                                    vmin=-0.5,
                                    cmap='vlag',
                                    cbar=False)
                g.set_facecolor('#c5c5c5')
                ax[i].set_title(f'initial cue: {cues[i-1]}\n', size=20)
                stim_starts = [15.5 + 47 * s for s in np.arange(len(stages))]
                stim_labels = [f'0\n\n{s}' if c % 2 == 0 else f'0\n{s}' for c, s in enumerate(stages)]
                ax[i].set_xticks(stim_starts)
                ax[i].set_xticklabels(stim_labels, rotation=0)
                if i == 1:
                    ax[i].set_ylabel('cell number', size=18)
                ax[i].set_xlabel('\ntime from stimulus onset (sec)', size=18)
                if i > 1:
                    ax[i].set_yticks([])
            plt.suptitle(f'cue: {name1} {sort_by} {mod}', position=(0.5, 0.98), size=20)
            plt.savefig(os.path.join(save_folder, f'{mod}_{sort_by}_{name1}_rank{rr}_heatmap.png'), bbox_inches='tight')
            #         plt.savefig(os.path.join(save_folder, f'{mod}_{sort_by}_rank{rr}_heatmap.png'), bbox_inches='tight', dpi=300)


def _row_norm(any_mega_tensor_flat):
    """
    Normalize across a row. Normalize each cell to its max across conditions/cues.
    """
    cell_max = np.nanmax(np.nanmax(any_mega_tensor_flat, axis=1), axis=1)
    any_mega_tensor_flat_norm = any_mega_tensor_flat / cell_max[:, None, None]

    return any_mega_tensor_flat_norm, cell_max


def _average_sem_over_cell_cats(tensor_stack, cell_cats, shared_cat=None):
    """Helper function to generate averages and SEM for categories of cells. SEM over cells.

    Parameters
    ----------
    tensor_stack : numpy.ndarray
        Array of cells x times-&-stages x tuning/conditition/cue.
    cell_cats : list
        List of categores as integers that a cell belongs to.
    shared_cat : int or list
        Category in cell_cats to use in all calculations. Must be negative to prevent it being incuded as
        a stand alone category as well. i.e., a cell category of -2.

    Returns
    -------
    Two numpy arrays. 
        Numpy array of average traces and SEM.
    """

    all_cats = np.unique(cell_cats)
    all_cats = all_cats[all_cats >= 0]  # negatives are unassigned cells

    # optionally us a subset of cats in all calcutions
    if shared_cat is not None:
        if not isinstance(shared_cat, list):
            if isinstance(shared_cat, int):
                shared_cat = [shared_cat]
            else:
                shared_cat = list(shared_cat)
        assert all([s < 0 for s in shared_cat])

    avg_ten = np.zeros((len(all_cats), tensor_stack.shape[1], tensor_stack.shape[2])) + np.nan
    sem_ten = np.zeros((len(all_cats), tensor_stack.shape[1], tensor_stack.shape[2])) + np.nan

    for cc, cat in enumerate(all_cats):

        # cat_vec = cell_cats == cat
        cat_vec = np.isin(cell_cats, cat) | np.isin(cell_cats, shared_cat)
        avg_ten[cc, :, :] = np.nanmean(tensor_stack[cat_vec, :, :], axis=0)
        sem_ten[cc, :, :] = np.nanstd(tensor_stack[cat_vec, :, :], axis=0) / np.sqrt(
            np.sum(~np.isnan(tensor_stack[cat_vec, :, :]), axis=0))

    return avg_ten, sem_ten


def _average_sem_over_mice_cats(tensor_stack, cell_cats, mouse_vec, shared_cat=None):
    """Helper function to generate averages and SEM for categories of cells. SEM over mice.

    Parameters
    ----------
    tensor_stack : numpy.ndarray
        Array of cells x times-&-stages x tuning/conditition/cue.
    cell_cats : list
        List of categores as integers that a cell belongs to. 
    mouse_vec : numpy.ndarray or list
        Vector as long as cell numbers, that specifies mouse identity. 
    shared_cat : int or list
        Category in cell_cats to use in all calculations. Must be negative to prevent it being incuded as
        a stand alone category as well. i.e., a cell category of -2.

    Returns
    -------
    Two numpy arrays. 
        Numpy array of average traces and SEM.
    """

    all_mice = np.unique(mouse_vec)
    all_cats = np.unique(cell_cats)
    all_cats = all_cats[all_cats >= 0]  # negatives are unassigned cells

    # optionally us a subset of cats in all calcutions
    if shared_cat is not None:
        if not isinstance(shared_cat, list):
            if isinstance(shared_cat, int):
                shared_cat = [shared_cat]
            else:
                shared_cat = list(shared_cat)
        assert all([s < 0 for s in shared_cat])

    avg_ten = np.zeros((len(all_cats), tensor_stack.shape[1], tensor_stack.shape[2])) + np.nan
    sem_ten = np.zeros((len(all_cats), tensor_stack.shape[1], tensor_stack.shape[2])) + np.nan

    for cc, cat in enumerate(all_cats):

        # cat_vec = cell_cats == cat
        cat_vec = np.isin(cell_cats, cat) | np.isin(cell_cats, shared_cat)
        cat_avg = np.zeros((len(all_mice), tensor_stack.shape[1], tensor_stack.shape[2])) + np.nan

        for mi, mouse in enumerate(all_mice):

            mouse_bool = mouse_vec == mouse
            cat_avg[mi, :, :] = np.nanmean(tensor_stack[cat_vec & mouse_bool, :, :], axis=0)

        avg_ten[cc, :, :] = np.nanmean(cat_avg, axis=0)
        sem_ten[cc, :, :] = np.nanstd(cat_avg, axis=0) / np.sqrt(np.sum(~np.isnan(cat_avg), axis=0))

    return avg_ten, sem_ten

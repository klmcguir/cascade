import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy import optimize

from . import utils, lookups

# TODO
#  1. split your data by center of mass, and offset based splits
#  2. then calculate bias
#  3. then stim history/reward history/punish history

def best_comp_cats(factors, cell_frac_thresh=0):

    # rescale factors so most of the var is in cell_weights
    # factors = ensemble[mod].results[rr][0].factors
    scaled_factors = utils.rescale_factors(factors)

    # get cell threshold vector
    frac_weight = scaled_factors[0].T/np.sum(scaled_factors[0], axis=1)
    frac_weight[np.isnan(frac_weight)] = 0
    max_fac_ind = np.argmax(frac_weight, axis=0)
    assert len(max_fac_ind) == factors[0].shape[0]

    return max_fac_ind


def four_way_split(meta, pref_tensor, model=None, ):
    """
    Split cells into categories based in onset/offset peak and based on the center of mass for firing
    in each respective stage.

    # TODO
    #  1. consider a filter an driven-ness (again) to account for noisy baselines
    #  2. the shape of cells with an without running modulation can change. i.e., when an animal runs
    #  a cell's shape can go from transient to sustained (hiding the transient peak).

    :param meta:
    :param pref_tensor:
    :param model:
    :return:
    """


    # utils.filter_meta_bool(meta, meta_bool, filter_running=None, filter_licking=None, filter_hmm_engaged=True,
    #                  high_speed_thresh_cms=10,
    #                  low_speed_thresh_cms=4,
    #                  high_lick_thresh_ls=1.7,
    #                  low_lick_thresh_ls=1.7
    #                  )


    # Determine if cells have onset or offset peak activity
    if by_cells:
        off_vec = utils.get_offset_cells(meta, pref_tensor) # or cell-wise version
    else:
        off_vec = cell_factor_is_offset(meta, model, rank_num=15, buffer_s=0.100, cell_fac_std_thresh=1)


def trans_center_of_mass_from_cells(meta, pref_tensor, staging='parsed_11stage', buffer_s=0.300,
                                    stages_for_calc=('L4 learning', 'L5 learning'),
                                    cm_off_thresh=12, cm_stim_l=12, cm_stim_h=14, shuffle=False):

    # make sure your input is a list
    if isinstance(stages_for_calc, str):
        stages_for_calc = [stages_for_calc]

    # get off vec
    offset_bool = utils.get_offset_cells(meta, pref_tensor, buffer_s=buffer_s)

    # get mean cell responses for high dprime learning
    meta_bool = meta[staging].isin(stages_for_calc)
    mean_mat = utils.simple_mean_per_day(meta, pref_tensor, meta_bool=meta_bool, filter_running=None,
                                         filter_licking=None, filter_hmm_engaged=False)
    tr = np.nanmean(mean_mat, axis=2).T

    # rectify traces before calc
    tr[tr < 0] = 0

    # get range of response windows
    mouse = utils.meta_mouse(meta)
    times = np.arange(-1, 6, 1 / 15.5)[:108]
    # stim_bool = (times > buffer_s) & (times < lookups.stim_length[mouse])
    stim_bool = (times > buffer_s) & (times < 2)  # use first two seconds of stim for all mice, why? for thresholding
    response_bool = (times > lookups.stim_length[mouse] + buffer_s) & (times < lookups.stim_length[mouse] + 2)

    # create array of the total number of time points, make it a little longer, then shorten for rounding error
    resp_n = np.sum(response_bool)
    stim_n = np.sum(stim_bool)
    stim_pos = np.arange(1, np.sum(stim_bool) + 1)
    response_pos = np.arange(1, np.sum(response_bool) + 1)

    # calculate center of mass
    cell_center_of_mass = []
    dead_cells = []
    shuffle_cm = []
    for i in range(tr.shape[1]):
        temp_fac = tr[:, i]
        if offset_bool[i]:
            off_fac = temp_fac[response_bool]
            cm = np.sum(off_fac * response_pos) / np.sum(off_fac)
            dead_cells.append(
                (np.sum(off_fac <= 0)/resp_n) > 0.9)  # if 90% of time points of sub-zero for that window

            if shuffle:
                shuff_fac = deepcopy(off_fac)
                np.random.shuffle(shuff_fac)
                cm_shuff = np.sum(shuff_fac * response_pos) / np.sum(shuff_fac)
                shuffle_cm.append(cm_shuff)
        else:
            stim_fac = temp_fac[stim_bool]
            cm = np.sum(stim_fac * stim_pos) / np.sum(stim_fac)
            dead_cells.append(
                (np.sum(stim_fac <= 0) / stim_n) > 0.9)  # if 90% of time points of sub-zero for that window

            if shuffle:
                shuff_fac = deepcopy(stim_fac)
                np.random.shuffle(shuff_fac)
                cm_shuff = np.sum(shuff_fac * stim_pos) / np.sum(shuff_fac)
                shuffle_cm.append(cm_shuff)
        cell_center_of_mass.append(cm)
    cell_center_of_mass = np.array(cell_center_of_mass)

    # loop over CMs and test if they are high/low given offset or stim drive
    trans_cm = []
    delay_cm = []
    other_cm = []
    for i, cm_s in enumerate(cell_center_of_mass):
        if offset_bool[i]:
            trans_cm.append(cm_s < cm_off_thresh)
            delay_cm.append(cm_s > cm_off_thresh)
            other_cm.append(False)
        else:
            trans_cm.append(cm_s < cm_stim_l)
            delay_cm.append(cm_s > cm_stim_h)
            other_cm.append((cm_s >= cm_stim_l) & (cm_s <= cm_stim_h))

    # double check ramping and sharp offset cells using a max in the 500 ms pre and post reversal with buffer
    for i in range(tr.shape[1]):

        # transient offset cells:
        if trans_cm[i] & offset_bool[i] & ~dead_cells[i]:
            temp_fac = deepcopy(tr[:, i])
            first_sec_off = np.nanmax(temp_fac[response_bool][:8])
            back_bool = times < lookups.stim_length[mouse]
            last_sec_on = np.nanmax(temp_fac[back_bool][-8:])
            # update category to a ramp onset cell if the max is lower 500 ms after offset
            if first_sec_off < last_sec_on:
                stim_fac = deepcopy(temp_fac[stim_bool])
                trans_cm[i] = False
                offset_bool[i] = False
                delay_cm[i] = True
                cm = np.sum(stim_fac * stim_pos) / np.sum(stim_fac)
                cell_center_of_mass[i] = cm
                dead_cells[i] = (np.sum(stim_fac <= 0) / stim_n) > 0.9

        # ramping/delayed onset cells
        elif delay_cm[i] & ~offset_bool[i] & ~dead_cells[i]:
            temp_fac = deepcopy(tr[:, i])
            first_sec_off = np.nanmax(temp_fac[response_bool][:8])
            back_bool = times < lookups.stim_length[mouse]
            last_sec_on = np.nanmax(temp_fac[back_bool][-8:])
            # update category to a fast ofset cell if the max is higher 500 ms after offset
            if first_sec_off > last_sec_on:
                resp_fac = deepcopy(temp_fac[response_bool])
                trans_cm[i] = True
                offset_bool[i] = True
                delay_cm[i] = False
                cm = np.sum(resp_fac * response_pos) / np.sum(resp_fac)
                cell_center_of_mass[i] = cm
                dead_cells[i] = (np.sum(resp_fac <= 0) / resp_n) > 0.9

    cm_df = pd.DataFrame(data={
        'mouse': [mouse] * len(cell_center_of_mass),
        'cell_n': np.arange(len(cell_center_of_mass)) + 1,
        'cell_center_of_mass': cell_center_of_mass,
        'cell_center_of_mass_shuffle': shuffle_cm if shuffle else [False] * len(cell_center_of_mass),
        'trans_cm': trans_cm,
        'delay_cm': delay_cm,
        'other_cm': other_cm,
        'dead_cell': dead_cells,
        'offset_cell': offset_bool,
        'no_data': np.isnan(tr[0, :])
    }).set_index(['mouse', 'cell_n'])

    return cm_df


def trans_center_of_mass_from_model(meta, model, rank_num=15, buffer_s=0.200, stim_buffer_s=0,
                              cell_fac_std_thresh=1, cm_threshold_off=None, cm_threshold_stim=None):

    # get offset factors from model
    offset_bool = temp_factor_is_offset(meta, model, rank_num=rank_num, buffer_s=buffer_s)

    # get mouse from metadata
    mouse = utils.meta_mouse(meta)

    # get array of temporal factors
    tr = deepcopy(model.results[rank_num][0].factors[1][:, :])

    # get range of response windows
    times = np.arange(-1, 6, 1 / 15.5)[:108]
    stim_bool = (times > stim_buffer_s) & (times < lookups.stim_length[mouse])
    response_bool = (times > lookups.stim_length[mouse]) & (times < lookups.stim_length[mouse] + 2)

    # create array of the total number of time points
    stim_pos = np.arange(1, np.sum(stim_bool)+1)
    response_pos = np.arange(1, np.sum(response_bool)+1)
    stim_mid = stim_pos[int(np.floor(len(stim_pos)/2))]
    response_mid = response_pos[int(np.floor(len(response_pos) / 2))]

    # print(stim_mid, response_mid)

    # calculate center of mass
    fac_center_of_mass = []
    for i in range(tr.shape[1]):
        temp_fac = tr[:, i]
        if offset_bool[i]:
            off_fac = temp_fac[response_bool]
            cm = np.sum(off_fac * response_pos) / np.sum(off_fac)
        else:
            stim_fac = temp_fac[stim_bool]
            cm = np.sum(stim_fac * stim_pos) / np.sum(stim_fac)
        fac_center_of_mass.append(cm)
    fac_center_of_mass = np.array(fac_center_of_mass)

    # loop over CMs and test if they are high/low given offset or stim drive
    trans_cm = []
    delay_cm = []
    other_cm = []
    for i, cm_s in enumerate(fac_center_of_mass):
        if offset_bool[i]:
            trans_cm.append(cm_s < response_mid)
            # delay_cm.append(cm_s > response_mid)
            # other_cm.append(cm_s)
        else:
            trans_cm.append(cm_s + stim_buffer_s < stim_mid)
    trans_compn = np.where(trans_cm)[0]
    print(trans_compn)

    # TODO
    #   1. could add middle group (flat)
    #   2. could return "best" cm from highest relative std weight

    # get cells that had an transient (sharp onset) or sharp offset component
    cell_facs = deepcopy(model.results[rank_num][0].factors[0][:, :])
    cell_facs_std = np.std(cell_facs, axis=0)
    mask = np.zeros((cell_facs.shape[0], len(trans_compn)))
    for c, transn in enumerate(trans_compn):
        mask[:, c] = cell_facs[:, transn] > cell_facs_std[transn]*cell_fac_std_thresh
    any_trans_bool = np.any(mask, axis=1)

    return any_trans_bool


def cell_factor_is_offset(meta, model, rank_num=15, buffer_s=0.200, cell_fac_std_thresh=1):
    """
    Function to return offset calculated on temporal factors from TCA.

    # TODO adding a filter on components that explain a certain amount of variance might be useful

    :param meta: pandas.DataFrame
        Trial metadata
    :param model: tensortools.Ensemble
        TCA model.
    :param rank_num: int
        tensortools TCA model rank.
    :param buffer_s: float
        Seconds to buffer stimulus offset to prevent confusion with GECI tail.
    :param cell_fac_std_thresh: float/int
        Standard deviation threshold for cell factor to say a cell participates in a component.
    :return: boolean vector, True for cells with peak after offset
    """

    # get offset factors from model
    offset_bool = temp_factor_is_offset(meta, model, rank_num=rank_num, buffer_s=buffer_s)
    offset_compn = np.where(offset_bool)[0]

    # get cells that had an offset component
    cell_facs = deepcopy(model.results[rank_num][0].factors[0][:, :])
    cell_facs_std = np.std(cell_facs, axis=0)
    mask = np.zeros((cell_facs.shape[0], len(offset_compn)))
    for c, offn in enumerate(offset_compn):
        mask[:, c] = cell_facs[:, offn] > cell_facs_std[offn]*cell_fac_std_thresh
    any_offset_bool = np.any(mask, axis=1)

    return any_offset_bool


def temp_factor_is_offset(meta, model, rank_num=15, buffer_s=0.200):
    """
    Function to return offset calculated on temporal factors from TCA.

    # TODO adding a filter on components that explain a certain amount of variance might be useful

    :param meta: pandas.DataFrame
        Trial metadata
    :param model: tensortools.Ensemble
        TCA model.
    :param rank_num: int
        tensortools TCA model rank.
    :param buffer_s: float
        Seconds to buffer stimulus offset to prevent confusion with GECI tail.
    :param cell_fac_std_thresh: float/int
        Standard deviation threshold for cell factor to say a cell participates in a component.
    :return: boolean vector, True for cells with peak after offset
    """

    # get mouse from metadata
    mouse = utils.meta_mouse(meta)

    # determine cells with offset responses
    trace_mean = deepcopy(model.results[rank_num][0].factors[1][:, :].T)

    # buffer around offset to avoid GECI tail confusion
    stim_offset_buffer_start = int(np.floor(15.5 * (1 + lookups.stim_length[mouse] - buffer_s)))
    stim_offset_buffer_end = int(np.floor(15.5 * (1 + lookups.stim_length[mouse] + buffer_s)))
    trace_mean[:, stim_offset_buffer_start:stim_offset_buffer_end] = np.nan

    # buffer baseline with a few extra frames, ~200 ms
    trace_mean[:, :18] = np.nan

    # Get cells with peak after offset
    offset_bool = np.nanargmax(trace_mean, axis=1)
    offset_bool = offset_bool > 15.5 * (1 + lookups.stim_length[mouse])

    return offset_bool


def std_thresholds(cm_df, std_num=1):
    """
    Center of mass DataFrame helper function for getting lower and upper center of mass bounds.

    :param cm_df: pandas.DataFrame
        Center of mass DataFrame for single mouse or probably across mice.
    :return: list of lists of lower and upper std bounds for stimulus res
    """

    # get stimulus std bounds
    mn = np.nanmean(cm_df.cell_center_of_mass_shuffle.loc[~cm_df.offset_cell & ~cm_df.dead_cell])
    std = np.nanstd(cm_df.cell_center_of_mass_shuffle.loc[~cm_df.offset_cell & ~cm_df.dead_cell])
    stim_upper = mn + std * std_num
    stim_lower = mn - std * std_num

    # get stimulus std bounds
    mn = np.nanmean(cm_df.cell_center_of_mass_shuffle.loc[cm_df.offset_cell & ~cm_df.dead_cell])
    std = np.nanstd(cm_df.cell_center_of_mass_shuffle.loc[cm_df.offset_cell & ~cm_df.dead_cell])
    off_upper = mn + std * std_num
    off_lower = mn - std * std_num

    return [stim_lower, stim_upper], [off_lower, off_upper]


def category_vector(cm_df):
    """
    Center of mass DataFrame helper function for getting categories as integers.

    :param cm_df: pandas.DataFrame
        Center of mass DataFrame for single mouse or probably across mice.
    :param cm_df:
    :return:
    """

    # preallocate
    cat_vec = np.zeros(len(cm_df)) + np.nan

    # get standard deviation thresholds
    [stim_lower, stim_upper], [off_lower, off_upper] = std_thresholds(cm_df)
    [stim_lower2, stim_upper2], [off_lower2, off_upper2] = std_thresholds(cm_df, std_num=2)

    # transient stim cells
    # cat_vec[(cm_df.cell_center_of_mass < stim_lower) & ~cm_df.offset_cell & ~cm_df.dead_cell] = 0
    cat_vec[(cm_df.cell_center_of_mass < stim_lower2) & ~cm_df.offset_cell & ~cm_df.dead_cell] = 0

    # trasnient/sharp offset cell
    cat_vec[(cm_df.cell_center_of_mass < off_lower) & cm_df.offset_cell & ~cm_df.dead_cell] = 1

    # ramping stimulus cell
    cat_vec[(cm_df.cell_center_of_mass > stim_upper) & ~cm_df.offset_cell & ~cm_df.dead_cell] = 2

    # delayed offset cell
    cat_vec[(cm_df.cell_center_of_mass > off_upper) & cm_df.offset_cell & ~cm_df.dead_cell] = 3

    # FLAT stimulus cell
    cat_vec[(cm_df.cell_center_of_mass < stim_upper) & (cm_df.cell_center_of_mass > stim_lower)
            & ~cm_df.offset_cell & ~cm_df.dead_cell] = 4

    # FLAT/ambiguous offset cell
    cat_vec[(cm_df.cell_center_of_mass < off_upper) & (cm_df.cell_center_of_mass > off_lower)
            & cm_df.offset_cell & ~cm_df.dead_cell] = 5

    return cat_vec


def category_from_cm(meta, pref_tensor, staging='parsed_11stage',
                     buffer_s=0.300, stages_for_calc=('L4 learning', 'L5 learning'),):

    # break cells up according to offsets and
    cm_df = trans_center_of_mass_from_cells(meta, pref_tensor, staging=staging,
                                    buffer_s=buffer_s, stages_for_calc=stages_for_calc,
                                    shuffle=True)

    # get category as numerical value
    cat_col = category_vector(cm_df)

    # make DataFrame
    cat_df = pd.DataFrame(data={'categories_v1': cat_col}, index=cm_df.index)

    return cat_df



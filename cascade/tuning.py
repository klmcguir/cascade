"""Functions for analyzing tuning of neurons and TCA components."""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import numpy as np
import os
from . import load, paths, utils, lookups, bias
from copy import deepcopy
import scipy as sp


# TODO cell_tuning
#  how should I deal with offset components here?
#  it seems that offset compoenents (including their ramping activity)
#  may account for a lot of the FC bias and certainly account for the


def cell_tuning(meta, tensor, model, rank, by_stage=False, nan_lick=False,
                staging='parsed_11stage', tuning_type='initial'):
    """
    Function for calculating tuning for different stages of learning for the components
    from TCA.

    :param meta: pandas.DataFrame, trial metadata
    :param model: tensortools.ensemble, TCA results
    :param rank: int, rank of TCA model to use for components
    :param by_stage: boolean, choose to use stages or simply caclulate tuning over all time
    :param staging: str, binning used to define stages of learning
    :param tuning_type: str, way to define stimulus type. 'initial', 'orientation', or defaults to 'condition'
    :return: stage_tuning_df: pandas.DataFrame, columns are stages
    """

    # baseline period boolean
    timepts = np.arange(tensor.shape[1])

    # get tensor-shaped mask of values that exclude licking
    # this accounts for differences in stimulus length between mice
    # baseline is also set to false
    if nan_lick:
        mask = bias.get_lick_mask(meta, tensor)
        ablated_tensor = deepcopy(tensor)
        ablated_tensor[~mask] = np.nan  # this sets baseline to NaN as well
    else:
        ablated_tensor = tensor

    # get mean for each trial
    stim_mean_nolick = np.nanmean(ablated_tensor, axis=1)

    # baseline period boolean
    timepts = np.arange(tensor.shape[1])
    base_boo = timepts <= 15.5
    # baseline_mean = np.nanmean(tensor[:, base_boo, :], axis=1)


def component_tuning(meta, model, rank, by_stage=False, staging='parsed_11stage', tuning_type='initial'):
    """
    Function for calculating tuning for different stages of learning for the components
    from TCA.

    :param meta: pandas.DataFrame, trial metadata
    :param model: tensortools.ensemble, TCA results
    :param rank: int, rank of TCA model to use for components
    :param by_stage: boolean, choose to use stages or simply caclulate tuning over all time
    :param staging: str, binning used to define stages of learning
    :param tuning_type: str, way to define stimulus type. 'initial', 'orientation', or defaults to 'condition'
    :return: stage_tuning_df: pandas.DataFrame, columns are stages
    """
    # meta can only be a DataFrame of a single mouse
    assert len(meta.reset_index()['mouse'].unique()) == 1
    mouse = meta.reset_index()['mouse'].unique()[0]

    # select rank of model to use
    trial_factors = model.results[rank][0].factors[2]
    offset = np.argmax(model.results[rank][0].factors[1][:, :], axis=0) > 15.5 * (1 + lookups.stim_length[mouse])

    df_list = []
    for comp_n in range(trial_factors.shape[1]):

        # calculate tuning either for stages or whole component
        trial_avg_vec = trial_factors[:, comp_n]
        if by_stage:
            tuning_df = tuning_by_stage(meta, trial_avg_vec, staging=staging, tuning_type=tuning_type)
        else:
            tuning_df = tuning_not_by_stage(meta, trial_avg_vec, tuning_type=tuning_type)

        # add component to index
        tuning_df['offset component'] = offset[comp_n]
        tuning_df['component'] = comp_n + 1
        tuning_df.reset_index(inplace=True)
        tuning_df.set_index(['mouse', 'component'], inplace=True)
        df_list.append(tuning_df)

    # create final df for return
    tuning_df_all_comps = pd.concat(df_list, axis=0)

    return tuning_df_all_comps


def tuning_by_stage(meta, trial_avg_vec, staging='parsed_11stage', tuning_type='initial'):
    """
    Function for calculating tuning for different stages of learning for the same cell or
    component.

    :param meta: pandas.DataFrame, trial metadata
    :param trial_avg_vec: vector of responses, one per trial, must be same length as meta
    :param staging: str, binning used to define stages of learning
    :param tuning_type: str, way to define stimulus type. 'initial', 'orientation', or defaults to 'condition'
    :return: stage_tuning_df: pandas.DataFrame, columns are stages
    """
    # meta can only be a DataFrame of a single mouse
    assert len(meta.reset_index()['mouse'].unique()) == 1

    # make sure parsed stage exists
    meta = utils.add_stages_to_meta(meta, staging)

    # split meta and trial avg vec by stage
    temp_meta = deepcopy(meta)
    temp_meta['responses for tuning calc'] = trial_avg_vec

    # loop over groupings
    tuning_words = []
    tuning_vecs = []
    stage_words = []
    groupings = temp_meta.groupby(staging)
    for name, gri in groupings:
        stage_vec = gri['responses for tuning calc'].values

        # pass split meta and trial averages to tuning calculations
        tuning_words.append(calc_tuning_from_meta_and_vec(gri, stage_vec, tuning_type=tuning_type))
        tuning_vecs.append(cosine_distance_from_meta_and_vec(gri, stage_vec, tuning_type=tuning_type))
        stage_words.append(name)

    # create a DataFrame for output
    data = {staging: stage_words, 'preferred tuning': tuning_words, 'cosine tuning': tuning_vecs}
    stage_tuning_df = pd.DataFrame(data=data)
    stage_tuning_df['mouse'] = meta.reset_index()['mouse'].unique()[0]
    stage_tuning_df.set_index(['mouse'], inplace=True)

    return stage_tuning_df


def tuning_not_by_stage(meta, trial_avg_vec, tuning_type='initial'):
    """
    Function for calculating tuning not accounting for stages of learning for the same cell or
    component.

    :param meta: pandas.DataFrame, trial metadata
    :param trial_avg_vec: vector of responses, one per trial, must be same length as meta
    :param tuning_type: str, way to define stimulus type. 'initial', 'orientation', or defaults to 'condition'
    :return: stage_tuning_df: pandas.DataFrame, columns are stages
    """
    # meta can only be a DataFrame of a single mouse
    assert len(meta.reset_index()['mouse'].unique()) == 1

    # pass split meta and trial averages to tuning calculations
    tuning_words = calc_tuning_from_meta_and_vec(meta, trial_avg_vec, tuning_type=tuning_type)
    tuning_vecs = cosine_distance_from_meta_and_vec(meta, trial_avg_vec, tuning_type=tuning_type)

    # create a DataFrame for output
    data = {'preferred tuning': tuning_words, 'cosine tuning': tuning_vecs}
    stage_tuning_df = pd.DataFrame(data=data)
    stage_tuning_df['mouse'] = meta.reset_index()['mouse'].unique()[0]
    stage_tuning_df.set_index(['mouse'], inplace=True)

    return stage_tuning_df


def calc_tuning_from_meta_and_vec(meta, trial_avg_vec, tuning_type='initial'):
    """
    Helper function for calculating the preferred tuning of any vector (trial averaged responses)
    and metadata DataFrame (subset to match vector of reponses). Based on initial CS values.

    :param meta: pandas.DataFrame, trial metadata
    :param trial_avg_vec: vector of responses, one per trial, must be same length as meta
    :param tuning_type: str, way to define stimulus type. 'initial', 'orientation', or defaults to 'condition'
    :return: preferred_tuning: str, string of preferred tuning of this vector
    """

    # calculate cosine distances for respones to each cue compared to canonical basis set of tuning
    distances = cosine_distance_from_meta_and_vec(meta, trial_avg_vec, tuning_type=tuning_type)

    # get conditions
    if 'initial' in tuning_type:
        cond_type = 'initial_condition'
    elif 'ori' in tuning_type:
        cond_type = 'orientation'
    else:
        cond_type = 'condition'
    u_conds = sorted(meta[cond_type].unique())

    # assume you are getting 3 cues
    assert len(u_conds) == 3

    # pick tuning type
    if np.sum(distances < 0.6) == 1:  # ~ cosine_dist([1, 0], [0.85, 1]) to allow for joint tuning
        preferred_tuning = f'{u_conds[np.argsort(distances)[0]]}'
    elif np.sum(distances < 0.6) == 2:
        preferred_tuning = f'{u_conds[np.argsort(distances)[0]]}-{u_conds[np.argsort(distances)[1]]}'
    elif np.sum(distances == 0) == 3 or np.sum(~np.isfinite(distances)) == 3:
        preferred_tuning = 'none'
    else:
        preferred_tuning = 'broad'

    return preferred_tuning


def cosine_distance_from_meta_and_vec(meta, trial_avg_vec, tuning_type='initial'):
    """
    Helper function for calculating cosine distances
    and metadata DataFrame (subset to match vector of responses). Based on initial CS values.

    :param meta: pandas.DataFrame, trial metadata
    :param trial_avg_vec: vector of responses, one per trial, must be same length as meta
    :param tuning_type: str, way to define stimulus type. 'initial', 'orientation', or defaults to 'condition'
    :return: distances: [float float float], cosine distances
    """

    # get initial conditions
    if 'initial' in tuning_type:
        cond_type = 'initial_condition'
    elif 'ori' in tuning_type:
        cond_type = 'orientation'
    else:
        cond_type = 'condition'
    u_conds = sorted(meta[cond_type].unique())

    # assume you are getting 3 cues
    assert len(u_conds) == 3

    # rectify vector
    rect_trial_avg_vec = deepcopy(trial_avg_vec)
    rect_trial_avg_vec[rect_trial_avg_vec < 0] = 0

    # get mean response per cue across all trials provided as a single tuning vector
    mean_cue = []
    for cue in u_conds:
        cue_boo = meta[cond_type].isin([cue])
        mean_cue.append(np.nanmean(rect_trial_avg_vec[cue_boo]))

    # get unit vector for tuning vector
    unit_vec_tuning = mean_cue / np.linalg.norm(mean_cue)

    # compare tuning standards for perfect tuning to one of the three cues to tuning vector
    standard_vecs = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    distances = []
    for standi in standard_vecs:
        # calculate cosine distance by hand (for unit vectors this is just 1 - inner prod)
        distances.append(1 - np.inner(standi, unit_vec_tuning))
    distances = np.array(distances)

    return distances

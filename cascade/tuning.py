"""Functions for analyzing tuning of neurons and TCA components."""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import numpy as np
import os
from . import load, paths, utils, lookups
from copy import deepcopy
import scipy as sp


def tuning_by_stage(meta, trial_avg_vec, staging='parsed_11stage'):
    """
    Function for calculating tuning for different stages of learning for the same cell or
    component.

    :param meta: pandas.DataFrame, trial metadata
    :param trial_avg_vec: vector of responses, one per trial, must be same length as meta
    :param staging: str, binning used to define stages of learning
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
        tuning_words.append(calc_tuning_from_meta_and_vec(gri, stage_vec))
        tuning_vecs.append(cosine_distance_from_meta_and_vec(gri, stage_vec))
        stage_words.append(name)

    # create a DataFrame for output
    data = {staging: stage_words, 'preferred tuning': tuning_words, 'cosine tuning': tuning_vecs}
    stage_tuning_df = pd.DataFrame(data=data, columns=temp_meta[staging].unique())
    stage_tuning_df['mouse'] = meta.reset_index()['mouse'].unique()[0]
    stage_tuning_df.set_index(['mouse'], inplace=True)

    return stage_tuning_df


def calc_tuning_from_meta_and_vec(meta, trial_avg_vec):
    """
    Helper function for calculating the preferred tuning of any vector (trial averaged responses)
    and metadata DataFrame (subset to match vector of reponses). Based on initial CS values.

    :param meta: pandas.DataFrame, trial metadata
    :param trial_avg_vec: vector of responses, one per trial, must be same length as meta
    :return: preferred_tuning: str, string of preferred tuning of this vector
    """

    # calculate cosine distances for respones to each cue compared to canonical basis set of tuning
    distances = cosine_distance_from_meta_and_vec(meta, trial_avg_vec)

    # pick tuning type
    if np.sum(distances < 0.6) == 1:  # ~ cosine_dist([1, 0], [0.85, 1]) to allow for joint tuning
        preferred_tuning = u_conds[np.argsort(distances)[0]]
    elif np.sum(distances < 0.6) == 2:
        preferred_tuning = u_conds[np.argsort(distances)[0]] + '-' + u_conds[np.argsort(distances)[1]]
    elif np.sum(distances == 0) == 3 or np.sum(~np.isfinite(distances)) == 3:
        preferred_tuning = 'none'
    else:
        preferred_tuning = 'broad'

    return preferred_tuning


def cosine_distance_from_meta_and_vec(meta, trial_avg_vec):
    """
    Helper function for calculating cosine distances
    and metadata DataFrame (subset to match vector of reponses). Based on initial CS values.

    :param meta: pandas.DataFrame, trial metadata
    :param trial_avg_vec: vector of responses, one per trial, must be same length as meta
    :return: distances: [float float float], cosine distances
    """

    # get initial conditions
    u_conds = meta['initial_condition'].unique()

    # assume you are getting 3 cues
    assert len(u_conds) == 3

    # rectify vector
    rect_trial_avg_vec = deepcopy(trial_avg_vec)
    rect_trial_avg_vec[rect_trial_avg_vec < 0] = 0

    # get mean response per cue across all trials provided as a single tuning vector
    mean_cue = []
    for cue in u_conds:
        cue_boo = meta['initial_condition'].isin([cue])
        mean_cue.append(np.nanmean(rect_trial_avg_vec[cue_boo]))

    # get unit vector for tuning vector
    unit_vec_tuning = mean_cue/np.linalg.norm(mean_cue)

    # compare tuning standards for perfect tuning to one of the three cues to tuning vector
    standard_vecs = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    distances = []
    for standi in standard_vecs:
        # calculate cosine distance by hand (for unit vectors this is just 1 - inner prod)
        distances.append(1 - np.inner(standi, unit_vec_tuning))
    distances = np.array(distances)

    return distances

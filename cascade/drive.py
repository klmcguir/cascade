"""Functions for dealing with cell drivenness across days"""

from . import utils, lookups
import flow
import pool
import numpy as np
import pandas as pd


def drive_map_from_meta_and_ids(meta, ids, drive_type='trial'):
    """
    Get a matrix of -log10(p-values) for drivenness values.

    :param meta: pandas.DataFrame
        Trial metadata.
    :param ids: numpy.array
        Vector of cell IDs.
    :param drive_type: str
        Type of drivenness to be calculated ('visual', 'trial', etc.).
    :return: matrix of -log10(p-values) of drivenness, cells x days x cues
    """

    # get dates
    dates = meta.reset_index().date.unique()
    no_half_dates = [int(s) for s in dates if s % 1 == 0]

    # get oris sorted alphabetically by initial condition type: e.g., ['minus', 'neutral', 'plus']
    oris_to_eval = np.array([0, 135, 270])
    cond_type = [lookups.lookup_ori[utils.meta_mouse(meta)][oi] for oi in oris_to_eval]
    sorted_oris = oris_to_eval[np.argsort(cond_type)]

    # create DateSorter object
    days = flow.DateSorter.frommeta(
        mice=[utils.meta_mouse(meta)], dates=no_half_dates, tags=None, exclude_tags=['bad'])

    # for every day, load ids, and calculate drivenness for each ori
    drive_map = np.zeros((len(ids), len(no_half_dates), 3)) + np.nan
    for cday, day1 in enumerate(days):

        # get single day cell_ids
        d1_ids = flow.xday._read_crossday_ids(day1.mouse, day1.date)
        d1_ids = np.array([int(s) for s in d1_ids])

        # get drivenness
        for cori, ori in enumerate(sorted_oris):
            drive_vec_daysort = get_drive_vec_for_cue(day1, ori, drive_type=drive_type)
            if drive_vec_daysort is None:
                continue

            # add each cell's drive in the same ordering as ids input
            for cid, day_id in enumerate(d1_ids):
                id_location_in_master = np.where(ids == day_id)
                if len(id_location_in_master) > 0:
                    drive_map[id_location_in_master, cday, cori] = drive_vec_daysort[cid]

    return drive_map


def get_drive_vec_for_cue(day, cs, drive_type='trial'):
    """
    Function to run one of many driven-ness calculations.

    :param day: flow.DateSorter
        DateSorter object of day to test cell driven-ness.
    :param cs: str
        Orientation or cue as string.
    :param drive_type: str
        Type of calculation to perform.
    :return:
    """

    # cs must be a str (this is only for recalculation, callback seems to be insensitive to type)
    if ~isinstance(cs, str):
        cs = str(cs)

    if drive_type.lower() == 'trial':
        d_vec = pool.calc.driven.trial(day, cs)
    elif drive_type.lower() == 'trial_abs':
        d_vec = pool.calc.driven.trial_abs(day, cs)
    elif drive_type.lower() == 'visual':
        d_vec = pool.calc.driven.visually(day, cs)
    elif drive_type.lower() == 'early' or drive_type.lower() == 'visual_early':
        d_vec = pool.calc.driven.visually_early(day, cs)
    elif drive_type.lower() == 'inhib':
        d_vec = pool.calc.driven.visually_inhib(day, cs)
    elif drive_type.lower() == 'offset':
        d_vec = pool.calc.driven.offset(day, cs)
    else:
        print(str(day) + ' requested ' + cs + ' ' + drive_type +
              ': no match to what was shown (probably pav only).')
        d_vec = None

    return d_vec


def isdriven_stage(meta, ids, drive_type='trial', staging='parsed_11stage', initial_cue=None):
    """
    Build a map that is over stages instead of days.

    Basic algorithm is to look at all finite-value days and check the drivenness -log10(p-value)
    with additional Bonferroni correction for the number of days. If any day is significant
    then the cell is considered driven for that stage.

    NOTE: this calculation excludes the drivenness calc for 0.5 days from reversal since these are
    based on pre-reversal as well.

    :param meta: pandas.DataFrame
        Trial metadata.
    :param ids: numpy.array
        Vector of cell IDs.
    :param drive_type: str
        Type of drivenness to be calculated ('visual', 'trial', etc.).
    :param staging: str
        Type of binning to do when defining stages of learning.
    :param initial_cue: str
        Optionally get a drive map for a single cue type
    :return:
    """

    # get drive map over days
    drive_map_days = drive_map_from_meta_and_ids(meta, ids, drive_type=drive_type)

    # get drivenness matrix
    if initial_cue is None:
        driven_mat = np.nanmax(drive_map_days, axis=2)
    else:
        sorted_cue_order = np.array(['minus', 'neutral', 'plus'])  # alphabetical to match drive_map_days
        mat_ind = np.where(sorted_cue_order == initial_cue)[0][0]
        driven_mat = drive_map_days[:, :, mat_ind]

    # get dates
    dates = meta.reset_index().date.unique()
    no_half_dates = np.array([s for s in dates if s % 1 == 0])  # keep this float

    # loop over stages to build new matrix
    stages = lookups.staging[staging]
    stage_mat = np.zeros((len(ids), len(stages)))
    for stagec, stagei in enumerate(stages):
        stage_days = meta.loc[meta[staging].isin([stagei])].reset_index().date.unique()
        days_to_check = np.isin(no_half_dates, stage_days)
        n_days = np.sum(days_to_check)
        if n_days == 0:
            stage_mat[:, stagec] = np.nan
            continue

        # get days for stage and Bonferroni correct for days
        stage_ps = driven_mat[:, days_to_check]
        cell_days = np.nansum(~np.isnan(stage_ps), axis=1)
        corrected_ps = stage_ps/cell_days[:, None]

        # add 1 for driven, nan for all nan and leave 0 if not driven
        driven_count = np.sum(corrected_ps > 1.31, axis=1)  # 1.31 is the -log10(p-value) for 0.05
        stage_mat[driven_count > 0, stagec] = 1
        all_nan = np.isnan(np.nanmean(corrected_ps, axis=1))
        stage_mat[all_nan, stagec] = np.nan

    return stage_mat


def isdriven_stage_col(meta, ids, drive_type='trial', staging='parsed_11stage', initial_cue=None):
    """
    Build a pandas.DataFrame with a single column for drivenness.
        1 = driven.
        0 = not driven.
        nan = not found.

    :param meta: pandas.DataFrame
        Trial metadata.
    :param ids: numpy.array
        Vector of cell IDs.
    :param drive_type: str
        Type of drivenness to be calculated ('visual', 'trial', etc.).
    :param staging: str
        Type of binning to do when defining stages of learning.
    :param initial_cue: str
        Optionally get a drive map for a single cue type
    :return:
    """

    # get necessary info for building DataFrame of drivenness
    drive_mat = isdriven_stage(meta, ids, drive_type=drive_type, staging=staging, initial_cue=initial_cue)
    mouse = utils.meta_mouse(meta)
    stages = lookups.staging[staging]

    # build list of dfs across stages, then concatenate
    stage_list = []
    for stagec, stagei in enumerate(stages):
        data = {'mouse': [mouse] * len(ids), 'cell_id': ids, 'isdriven': drive_mat[:, stagec],
                staging: [stagei] * len(ids)}
        stage_list.append(pd.DataFrame(data=data).set_index(['mouse', staging, 'cell_id']))
    df = pd.concat(stage_list, axis=0)

    return df


def isdriven_trial_mat(meta, ids, drive_type='trial', nan_half_days=True):
    """
    Get a matrix of -log10(p-values) for drivenness values.

    :param meta: pandas.DataFrame
        Trial metadata.
    :param ids: numpy.array
        Vector of cell IDs.
    :param drive_type: str
        Type of drivenness to be calculated ('visual', 'trial', etc.).
    :return: matrix of -log10(p-values) of drivenness, cells x trials x cues
    """

    drive_mat = drive_map_from_meta_and_ids(meta, ids, drive_type=drive_type)

    # get dates
    dates = meta.reset_index().date.unique()
    dates_full = meta.reset_index().date
    no_half_dates = [int(s) for s in dates if s % 1 == 0]

    # cues
    initial_cue = meta.reset_index().initial_condition

    # loop over dates filling in trials for that date with 1 for driven and 0 for not driven
    trial_mat = np.zeros((len(ids), len(meta))) + np.nan
    for cuec, cue in enumerate(['minus', 'neutral', 'plus']):  # alphabetical

        # get a boolean
        cue_bool = initial_cue.isin([cue])

        for datec, datei in enumerate(no_half_dates):

            # if the date is a half date (0.5) it has the same drivenness as the -0.5 date
            if nan_half_days:
                date_bool = dates_full.isin([datei])
            else:
                date_bool = dates_full.isin([datei, datei + 0.5])

            for celli in range(len(ids)):

                # significant set to 1, not sig to 0, not found left as nan
                if drive_mat[celli, datec, cuec] > 1.31:
                    trial_mat[celli, date_bool & cue_bool] = 1
                elif drive_mat[celli, datec, cuec] <= 1.31:
                    trial_mat[celli, date_bool & cue_bool] = 0

    return trial_mat

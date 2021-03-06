"""Functions for dealing with cell drivenness across days"""

from . import utils, lookups, load
import flow
import pool
import numpy as np
import pandas as pd
from scipy import stats
import os


def preferred_drive_mat(match_to='onsets', staging='parsed_11stage', FOV_data=False):
    """Wrapper function to get a driven matrix per stage, but only return drive 
    for that cells preferred cue defined by TCA.

    Parameters
    ----------
    match_to : str, optional
        Which dataset o use for matching ('onsets' or 'offsets'), by default 'onsets'
    FOV_data : bool
        Optionally run calcualtions for LML5 dataset that does not rely on TCA groups
        for preferred tuning.

    Returns
    -------
    numpy.ndarray
        Numpy array with 0 for not driven, 1 for yes driven, and NaN for non-existent.

    Raises
    ------
    ValueError
        Needs groupings of components for offset cells rank 8 or onsets rank 9. Value error if you 
        try and use something other than this. 
    """

    if FOV_data:
        return _preferred_drive_mat_FOV(match_to=match_to, staging=staging)

    # load tca reversal n=7 data
    tca_dict = load.core_tca_data(limit_to=None, match_to=match_to)
    drive_mat = drive_mat_from_core_reversal_data(match_to=match_to, staging=staging)

    # set groupings
    if match_to == 'onsets':
        cat_groups_pref_tuning = lookups.tuning_groups['rank9_onset']
    elif match_to == 'offsets':
        cat_groups_pref_tuning = lookups.tuning_groups['rank8_offset']
    else:
        raise ValueError

    # preallocate
    flat_drive_mat = np.zeros((drive_mat.shape[0], drive_mat.shape[1])) + np.nan
    for c, cue in enumerate(['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']):
        cue_cells = np.isin(tca_dict['cell_cats'], cat_groups_pref_tuning[cue])
        flat_drive_mat[cue_cells, :] = drive_mat[cue_cells, :, c]

    # deal with special cases of tuning
    # broad tuning
    if any(['broad' in s for s in cat_groups_pref_tuning.keys()]):
        cue_cells = np.isin(tca_dict['cell_cats'], cat_groups_pref_tuning['broad'])
        flat_drive_mat[cue_cells, :] = np.nanmax(drive_mat[cue_cells, :, :], axis=2)  # over 0s 1s and nans

    # joint tuning
    if any(['joint' in s for s in cat_groups_pref_tuning.keys()]):
        tensor_cues = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']
        jcues = [s for s in cat_groups_pref_tuning.keys() if 'joint' in s]
        for cue in jcues:
            cue_cells = np.isin(tca_dict['cell_cats'], cat_groups_pref_tuning[cue])
            cue_inds = np.array([int(c) for c, s in enumerate(tensor_cues) if s in cue])
            assert len(cue_inds) == 2  # joint tuning should always be two cues
            flat_drive_mat[cue_cells, :] = np.nanmax(drive_mat[cue_cells, :, :][:, :, cue_inds], axis=2)

    return flat_drive_mat


def _preferred_drive_mat_FOV(match_to='onsets', staging='parsed_11stage'):

    # TODO
    raise NotImplementedError


def preferred_cue_ind(match_to='onsets', FOV_data=False):
    """Wrapper function to get the preferred CUE index of a cells x stages (or trials) x CUES matrix.

    NOTE: indices refer to "unforced" cue order in mismatch_condition. The order assumes:
    ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']

    Parameters
    ----------
    match_to : str, optional
        Which dataset o use for matching ('onsets' or 'offsets'), by default 'onsets'

    Returns
    -------
    list of lists
       lists of teh index 

    Raises
    ------
    ValueError
        Needs groupings of components for offset cells rank 8 or onsets rank 9. Value error if you 
        try and use something other than this. 
    """

    if FOV_data:
        return _preferred_cue_ind_FOV(match_to=match_to)

    # load tca reversal n=7 data
    tca_dict = load.core_tca_data(limit_to=None, match_to=match_to)

    # set groupings
    if match_to == 'onsets':
        cat_groups_pref_tuning = lookups.tuning_groups['rank9_onset']
    elif match_to == 'offsets':
        cat_groups_pref_tuning = lookups.tuning_groups['rank8_offset']
    else:
        raise ValueError

    # preallocate
    preferred_cue_level = np.zeros((len(tca_dict['cell_cats']), 3)) + np.nan
    for c, cue in enumerate(['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']):
        cue_cells = np.isin(tca_dict['cell_cats'], cat_groups_pref_tuning[cue])
        preferred_cue_level[cue_cells, np.isin([0, 1, 2], c)] = 1

    # deal with special cases of tuning
    # broad tuning
    if any(['broad' in s for s in cat_groups_pref_tuning.keys()]):
        cue_cells = np.isin(tca_dict['cell_cats'], cat_groups_pref_tuning['broad'])
        preferred_cue_level[cue_cells, :] = 1

    # joint tuning
    if any(['joint' in s for s in cat_groups_pref_tuning.keys()]):
        tensor_cues = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']
        jcues = [s for s in cat_groups_pref_tuning.keys() if 'joint' in s]
        for cue in jcues:
            cue_cells = np.isin(tca_dict['cell_cats'], cat_groups_pref_tuning[cue])
            cue_inds = np.array([int(c) for c, s in enumerate(tensor_cues) if s in cue])
            assert len(cue_inds) == 2  # joint tuning should always be two cues
            start_end = np.where(np.isin([0, 1, 2], cue_inds))[0]
            preferred_cue_level[cue_cells, start_end[0]:(start_end[1] + 1)] = 1

    return preferred_cue_level


def _preferred_cue_ind_FOV(match_to='onsets'):
    """ Helper function to deal with data that does not have a preferred group from TCA

    array([[ 1., nan, nan],
       [nan, nan,  1.],
       [nan, nan,  1.],
       ...,
       [nan, nan,  1.],
       [nan, nan,  1.],
       [nan,  1., nan]])

    Parameters
    ----------
    match_to : str, optional
        Portion of response to analyze, by default 'onsets'
    """

    # TODO
    raise NotImplementedError



def preferred_drive_day_mat(match_to='onsets', reduce_to='stagedays'):
    """Wrapper function to get a driven matrix per day, but only return drive 
    for that cell's preferred cue defined by TCA.

    Parameters
    ----------
    match_to : str, optional
        Which dataset o use for matching ('onsets' or 'offsets'), by default 'onsets'
    reduce_to : str, optional
        Full output is trials long (meta), but can be reduced to 'days' or 'stagedays'.

    Returns
    -------
    list of numpy.ndarray (one per mouse)
        Numpy array with 0 for not driven, 1 for yes driven, and NaN for non-existent.

    Raises
    ------
    ValueError
        Needs groupings for offset cells rank 8. Will need joint tuning special cases. 
    """

    # load tca reversal n=7 data
    tca_dict = load.core_tca_data(limit_to=None, match_to=match_to)
    drive_mat = drive_day_mat_from_core_reversal_data(match_to=match_to)
    if reduce_to is not None:
        load_dict = load.core_reversal_data(limit_to=None, match_to=match_to)

    # set groupings
    if match_to == 'onsets':
        cat_groups_pref_tuning = lookups.tuning_groups['rank9_onset']
    elif match_to == 'offsets':
        cat_groups_pref_tuning = lookups.tuning_groups['rank8_offset']
    else:
        raise ValueError

    flat_mat_list = []
    mouse_vec = tca_dict['mouse_vec']
    for mc, m in enumerate(pd.unique(mouse_vec)):

        # get category vectors for each mouse
        mouse_cats = tca_dict['cell_cats'][mouse_vec == m]

        flat_drive_mat = np.zeros((drive_mat[mc].shape[0], drive_mat[mc].shape[1])) + np.nan
        for c, cue in enumerate(['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']):
            cue_cells = np.isin(mouse_cats, cat_groups_pref_tuning[cue])
            flat_drive_mat[cue_cells, :] = drive_mat[mc][cue_cells, :, c]

        # deal with special cases of tuning
        cue_cells = np.isin(mouse_cats, cat_groups_pref_tuning['broad'])
        flat_drive_mat[cue_cells, :] = np.nanmax(drive_mat[mc][cue_cells, :, :], axis=2)

        # deal with special cases of tuning
        if any(['joint' in s for s in cat_groups_pref_tuning.keys()]):

            cue_cells = np.isin(mouse_cats, cat_groups_pref_tuning['joint-becomes_unrewarded-remains_unrewarded'])
            # deal with edge case with only one cell per group
            joint_mat = drive_mat[mc][cue_cells, :, :2]
            if joint_mat.ndim == 2:  # if you only have one cell that dim will drop
                flat_drive_mat[cue_cells, :] = np.nanmax(joint_mat, axis=0)
            else:
                flat_drive_mat[cue_cells, :] = np.nanmax(joint_mat, axis=2)

            cue_cells = np.isin(mouse_cats, cat_groups_pref_tuning['joint-becomes_rewarded-remains_unrewarded'])
            joint_mat = drive_mat[mc][cue_cells, :, 1:]
            if joint_mat.ndim == 2:  # if you only have one cell that dim will drop
                flat_drive_mat[cue_cells, :] = np.nanmax(joint_mat, axis=0)
            else:
                flat_drive_mat[cue_cells, :] = np.nanmax(joint_mat, axis=2)

        # optionally reduce the expanded drive (trials long at this point)
        if reduce_to == 'stagedays':
            meta = load_dict['meta_list'][mc]
            # get all unique stage ay combos in order
            stage_day_df = (
                meta
                .groupby(['parsed_11stage', 'date'])
                .count()
                .reindex(lookups.staging['parsed_11stage'], level=0)
            )

            # preallocate
            flat_drive_mat_reduced = np.zeros((flat_drive_mat.shape[0], stage_day_df.shape[0])) + np.nan

            # use index from row iterator to get stage-day boolean
            for c, (ind, _) in enumerate(stage_day_df.iterrows()):

                # boolean
                stage_boo = meta.parsed_11stage.isin([ind[0]]).values
                day_boo = meta.reset_index().date.isin([ind[1]]).values
                sdb = stage_boo & day_boo
                flat_drive_mat_reduced[:, c] = np.nanmean(flat_drive_mat[:, sdb], axis=1)
            assert np.isin(np.unique(flat_drive_mat_reduced[~np.isnan(flat_drive_mat_reduced)]), [0, 1]).all()
            flat_drive_mat = flat_drive_mat_reduced
        else:
            raise ValueError

        flat_mat_list.append(flat_drive_mat)

    return flat_mat_list


def drive_mat_from_core_reversal_data(match_to='onsets', staging='parsed_11stage'):
    """ Wrapper function to run drivenness calc on the reversal data and return as a matrix. 
    
    Returns
    -------
    numpy.ndarray
        Boolean matrix cells x stages x cues, of a cell's drivenness for each stage. 
    """

    # load if the file already exists
    # save_path = os.path.join(lookups.coreroot, f'{match_to}_drive_mat_stack.npy')
    save_path = os.path.join(lookups.coreroot, f'{match_to}_{staging}_drive_mat_stack.npy')
    if os.path.isfile(save_path):
        drive_stack = np.load(save_path, allow_pickle=True)
        return drive_stack

    # load all of your raw reversal n=7 data
    load_dict = load.core_reversal_data(limit_to=None, match_to=match_to)

    # calculate drivenness on each cell looping over mice
    drive_dfs = []
    for meta, ids, tensor in zip(load_dict['meta_list'], load_dict['id_list'], load_dict['tensor_list']):
        offset_spoof = np.ones(len(ids)) == (0 if match_to == 'onsets' else 1)
        df = multi_stat_drive(
            meta, ids, tensor, alternative='less', offset_bool=offset_spoof, neg_log10_pv_thresh=4,
            staging=staging
            )
        drive_dfs.append(df)

    # reshape dataframe into array cells x stages x cues
    drive_stack = drive_stack_from_dfs(drive_dfs, load_dict, staging=staging)

    # save your drive mat list
    try:
        np.save(save_path, drive_stack)
    except:
        print('File not saved: drive_day_mat_from_core_reversal_data().')

    return drive_stack


def drive_mat_from_FOV_reversal_data(match_to='onsets', staging='parsed_11stage'):
    """ Wrapper function to run drivenness calc on the reversal data and return as a matrix. 
    FOV data are FOVs from LM or L5 POR.
    
    Returns
    -------
    numpy.ndarray
        Boolean matrix cells x stages x cues, of a cell's drivenness for each stage. 
    """

    # load if the file already exists
    # save_path = os.path.join(lookups.coreroot, f'{match_to}_drive_mat_stack.npy')
    save_path = os.path.join(lookups.coreroot, f'{match_to}_{staging}_drive_mat_stack_FOV.npy')
    if os.path.isfile(save_path):
        drive_stack = np.load(save_path, allow_pickle=True)
        return drive_stack

    # load all of your raw reversal n=7 data
    load_dict = load.core_FOV_data(limit_to=None, match_to=match_to)

    # calculate drivenness on each cell looping over mice
    drive_dfs = []
    for meta, ids, tensor in zip(load_dict['meta_list'], load_dict['id_list'], load_dict['tensor_list']):
        offset_spoof = np.ones(len(ids)) == (0 if match_to == 'onsets' else 1)
        df = multi_stat_drive(
            meta, ids, tensor, alternative='less', offset_bool=offset_spoof, neg_log10_pv_thresh=4,
            staging=staging
            )
        drive_dfs.append(df)

    # reshape dataframe into array cells x stages x cues
    drive_stack = drive_stack_from_dfs(drive_dfs, load_dict, staging=staging)

    # save your drive mat list
    try:
        np.save(save_path, drive_stack)
    except:
        print('File not saved: drive_day_mat_from_core_reversal_data().')

    return drive_stack


def drive_stack_from_dfs(drive_dfs, load_dict, staging='parsed_11stage'):
    """Reshape a list of drive dfs into a single matrix cells x stages x cues.

    Parameters
    ----------
    drive_dfs : list of pandas.DataFrame
        Drive df from multi_stat_drive().
    load_dict : list of arrays
        List of cell ids you wish to use in your final array.
    """

    # set cue order
    cues = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']

    # organize dataframe into matrix cells x stages x cues
    drive_stack = []
    for m_df, ids in zip(drive_dfs, load_dict['id_list']):
        cue_stack = []
        for cue in cues:
            cue_lookup = lookups.lookup_mm_inv[utils.meta_mouse(m_df)]
            cue_df = m_df.loc[m_df.reset_index().initial_cue.isin([cue_lookup[cue]]).values, :]
            cue_df_unfolded = cue_df.pivot_table(values='driven', index='cell_id', columns=staging)
            # add "missing" stage columns to df
            add_cols = [s for s in lookups.staging[staging] if s not in cue_df_unfolded.columns.values]
            if len(add_cols) > 0:
                for col in add_cols:
                    cue_df_unfolded[col] = np.nan
            cue_df_binary = cue_df_unfolded.loc[:, lookups.staging[staging]].replace({True: 1, False: 0})
            # assert np.array_equal(cue_df_binary.reset_index().cell_id.values, ids)
            # match ids if any are missing (this has only happend in the FOV dataset)
            if not np.array_equal(cue_df_binary.reset_index().cell_id.values, ids):
                spoof_stack = np.zeros((len(ids), cue_df_binary.values.shape[1])) + np.nan
                existing_ids = np.isin(ids, cue_df_binary.reset_index().cell_id.values)
                # should be in the same order
                assert(np.array_equal(cue_df_binary.reset_index().cell_id.values, ids[existing_ids]))
                spoof_stack[existing_ids, :] = cue_df_binary.values
                cue_stack.append(spoof_stack)
            else:
                cue_stack.append(cue_df_binary.values)
        cue_stack = np.dstack(cue_stack)
        drive_stack.append(cue_stack)
    drive_stack = np.vstack(drive_stack)

    return drive_stack


def drive_day_mat_from_core_reversal_data(match_to='onsets'):
    """ Wrapper function to run drivenness calc on the reversal data and return as a matrix. 
    
    Returns
    -------
    list of numpy.ndarray
        Boolean matrix cells x trials x cues, of a cell's drivenness for each stage. 
    """

    save_path = os.path.join(lookups.coreroot, f'{match_to}_drive_day_mats.npy')
    if os.path.isfile(save_path):
        drive_mat_load = np.load(save_path, allow_pickle=True)
        return drive_mat_load

    # load all of your raw reversal n=7 data
    load_dict = load.core_reversal_data(limit_to=None, match_to=match_to)

    drive_mat_list = []
    for meta, tensor, ids in zip(load_dict['meta_list'], load_dict['tensor_list'], load_dict['id_list']):

        # add reversal condition to be used as your cue
        meta = utils.add_reversal_mismatch_condition_to_meta(meta)

        if match_to == 'onsets':
            offset_spoof = np.ones(tensor.shape[0]) == 0
        elif match_to == 'offsets':
            offset_spoof = np.ones(tensor.shape[0]) == 1

        df = multi_stat_drive_day(meta, ids, tensor, alternative='less', offset_bool=offset_spoof, neg_log10_pv_thresh=4)

        # Reshape a driven-ness matrix into cells x trials x cues
        cues = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']
        day_drive_map = np.zeros((tensor.shape[0], len(meta), len(cues))) + np.nan
        trial_date_vec = meta.reset_index().date.values

        for cuei, cue in enumerate(cues):
            cue_df = pd.pivot_table(df.loc[df.mismatch_condition.isin([cue]), :],
                                    values='driven', index='cell_id', columns='date',
                                    aggfunc='any', fill_value=np.nan, dropna=False)
            cue_df = cue_df.replace({False: 0, True: 1})
            assert np.array_equal(cue_df.index.values, ids)
            for di, day in enumerate(pd.unique(trial_date_vec)): # don't sort pd.
                day_vec = cue_df.loc[:, day]
                day_boo = np.isin(trial_date_vec, day)
                day_drive_map[:, day_boo, cuei] = np.tile(day_vec.values[:, None], np.sum(day_boo))

        drive_mat_list.append(day_drive_map)

        # save your drive mat list
        try:
            np.save(save_path, drive_mat_list)
        except:
            print('File not saved: drive_day_mat_from_core_reversal_data().')

    return drive_mat_list


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


def isdriven_stage(meta, ids, drive_type='trial', staging='parsed_11stage', initial_cue=None, threshold=1.31):
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
    :param threshold: float
        -log10(p-value) threshold for calling a cells significantly driven. Default is for p=0.05 --> 1.31
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
        corrected_ps = stage_ps / cell_days[:, None]

        # add 1 for driven, nan for all nan and leave 0 if not driven
        driven_count = np.sum(corrected_ps > threshold, axis=1)  # 1.31 is the -log10(p-value) for 0.05
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


def cell_driven_one_stage_of_two(meta, ids, drive_type='trial', staging='parsed_11stage',
                                 initial_cue=None, threshold=1.31):
    """
    Build a vector that is 1 (one stage) or 2 (both stages) if a cell if driven. This allows you to calculate
    a meaningful difference if a cell goes offline or comes online between two stages, but limits the differences
    being calculated on zeros/low values.

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
    :param threshold: float
        -log10(p-value) threshold for calling a cells significantly driven. Default is for p=0.05 --> 1.31
    :return:
    """

    # get boolean drivenness mat
    drive_mat = isdriven_stage(meta, ids, drive_type=drive_type, staging=staging,
                               initial_cue=initial_cue, threshold=threshold)

    # get all pairwise stages
    pairwise_drive = {}
    stages = lookups.staging[staging]
    for c1, s1 in enumerate(stages):

        curr_stage_vec = drive_mat[:, c1]

        driven_pairs = np.zeros(drive_mat.shape) + np.nan
        for c2, s2 in enumerate(stages):
            comparison_vec = drive_mat[:, c2]

            # fill in vectors with 0 for not driven (but calculated), and 1 or 2 for driven stage pairs
            driven_pairs[(curr_stage_vec == 0) & (comparison_vec == 0), c2] = 0  # both not driven
            driven_pairs[(curr_stage_vec == 1) | (comparison_vec == 1), c2] = 1  # one of the two is driven
            driven_pairs[(curr_stage_vec == 1) & (comparison_vec == 1), c2] = 2  # both are driven

        driven_pairs[:, c1] = np.nan  # nan the current period for clarity
        pairwise_drive[s1] = driven_pairs

    return pairwise_drive


def multi_stat_drive(meta,
                     ids,
                     tensor,
                     alternative='less',
                     offset_bool=None,
                     neg_log10_pv_thresh=4,
                     staging='parsed_11stage'):
    """
    Calculate drivenness on all trials for each stage for each cell. For ONSET/STIM cells it tests the whole
    stimulus window and a 500 ms window from 200 - 700 ms after stimulus onset with Bonferroni correction for
    multiple comparisons against the 1 sec baseline preceding stimulus onset. For OFFSET cells, the same procedure is
    performed for a 1 sec baseline preceding stimulus onset, and a 1 sec baseline preceding stimulus offset. The
    np.max is taken for these two offset p-values to choose the worst of the two. For a cell to be considered
    driven, it needs to pass both baseline comparisons.

    :param meta: pandas.DataFrame
        Trial metadata.
    :param ids: numpy.array
        Vector of cell IDs.
    :param tensor: numpy.array
        Imaging traces, 3D matrix oraganized --> [cells, times, trials]
    :param alternative: str
        Kwarg for scipy.stats.wilcoxon, 'less' for one-tailed, 'two-sided' for two,
    :param offset_bool: boolean
        Boolean vector, the same length and order as ids, and tensor --> [cells, :, :].
        True = Offset cells. False = not Offset cells.
    :param neg_log10_pv_thresh: float or int
        Threshold on -np.log10(p-value) >= threshold, for calling a cell driven.

    :return: pandas.DataFrame of drivenness p-values and -log10(p-values) for each cell for each stage and cue.
    """

    # offset_bool = cas.utils.get_offset_cells(meta, pref_tensor)
    if offset_bool is None:
        offset_bool = utils.get_offset_cells(meta, tensor)

    # You can't use 'greater', it will test an unintended direction
    assert alternative.lower() in ['less', 'two-sided']

    # mouse
    mouse = utils.meta_mouse(meta)

    # assume data is the usual 15.5 hz 7 sec 108 frame vector
    assert tensor.shape[1] == 108

    # check that staging is in meta
    if staging not in meta.columns:
        meta = utils.add_stages_to_meta(meta, staging)

    # get windows for additional comparisons --> 500 ms with a 200 ms delay to account for GECI tau
    stim_length = lookups.stim_length[mouse]
    time_vec = np.arange(-1, 6, 1 / 15.5)[:108]
    ms200 = np.where(time_vec > 0.2)[0][0]
    ms700 = np.where(time_vec > 0.7)[0][0]
    off_ms200 = np.where(time_vec > stim_length + 0.2)[0][0]
    off_ms700 = np.where(time_vec > stim_length + 0.7)[0][0]
    stim_off_frame = np.where(time_vec > stim_length)[0][0]
    stim_off_frame_minus1s = np.where(time_vec > stim_length - 1)[0][0]

    # get mean trial response matriices, cells x trials
    mean_t_tensor = utils.tensor_mean_per_trial(meta, tensor, nan_licking=False, account_for_offset=True)
    mean_t_baselines = np.nanmean(tensor[:, :15, :], axis=1)
    mean_t_1stsec = np.nanmean(tensor[:, ms200:ms700, :], axis=1)

    # for offset cells it is useful to have averages of 1sec before offset and 1 sec after as well
    mean_t_off_baselines = np.nanmean(tensor[:, stim_off_frame_minus1s:stim_off_frame, :], axis=1)
    mean_t_off_1st_sec = np.nanmean(tensor[:, off_ms200:off_ms700, :], axis=1)

    # preallocate
    data = {
        'mouse': [],
        staging: [],
        'initial_cue': [],
        'cell_id': [],
        'cell_n': [],
        'offset_cell': [],
        'pv': [],
        'neg_log10_pv': [],
        'driven': []
    }
    for si, stage in enumerate(lookups.staging[staging]):
        stage_boo = meta[staging].isin([stage]).values

        for icue in ['plus', 'minus', 'neutral']:
            cue_boo = meta.initial_condition.isin([icue]).values

            for celli in range(mean_t_tensor.shape[0]):
                cell_boo = ~np.isnan(mean_t_tensor[celli, :])

                # get boolean of stage, cue, and existing trials
                existing_epoch_trial_bool = stage_boo & cue_boo & cell_boo
                if np.sum(existing_epoch_trial_bool) == 0:  # no matched trials for cell
                    continue

                # get baseline vectors
                # mean_t_tensor accounts for offset, so this is the first test for both cases full-stim or full-offset
                epoch_bases = mean_t_baselines[celli, existing_epoch_trial_bool]
                epoch_means = mean_t_tensor[celli, existing_epoch_trial_bool]

                # different calcs if a cell is a stimulus peaking or offset peaking cell
                if not offset_bool[celli]:

                    # additional test, check 1st second of stim period to not punish transient responses
                    epoch_1s = mean_t_1stsec[celli, existing_epoch_trial_bool]

                else:

                    # additional baseline values defined from last second of the stimulus
                    off_epoch_bases = mean_t_off_baselines[celli, existing_epoch_trial_bool]  # last second of stimulus

                    # additional data for 1st second following offset
                    epoch_1s = mean_t_off_1st_sec[celli, existing_epoch_trial_bool]  # first second following offset

                # check for drivenness
                # this compares baseline to the full stim or offset period as well as to the first second of both
                # tests bases-means delta is negative, H0: symmetric
                pv_epoch = stats.wilcoxon(epoch_bases, epoch_means, alternative=alternative).pvalue
                pv_epoch_1s = stats.wilcoxon(epoch_bases, epoch_1s, alternative=alternative).pvalue

                # reset pv_epoch to be best with bonferroni
                pv_epoch = np.nanmin([pv_epoch, pv_epoch_1s]) * 2

                # additional tests specific offset using the 1s before stim offset as a baseline
                if offset_bool[celli]:
                    # this compares 1s pre offset to stim or offset period as well as to the first second of both
                    off_pv_epoch = stats.wilcoxon(off_epoch_bases, epoch_means, alternative=alternative).pvalue
                    off_pv_epoch_1s = stats.wilcoxon(off_epoch_bases, epoch_1s, alternative=alternative).pvalue

                    # reset pv_epoch to be best with bonferroni
                    off_pv_epoch = np.nanmin([off_pv_epoch, off_pv_epoch_1s]) * 2

                    # select your final p-value for offset cells to be the worst of the two comparisons.
                    # 1. baseline-(full or 1s)
                    # 2. last_second_of_stim-(full or 1s)
                    # both must be significant for a cell to pass so select the worst of the two for offset cells.

                    # pick the worst of your 2 p_values.
                    pv_epoch = np.nanmax([pv_epoch, off_pv_epoch])

                # build your dict for a dataframe
                data['mouse'].append(mouse)
                data[staging].append(stage)
                data['initial_cue'].append(icue)
                data['cell_id'].append(ids[celli])
                data['cell_n'].append(celli + 1)
                data['offset_cell'].append(offset_bool[celli])
                data['pv'].append(pv_epoch)
                data['neg_log10_pv'].append(-np.log10(pv_epoch))
                data['driven'].append(-np.log10(pv_epoch) >= neg_log10_pv_thresh)

    df = pd.DataFrame(data=data).set_index(['mouse', staging, 'initial_cue'])

    return df


def multi_stat_pvalues(meta, ids, tensor, alternative='less'):
    """
    Calculate pvalues on all trials for each stage for each cell. Does all tests for onset and offset cells for
    each cell.

    For ONSET/STIM cells it tests the whole stimulus window and a 500 ms window from 200 - 700 ms after stimulus onset
    with Bonferroni correction for multiple comparisons against the 1 sec baseline preceding stimulus onset.

    For OFFSET cells, the same procedure is performed for a 1 sec baseline preceding stimulus onset, and a 1 sec
    baseline preceding stimulus offset. The np.ma()x is taken for these two offset p-values to choose the worst of the
    two.

    :param meta: pandas.DataFrame
        Trial metadata.
    :param ids: numpy.array
        Vector of cell IDs.
    :param tensor: numpy.array
        Imaging traces, 3D matrix oraganized --> [cells, times, trials]
    :param alternative: str
        Kwarg for scipy.stats.wilcoxon, 'less' for one-tailed, 'two-sided' for two,
    :param neg_log10_pv_thresh: float or int
        Threshold on -np.log10(p-value) >= threshold, for calling a cell driven.

    :return: pandas.DataFrame of p-values for drivenness relative to different periods of the trial.
    """

    # You can't use 'greater', it will test an unintended direction
    assert alternative.lower() in ['less', 'two-sided']

    # mouse
    mouse = utils.meta_mouse(meta)

    # assume data is the usual 15.5 hz 7 sec 108 frame vector
    assert tensor.shape[1] == 108

    # get windows for additional comparisons --> 500 ms with a 200 ms delay to account for GECI tau
    stim_length = lookups.stim_length[mouse]
    time_vec = np.arange(-1, 6, 1 / 15.5)[:108]
    ms200 = np.where(time_vec > 0.2)[0][0]
    ms700 = np.where(time_vec > 0.7)[0][0]
    off_ms200 = np.where(time_vec > stim_length + 0.2)[0][0]
    off_ms700 = np.where(time_vec > stim_length + 0.7)[0][0]
    stim_off_frame = np.where(time_vec > stim_length)[0][0]
    stim_off_frame_minus1s = np.where(time_vec > stim_length - 1)[0][0]
    end_off_response_window = np.where(time_vec > stim_length + 2)[0][0]

    # get mean trial response matrices, cells x trials
    mean_t_stim = np.nanmean(tensor[:, ms200:stim_off_frame, :], axis=1)
    mean_t_baselines = np.nanmean(tensor[:, :15, :], axis=1)
    mean_t_stim500 = np.nanmean(tensor[:, ms200:ms700, :], axis=1)

    # for offset cells it is useful to have averages of 1sec before offset and 1 sec after as well
    mean_t_off = np.nanmean(tensor[:, off_ms200:end_off_response_window, :], axis=1)
    mean_t_off_baselines = np.nanmean(tensor[:, stim_off_frame_minus1s:stim_off_frame, :], axis=1)
    mean_t_off500 = np.nanmean(tensor[:, off_ms200:off_ms700, :], axis=1)

    # preallocate
    data = {
        'mouse': [],
        'parsed_11stage': [],
        'initial_cue': [],
        'cell_id': [],
        'cell_n': [],
        'pv_stim_long': [],
        'pv_stim_short': [],
        'pv_off_long': [],
        'pv_off_short': [],
        'pv_off_long_stimbase': [],
        'pv_off_short_stimbase': []
    }
    for si, stage in enumerate(lookups.staging['parsed_11stage']):
        stage_boo = meta.parsed_11stage.isin([stage]).values

        for icue in ['plus', 'minus', 'neutral']:
            cue_boo = meta.initial_condition.isin([icue]).values

            for celli in range(mean_t_stim.shape[0]):
                cell_boo = ~np.isnan(mean_t_stim[celli, :])

                # get boolean of stage, cue, and existing trials
                existing_epoch_trial_bool = stage_boo & cue_boo & cell_boo
                if np.sum(existing_epoch_trial_bool) == 0:  # no matched trials for cell
                    continue

                # get baseline vectors
                # mean_t_tensor accounts for offset, so this is the first test for both cases full-stim or full-offset
                epoch_bases = mean_t_baselines[celli, existing_epoch_trial_bool]
                off_epoch_bases = mean_t_off_baselines[celli, existing_epoch_trial_bool]  # last second of stimulus

                # get test windows
                stim500 = mean_t_stim500[celli, existing_epoch_trial_bool]
                off500 = mean_t_off500[celli, existing_epoch_trial_bool]  # first second following offset
                offall = mean_t_off[celli, existing_epoch_trial_bool]
                stimall = mean_t_stim[celli, existing_epoch_trial_bool]

                # check for drivenness
                # this compares baseline to the full stim or offset period as well as to the first second of both
                # tests bases-means delta is negative, H0: symmetric
                pv_stimall = stats.wilcoxon(epoch_bases, stimall, alternative=alternative).pvalue
                pv_stim500 = stats.wilcoxon(epoch_bases, stim500, alternative=alternative).pvalue

                # for the baseline of the trial
                pv_offall = stats.wilcoxon(epoch_bases, offall, alternative=alternative).pvalue
                pv_off500 = stats.wilcoxon(epoch_bases, off500, alternative=alternative).pvalue

                # for the baseline last second of the stimulus
                pv_offall_vstim = stats.wilcoxon(off_epoch_bases, offall, alternative=alternative).pvalue
                pv_off500_vstim = stats.wilcoxon(off_epoch_bases, off500, alternative=alternative).pvalue

                # build your dict for a dataframe
                data['mouse'].append(mouse)
                data['parsed_11stage'].append(stage)
                data['initial_cue'].append(icue)
                data['cell_id'].append(ids[celli])
                data['cell_n'].append(celli + 1)
                data['pv_stim_long'].append(pv_stimall)
                data['pv_stim_short'].append(pv_stim500)
                data['pv_off_long'].append(pv_offall)
                data['pv_off_short'].append(pv_off500)
                data['pv_off_long_stimbase'].append(pv_offall_vstim)
                data['pv_off_short_stimbase'].append(pv_off500_vstim)

    df = pd.DataFrame(data=data).set_index(['mouse', 'parsed_11stage', 'initial_cue'])

    return df


def multi_stat_drive_run(meta, ids, tensor, alternative='less', offset_bool=None, neg_log10_pv_thresh=4):
    """
    Calculate drivenness on all trials for each RUN for each cell. For ONSET/STIM cells it tests the whole
    stimulus window and a 500 ms window from 200 - 700 ms after stimulus onset with Bonferroni correction for
    multiple comparisons against the 1 sec baseline preceding stimulus onset. For OFFSET cells, the same procedure is
    performed for a 1 sec baseline preceding stimulus onset, and a 1 sec baseline preceding stimulus offset. The
    np.max is taken for these two offset p-values to choose the worst of the two. For a cell to be considered
    driven, it needs to pass both baseline comparisons.

    :param meta: pandas.DataFrame
        Trial metadata.
    :param ids: numpy.array
        Vector of cell IDs.
    :param tensor: numpy.array
        Imaging traces, 3D matrix oraganized --> [cells, times, trials]
    :param alternative: str
        Kwarg for scipy.stats.wilcoxon, 'less' for one-tailed, 'two-sided' for two,
    :param offset_bool: boolean
        Boolean vector, the same length and order as ids, and tensor --> [cells, :, :].
        True = Offset cells. False = not Offset cells.
    :param neg_log10_pv_thresh: float or int
        Threshold on -np.log10(p-value) >= threshold, for calling a cell driven.

    :return: pandas.DataFrame of drivenness p-values and -log10(p-values) for each cell for each stage and cue.
    """

    # offset_bool = cas.utils.get_offset_cells(meta, pref_tensor)
    if offset_bool is None:
        offset_bool = utils.get_offset_cells(meta, tensor)

    # You can't use 'greater', it will test an unintended direction
    assert alternative.lower() in ['less', 'two-sided']

    # mouse
    mouse = utils.meta_mouse(meta)

    # assume data is the usual 15.5 hz 7 sec 108 frame vector
    assert tensor.shape[1] == 108

    # get windows for additional comparisons --> 500 ms with a 200 ms delay to account for GECI tau
    stim_length = lookups.stim_length[mouse]
    time_vec = np.arange(-1, 6, 1 / 15.5)[:108]
    ms200 = np.where(time_vec > 0.2)[0][0]
    ms700 = np.where(time_vec > 0.7)[0][0]
    off_ms200 = np.where(time_vec > stim_length + 0.2)[0][0]
    off_ms700 = np.where(time_vec > stim_length + 0.7)[0][0]
    stim_off_frame = np.where(time_vec > stim_length)[0][0]
    stim_off_frame_minus1s = np.where(time_vec > stim_length - 1)[0][0]

    # get mean trial response matriices, cells x trials
    mean_t_tensor = utils.tensor_mean_per_trial(meta, tensor, nan_licking=False, account_for_offset=True)
    mean_t_baselines = np.nanmean(tensor[:, :15, :], axis=1)
    mean_t_1stsec = np.nanmean(tensor[:, ms200:ms700, :], axis=1)

    # for offset cells it is useful to have averages of 1sec before offset and 1 sec after as well
    mean_t_off_baselines = np.nanmean(tensor[:, stim_off_frame_minus1s:stim_off_frame, :], axis=1)
    mean_t_off_1st_sec = np.nanmean(tensor[:, off_ms200:off_ms700, :], axis=1)

    # preallocate
    data = {
        'mouse': [],
        'date': [],
        'run': [],
        'initial_cue': [],
        'cell_id': [],
        'cell_n': [],
        'offset_cell': [],
        'pv': [],
        'neg_log10_pv': [],
        'driven': []
    }
    for day in meta.reset_index().date.unique():
        day_boo = meta.reset_index().date.isin([day]).values
        day_runs = meta.loc[day_boo, :].reset_index().run.unique()

        for run in day_runs:
            run_boo = meta.reset_index().run.isin([run]).values

            for icue in ['plus', 'minus', 'neutral']:
                cue_boo = meta.initial_condition.isin([icue]).values

                for celli in range(mean_t_tensor.shape[0]):
                    cell_boo = ~np.isnan(mean_t_tensor[celli, :])

                    # get boolean of stage, cue, and existing trials
                    existing_epoch_trial_bool = day_boo & run_boo & cue_boo & cell_boo
                    if np.sum(existing_epoch_trial_bool) == 0:  # no matched trials for cell
                        continue

                    # get baseline vectors
                    # mean_t_tensor accounts for offset, so this is the first test for both cases full-stim or full-offset
                    epoch_bases = mean_t_baselines[celli, existing_epoch_trial_bool]
                    epoch_means = mean_t_tensor[celli, existing_epoch_trial_bool]

                    # different calcs if a cell is a stimulus peaking or offset peaking cell
                    if not offset_bool[celli]:

                        # additional test, check 1st second of stim period to not punish transient responses
                        epoch_1s = mean_t_1stsec[celli, existing_epoch_trial_bool]

                    else:

                        # additional baseline values defined from last second of the stimulus
                        off_epoch_bases = mean_t_off_baselines[celli, existing_epoch_trial_bool]  # last second of stimulus

                        # additional data for 1st second following offset
                        epoch_1s = mean_t_off_1st_sec[celli, existing_epoch_trial_bool]  # first second following offset

                    # check for drivenness
                    # this compares baseline to the full stim or offset period as well as to the first second of both
                    # tests bases-means delta is negative, H0: symmetric
                    pv_epoch = stats.wilcoxon(epoch_bases, epoch_means, alternative=alternative).pvalue
                    pv_epoch_1s = stats.wilcoxon(epoch_bases, epoch_1s, alternative=alternative).pvalue

                    # reset pv_epoch to be best with bonferroni
                    pv_epoch = np.nanmin([pv_epoch, pv_epoch_1s]) * 2

                    # additional tests specific offset using the 1s before stim offset as a baseline
                    if offset_bool[celli]:
                        # this compares 1s pre offset to stim or offset period as well as to the first second of both
                        off_pv_epoch = stats.wilcoxon(off_epoch_bases, epoch_means, alternative=alternative).pvalue
                        off_pv_epoch_1s = stats.wilcoxon(off_epoch_bases, epoch_1s, alternative=alternative).pvalue

                        # reset pv_epoch to be best with bonferroni
                        off_pv_epoch = np.nanmin([off_pv_epoch, off_pv_epoch_1s]) * 2

                        # select your final p-value for offset cells to be the worst of the two comparisons.
                        # 1. baseline-(full or 1s)
                        # 2. last_second_of_stim-(full or 1s)
                        # both must be significant for a cell to pass so select the worst of the two for offset cells.

                        # pick the worst of your 2 p_values.
                        pv_epoch = np.nanmax([pv_epoch, off_pv_epoch])

                    # build your dict for a dataframe
                    data['mouse'].append(mouse)
                    data['date'].append(day)
                    data['run'].append(run)
                    data['initial_cue'].append(icue)
                    data['cell_id'].append(ids[celli])
                    data['cell_n'].append(celli + 1)
                    data['offset_cell'].append(offset_bool[celli])
                    data['pv'].append(pv_epoch)
                    data['neg_log10_pv'].append(-np.log10(pv_epoch))
                    data['driven'].append(-np.log10(pv_epoch) >= neg_log10_pv_thresh)

    df = pd.DataFrame(data=data).set_index(['mouse', 'parsed_11stage', 'initial_cue'])

    return df


def multi_stat_drive_day(meta, ids, tensor, alternative='less', offset_bool=None, neg_log10_pv_thresh=4):
    """
    Calculate drivenness on all trials for each DAY for each cell. For ONSET/STIM cells it tests the whole
    stimulus window and a 500 ms window from 200 - 700 ms after stimulus onset with Bonferroni correction for
    multiple comparisons against the 1 sec baseline preceding stimulus onset. For OFFSET cells, the same procedure is
    performed for a 1 sec baseline preceding stimulus onset, and a 1 sec baseline preceding stimulus offset. The
    np.max is taken for these two offset p-values to choose the worst of the two. For a cell to be considered
    driven, it needs to pass both baseline comparisons.

    :param meta: pandas.DataFrame
        Trial metadata.
    :param ids: numpy.array
        Vector of cell IDs.
    :param tensor: numpy.array
        Imaging traces, 3D matrix oraganized --> [cells, times, trials]
    :param alternative: str
        Kwarg for scipy.stats.wilcoxon, 'less' for one-tailed, 'two-sided' for two,
    :param offset_bool: boolean
        Boolean vector, the same length and order as ids, and tensor --> [cells, :, :].
        True = Offset cells. False = not Offset cells.
    :param neg_log10_pv_thresh: float or int
        Threshold on -np.log10(p-value) >= threshold, for calling a cell driven.

    :return: pandas.DataFrame of drivenness p-values and -log10(p-values) for each cell for each stage and cue.
    """

    # offset_bool = cas.utils.get_offset_cells(meta, pref_tensor)
    if offset_bool is None:
        offset_bool = utils.get_offset_cells(meta, tensor)

    # You can't use 'greater', it will test an unintended direction
    assert alternative.lower() in ['less', 'two-sided']

    # mouse
    mouse = utils.meta_mouse(meta)

    # assume data is the usual 15.5 hz 7 sec 108 frame vector
    assert tensor.shape[1] == 108

    # get windows for additional comparisons --> 500 ms with a 200 ms delay to account for GECI tau
    stim_length = lookups.stim_length[mouse]
    time_vec = np.arange(-1, 6, 1 / 15.5)[:108]
    ms200 = np.where(time_vec > 0.2)[0][0]
    ms700 = np.where(time_vec > 0.7)[0][0]
    off_ms200 = np.where(time_vec > stim_length + 0.2)[0][0]
    off_ms700 = np.where(time_vec > stim_length + 0.7)[0][0]
    stim_off_frame = np.where(time_vec > stim_length)[0][0]
    stim_off_frame_minus1s = np.where(time_vec > stim_length - 1)[0][0]

    # get mean trial response matriices, cells x trials
    mean_t_tensor = utils.tensor_mean_per_trial(meta, tensor, nan_licking=False, account_for_offset=True)
    mean_t_baselines = np.nanmean(tensor[:, :15, :], axis=1)
    mean_t_1stsec = np.nanmean(tensor[:, ms200:ms700, :], axis=1)

    # for offset cells it is useful to have averages of 1sec before offset and 1 sec after as well
    mean_t_off_baselines = np.nanmean(tensor[:, stim_off_frame_minus1s:stim_off_frame, :], axis=1)
    mean_t_off_1st_sec = np.nanmean(tensor[:, off_ms200:off_ms700, :], axis=1)

    # preallocate
    data = {
        'mouse': [],
        'date': [],
        'initial_cue': [],
        'mismatch_condition': [],
        'cell_id': [],
        'cell_n': [],
        'offset_cell': [],
        'pv': [],
        'neg_log10_pv': [],
        'driven': []
    }
    for day in meta.reset_index().date.unique():
        day_boo = meta.reset_index().date.isin([day]).values

        for icue in ['plus', 'minus', 'neutral']:
            cue_boo = meta.initial_condition.isin([icue]).values

            for celli in range(mean_t_tensor.shape[0]):
                cell_boo = ~np.isnan(mean_t_tensor[celli, :])

                # get boolean of stage, cue, and existing trials
                existing_epoch_trial_bool = day_boo & cue_boo & cell_boo
                if np.sum(existing_epoch_trial_bool) == 0:  # no matched trials for cell
                    continue

                # get baseline vectors
                # mean_t_tensor accounts for offset, so this is the first test for both cases full-stim or full-offset
                epoch_bases = mean_t_baselines[celli, existing_epoch_trial_bool]
                epoch_means = mean_t_tensor[celli, existing_epoch_trial_bool]

                # different calcs if a cell is a stimulus peaking or offset peaking cell
                if not offset_bool[celli]:

                    # additional test, check 1st second of stim period to not punish transient responses
                    epoch_1s = mean_t_1stsec[celli, existing_epoch_trial_bool]

                else:

                    # additional baseline values defined from last second of the stimulus
                    off_epoch_bases = mean_t_off_baselines[celli, existing_epoch_trial_bool]  # last second of stimulus

                    # additional data for 1st second following offset
                    epoch_1s = mean_t_off_1st_sec[celli, existing_epoch_trial_bool]  # first second following offset

                # check for drivenness
                # this compares baseline to the full stim or offset period as well as to the first second of both
                # tests bases-means delta is negative, H0: symmetric
                pv_epoch = stats.wilcoxon(epoch_bases, epoch_means, alternative=alternative).pvalue
                pv_epoch_1s = stats.wilcoxon(epoch_bases, epoch_1s, alternative=alternative).pvalue

                # reset pv_epoch to be best with bonferroni
                pv_epoch = np.nanmin([pv_epoch, pv_epoch_1s]) * 2

                # additional tests specific offset using the 1s before stim offset as a baseline
                if offset_bool[celli]:
                    # this compares 1s pre offset to stim or offset period as well as to the first second of both
                    off_pv_epoch = stats.wilcoxon(off_epoch_bases, epoch_means, alternative=alternative).pvalue
                    off_pv_epoch_1s = stats.wilcoxon(off_epoch_bases, epoch_1s, alternative=alternative).pvalue

                    # reset pv_epoch to be best with bonferroni
                    off_pv_epoch = np.nanmin([off_pv_epoch, off_pv_epoch_1s]) * 2

                    # select your final p-value for offset cells to be the worst of the two comparisons.
                    # 1. baseline-(full or 1s)
                    # 2. last_second_of_stim-(full or 1s)
                    # both must be significant for a cell to pass so select the worst of the two for offset cells.

                    # pick the worst of your 2 p_values.
                    pv_epoch = np.nanmax([pv_epoch, off_pv_epoch])

                # build your dict for a dataframe
                data['mouse'].append(mouse)
                data['date'].append(day)
                data['initial_cue'].append(icue)
                data['mismatch_condition'].append(lookups.lookup_mm[mouse][icue])
                data['cell_id'].append(ids[celli])
                data['cell_n'].append(celli + 1)
                data['offset_cell'].append(offset_bool[celli])
                data['pv'].append(pv_epoch)
                data['neg_log10_pv'].append(-np.log10(pv_epoch))
                data['driven'].append(-np.log10(pv_epoch) >= neg_log10_pv_thresh)

    df = pd.DataFrame(data=data).set_index(['mouse', 'date', 'initial_cue'])

    return df


def add_drive_type_column(pvalue_df, neg_log10_pv_thresh=4):
    """
    Add a column that categorizes cells as driven to onset, offset, or both. Also add Bonferroni corrected
    p-value columns to a stage drive p-value dataframe as part of the calculation.

    :param pvalue_df: pandas.DataFrame
        cascade.drive.multi_stat_pvalues() output DataFrame
    :param neg_log10_pv_thresh: float or int
        Threshold on -np.log10(p-value) >= threshold, for calling a cell driven.
    :return: pvalue_df with new columns
    """

    def _nl10(x):
        """negative log 10"""
        return -np.log10(x)

    # add Bonferroni corrected columns for your three test comparisons
    pvalue_df = _add_bonferroni_pvalue_columns(pvalue_df)

    # create a column that classifies drive type as onset-offset-both
    pvalue_df['driven_to'] = 'not driven'
    onbool = pvalue_df['bc_pv_stim'].apply(_nl10).ge(neg_log10_pv_thresh)
    pvalue_df.loc[onbool, 'driven_to'] = 'onset'
    offbool = pvalue_df['bc_pv_off'].apply(_nl10).ge(neg_log10_pv_thresh) \
              & pvalue_df['bc_pv_off_stimbase'].apply(_nl10).ge(neg_log10_pv_thresh)
    pvalue_df.loc[offbool, 'driven_to'] = 'offset'
    pvalue_df.loc[offbool & onbool, 'driven_to'] = 'both'

    return pvalue_df


def _add_bonferroni_pvalue_columns(pvalue_df):
    """
    Helper function to add Bonferroni corrected p-value columns to a stage drive p-value dataframe
    :param pvalue_df: pandas.DataFrame
        cascade.drive.multi_stat_pvalues() output DataFrame
    :return: pvalue_df with new columns
    """

    # Bonferroni correct your 3 calculations for the two full window and 500 ms window tests
    bc_stim = pvalue_df.loc[:, ['pv_stim_short', 'pv_stim_long']].min(axis=1) * 2
    bc_off = pvalue_df.loc[:, ['pv_off_short', 'pv_off_long']].min(axis=1) * 2
    bc_off_stimbase = pvalue_df.loc[:, ['pv_off_short_stimbase', 'pv_off_long_stimbase']].min(axis=1) * 2

    pvalue_df['bc_pv_stim'] = bc_stim
    pvalue_df['bc_pv_off'] = bc_off
    pvalue_df['bc_pv_off_stimbase'] = bc_off_stimbase

    return pvalue_df

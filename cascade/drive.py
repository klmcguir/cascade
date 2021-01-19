"""Functions for dealing with cell drivenness across days"""

from . import utils, lookups
import flow
import pool
import numpy as np
import pandas as pd
from scipy import stats


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
        corrected_ps = stage_ps/cell_days[:, None]

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


def multi_stat_drive(meta, ids, tensor, alternative='less', offset_bool=None):
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
        'parsed_11stage': [],
        'initial_cue': [],
        'cell_id': [],
        'cell_n':[],
        'pv': [],
        'neg_log10_pv': []
    }
    for si, stage in enumerate(lookups.staging['parsed_11stage']):
        stage_boo = meta.parsed_11stage.isin([stage]).values

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
                data['parsed_11stage'].append(stage)
                data['initial_cue'].append(icue)
                data['cell_id'].append(ids[celli])
                data['cell_n'].append(celli + 1)
                data['pv'].append(pv_epoch)
                data['neg_log10_pv'].append(-np.log10(pv_epoch))

    df = pd.DataFrame(data=data).set_index(['mouse', 'parsed_11stage', 'initial_cue'])

    return df

"""Functions for loading tensors filtering on different tags, etc."""
import tensortools as tt
import numpy as np
import flow
from flow.misc import wordhash
import pool
import pandas as pd
import os
from . import utils
from . import paths
from .tca import _trialmetafromrun
from copy import deepcopy


def singleday_tensor(
        mouse,
        date,
        tags=None,

        # tensor params
        trace_type='zscore_day',
        cs='',
        downsample=True,
        start_time=-1,
        end_time=6,
        clean_artifacts=None,
        thresh=20,
        warp=False,
        smooth=True,
        smooth_win=5,
        verbose=True,

        # filtering params
        exclude_tags=('orientation_mapping', 'contrast', 'retinotopy'),
        exclude_conds=('blank', 'blank_reward', 'pavlovian'),
        driven=True,
        drive_css=('0', '135', '270'),
        drive_threshold=15):
    """
    Build tensor component analysis (TCA) on for a single day.

    Parameters
    ----------

    Returns
    -------

    """

    # create folder structure and save dir
    pars = {'tags': tags,
            'trace_type': trace_type, 'cs': cs, 'downsample': downsample,
            'start_time': start_time, 'end_time': end_time,
            'clean_artifacts': clean_artifacts, 'thresh': thresh,
            'warp': warp, 'smooth': smooth, 'smooth_win': smooth_win,
            'exclude_tags': exclude_tags, 'exclude_conds': exclude_conds,
            'driven': driven, 'drive_css': drive_css,
            'drive_threshold': drive_threshold}
    save_dir = paths.tca_path(mouse, 'single', pars=pars)

    day1 = flow.DateSorter.frommeta(
        mice=[mouse], dates=[date], tags=tags)[0]

    # get cell_ids
    d1_ids = flow.xday._read_crossday_ids(day1.mouse, day1.date)
    d1_ids = np.array([int(s) for s in d1_ids])

    # filter cells based on visual/trial drive across all cs, prevent
    # breaking when only pavs are shown
    if driven:
        d1_drive = []
        for dcs in drive_css:
            try:
                d1_drive.append(pool.calc.driven.trial(day1, dcs))
            except KeyError:
                print(str(day1) + ' requested ' + dcs +
                      ': no match to what was shown (probably pav only).')
        d1_drive = np.max(d1_drive, axis=0)
        # account for rare cases where lost xday ids are final id (making _ids
        # 1 shorter than _drive). Add a fake id to the end and force drive to
        # be false for that id
        if len(d1_drive) > len(d1_ids):
            print('Warning: ' + str(day1) + ': _ids was ' +
                  str(len(d1_drive)-len(d1_ids)) +
                  ' shorter than _drive: added pseudo-id.')
            d1_drive[-1] = 0
            d1_ids = np.concatenate((d1_ids, np.array([-1])))
        d1_ids_bool = np.array(d1_drive) > drive_threshold
        d1_drive_ids = d1_ids[np.array(d1_drive) > drive_threshold]
        d1_sorter = np.argsort(d1_ids[d1_ids_bool])
    else:
        d1_ids_bool = np.ones(np.shape(d1_ids)) > 0
        d1_sorter = np.argsort(d1_ids[d1_ids_bool])
    ids = d1_ids[d1_ids_bool][d1_sorter]

    # TODO add in additional filter for being able to check for quality of xday alignment

    # get all runs for both days
    d1_runs = day1.runs()

    # filter for only runs without certain tags
    d1_runs = [run for run in d1_runs if not any(np.isin(run.tags, exclude_tags))]

    # build tensors for all correct runs and trials after filtering
    if d1_runs:
        d1_tensor_list = []
        d1_meta = []
        for run in d1_runs:
            t2p = run.trace2p()
            # trigger all trials around stimulus onsets
            run_traces = utils.getcstraces(run, cs=cs, trace_type=trace_type,
                                     start_time=start_time, end_time=end_time,
                                     downsample=True, clean_artifacts=clean_artifacts,
                                     thresh=thresh, warp=warp, smooth=smooth,
                                     smooth_win=smooth_win)
            # filter and sort
            run_traces = run_traces[d1_ids_bool, :, :][d1_sorter, :, :]
            # get matched trial metadata/variables
            dfr = _trialmetafromrun(run)
            # subselect metadata if you are only running certain cs
            if cs != '':
                if cs == 'plus' or cs == 'minus' or cs == 'neutral':
                    dfr = dfr.loc[(dfr['condition'].isin([cs])), :]
                elif cs == '0' or cs == '135' or cs == '270':
                    dfr = dfr.loc[(dfr['orientation'].isin([cs])), :]
                else:
                    print('ERROR: cs called - "' + cs + '" - is not\
                          a valid option.')

            # subselect metadata to remove certain condtions
            if len(exclude_conds) > 0:
                run_traces = run_traces[:, :, (~dfr['condition'].isin(exclude_conds))]
                dfr = dfr.loc[(~dfr['condition'].isin(exclude_conds)), :]

            # drop trials with nans and add to lists
            keep = np.sum(np.sum(np.isnan(run_traces), axis=0, keepdims=True),
                          axis=1, keepdims=True).flatten() == 0
            dfr = dfr.iloc[keep, :]
            d1_tensor_list.append(run_traces[:, :, keep])
            d1_meta.append(dfr)

        # concatenate matched cells across trials 3rd dim (aka, 2)
        tensor = np.concatenate(d1_tensor_list, axis=2)

        # concatenate all trial metadata in pd dataframe
        meta = pd.concat(d1_meta, axis=0)

        # concatenate and save df for the day
        meta_path = os.path.join(save_dir, str(day1.mouse) + '_'
                                 + str(day1.date) + '_df_single_meta.pkl')
        input_tensor_path = os.path.join(save_dir, str(day1.mouse) + '_'
                                         + str(day1.date) + '_single_tensor_'
                                         + str(trace_type) + '.npy')
        input_ids_path = os.path.join(save_dir, str(day1.mouse) + '_'
                                      + str(day1.date) + '_single_ids_'
                                      + str(trace_type) + '.npy')

        # meta.to_pickle(meta_path)
        # np.save(input_tensor_path, tensor)
        # np.save(input_ids_path, ids)

        # print output so you don't go crazy waiting
        if verbose:
            print('Day: ' + str(day1.date) + ': ' + str(day1.mouse) + ': done.')

        return tensor, meta, ids

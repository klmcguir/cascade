"""Functions for building dataframes."""
import os
import flow
import pool
import numpy as np
import pandas as pd
import warnings
from copy import deepcopy
from . import utils
from . import paths
from . import tca

# ----------------------- EARLY STAGE DFs -----------------------


def trigger(mouse, trace_type='zscore_day', cs='', downsample=True,
            start_time=-1, end_time=6, clean_artifacts=None,
            thresh=20, warp=False, smooth=True, smooth_win=5,
            verbose=True):
    """
    Create a pandas dataframe of all of your triggered traces for a mouse.

    Parameters:
    -----------
    mouse : str
        Mouse name.
    cs : str
        Type of CS. e.g., plus, minus, neutral, 0, 135, 270, ...
    trace_type : str
        dff, zscore, zscore_iti, deconvolved
    downsample : bool
        Downsample from 31 to 15 Hz sampling rate
    clean_artifacts : str
        nan, interp; Remove huge artifacts in dff traces by interpolating
        or adding in nan values
        Note: setting either value here will cause zscoring to nan artifacts
        before calculating mu/sigma
    thresh : int
        Threshold for removing artifacts
    warp : bool
        Warp the outcome to a particular time point using interpolation.
        Calls flow.Trace2P.warpcstraces()
    smooth : bool
        Smooth your signal by convolution
    smooth_win : int
        Window in sampling points over which to smooth
    smooth_win_dec : int
        Window in sampling points over which to smooth deconvolved data
        Note: this step follows downsampling so window should probably be
        smaller than that for smoothing z-score.

    Returns:
    --------
    Pandas dataframe of all triggered traces and saves to .../output folder.
    """

    # create dir with hashed parameters
    pars = {'trace_type': trace_type, 'cs': cs, 'downsample': downsample,
            'start_time': start_time, 'end_time': end_time,
            'clean_artifacts': clean_artifacts, 'thresh': thresh,
            'warp': warp, 'smooth': smooth, 'smooth_win': smooth_win}
    save_dir = paths.df_path(mouse, pars=pars)

    # build your runs object
    dates = flow.DateSorter.frommeta(mice=[mouse], exclude_tags=['bad'])

    trial_list = []
    # loop through all days for a mouse, build and save pandas df
    for count, d in enumerate(dates):

        # loop through runs on a particular day
        for run in d.runs(exclude_tags=['bad']):

            # get your t2p object
            t2p = run.trace2p()

            # get your cell# from xday alignment
            # use to index along axis=0 in cstraces/run_traces
            cell_ids = flow.xday._read_crossday_ids(run.mouse, run.date)
            cell_ids = [int(s) for s in cell_ids]

            # trigger all trials around stimulus onsets
            run_traces = utils.getcstraces(run, cs=cs, trace_type=trace_type,
                                     start_time=start_time, end_time=end_time,
                                     downsample=True, clean_artifacts=clean_artifacts,
                                     thresh=thresh, warp=warp, smooth=smooth,
                                     smooth_win=smooth_win)

            # make timestamps, downsample is necessary
            timestep = 1/t2p.d['framerate']
            timestamps = np.arange(start_time, end_time, timestep)

            if (t2p.d['framerate'] > 30) and downsample:
                timestamps = timestamps[::2][:np.shape(run_traces)[1]]

            # check that you don't have extra cells
            if len(cell_ids) != np.shape(run_traces)[0]:
                run_traces = run_traces[range(0,len(cell_ids)), :, :]
                warnings.warn(str(run) + ': You have more cell traces than cell_idx: skipping extra cells.')

            # build matrices to match cell, trial, time variables to traces
            trial_mat = np.ones(np.shape(run_traces))
            for trial in range(np.shape(run_traces)[2]):
                trial_mat[:, :, trial] = trial

            cell_mat = np.ones(np.shape(run_traces))
            for cell in range(np.shape(run_traces)[0]):
                cell_mat[cell, :, :] = cell_ids[cell]

            time_mat = np.ones(np.shape(run_traces))
            for timept in range(np.shape(run_traces)[1]):
                time_mat[:, timept, :] = timestamps[timept]

            # reshape and build df
            vec_sz = run_traces.size
            index = pd.MultiIndex.from_arrays([
                [run.mouse] * vec_sz,
                [run.date] * vec_sz,
                [run.run] * vec_sz,
                trial_mat.reshape(vec_sz),
                cell_mat.reshape(vec_sz),
                time_mat.reshape(vec_sz)
                ],
                names=['mouse', 'date', 'run', 'trial_idx',
                       'cell_idx', 'timestamp'])

            # append all runs across a day together in a list
            trial_list.append(pd.DataFrame({'trace': run_traces.reshape(vec_sz)}, index=index))

            # clear your t2p to save memory
            run._t2p = None

        # concatenate and save df for the day
        trial_df = pd.concat(trial_list, axis=0)
        save_path = os.path.join(save_dir, str(d.mouse) + '_' + str(d.date)
                                 + '_df_' + trace_type + '.pkl')
        trial_df.to_pickle(save_path)

        # print output so you don't go crazy waiting
        if verbose:
            print('Day: ' + str(count+1) + ': ' + str(d.mouse)
                  + '_' + str(d.date) + ': ' + str(len(trial_list)))

        # reset trial list before starting new day
        trial_list = []


def trialmeta(mouse, downsample=True, verbose=True):
    """
    Create a pandas dataframe of all of your trial metadata
    for a mouse.

    Parameters:
    -----------
    mouse : str
        Mouse name.
    downsample : bool
        Downsample from 30 to 15 Hz. Must match trigger
        params.

    Returns:
    --------
    Pandas dataframe of all trial metadata saved
    to .../output folder.

    """

    # time before stimulus in triggered data, relative to onset
    start_time = -1

    runs = flow.RunSorter.frommeta(mice=[mouse], exclude_tags=['bad'])

    trial_list = []
    for run in runs:

        # get your t2p object
        t2p = run.trace2p()

        # get the number of trials in your run
        try:
            ntrials = t2p.ntrials
            trial_idx = range(ntrials)
        except:
            run_traces = t2p.cstraces(
                '', start_s=-1, end_s=6,
                trace_type='dff', cutoff_before_lick_ms=-1,
                errortrials=-1, baseline=(-1, 0),
                baseline_to_stimulus=True)
            ntrials = np.shape(run_traces)[2]
            trial_idx = range(ntrials)

        # if there are no stimulus presentations skip "trials"
        if ntrials == 0:
            if verbose:
                print('No CS presentations on', run)
            continue

        # get your learning-state
        run_tags = run.tags
        if 'naive' in run_tags:
            learning_state = 'naive'
        elif 'learning' in run_tags:
            learning_state = 'learning'
        elif 'reversal1' in run_tags:
            learning_state = 'reversal1'
        elif 'reversal2' in run_tags:
            learning_state = 'reversal2'
        learning_state = [learning_state]*len(trial_idx)

        # get hunger-state for all trials, consider hungry if not sated
        if 'sated' in run_tags:
            hunger = 'sated'
        else:
            hunger = 'hungry'
        hunger = [hunger]*len(trial_idx)

        # get relevant trial-distinguising tags excluding kelly, hunger-state, learning-state
        tags = [str(run_tags[s]) for s in range(len(run_tags)) if run_tags[s] != hunger[0]
                and run_tags[s] != learning_state[0]
                and run_tags[s] != 'kelly'
                and run_tags[s] != 'learning_start'
                and run_tags[s] != 'reversal1_start'
                and run_tags[s] != 'reversal2_start']
        if tags == []:  # define as "standard" if the run is not another option
            tags = ['standard']
        tags = [tags[0]]*len(trial_idx)

        # get trialerror ensuring you don't include runthrough at end of trials
        trialerror = np.array(t2p.d['trialerror'][trial_idx])

        # get cs and orientation info for each trial
        lookup = {v: k for k, v in t2p.d['codes'].items()}  # invert dict
        css = [lookup[s] for s in t2p.d['condition'][trial_idx]]
        oris = [t2p.d['orientations'][lookup[s]] for s in t2p.d['condition'][trial_idx]]

        # get mean running speed for time stim is on screen
        # TODO offset may be hardcoded from write_simpcell
        all_onsets = t2p.csonsets()
        try:
            all_offsets = t2p.d['offsets'][0:len(all_onsets)]
        except KeyError:
            all_offsets = all_onsets + (np.round(t2p.d['framerate'])*3)
        if all_offsets[-1] > t2p.nframes:
            all_offsets[-1] = t2p.nframes
        if t2p.d['running'].size > 0:
            speed_vec = t2p.speed()
            speed_vec = speed_vec.astype('float')
            speed = []
            for s in trial_idx:
                try:
                    speed.append(np.nanmean(speed_vec[all_onsets[s]:all_offsets[s]]))
                except:
                    speed.append(np.nan)
            speed = np.array(speed)
        else:
            speed = np.full(len(trial_idx), np.nan)
        # get mean brainmotion for time stim is on screen
        # use first 3 seconds after onset if there is no offset
        if t2p.d['brainmotion'].size > 0:
            xy_vec = t2p.d['brainmotion']
            xy_vec = xy_vec.astype('float')
            brainmotion = []
            for s in trial_idx:
                try:
                    brainmotion.append(np.nanmean(xy_vec[all_onsets[s]:all_offsets[s]]))
                except:
                    brainmotion.append(np.nan)
            brainmotion = np.array(speed)
        else:
            brainmotion = np.full(len(trial_idx), np.nan)

        # get ensure/ensure/firstlick relative to triggered data
        ensure = t2p.ensure()
        ensure = ensure.astype('float')
        ensure[ensure == 0] = np.nan
        ensure = ensure - all_onsets + (np.abs(start_time)*np.round(t2p.d['framerate']))

        quinine = t2p.quinine()
        quinine = quinine.astype('float')
        quinine[quinine == 0] = np.nan
        quinine = quinine - all_onsets + (np.abs(start_time)*np.round(t2p.d['framerate']))

        firstlick = t2p.firstlick('')[trial_idx]
        firstlick = firstlick + (np.abs(start_time)*np.round(t2p.d['framerate']))

        # downsample all timestamps to 15Hz if framerate is 31Hz
        if (t2p.d['framerate'] > 30) and downsample:
            ensure = ensure/2
            quinine = quinine/2
            firstlick = firstlick/2

        # create your index out of relevant variables
        index = pd.MultiIndex.from_arrays([
                    [run.mouse]*len(trial_idx),
                    [run.date]*len(trial_idx),
                    [run.run]*len(trial_idx),
                    trial_idx
                    ],
                    names=['mouse', 'date', 'run', 'trial_idx'])

        data = {'orientation':  oris, 'condition': css,
                'trialerror': trialerror, 'hunger': hunger,
                'learning_state': learning_state, 'tag': tags,
                'firstlick': firstlick, 'ensure': ensure,
                'quinine': quinine, 'speed': speed,
                'brainmotion': brainmotion}

        # append all trials across all runs together into a list
        trial_list.append(pd.DataFrame(data, index=index))

        # clear your t2p to save RAM
        run._t2p = None
        if verbose:
            print('Run: ' + str(run) + ': ' + str(len(trial_list)))

    # concatenate all runs together in final dataframe
    trial_df = pd.concat(trial_list, axis=0)

    # create folder structure if needed
    save_dir = os.path.join(flow.paths.outd, str(mouse))
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # save (trace_type should be irrelevant for metadata)
    save_path = os.path.join(save_dir, str(runs[0].mouse) + '_df_trialmeta.pkl')
    trial_df.to_pickle(save_path)


def get_xdaymap(mouse):
    """
    Build crossday binary map to use for efficient loading/indexing.

    Parameters:
    ----------
    mouse : str
        Mouse name.

    Returns:
    --------
    ndarray
        ncells x ndays - 0 and 1
    """

    # get all days for a mouse
    days = flow.DateSorter.frommeta(mice=[mouse], exclude_tags=['bad'])

    # check all cell ids and build a list for looping over ids
    cell_mat = []
    cell_vec = []
    for day in days:
        cell_ids = flow.xday._read_crossday_ids(day.mouse, day.date)
        cell_ids = [int(s) for s in cell_ids]
        cell_mat.append(cell_ids)
        cell_vec.extend(cell_ids)
    all_cells = np.unique(cell_vec)

    # loop over ids and check if a cell exists for a given day
    cell_map = np.zeros((len(all_cells), len(days)))
    for day_num in range(len(days)):
        for cell_num in range(len(all_cells)):
            if np.isin(all_cells[cell_num], cell_mat[day_num]):
                cell_map[cell_num, day_num] = 1

    return cell_map


def singlecell(mouse, trace_type, cell_idx, xmap=None, word=None):
    """Build df for a single cell loading/indexing efficiently.

    Parameters:
    -----------
    mouse : str
        Mouse name.
    trace_type : str
         {'dff', 'zscore', 'deconvolved'}
    cell_idx : int
        one-indexed
    xmap : ndarray
        xdaymap, can be passed optionally to prevent build

    Returns:
    --------
    pandas df
        df of all traces over all days for a cell
    """

    # get all days for mouse
    days = flow.DateSorter.frommeta(mice=[mouse], exclude_tags=['bad'])

    # build crossday binary map to use for efficient loading/indexing.
    if xmap is None:
        xmap = get_xdaymap(mouse)

    # correct cell_idx (1 indexed) for use indexing (0 indexed)
    cell_num = int(cell_idx - 1)

    # assign folder structure for loading
    if word:
        word_tag = '-' + word
    save_dir = os.path.join(flow.paths.outd, str(mouse),
                            'dfs-' + str(trace_type) + word_tag)

    # load all dfs into list that contain the cell of interest
    cell_xday = []
    for d in np.where(xmap[cell_num, :] == 1)[0]:

        path = os.path.join(save_dir, str(days[d].mouse) + '_'
                            + str(days[d].date) + '_df_' + trace_type + '.pkl')
        dft = pd.read_pickle(path)

        cell_indexer = dft.index.get_level_values('cell_idx') == cell_idx
        dft = dft.loc[cell_indexer, :]
        cell_xday.append(dft)

    dft = pd.concat(cell_xday)

    return dft


def trialbhv(mouse, start_time=-1, end_time=6, verbose=True):
    """ Create a pandas dataframe of all of your triggered behavior traces for a mouse

    Parameters:
    -----------
    trace_type : str; dff, zscore, deconvolved
    start_time : int; in seconds relative to stim onsets, -1 is default
    end_time   : int; in seconds relative to stim onsets, 6 is default

    Returns:
    ________
    Pandas dataframe of all triggered traces and saves to .../output folder
    """

    # build your runs object
    runs = flow.RunSorter.frommeta(mice=[mouse], exclude_tags=['bad'])

    trial_list = []
    for r in range(len(runs)):

        run = runs[r]

        # get your t2p object
        t2p = run.trace2p()

        # trigger all running speed around stim onsets
        speed_traces = behaviortraces(t2p, '', start_s=start_time, end_s=end_time,
                                           trace_type='speed', cutoff_before_lick_ms=-1)
        print(np.shape(speed_trace))
        # trigger all running speed around stim onsets
        lick_traces = behaviortraces(t2p, '', start_s=start_time, end_s=end_time,
                                           trace_type='lick', cutoff_before_lick_ms=-1)
        print(np.shape(lick_trace))
        # make timestamps
        timestep = 1/np.round(t2p.d['framerate'])
        timestamps = np.concatenate((np.arange(start_time, 0, timestep),
                                     np.arange(0, end_time, timestep)))

        # loop through and append each trial (slice of cstraces)
        for trial in range(np.shape(run_traces)[2]):
            index = pd.MultiIndex.from_arrays([
                        [run.mouse] * np.shape(run_traces)[1],
                        [run.date] * np.shape(run_traces)[1],
                        [run.run] * np.shape(run_traces)[1],
                        [int(trial)] * np.shape(run_traces)[1],
                        timestamps
                        ],
                        names=['mouse', 'date', 'run', 'trial_idx', 'timestamp'])

            data = {'speed': np.squeeze(speed_traces[0, :, trial]),
                    'licking': np.squeeze(lick_traces[0, :, trial])}
            # append all trials across all runs together in a list
            trial_list.append(pd.DataFrame(data, index=index))

        # clear your t2p to save RAM
        run._t2p = None

    trial_df = pd.concat(trial_list, axis=0)
    save_path = os.path.join(flow.paths.outd, str(run.mouse) + '_df_bhv.pkl')
    trial_df.to_pickle(save_path)
    if verbose:
        print('Done.')


def behaviortraces(t2p, cs, start_s=-1, end_s=6, trace_type='speed',
                   cutoff_before_lick_ms=-1, errortrials=-1):
    """Return the triggered traces for a particular behavior with flexibility.

    Parameters
    ----------
    t2p : trace2P obj
    start_s : float
        Time before stim to include, in seconds. For backward compatability,
        can also be arg dict.
    end_s : float
        Time after stim to include, in seconds.
    trace_type : {'speed', 'lick', 'pupil'}
        Type of trace to return.
    cutoff_before_lick_ms : int
        Exclude all time around licks by adding NaN's this many ms before
        the first lick after the stim.
    errortrials : {-1, 0, 1}
        -1 is all trials, 0 is only correct trials, 1 is error trials

    Returns
    -------
    ndarray
        ncells x frames x nstimuli/onsets

    """

    if isinstance(start_s, dict):
        raise ValueError('Dicts are no longer accepted')

    start_frame = int(round(start_s*t2p.framerate))
    end_frame = int(round(end_s*t2p.framerate))
    cutoff_frame = int(round(cutoff_before_lick_ms/1000.0*t2p.framerate))

    if trace_type == 'speed' or trace_type == 'running':
        trace = t2p.speed()
    elif trace_type == 'lick' or trace_type == 'licking':
        trace = t2p.licking()
    else:
        print('Invalid trace_type: try speed/running or lick/licking.')
        return
    print('sz: ' + str(np.shape(trace)))
    # Get lick times and onsets
    licks = t2p.d['licking'].flatten()
    ons = t2p.csonsets(cs, errortrials=errortrials)
    out = np.empty((t2p.ncells, end_frame - start_frame, len(ons)))
    out.fill(np.nan)

    # Iterate through onsets, find the beginning and end, and add
    # the appropriate trace type to the output
    for i, onset in enumerate(ons):
        start = start_frame + onset
        end = end_frame + onset

        if i + start >= 0:
            if cutoff_before_lick_ms > -1:
                postlicks = licks[licks > onset]
                if len(postlicks) > 0 and postlicks[0] < end:
                    end = postlicks[0] - cutoff_frame
                    if end < onset:
                        end = start - 1

            if end > t2p.nframes:
                end = t2p.nframes
            if end > start:
                if np.shape(trace) > 0:
                    out[:, :end-start, i] = trace[:, start:end]

    return out


# ----------------------- LATE STAGE DFs -----------------------


def groupmouse_trialfac_summary_days(
        mice=['OA27', 'OA26', 'OA67', 'VF226', 'CC175'],
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        words=['orlando', 'already', 'already', 'already', 'already'],
        group_by='all',
        nan_thresh=0.85,
        score_threshold=None,
        speed_thresh=5,
        rank_num=18,
        verbose=False):

    """
    Cluster tca trial factors based on tuning to different oris, conditions,
    and trialerror values.

    Parameters:
    -----------
    mouse : str; mouse name
    trace_type : str; dff, zscore, deconvolved
    method : str; TCA fit method from tensortools

    Returns:
    --------
    Saves figures to .../analysis folder  .../qc
    """

    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}
    group_pars = {'group_by': group_by}

    conds_by_day = []
    oris_by_day = []
    trialerr_by_day = []
    # neuron_ids_by_day = []
    # neuron_clusters_by_day = []
    # factors_by_day = []
    day_list = []
    df_list_cm_learning = []
    df_list_tempo = []
    df_list_tuning_sc = []
    df_list_tuning = []
    df_list_conds = []
    df_list_error = []
    df_list_index = []
    df_list_runmod = []
    df_list_ramp = []
    df_list_fano = []
    df_list_dprime = []
    df_list_bias = []
    for mnum, mouse in enumerate(mice):

        # load your data
        load_kwargs = {'mouse': mouse,
                       'method': method,
                       'cs': cs,
                       'warp': warp,
                       'word': words[mnum],
                       'group_by': group_by
                       'nan_thresh': nan_thresh,
                       'score_threshold': score_threshold,
                       'rank': rank_num}
        ensemble, ids, clus = load.groupday_tca_model(
            load_kwargs, full_output=True)
        meta = load.groupday_tca_meta(load_kwargs)
        orientation = meta['orientation']
        condition = meta['condition']
        trialerror = meta['trialerror']
        hunger = deepcopy(meta['hunger'])
        speed = meta['speed']
        dates = pd.DataFrame(data={'date': meta.index.get_level_values('date')}, index=meta.index)
        dates = dates['date']  # turn into series for index matching for bool
        learning_state = meta['learning_state']

        df_mouse_tuning = []
        df_mouse_tuning_scaled = []
        df_mouse_conds = []
        df_mouse_error = []
        df_mouse_runmod = []
        df_mouse_ramp = []
        df_mouse_fano = []
        df_mouse_bias = []
        df_mouse_dprime = []

        for day in np.unique(dates):

            # set day indexer
            indexer = dates.isin([day])

            # if all(~np.isin(np.unique(learning_state[indexer]),
            #             ['naive', 'learning'])):
            #     continue

            # ------------ GET DPRIME

            date_obj = flow.Date(mouse, date=day, exclude_tags=['bad'])
            data = [pool.calc.performance.dprime(date_obj)]*rank_num
            dprime_data = {}
            dprime_data['dprime'] = data

            # ------------- GET TUNING

            trial_weights = sort_ensemble.results[rank_num][0].factors[2][:, :]
            tuning_weights = np.zeros((3, rank_num))
            oris_to_check = [0, 135, 270]
            for c, ori in enumerate(oris_to_check):
                tuning_weights[c, :] = np.nanmean(
                    trial_weights[(orientation == ori) & indexer, :], axis=0)
            # normalize using summed mean response to all three
            tuning_total = np.nansum(tuning_weights, axis=0)
            # if np.nansum(tuning_total) > 0:
            for c in range(len(oris_to_check)):
                tuning_weights[c, :] = np.divide(
                    tuning_weights[c, :], tuning_total)
            # dict for creating dataframe
            tuning_data = {}
            for c, errset in enumerate(oris_to_check):
                tuning_data['t' + str(errset)] = tuning_weights[c, :]

            # ------------- GET SCALED TUNING #1

            trial_weights = sort_ensemble.results[rank_num][0].factors[2][:, :]
            tuning_weights = np.zeros((3, rank_num))
            oris_to_check = [0, 135, 270]
            # scale_factor = (np.nanmean(trial_weights[indexer, :], axis=0)/
            #                 np.nanmax(trial_weights[:, :], axis=0))
            scale_factor = np.nanmean(trial_weights[indexer, :], axis=0)
            for c, ori in enumerate(oris_to_check):
                tuning_weights[c, :] = np.nanmean(
                    trial_weights[(orientation == ori) & indexer, :], axis=0)
            # normalize using summed mean response to all three
            tuning_total = np.nansum(tuning_weights, axis=0)
            # if np.nansum(tuning_total) > 0:
            for c in range(len(oris_to_check)):
                tuning_weights[c, :] = np.divide(
                    tuning_weights[c, :], tuning_total)
                # tuning_weights[c, :] = np.multiply(
                #     tuning_weights[c, :], scale_factor)
                tuning_weights[c, scale_factor < 0.01] = 0
            # dict for creating dataframe
            tuning_sc_data = {}
            for c, errset in enumerate(oris_to_check):
                tuning_sc_data['sc' + str(errset)] = tuning_weights[c, :]

            # ------------- GET NORMALIZED drive for preferred tuning

            oris_to_check = [0, 135, 270]
            pref_ori_idx = np.argmax(tuning_weights, axis=0)
            trial_weights = sort_ensemble.results[rank_num][0].factors[2][:, :]
            response_calc = np.zeros((2, rank_num))
            for c, ori in enumerate(pref_ori_idx):  # this is as long as rank #
                pref_indexer = (orientation == oris_to_check[ori])
                response_calc[0, c] = np.nanmax(
                    trial_weights[pref_indexer, c])
                response_calc[1, c] = np.nanmean(
                    trial_weights[pref_indexer & indexer, c])
            # normalize using summed mean response to both running states
            maxnorm = response_calc[1, :]  #/response_calc[0, :]
            # dict for creating dataframe
            # take only running/(running + stationary) value
            tuning_sc_data = {}
            tuning_sc_data['mag_pref_response'] = maxnorm

            # ------------- GET FANOFACTOR for preferred tuning

            oris_to_check = [0, 135, 270]
            pref_ori_idx = np.argmax(tuning_weights, axis=0)
            trial_weights = sort_ensemble.results[rank_num][0].factors[2][:, :]
            running_calc = np.zeros((2, rank_num))
            for c, ori in enumerate(pref_ori_idx):  # this is as long as rank #
                pref_indexer = (orientation == oris_to_check[ori])
                running_calc[0, c] = np.nanvar(
                    trial_weights[pref_indexer & indexer, c], axis=0)
                running_calc[1, c] = np.nanmean(
                    trial_weights[pref_indexer & indexer, c], axis=0)
            # normalize using summed mean response to both running states
            fano = running_calc[0, :]/running_calc[1, :]
            # dict for creating dataframe
            # take only running/(running + stationary) value
            fano_data = {}
            fano_data['fano_factor_pref'] = fano
            for c, ori in enumerate(oris_to_check):  # this is as long as rank #
                pref_indexer = (orientation == ori)
                running_calc[0, :] = np.nanvar(
                    trial_weights[pref_indexer & indexer, :], axis=0)
                running_calc[1, :] = np.nanmean(
                    trial_weights[pref_indexer & indexer, :], axis=0)
                fano = running_calc[0, :]/running_calc[1, :]
                fano_data['fano_factor_' + str(ori)] = fano

            # ------------- GET Condition TUNING

            trial_weights = sort_ensemble.results[rank_num][0].factors[2][:, :]
            conds_to_check = ['plus', 'minus', 'neutral']
            conds_weights = np.zeros((len(conds_to_check), rank_num))
            for c, conds in enumerate(conds_to_check):
                conds_weights[c, :] = np.nanmean(
                    trial_weights[(condition == conds) & indexer, :], axis=0)
            # normalize using summed mean response to all three
            conds_total = np.nansum(conds_weights, axis=0)
            # if np.nansum(conds_total) > 0:
            for c in range(len(conds_to_check)):
                conds_weights[c, :] = np.divide(
                    conds_weights[c, :], conds_total)
            # dict for creating dataframe
            conds_data = {}
            for c, errset in enumerate(conds_to_check):
                conds_data[errset] = conds_weights[c, :]

            # ------------- GET Trialerror TUNING

            if ~np.isin('naive', np.unique(learning_state[indexer])):
                trial_weights = sort_ensemble.results[rank_num][0].factors[2][:, :]
                err_to_check = [[0], [1], [2, 4], [3, 5]]  # hit, miss, CR, FA
                err_val = ['hit', 'miss', 'correct_reject', 'false_alarm']
                error_weights = np.zeros((len(err_to_check), rank_num))
                for c, errset in enumerate(err_to_check):
                    error_weights[c, :] = np.nanmean(
                        trial_weights[trialerror.isin(errset) & indexer, :], axis=0)
                # normalize using summed mean response to all three
                error_total = np.nansum(error_weights, axis=0)
                # if np.nansum(error_total) > 0:
                for c in range(len(err_to_check)):
                    error_weights[c, :] = np.divide(
                        error_weights[c, :], error_total)
                # dict for creating dataframe
                error_data = {}
                for c, errset in enumerate(err_val):
                    error_data[errset] = error_weights[c, :]
            else:
                error_data = []

            # ------------- RUNNING MODULATION for preferred ori

            oris_to_check = [0, 135, 270]
            pref_ori_idx = np.argmax(tuning_weights, axis=0)
            trial_weights = sort_ensemble.results[rank_num][0].factors[2][:, :]
            running_calc = np.zeros((2, rank_num))
            for c, ori in enumerate(pref_ori_idx):  # this is as long as rank #
                pref_indexer = (orientation == oris_to_check[ori])
                running_calc[0, c] = np.nanmean(
                    trial_weights[
                        (speed >= speed_thresh) &
                        pref_indexer
                        & indexer, c],
                    axis=0)
                running_calc[1, c] = np.nanmean(
                    trial_weights[
                        (speed < speed_thresh) &
                        pref_indexer &
                        indexer, c],
                    axis=0)
            # normalize using summed mean response to both running states
            running_mod = np.log2(running_calc[0, :]/running_calc[1, :])
            # dict for creating dataframe
            # take only running/(running + stationary) value
            running_data = {}
            running_data['running_mod'] = running_mod

            # ------------- EARLY/LATE RAMP INDEX within day for preferred ori

            oris_to_check = [0, 135, 270]
            pref_ori_idx = np.argmax(tuning_weights, axis=0)
            trial_weights = sort_ensemble.results[rank_num][0].factors[2][:, :]
            ramp_calc = np.zeros((2, rank_num))
            # build your date indexer for the first and last half of the day
            # need indexer df indices to match
            early_indexer = orientation.isin(['not_this'])
            late_indexer = orientation.isin(['not_this'])
            for d in np.unique(dates):
                day_idx = np.where(dates.isin([d]))[0]
                early_indexer[day_idx[0:int(len(day_idx)/2)]] = True
                late_indexer[day_idx[int(len(day_idx)/2):-1]] = True
            # get early vs late mean dff for preferred ori per component
            for c, ori in enumerate(pref_ori_idx):  # this is as long as rank #
                pref_indexer = (orientation == oris_to_check[ori])
                ramp_calc[0, c] = np.nanmean(
                    trial_weights[
                        early_indexer &
                        pref_indexer
                        & indexer, c],
                    axis=0)
                ramp_calc[1, c] = np.nanmean(
                    trial_weights[
                        late_indexer &
                        pref_indexer &
                        indexer, c],
                    axis=0)
            # normalize using summed mean response to both early/late
            ramp_index = np.log2(ramp_calc[1, :]/ramp_calc[0, :])
            ramp_data = {}
            ramp_data['ramp_index_trials'] = ramp_index

            # ------------ CREATE PANDAS DF

            index = pd.MultiIndex.from_arrays([
                [mouse] * rank_num,
                [day] * rank_num,
                range(1, rank_num+1)
                ],
                names=['mouse',
                       'date',
                       'component'])
            tuning_df = pd.DataFrame(tuning_data, index=index)
            tuning_sc_df = pd.DataFrame(tuning_sc_data, index=index)
            conds_df = pd.DataFrame(conds_data, index=index)
            error_df = pd.DataFrame(error_data, index=index)
            running_df = pd.DataFrame(running_data, index=index)
            ramp_df = pd.DataFrame(ramp_data, index=index)
            fano_df = pd.DataFrame(fano_data, index=index)
            dprime_df = pd.DataFrame(dprime_data, index=index)

            # create lists of dfs for concatenation
            df_mouse_tuning.append(tuning_df)
            df_mouse_tuning_scaled.append(tuning_sc_df)
            df_mouse_conds.append(conds_df)
            df_mouse_error.append(error_df)
            df_mouse_runmod.append(running_df)
            df_mouse_ramp.append(ramp_df)
            df_mouse_fano.append(fano_df)
            df_mouse_dprime.append(dprime_df)
            conds_by_day.append(condition)
            oris_by_day.append(orientation)
            trialerr_by_day.append(trialerror)

        # ------------- CENTER OF MASS for preferred ori trials across learning
        # only calculate once for all days
        # calculate center of mass for your trial factors for learning
        oris_to_check = [0, 135, 270]
        learning_indexer = learning_state.isin(['naive', 'learning'])
        trial_weights = sort_ensemble.results[rank_num][0].factors[2][:, :]
        pref_ori_idx = np.argmax(tuning_weights, axis=0)
        pos = np.arange(1, len(orientation)+1)
        cm_learning= []
        for c, ori in enumerate(pref_ori_idx):  # this is as long as rank #
            pref_indexer = (orientation == oris_to_check[ori])
            pos_pref = pos[learning_indexer & pref_indexer]
            weights_pref = trial_weights[learning_indexer & pref_indexer, c]
            cm_learning.append(
                np.sum(weights_pref * pos_pref)/np.sum(weights_pref))
        data = {'center_of_mass_trials_learning': cm_learning}
        index = pd.MultiIndex.from_arrays([
            [mouse] * rank_num,
            range(1, rank_num+1)
            ],
            names=['mouse',
                   'component'])
        cm_learning_df = pd.DataFrame(data=data, index=index)

        # only get the temporal factors once
        index = pd.MultiIndex.from_arrays([
            [mouse] * rank_num,
            range(1, rank_num+1)
            ],
            names=['mouse',
                   'component'])
        tempo_df = pd.DataFrame(
                sort_ensemble.results[rank_num][0].factors[1][:, :].T,
                index=index)

        # normalize mag of response        df_calc = df_mouse_dprime.pivot(
        df_calc = pd.concat(df_mouse_tuning_scaled, axis=0)
        df_calc = df_calc.unstack()
        df_calc = df_calc/df_calc.max(axis=0)
        df_calc = df_calc.stack()

        # concatenate different columns per mouse
        df_list_tempo.append(tempo_df)
        df_list_index.append(pd.DataFrame(index=index))
        df_list_tuning.append(pd.concat(df_mouse_tuning, axis=0))
        df_list_tuning_sc.append(df_calc)  # divided by max response
        df_list_conds.append(pd.concat(df_mouse_conds, axis=0))
        df_list_error.append(pd.concat(df_mouse_error, axis=0))
        df_list_runmod.append(pd.concat(df_mouse_runmod, axis=0))
        df_list_ramp.append(pd.concat(df_mouse_ramp, axis=0))
        df_list_fano.append(pd.concat(df_mouse_fano, axis=0))
        df_list_cm_learning.append(cm_learning_df)
        df_list_dprime.append(pd.concat(df_mouse_dprime, axis=0))

    # concatenate all mice/runs together in final dataframe
    all_tempo_df = pd.concat(df_list_tempo, axis=0)
    all_tuning_df = pd.concat(df_list_tuning, axis=0)
    all_tuning_sc_df = pd.concat(df_list_tuning_sc, axis=0)
    all_conds_df = pd.concat(df_list_conds, axis=0)
    all_error_df = pd.concat(df_list_error, axis=0)
    all_runmod_df = pd.concat(df_list_runmod, axis=0)
    all_ramp_df = pd.concat(df_list_ramp, axis=0)
    all_fano_df = pd.concat(df_list_fano, axis=0)
    all_cm_learning_df = pd.concat(df_list_cm_learning, axis=0)  # different index
    all_dprime_df = pd.concat(df_list_dprime, axis=0)

    # all_index_df = pd.concat(df_list_index, axis=0)
    trial_factor_df = pd.concat([all_conds_df, all_tuning_df, all_tuning_sc_df,
                                all_error_df, all_fano_df, all_dprime_df,
                                all_runmod_df, all_ramp_df], axis=1)

    # calculate center of mass for your temporal components
    tr = all_tempo_df.values
    pos = np.arange(1, np.shape(tr)[1]+1)
    center_of_mass = []
    for i in range(np.shape(tr)[0]):
        center_of_mass.append(np.sum(tr[i, :] * pos)/np.sum(tr[i, :]))
    data = {'center_of_mass': center_of_mass}
    new_tempo_df = pd.DataFrame(data=data, index=all_tempo_df.index)
    trial_factor_df = pd.merge(
        trial_factor_df.reset_index('date'), new_tempo_df, how='left',
        on=['mouse', 'component']).set_index('date', append=True)

    # merge in trial learning center of mass
    trial_factor_df = pd.merge(
        trial_factor_df.reset_index('date'), all_cm_learning_df, how='left',
        on=['mouse', 'component']).set_index('date', append=True)

    return trial_factor_df, all_tempo_df


def groupmouse_trialfac_summary_stages(
        mice=['OA27', 'OA26', 'OA67', 'VF226', 'CC175'],
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        words=['orlando', 'already', 'already', 'already', 'already'],
        group_by='all',
        nan_thresh=0.85,
        score_threshold=None,
        speed_thresh=5,
        rank_num=18,
        matched_only=True,
        verbose=False):

    """
    Cluster tca trial factors based on tuning to different oris, conditions,
    and trialerror values.

    Parameters:
    -----------
    mouse : str; mouse name
    trace_type : str; dff, zscore, deconvolved
    method : str; TCA fit method from tensortools

    Returns:
    --------
    Saves figures to .../analysis folder  .../qc
    """

    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}
    group_pars = {'group_by': group_by}

    conds_by_day = []
    oris_by_day = []
    trialerr_by_day = []
    # neuron_ids_by_day = []
    # neuron_clusters_by_day = []
    # factors_by_day = []
    day_list = []
    df_list_tempo = []
    df_list_tuning = []
    df_list_conds = []
    df_list_error = []
    df_list_index = []
    df_list_runmod = []
    df_list_ramp = []
    df_list_cm_learning = []
    df_list_ramp_learning = []
    df_list_speed_learning = []
    df_list_amplitude = []
    for mnum, mouse in enumerate(mice):

        # load your data
        load_kwargs = {'mouse': mouse,
                       'method': method,
                       'cs': cs,
                       'warp': warp,
                       'word': words[mnum],
                       'group_by': group_by
                       'nan_thresh': nan_thresh,
                       'score_threshold': score_threshold,
                       'rank': rank_num}
        ensemble, ids, clus = load.groupday_tca_model(
            load_kwargs, full_output=True)
        meta = load.groupday_tca_meta(load_kwargs)
        orientation = meta['orientation']
        condition = meta['condition']
        trialerror = meta['trialerror']
        hunger = deepcopy(meta['hunger'])
        speed = meta['speed']
        dates = meta.reset_index()['date']
        learning_state = meta['learning_state']

        # re-balance your factors ()
        print('Re-balancing factors.')
        for r in ensemble[method].results:
            for i in range(len(ensemble[method].results[r])):
                ensemble[method].results[r][i].factors.rebalance()

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


        # create dataframe of dprime values
        dprime_vec = []
        for date in dates:
            date_obj = flow.Date(mouse, date=date)  # EXCLUDE TAGS??
            dprime_vec.append(pool.calc.performance.dprime(date_obj))
        data = {'dprime': dprime_vec}
        dprime = pd.DataFrame(data=data, index=learning_state.index)
        dprime = dprime['dprime']  # make indices match to meta

        # update trial weights to nan days when all three stim are not shown
        if matched_only:
            keepbool = (condition.values != 'preallocating')
            conds_to_check = ['plus', 'minus', 'neutral']
            for day in np.unique(dates):
                indexer = dates.isin([day]).values
                for c, conds in enumerate(conds_to_check):
                    if np.sum((condition.values == conds) & indexer) <= 0:
                        keepbool[indexer] = False
                        continue
            sort_ensemble.results[rank_num][0].factors[2][~keepbool, :] = np.nan
            print('Removed ' + str(np.sum(keepbool == False)) + ' trials' +
                  ' from stimulus-unmatched days.')

        learning_stages = [
            'naive', 'low_dp_learning', 'high_dp_learning', 'low_dp_rev1',
            'high_dp_rev1', 'pre_rev1', 'pre_rev_wnaive']
        df_mouse_tuning = []
        df_mouse_conds = []
        df_mouse_error = []
        df_mouse_runmod = []
        df_mouse_ramp = []
        df_mouse_amplitude = []
        for stage in learning_stages:

            if stage == 'naive':
                indexer = learning_state.isin(['naive'])
            elif stage == 'low_dp_learning':
                indexer = learning_state.isin(['learning']) & (dprime < 2)
            elif stage == 'high_dp_learning':
                indexer = learning_state.isin(['learning']) & (dprime >= 2)
            elif stage == 'low_dp_rev1':
                indexer = learning_state.isin(['reversal1']) & (dprime < 2)
            elif stage == 'high_dp_rev1':
                indexer = learning_state.isin(['reversal1']) & (dprime >= 2)
            elif stage == 'pre_rev_wnaive':
                indexer = learning_state.isin(['naive', 'learning'])
            elif stage == 'pre_rev1':
                indexer = learning_state.isin(['learning'])

            # ------------- GET Condition Amplitude

            trial_weights = sort_ensemble.results[rank_num][0].factors[2][:, :]
            conds_to_check = ['plus', 'minus', 'neutral']
            conds_weights = np.zeros((len(conds_to_check), rank_num))
            for c, conds in enumerate(conds_to_check):
                conds_weights[c, :] = np.nanmean(
                    trial_weights[(condition == conds) & indexer, :], axis=0)
            # dict for creating dataframe
            ampl_data = {}
            for c, errset in enumerate(conds_to_check):
                ampl_data[errset + '_amp_' + stage] = conds_weights[c, :]

            # ------------- GET TUNING

            trial_weights = sort_ensemble.results[rank_num][0].factors[2][:, :]
            tuning_weights = np.zeros((3, rank_num))
            oris_to_check = [0, 135, 270]
            for c, ori in enumerate(oris_to_check):
                tuning_weights[c, :] = np.nanmean(
                    trial_weights[(orientation == ori) & indexer, :], axis=0)
            # normalize using summed mean response to all three
            tuning_total = np.nansum(tuning_weights, axis=0)
            # if np.nansum(tuning_total) > 0:
            for c in range(len(oris_to_check)):
                tuning_weights[c, :] = np.divide(
                    tuning_weights[c, :], tuning_total)
            # dict for creating dataframe
            tuning_data = {}
            for c, errset in enumerate(oris_to_check):
                 tuning_data['t' + str(errset) + '_' + stage] = tuning_weights[c, :]

            # ------------- GET Condition TUNING

            trial_weights = sort_ensemble.results[rank_num][0].factors[2][:, :]
            conds_to_check = ['plus', 'minus', 'neutral']
            conds_weights = np.zeros((len(conds_to_check), rank_num))
            for c, conds in enumerate(conds_to_check):
                conds_weights[c, :] = np.nanmean(
                    trial_weights[(condition == conds) & indexer, :], axis=0)
            # normalize using summed mean response to all three
            conds_total = np.nansum(conds_weights, axis=0)
            # if np.nansum(conds_total) > 0:
            for c in range(len(conds_to_check)):
                conds_weights[c, :] = np.divide(
                    conds_weights[c, :], conds_total)
            # dict for creating dataframe
            conds_data = {}
            for c, errset in enumerate(conds_to_check):
                conds_data[errset + '_' + stage] = conds_weights[c, :]

            # ------------- GET Trialerror TUNING

            if stage != 'naive':
                trial_weights = sort_ensemble.results[rank_num][0].factors[2][:, :]
                err_to_check = [[0], [1], [2, 4], [3, 5]]  # hit, miss, CR, FA
                err_val = ['hit', 'miss', 'correct_reject', 'false_alarm']
                error_weights = np.zeros((len(err_to_check), rank_num))
                for c, errset in enumerate(err_to_check):
                    error_weights[c, :] = np.nanmean(
                        trial_weights[trialerror.isin(errset) & indexer, :], axis=0)
                # normalize using summed mean response to all three
                error_total = np.nansum(error_weights, axis=0)
                # if np.nansum(error_total) > 0:
                for c in range(len(err_to_check)):
                    error_weights[c, :] = np.divide(
                        error_weights[c, :], error_total)
                # dict for creating dataframe
                error_data = {}
                for c, errset in enumerate(err_val):
                    error_data[errset + '_' + stage] = error_weights[c, :]
            else:
                error_data = []

            # ------------- RUNNING MODULATION for preferred ori

            oris_to_check = [0, 135, 270]
            pref_ori_idx = np.argmax(tuning_weights, axis=0)
            trial_weights = sort_ensemble.results[rank_num][0].factors[2][:, :]
            running_calc = np.zeros((2, rank_num))
            for c, ori in enumerate(pref_ori_idx):  # this is as long as rank #
                pref_indexer = (orientation == oris_to_check[ori])
                running_calc[0, c] = np.nanmean(
                    trial_weights[
                        (speed >= speed_thresh) &
                        pref_indexer
                        & indexer, c],
                    axis=0)
                running_calc[1, c] = np.nanmean(
                    trial_weights[
                        (speed < speed_thresh) &
                        pref_indexer &
                        indexer, c],
                    axis=0)
            # normalize using summed mean response to both running states
            # run_total = np.nansum(running_calc, axis=0)
            # running_mod = running_calc[0, :]/(running_calc[0, :] +
            #                                   running_calc[1, :])
            running_mod = np.log2(running_calc[0, :]/running_calc[1, :])
            # dict for creating dataframe
            # take only running/(running + stationary) value
            running_data = {}
            running_data['running_modulation_' + stage] = running_mod

            # ------------- EARLY/LATE RAMP INDEX within days for preferred ori

            oris_to_check = [0, 135, 270]
            pref_ori_idx = np.argmax(tuning_weights, axis=0)
            trial_weights = sort_ensemble.results[rank_num][0].factors[2][:, :]
            ramp_calc = np.zeros((2, rank_num))
            # build your date indexer for the first and last half of the day
            # need indexer df indices to match
            early_indexer = orientation.isin(['not_this'])
            late_indexer = orientation.isin(['not_this'])
            for day in np.unique(dates):
                day_idx = np.where(dates.isin([day]))[0]
                early_indexer[day_idx[0:int(len(day_idx)/2)]] = True
                late_indexer[day_idx[int(len(day_idx)/2):-1]] = True
            # get early vs late mean dff for preferred ori per component
            for c, ori in enumerate(pref_ori_idx):  # this is as long as rank #
                pref_indexer = (orientation == oris_to_check[ori])
                ramp_calc[0, c] = np.nanmean(
                    trial_weights[
                        early_indexer &
                        pref_indexer
                        & indexer, c],
                    axis=0)
                ramp_calc[1, c] = np.nanmean(
                    trial_weights[
                        late_indexer &
                        pref_indexer &
                        indexer, c],
                    axis=0)
            # normalize using summed mean response to both late/early
            ramp_index = np.log2(ramp_calc[1, :]/ramp_calc[0, :])
            ramp_data = {}
            ramp_data['ramp_index_trials_' + stage] = ramp_index

            # ------------ CREATE PANDAS DF

            index = pd.MultiIndex.from_arrays([
                [mouse] * rank_num,
                range(1, rank_num+1)
                ],
                names=['mouse',
                'component'])
            tuning_df = pd.DataFrame(tuning_data, index=index)
            conds_df = pd.DataFrame(conds_data, index=index)
            error_df = pd.DataFrame(error_data, index=index)
            running_df = pd.DataFrame(running_data, index=index)
            ramp_df = pd.DataFrame(ramp_data, index=index)
            ampl_df = pd.DataFrame(ampl_data, index=index)

            # create lists of dfs for concatenation
            df_mouse_tuning.append(tuning_df)
            df_mouse_conds.append(conds_df)
            df_mouse_error.append(error_df)
            df_mouse_runmod.append(running_df)
            df_mouse_ramp.append(ramp_df)
            conds_by_day.append(condition)
            oris_by_day.append(orientation)
            trialerr_by_day.append(trialerror)
            df_mouse_amplitude.append(ampl_df)

        # only get the temporal factors once
        tempo_df = pd.DataFrame(
                sort_ensemble.results[rank_num][0].factors[1][:, :].T,
                index=index)

        # ------------- CENTER OF MASS for preferred ori trials across learning
        # only calculate once for all days
        # calculate center of mass for your trial factors for learning
        oris_to_check = [0, 135, 270]
        learning_indexer = learning_state.isin(['naive', 'learning'])
        trial_weights = sort_ensemble.results[rank_num][0].factors[2][:, :]
        pref_ori_idx = np.argmax(tuning_weights, axis=0)
        pos = np.arange(1, len(orientation)+1)
        cm_learning = []
        for c, ori in enumerate(pref_ori_idx):  # this is as long as rank #
            pref_indexer = (orientation == oris_to_check[ori])
            pos_pref = pos[learning_indexer & pref_indexer]
            weights_pref = trial_weights[learning_indexer & pref_indexer, c]
            cm_learning.append(
                np.sum(weights_pref * pos_pref)/np.sum(weights_pref))
        data = {'center_of_mass_trials_learning': cm_learning}
        index = pd.MultiIndex.from_arrays([
            [mouse] * rank_num,
            range(1, rank_num+1)
            ],
            names=['mouse',
                   'component'])
        cm_learning_df = pd.DataFrame(data=data, index=index)

        # ------------- RAMP INDEX for preferred ori trials across learning
        # only calculate once for all days
        oris_to_check = [0, 135, 270]
        learning_indexer = learning_state.isin(['learning'])
        trial_weights = sort_ensemble.results[rank_num][0].factors[2][:, :]
        pref_ori_idx = np.argmax(tuning_weights, axis=0)
        ramp_learning = []
        for c, ori in enumerate(pref_ori_idx):  # this is as long as rank #
            pref_indexer = (orientation == oris_to_check[ori])
            weights_pref_low_dp = np.nanmean(trial_weights[
                learning_indexer & (dprime < 2) & pref_indexer, c],
                axis=0)
            weights_pref_high_dp = np.nanmean(trial_weights[
                learning_indexer & (dprime >= 2) & pref_indexer, c],
                axis=0)
            ramp_learning.append(
                np.log2(weights_pref_high_dp/weights_pref_low_dp))
        data = {'ramp_index_learning': ramp_learning}
        index = pd.MultiIndex.from_arrays([
            [mouse] * rank_num,
            range(1, rank_num+1)
            ],
            names=['mouse',
                   'component'])
        ramp_learning_df = pd.DataFrame(data=data, index=index)

        # ------------- RAMP SPEED for preferred ori trials across learning
        # only calculate once for all days
        oris_to_check = [0, 135, 270]
        learning_indexer = learning_state.isin(['learning'])
        trial_weights = speed.values
        pref_ori_idx = np.argmax(tuning_weights, axis=0)
        ramp_learning = []
        for c, ori in enumerate(pref_ori_idx):  # this is as long as rank #
            pref_indexer = (orientation == oris_to_check[ori])
            weights_pref_low_dp = np.nanmean(trial_weights[
                learning_indexer & (dprime < 2) & pref_indexer])
            weights_pref_high_dp = np.nanmean(trial_weights[
                learning_indexer & (dprime >= 2) & pref_indexer])
            ramp_learning.append(
                np.log2(weights_pref_high_dp/weights_pref_low_dp))
        data = {'ramp_index_speed_learning': ramp_learning}
        index = pd.MultiIndex.from_arrays([
            [mouse] * rank_num,
            range(1, rank_num+1)
            ],
            names=['mouse',
                   'component'])
        speed_learning_df = pd.DataFrame(data=data, index=index)

        # concatenate different columns per mouse
        df_list_amplitude.append(pd.concat(df_mouse_amplitude, axis=1))
        df_list_tempo.append(tempo_df)
        df_list_index.append(pd.DataFrame(index=index))
        df_list_tuning.append(pd.concat(df_mouse_tuning, axis=1))
        df_list_conds.append(pd.concat(df_mouse_conds, axis=1))
        df_list_error.append(pd.concat(df_mouse_error, axis=1))
        df_list_runmod.append(pd.concat(df_mouse_runmod, axis=1))
        df_list_ramp.append(pd.concat(df_mouse_ramp, axis=1))
        df_list_cm_learning.append(cm_learning_df)
        df_list_ramp_learning.append(ramp_learning_df)
        df_list_speed_learning.append(speed_learning_df)

    # concatenate all mice/runs together in final dataframe
    all_amplitude_df = pd.concat(df_list_amplitude, axis=0)
    all_tempo_df = pd.concat(df_list_tempo, axis=0)
    all_tuning_df = pd.concat(df_list_tuning, axis=0)
    all_conds_df = pd.concat(df_list_conds, axis=0)
    all_error_df = pd.concat(df_list_error, axis=0)
    all_runmod_df = pd.concat(df_list_runmod, axis=0)
    all_ramp_df = pd.concat(df_list_ramp, axis=0)
    all_cm_learning_df = pd.concat(df_list_cm_learning, axis=0)
    all_ramp_learning_df = pd.concat(df_list_ramp_learning, axis=0)
    all_speed_learning_df = pd.concat(df_list_speed_learning, axis=0)
    trial_factor_df = pd.concat([all_conds_df, all_tuning_df, all_error_df,
                                all_cm_learning_df, all_ramp_learning_df,
                                all_speed_learning_df, all_amplitude_df,
                                all_runmod_df, all_ramp_df], axis=1)

    # calculate center of mass amd ramp index for your temporal components
    tr = all_tempo_df.values
    pos = np.arange(1, np.shape(tr)[1]+1)
    center_of_mass = []
    ramp_index_trace = []
    offset_index_trace = []
    for i in range(np.shape(tr)[0]):
        center_of_mass.append(np.sum(tr[i, :] * pos)/np.sum(tr[i, :]))
        ramp_index_trace.append(
            np.log2(np.nanmean(tr[i, 39:62])/np.nanmean(tr[i, 16:39])))
        offset_index_trace.append(
            np.log2(np.nanmean(tr[i, 62:93])/np.nanmean(tr[i, 16:62])))
    trial_factor_df['center_of_mass'] = center_of_mass
    trial_factor_df['ramp_index_trace'] = ramp_index_trace
    trial_factor_df['ramp_index_trace_offset'] = offset_index_trace

    return trial_factor_df, all_tempo_df

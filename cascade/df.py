"""Functions for building dataframes."""
import os
import flow
import numpy as np
import pandas as pd
import warnings
from . import utils
from . import paths


def trigger(mouse, trace_type='zscore_day', cs='', downsample=True,
            start_time=-1, end_time=6, clean_artifacts='interp',
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
    dates = flow.metadata.DateSorter.frommeta(mice=[mouse])

    trial_list = []
    # loop through all days for a mouse, build and save pandas df
    for count, d in enumerate(dates):

        # loop through runs on a particular day
        for run in d.runs():

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

    runs = flow.metadata.RunSorter.frommeta(mice=[mouse])

    trial_list = []
    for run in runs:

        # get your t2p object
        t2p = run.trace2p()

        # get the number of trials in your run
        try:
            trial_idx = range(t2p.ntrials)
        except:
            run_traces = t2p.cstraces(
                '', start_s=-1, end_s=6,
                trace_type='dff', cutoff_before_lick_ms=-1,
                errortrials=-1, baseline=(-1, 0),
                baseline_to_stimulus=True)
            trial_idx = range(np.shape(run_traces)[2])

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
    days = flow.metadata.DateSorter.frommeta(mice=[mouse])

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
    days = flow.metadata.DateSorter.frommeta(mice=[mouse])

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
    runs = flow.metadata.RunSorter.frommeta(mice=[mouse])

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

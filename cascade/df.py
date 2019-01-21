"""Functions for building dataframes."""
import os
import flow
import numpy as np
import pandas as pd


def trigger(mouse, trace_type='dff', start_time=-1, end_time=6, verbose=True):
    """ Create a pandas dataframe of all of your triggered traces for a mouse

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
    count = 1
    for r in range(len(runs)):

        run = runs[r]

        # get your t2p object
        t2p = run.trace2p()

        # get your cell# from xday alignment
        # use to index along axis=0 in cstraces/run_traces
        cell_ids = flow.xday._read_crossday_ids(run.mouse, run.date)
        cell_ids = [int(s) for s in cell_ids]

        # get all of your trial variables of interest
        # use to index along axis=2 in cstraces/run_traces
        oris = []
        css = []
        for trial in t2p.d['condition']:
            codename = t2p.d['codes'].keys()[t2p.d['codes'].values().index(t2p.d['condition'][trial])]
            oriname = t2p.d['orientations'][codename]
            css.append(codename)
            oris.append(oriname)

        oris = np.array(oris)

        # trigger all trials around stimulus onsets
        run_traces = t2p.cstraces('', start_s=start_time, end_s=end_time, trace_type=trace_type,
                                  cutoff_before_lick_ms=-1, errortrials=-1, baseline=None,
                                  baseline_to_stimulus=True)
        # add downsample option

        # make timestamps
        timestep = 1/np.round(t2p.d['framerate'])
        timestamps = np.concatenate((np.arange(start_time, 0, timestep),
                                     np.arange(0, end_time, timestep)))

        # loop through and append each trial (slice of cstraces)
    #     print(len([int(trial)] * np.shape(run_traces)[1]))
        for trial in range(np.shape(run_traces)[2]):
            for cell in range(np.shape(run_traces)[0]):
                # to take car of an indexing error from origianl pull don't look beyond known ids
                if cell >= len(cell_ids):
                    continue
                index = pd.MultiIndex.from_arrays([
                            [run.mouse] * np.shape(run_traces)[1],
                            [run.date] * np.shape(run_traces)[1],
                            [run.run] * np.shape(run_traces)[1],
                            [int(trial)] * np.shape(run_traces)[1],
                            [cell_ids[cell]] * np.shape(run_traces)[1],
                            timestamps
                            ],
                            names=['mouse', 'date', 'run', 'trial_idx', 'cell_idx', 'timestamp'])

                # append all trials across all runs together in a list
                trial_list.append(pd.DataFrame({'trace': np.squeeze(run_traces[cell, :, trial])}, index=index))

        # clear your t2p to save RAM
        run._t2p = None

        try:
            run_fut = runs[r+1]
            if run.date < run_fut.date:
                trial_df = pd.concat(trial_list, axis=0)
                save_path = os.path.join(flow.paths.outd, str(run.mouse) + '_' + str(run.date)
                                         + '_df_' + trace_type + '.pkl')
                trial_df.to_pickle(save_path)
                if verbose:
                    print('Day: ' + str(count) + ': ' + str(run.mouse)
                          + '_' + str(run.date) + ': ' + str(len(trial_list)))
                trial_list = []
                count = count + 1
        except IndexError:
            trial_df = pd.concat(trial_list, axis=0)
            save_path = os.path.join(flow.paths.outd, str(run.mouse) + '_' + str(run.date)
                                     + '_df_' + trace_type + '.pkl')
            trial_df.to_pickle(save_path)
            if verbose:
                print('Day: ' + str(count) + ': ' + str(run.mouse)
                      + '_' + str(run.date) + ': ' + str(len(trial_list)))


def trialmeta(mouse, trace_type='dff', start_time=-1, end_time=6):
    """ Create a pandas dataframe of all of your trial metadata for a mouse

    Parameters:
    -----------
    trace_type : str, must match trigger params
    start_time : int, must match trigger params
    end_time   : int, must match trigger params

    Returns:
    ________
    Pandas dataframe of all trial metadata and saves to .../output folder

    """

    runs = flow.metadata.RunSorter.frommeta(mice=[mouse])

    # triggering parameters
    start_time = start_time
    end_time = end_time
    trace_type = trace_type

    trial_list = []
    for run in runs:

        # get your t2p object
        t2p = run.trace2p()

        # trigger all trials around stimulus onsets to get trial number
        run_traces = t2p.cstraces('', start_s=start_time, end_s=end_time, trace_type=trace_type,
                        cutoff_before_lick_ms=-1, errortrials=-1, baseline=None,
                        baseline_to_stimulus=True)
        trial_idx = range(np.shape(run_traces)[2])

        # get your runtype and relevant tags
        run_type = run.run_type
        run_type = [run_type]*len(trial_idx)

        # get hunger state for all trials, consider hungry if not sated
        run_tags = run.tags
        hunger = ['hungry' for s in range(len(run_tags)) if run_tags[s] == 'hungry']
        if hunger == []:
            hunger = ['hungry' for s in range(len(run_tags)) if run_tags[s] != 'sated']
        hunger = np.unique(hunger)[0]
        hunger = [hunger]*len(trial_idx)

        # get relevant trial-distinguising tags excluding kelly, hunger state, and run_type
        tags = [run_tags[s] for s in range(len(run_tags)) if run_tags[s] != hunger[0]
                and run_tags[s] != 'kelly'
                and run_tags[s] != run_type[0]]
        if tags == []:  # define as "standard" if the run is not another option
            tags = ['standard']
        tags = tags[0]
        tags = [tags]*len(trial_idx)

        # get trialerror ensureing you don't include runthrough at end of trials
        trialerror = np.array(t2p.d['trialerror'][trial_idx])

        # get cs and orientation infor for each trial
        oris = []
        css = []
        for trial in t2p.d['condition'][trial_idx]:
            # get cs and ori
            codename = t2p.d['codes'].keys()[t2p.d['codes'].values().index(trial)]
            oriname = t2p.d['orientations'][codename]
            css.append(codename)
            oris.append(oriname)

        # get mean running speed for time stim is on screen
        all_onsets = t2p.csonsets()
        all_offsets = t2p.d['offsets'][0:len(all_onsets)]
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

        # get offset relative to triggered data
    #     offsets = all_offsets - all_onsets + (np.abs(start_time)*np.round(t2p.d['framerate']))
    #     offsets = offsets.flatten()

        # get ensure relative to triggered data
        ensure = t2p.ensure()
        ensure = ensure.astype('float')
        ensure[ensure == 0] = np.nan
        ensure = ensure - all_onsets + (np.abs(start_time)*np.round(t2p.d['framerate']))

        # get quinine relative to triggered data
        quinine = t2p.quinine()
        quinine = quinine.astype('float')
        quinine[quinine == 0] = np.nan
        quinine = quinine - all_onsets + (np.abs(start_time)*np.round(t2p.d['framerate']))

        # get firstlick for trial
        firstlick = t2p.firstlick('')[trial_idx]
        firstlick = firstlick + (np.abs(start_time)*np.round(t2p.d['framerate']))

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
                'run_type': run_type, 'tag': tags,
                'firstlick': firstlick, 'ensure': ensure,
                'quinine': quinine, 'speed': speed}

        # append all trials across all runs together into a list
        trial_list.append(pd.DataFrame(data, index=index))

        # clear your t2p to save RAM
        run._t2p = None
        print('Run: ' + str(run) + ': ' + str(len(trial_list)))

    # concatenate all runs together in final dataframe
    trial_df = pd.concat(trial_list, axis=0)

    # save
    save_path = os.path.join(flow.paths.outd, str(runs[0].mouse) + '_df_' + trace_type + '_trialmeta.pkl')
    trial_df.to_pickle(save_path)


def trialbhv(mouse, start_time=-1, end_time=6, verbose=True):
    """ Create a pandas dataframe of all of your triggered behvaior traces for a mouse

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

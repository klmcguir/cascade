"""Functions for building dataframes."""
import os
import flow
import numpy as np
import pandas as pd


def trigger(mouse, trace_type='dff', start_time=-1, end_time=6, verbose=True)
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

    # triggering parameters
    start_time = start_time
    end_time = end_time
    trace_type = trace_type

    trial_list = []
    chunk = 1
    for run in runs:

        # get your t2p object
        t2p = run.trace2p()

        # get your cell# from xday alignment
        # use to index along axis=0 in cstraces/run_traces
        cell_ids = flow.xday._read_crossday_ids(run.mouse, run.date)
        cell_ids = [int(s) for s in cell_ids]
        max_id = np.max(cell_ids)
        # print(np.shape(cell_ids))

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
        # print(np.shape(oris))
        # print(oris)
        # print(css)

        # trigger all trials around stimulus onsets
        run_traces = t2p.cstraces('', start_s=start_time, end_s=end_time, trace_type=trace_type,
                                  cutoff_before_lick_ms=-1, errortrials=-1, baseline=None,
                                  baseline_to_stimulus=True)

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
                trial_list.append(pd.DataFrame({'trace': np.squeeze(run_traces[cell,:,trial])}, index=index))

        # clear your t2p to save RAM
        run._t2p = None

        if verbose:
            print('Run: ' + str(run) + ': ' + str(len(trial_list)))
        trial_df = pd.concat(trial_list, axis=0)
        save_path = os.path.join(flow.paths.outd, str(run) + '_df_' + trace_type + '.pkl')
        trial_df.to_pickle(save_path)

        # reset trial_list
        trial_list = []


def trialmeta(mouse, trace_type='dff', start_time=-1, end_time=6)
    """ Create a pandas dataframe of all of your trial metadata for a mouse

    Parameters:
    -----------
    trace_type : str, does not matter for df_metadata, only included for consistency
    start_time : int, does not matter for df_metadata, only included for consistency
    end_time   : int, does not matter for df_metadata, only included for consistency

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

        # get your runtype and relevant tags
        run_type = run.run_type
        run_tags = run.tags
        # get hunger state for all trials, consider hungry if not sated
        hunger = ['hungry' for s in range(len(run_tags)) if run_tags[s] == 'hungry']
        if hunger == []:
            hunger = ['hungry' for s in range(len(run_tags)) if run_tags[s] != 'sated']
        hunger = np.unique(hunger)[0]
        # get relevant trial-distinguising tags excluding kelly, hunger state, and run_type
        tags = [run_tags[s] for s in range(len(run_tags)) if run_tags[s] != hunger
                and run_tags[s] != 'kelly'
                and run_tags[s] != run_type]
        if tags == []:  # define as "standard" if the run is not another option
            tags = ['standard']
        tags = tags[0]

        # get all of your trial variables of interest
        # use to index along axis=2 in cstraces/run_traces
        oris = []
        css = []
        trialerror = []
        for trial in t2p.d['condition']:

            # get cs and ori
            codename = t2p.d['codes'].keys()[t2p.d['codes'].values().index(t2p.d['condition'][trial])]
            oriname = t2p.d['orientations'][codename]
            css.append(codename)
            oris.append(oriname)

            # get trialerror
            trialerror.append(t2p.d['trialerror'][trial])

        oris = np.array(oris)

        # trigger all trials around stimulus onsets to get trial number
        run_traces = t2p.cstraces('', start_s=start_time, end_s=end_time, trace_type=trace_type,
                                  cutoff_before_lick_ms=-1, errortrials=-1, baseline=None,
                                  baseline_to_stimulus=True)

        # loop through and append each trial (slice of cstraces)
        for trial in range(np.shape(run_traces)[2]):
            index = pd.MultiIndex.from_arrays([
                        [run.mouse],
                        [run.date],
                        [run.run],
                        [oris[trial]],
                        [css[trial]],
                        [trialerror[trial]],
                        [hunger],
                        [run_type],
                        [tags]
                        ],
                        names=['mouse', 'date', 'run', 'orientation', 'condition', 'trialerror', 'hunger', 'run_type', 'tag'])

            # append all trials across all runs together in a list
            trial_list.append(pd.DataFrame({'trial_idx': trial}, index=index))

        # clear your t2p to save RAM
        run._t2p = None
        print('Run: ' + str(run) + ': ' + str(len(trial_list)))

    # concatenate all runs together in final dataframe
    trial_df = pd.concat(trial_list, axis=0)

    # save
    save_path = os.path.join(flow.paths.outd, str(runs[0].mouse) + '_df_klg_' + trace_type + '_trialmeta.pkl')
    trial_df.to_pickle(save_path)

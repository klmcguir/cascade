"""Functions for general calculations and data management."""
import flow
import pool
import numpy as np


def getcstraces(run, cs='', trace_type='zscore_iti', start_time=-1, end_time=6,
                downsample=True, clean_artifacts=None, thresh=10,
                warp=False):
    """
    Wrapper function for flow.Trace2P.cstraces() or .warpsctraces().
    Adds in artifact removal, and multiple types of z-score calc.

    Parameters
    ----------
    run : Run object
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
    warp_ : bool
        Warp the outcome to a particular time point using interpolation.
        Calls flow.Trace2P.warpcstraces()

    Result
    ------
    np.ndarray
        ncells x frames x nstimuli/onsets

    """

    t2p = run.trace2p()
    date = flow.metadata.sorters.Date(mouse=run.mouse, date=run.date)

    # trigger all trials around stimulus onsets
    if warp:
        print('warping')
        run_traces = t2p.warpcstraces(cs, start_s=start_time, end_s=end_time,
                                      trace_type='dff', cutoff_before_lick_ms=-1,
                                      errortrials=-1, baseline=(-1, 0),
                                      move_outcome_to=4, baseline_to_stimulus=True)
    else:
        run_traces = t2p.cstraces(cs, start_s=start_time, end_s=end_time,
                                  trace_type='dff', cutoff_before_lick_ms=-1,
                                  errortrials=-1, baseline=(-1, 0),
                                  baseline_to_stimulus=True)

    # clean artifacts
    if clean_artifacts:
        sz = np.shape(run_traces)  # dims: (cells, time, trials)
        run_traces = run_traces.reshape((sz[0], sz[1]*sz[2]))
        nanpad = np.zeros(np.shape(run_traces))
        nanpad[np.abs(run_traces) > thresh] = 1
        # dialate around threshold crossings
        for cell in range(sz[0]):
            nanpad[cell, :] = np.convolve(nanpad[cell, :], np.ones(3), mode='same')
        # clear with nans or interpolation
        if clean_artifacts.lower() == 'nan':
            print('cleaning with nan')
            run_traces[nanpad != 0] = np.nan
        elif clean_artifacts.lower() == 'interp':
            print('cleaning with nan')
            x = np.arange(0, sz[1]*sz[2])
            for cell in range(sz[0]):
                # x = np.where(np.isfinite(run_traces[cell, :]))[0]
                if np.nansum(nanpad[cell, :]) > 0:
                    blank = np.where(nanpad[cell, :] != 0)[0]
                    keep = np.where(nanpad[cell, :] == 0)[0]
                    run_traces[cell, blank] = np.interp(x[blank], x[keep],
                                                        run_traces[cell, keep])
        run_traces = run_traces.reshape((sz[0], sz[1], sz[2]))

    # z-score
    if 'zscore' in trace_type:
        sz = np.shape(run_traces)  # dims: (cells, time, trials)
        arti = False if clean_artifacts is None else True
        if trace_type == 'zscore':
            print('zscoring')
            mu = pool.calc.zscore.mu(date, nan_artifacts=arti,
                                     thresh=thresh)
            sigma = pool.calc.zscore.sigma(date, nan_artifacts=arti,
                                           thresh=thresh)
        elif trace_type == 'zscore_iti':
            print('zscoring_iti')
            mu = pool.calc.zscore.iti_mu(date, nan_artifacts=arti)
            sigma = pool.calc.zscore.iti_sigma(date, nan_artifacts=arti)
        run_traces = ((((run_traces
                      .reshape((sz[0], sz[1]*sz[2])).T - mu)/sigma).T)
                      .reshape((sz[0], sz[1], sz[2])))


    # downsample all traces/timestamps to 15Hz if framerate is 31Hz
    if (t2p.d['framerate'] > 30) and downsample:

        # make sure divisible by 2
        sz = np.shape(run_traces)  # dims: (cells, time, trials)
        if sz[1] % 2 == 1:
            run_traces = run_traces[:, :-1, :]
            sz = np.shape(run_traces)

        # downsample
        ds_traces = np.zeros((sz[0], int(sz[1]/2), sz[2]))
        for trial in range(sz[2]):
            a = run_traces[:, :, trial].reshape(sz[0], int(sz[1]/2), 2)
            ds_traces[:, :, trial] = np.nanmean(a, axis=2)

        run_traces = ds_traces

    return run_traces

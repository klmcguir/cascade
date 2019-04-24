"""Functions for general calculations and data management."""
import flow
import pool
import numpy as np
import warnings
import pandas as pd
from . import tca


def getdailycstraces(
        # DateSorter params
        DateSorter,

        # cstrace params
        cs='',
        trace_type='zscore_day',
        start_time=-1,
        end_time=6,
        downsample=True,
        clean_artifacts=None,
        thresh=17.5,
        warp=False,
        smooth=False,
        smooth_win=5,
        smooth_win_dec=3):
    """
    Wrapper function for getcstraces. Gets cstraces for a DateSorter object.
    """

    if isinstance(DateSorter, flow.sorters.Date):
        runs = DateSorter.runs(run_types='training', tags='hungry')

    runlist = []
    for run in runs:
        trs = getcstraces(
                run, cs=cs, trace_type=trace_type,
                start_time=start_time, end_time=end_time,
                downsample=downsample, clean_artifacts=clean_artifacts,
                thresh=thresh, warp=warp, smooth=smooth,
                smooth_win=smooth_win, smooth_win_dec=smooth_win_dec)
        runlist.append(trs)
    cstraces = np.concatenate(runlist, axis=2)

    return cstraces


def getdailymeta(
        DateSorter,
        tags='hungry',
        run_types='training'):
    """
    Wrapper function for tca._trialmetafromrun(run). Gets trial metadata for a
    DateSorter object.
    """
    if isinstance(DateSorter, flow.sorters.Date):
        runs = DateSorter.runs(run_types=run_types, tags=tags)

    metalist = []
    for run in runs:
        metalist.append(tca._trialmetafromrun(run))
    meta = pd.concat(metalist, axis=0)

    return meta


def getcstraces(
        run,
        cs='',
        trace_type='zscore_day',
        start_time=-1,
        end_time=6,
        downsample=True,
        clean_artifacts=None,
        thresh=17.5,
        warp=False,
        smooth=True,
        smooth_win=5,
        smooth_win_dec=3):
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

    Result
    ------
    np.ndarray
        ncells x frames x nstimuli/onsets

    """

    t2p = run.trace2p()
    date = run.parent
    date.set_subset(run.cells)

    # z-score
    if 'zscore' in trace_type.lower():

        # get dff for creation of alternative trace_types
        traces = t2p.trace('dff')

        # clean artifacts
        if clean_artifacts:
            nanpad = np.zeros(np.shape(traces))
            nanpad[np.abs(traces) > thresh] = 1
            print(np.sum(nanpad.flatten()))
            # dialate around threshold crossings
            for cell in range(np.shape(traces)[0]):
                nanpad[cell, :] = np.convolve(nanpad[cell, :], np.ones(3), mode='same')
            # clear with nans or interpolation
            if clean_artifacts.lower() == 'nan':
                traces[nanpad != 0] = np.nan
            elif clean_artifacts.lower() == 'interp':
                x = np.arange(0, np.shape(traces)[1])
                for cell in range(np.shape(traces)[0]):
                    # x = np.where(np.isfinite(run_traces[cell, :]))[0]
                    if np.nansum(nanpad[cell, :]) > 0:
                        blank = np.where(nanpad[cell, :] != 0)[0]
                        keep = np.where(nanpad[cell, :] == 0)[0]
                        traces[cell, blank] = np.interp(x[blank], x[keep],
                                                        traces[cell, keep])

        # z-score
        if 'zscore' in trace_type.lower():
            arti = False if clean_artifacts is None else True
            if trace_type.lower() == 'zscore_day':
                mu = pool.calc.zscore.mu(date, nan_artifacts=arti,
                                         thresh=thresh)
                sigma = pool.calc.zscore.sigma(date, nan_artifacts=arti,
                                               thresh=thresh)
            elif trace_type.lower() == 'zscore_iti':
                mu = pool.calc.zscore.iti_mu(date, window=4, nan_artifacts=arti,
                                             thresh=thresh)
                sigma = pool.calc.zscore.iti_sigma(date, window=4, nan_artifacts=arti,
                                                   thresh=thresh)
            elif trace_type.lower() == 'zscore_run':
                mu = pool.calc.zscore.run_mu(run, nan_artifacts=arti,
                                             thresh=thresh)
                sigma = pool.calc.zscore.run_sigma(run, nan_artifacts=arti,
                                                   thresh=thresh)
            else:
                print('WARNING: did not recognize z-scoring method.')
            traces = ((traces.T - mu)/sigma).T

        #
        if smooth:
            for cell in range(np.shape(traces)[0]):
                traces[cell, :] = np.convolve(traces[cell, :], np.ones(smooth_win,
                                              dtype=np.float64)/smooth_win, 'same')

        # add new trace type into t2p
        t2p.add_trace(trace_type, traces)

    # trigger all trials around stimulus onsets
    if warp:
        run_traces = t2p.warpcstraces(cs, start_s=start_time, end_s=end_time,
                                      trace_type=trace_type, cutoff_before_lick_ms=-1,
                                      errortrials=-1, baseline=(-1, 0),
                                      move_outcome_to=4, baseline_to_stimulus=True)
    else:
        run_traces = t2p.cstraces(cs, start_s=start_time, end_s=end_time,
                                  trace_type=trace_type, cutoff_before_lick_ms=-1,
                                  errortrials=-1, baseline=(-1, 0),
                                  baseline_to_stimulus=True)

    # downsample all traces/timestamps to 15Hz if framerate is 31Hz
    if (t2p.d['framerate'] > 30) and downsample:
        # make sure divisible by 2
        sz = np.shape(run_traces)  # dims: (cells, time, trials)
        if sz[1] % 2 == 1:
            run_traces = run_traces[:, :-1, :]
            sz = np.shape(run_traces)
        # downsample
        # ignore python and numpy divide by zero warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with np.errstate(invalid='ignore', divide='ignore'):
                ds_traces = np.zeros((sz[0], int(sz[1]/2), sz[2]))
                for trial in range(sz[2]):
                    a = run_traces[:, :, trial].reshape(sz[0], int(sz[1]/2), 2)
                    if trace_type.lower() == 'deconvolved':
                        ds_traces[:, :, trial] = np.nanmax(a, axis=2)
                    else:
                        ds_traces[:, :, trial] = np.nanmean(a, axis=2)

        run_traces = ds_traces

    # smooth deconvolved data
    if smooth and (trace_type.lower() == 'deconvolved'):
        sz = np.shape(run_traces)  # dims: (cells, time, trials)
        run_traces = run_traces.reshape((sz[0], sz[1]*sz[2]))
        for cell in range(sz[0]):
            run_traces[cell, :] = np.convolve(run_traces[cell, :],
                                              np.ones(smooth_win_dec,
                                              dtype=np.float64)/smooth_win_dec,
                                              'same')
        run_traces = run_traces.reshape((sz[0], sz[1], sz[2]))

    return run_traces

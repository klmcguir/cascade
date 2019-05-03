"""Calculations to be saved to mongoDB database"""
from pool.database import memoize
from .. import paths
import tensortools as tt
import numpy as np
import pandas as pd
import os


@memoize(across='mouse', updated=190429, returns='other', large_output=False)
def groupday_varex_byday(
        mouse,
        trace_type='zscore_day',
        method='ncp_bcd',
        cs='',
        warp=False,
        word=None,
        group_by='all',
        nan_thresh=0.85,
        rectified=False,
        verbose=False):
    """
    Plot reconstruction error as variance explained across all whole groupday
    TCA decomposition ensemble.

    Parameters:
    -----------
    mouse : str; mouse object
    trace_type : str; dff, zscore, deconvolved
    method : str; TCA fit method from tensortools

    Returns:
    --------
    Saves figures to .../analysis folder  .../qc
    """

    mouse = mouse.mouse
    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}
    group_pars = {'group_by': group_by}

    # if cells were removed with too many nan trials
    if nan_thresh:
        nt_tag = '_nantrial' + str(nan_thresh)
    else:
        nt_tag = ''

    # load dir
    load_dir = paths.tca_path(
        mouse, 'group', pars=pars, word=word, group_pars=group_pars)
    tensor_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_group_decomp_' + str(trace_type) + '.npy')
    input_tensor_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_group_tensor_' + str(trace_type) + '.npy')
    meta_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_df_group_meta.pkl')

    # load your data
    ensemble = np.load(tensor_path)
    ensemble = ensemble.item()
    V = ensemble[method]
    X = np.load(input_tensor_path)
    meta = pd.read_pickle(meta_path)
    orientation = meta['orientation']
    condition = meta['condition']
    dates = meta.reset_index()['date']

    # rectify input tensor (only look at nonnegative variance)
    if rectified:
        X[X < 0] = 0

    # create vectors for dataframe
    dayframe = {}
    varex = []
    varex_smu = []
    varex_mu = []
    date = []
    rank = []
    for r in V.results:
        # model
        bU = V.results[r][0].factors.full()
        # mean response of neuron across trials
        mU = np.nanmean(
            X, axis=2, keepdims=True) * np.ones((1, 1, np.shape(X)[2]))
        # smoothed response of neuron across time
        smU = np.convolve(
            X.reshape((X.size)),
            np.ones(5, dtype=np.float64)/5, 'same').reshape(np.shape(X))
        # calculate variance explained per day
        for day in np.unique(dates):
            day_bool = dates.isin(day)
            bUd = bU[:, :, day_bool]
            mUd = mU[:, :, day_bool]
            smUd = smU[:, :, day_bool]
            bX = X[:, :, day_bool]
            rank.append(r)
            date.append(day)
            varex.append(
                (np.nanvar(bX) - np.nanvar(bX - bUd)) / np.nanvar(bX))
            varex_mu.append(
                (np.nanvar(bX) - np.nanvar(bX - mUd)) / np.nanvar(bX))
            varex_smu.append(
                (np.nanvar(bX) - np.nanvar(bX - smUd)) / np.nanvar(bX))

    # make dataframe of data
    # create your index out of relevant variables
    index = pd.MultiIndex.from_arrays(
        [[mouse]*len(varex)],
        names=['mouse'])

    data = {'rank': rank,
            'date':  date,
            'variance_explained_tcamodel': varex,
            'variance_explained_smoothmodel': varex_smu,
            'variance_explained_meanmodel': varex_mu}

    dfvar = pd.DataFrame(data, index=index)

    return dfvar


@memoize(across='mouse', updated=190429, returns='other', large_output=False)
def groupday_varex_byday_bycomp(
        mouse,
        trace_type='zscore_day',
        method='ncp_bcd',
        cs='',
        warp=False,
        word=None,
        group_by='all',
        nan_thresh=0.85,
        rectified=False,
        verbose=False):
    """
    Plot reconstruction error as variance explained across all whole groupday
    TCA decomposition ensemble.

    Parameters:
    -----------
    mouse : str; mouse object
    trace_type : str; dff, zscore, deconvolved
    method : str; TCA fit method from tensortools

    Returns:
    --------
    Saves figures to .../analysis folder  .../qc
    """

    mouse = mouse.mouse
    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}
    group_pars = {'group_by': group_by}

    # if cells were removed with too many nan trials
    if nan_thresh:
        nt_tag = '_nantrial' + str(nan_thresh)
    else:
        nt_tag = ''

    # load dir
    load_dir = paths.tca_path(
        mouse, 'group', pars=pars, word=word, group_pars=group_pars)
    tensor_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_group_decomp_' + str(trace_type) + '.npy')
    input_tensor_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_group_tensor_' + str(trace_type) + '.npy')
    meta_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_df_group_meta.pkl')

    # load your data
    ensemble = np.load(tensor_path)
    ensemble = ensemble.item()
    V = ensemble[method]
    X = np.load(input_tensor_path)
    meta = pd.read_pickle(meta_path)
    orientation = meta['orientation']
    condition = meta['condition']
    dates = meta.reset_index()['date']

    # rectify input tensor (only look at nonnegative variance)
    if rectified:
        X[X < 0] = 0

    # create vectors for dataframe
    varex = []
    date = []
    rank = []
    component = []
    for r in V.results:
        for fac_num in range(np.shape(V.results[r][0].factors[0][:, :])[1]):
            # reconstruct single component model
            a = V.results[r][0].factors[0][:, fac_num]
            b = V.results[r][0].factors[1][:, fac_num]
            c = V.results[r][0].factors[2][:, fac_num]
            ab = a[:, None] @ b[None, :]
            abc = ab[:, :, None] @ c[None, :]
            # calculate variance explained per day
            for day in np.unique(dates):
                day_bool = dates.isin([day])
                bUd = abc[:, :, day_bool]
                bX = X[:, :, day_bool]
                rank.append(r)
                date.append(day)
                component.append(fac_num+1)
                varex.append(
                    (np.nanvar(bX) - np.nanvar(bX - bUd)) / np.nanvar(bX))

    # make dataframe of data
    # create your index out of relevant variables
    index = pd.MultiIndex.from_arrays(
        [[mouse]*len(varex)],
        names=['mouse'])

    data = {'rank': rank,
            'date':  date,
            'component': component,
            'variance_explained_tcamodel': varex}

    dfvar = pd.DataFrame(data, index=index)

    return dfvar

@memoize(across='mouse', updated=190429, returns='other', large_output=False)
def groupday_varex_byday_bycomp_bycell(
        mouse,
        trace_type='zscore_day',
        method='ncp_bcd',
        cs='',
        warp=False,
        word=None,
        group_by='all',
        nan_thresh=0.85,
        rectified=False,
        verbose=False):
    """
    Plot reconstruction error as variance explained across all whole groupday
    TCA decomposition ensemble.

    Parameters:
    -----------
    mouse : str; mouse object
    trace_type : str; dff, zscore, deconvolved
    method : str; TCA fit method from tensortools

    Returns:
    --------
    Saves figures to .../analysis folder  .../qc
    """

    mouse = mouse.mouse
    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}
    group_pars = {'group_by': group_by}

    # if cells were removed with too many nan trials
    if nan_thresh:
        nt_tag = '_nantrial' + str(nan_thresh)
    else:
        nt_tag = ''

    # load dir
    load_dir = paths.tca_path(
        mouse, 'group', pars=pars, word=word, group_pars=group_pars)
    tensor_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_group_decomp_' + str(trace_type) + '.npy')
    ids_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_group_ids_' + str(trace_type) + '.npy')
    input_tensor_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_group_tensor_' + str(trace_type) + '.npy')
    meta_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_df_group_meta.pkl')

    # load your data
    ensemble = np.load(tensor_path)
    ensemble = ensemble.item()
    V = ensemble[method]
    X = np.load(input_tensor_path)
    ids = np.load(ids_path)
    meta = pd.read_pickle(meta_path)
    orientation = meta['orientation']
    condition = meta['condition']
    dates = meta.reset_index()['date']

    # rectify input tensor (only look at nonnegative variance)
    if rectified:
        X[X < 0] = 0

    # create vectors for dataframe
    varex = []
    date = []
    rank = []
    component = []
    cell_idx = []
    cell_id = []
    for r in V.results:
        for fac_num in range(np.shape(V.results[r][0].factors[0][:, :])[1]):
            # reconstruct single component model
            a = V.results[r][0].factors[0][:, fac_num]
            b = V.results[r][0].factors[1][:, fac_num]
            c = V.results[r][0].factors[2][:, fac_num]
            ab = a[:, None] @ b[None, :]
            abc = ab[:, :, None] @ c[None, :]
            # calculate variance explained per day
            for day in np.unique(dates):
                day_bool = dates.isin([day])
                abcd = abc[:, :, day_bool]
                xxxx = X[:, :, day_bool]
                for cell_num in range():
                    cell_id = ids[cell_num]
                    bX = xxxx[cell_num, :, :]
                    bU = abcd[cell, :, :]
                    rank.append(r)
                    date.append(day)
                    component.append(fac_num+1)
                    cell_idx.append(cell_num)
                    cell_id.append(cell_id)
                    varex.append(
                        (np.nanvar(bX) - np.nanvar(bX - bU)) / np.nanvar(bX))

    # make dataframe of data
    # create your index out of relevant variables
    index = pd.MultiIndex.from_arrays(
        [[mouse]*len(varex)],
        names=['mouse'])

    data = {'rank': rank,
            'date':  date,
            'component': component,
            'variance_explained_tcamodel': varex}

    dfvar = pd.DataFrame(data, index=index)

    return dfvar

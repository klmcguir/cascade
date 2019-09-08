"""Calculations to be saved to mongoDB database"""
from pool.database import memoize
from .. import paths
import tensortools as tt
import numpy as np
import pandas as pd
import bottleneck as bn
import os
from sklearn.decomposition import PCA
from copy import deepcopy


@memoize(across='mouse', updated=190805, returns='other', large_output=True)
def groupday_varex(
        mouse,
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        word=None,
        group_by='all',
        nan_thresh=0.85,
        score_threshold=None,
        rectified=True,
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

    # load your data
    load_kwargs = {'mouse': mouse,
                   'method': method,
                   'cs': cs,
                   'warp': warp,
                   'word': words[c],
                   'group_by': group_by,
                   'nan_thresh': nan_thresh,
                   'score_threshold': score_threshold}
    V, _ = load.groupday_tca_model(**load_kwargs, unsorted=True)
    X = load.groupday_tca_input_tensor(**load_kwargs)

    # rectify input tensor (only look at nonnegative variance)
    if rectified:
        X[X < 0] = 0

    # get reconstruction error as variance explained
    # create vectors for dataframe
    varex = []
    rank = []
    iteration = []
    for r in V.results:
        for it in range(0, len(V.results[r])):
            U = V.results[r][it].factors.full()
            varex.append(1 - (bn.nanvar(X - U)/bn.nanvar(X)))
            rank.append(r)
            iteration.append(it)

    # mean response of neuron across trials
    mU = np.nanmean(X, axis=2, keepdims=True) * np.ones((1, 1, np.shape(X)[2]))
    varex_mu = 1 - (bn.nanvar(X - mU)/bn.nanvar(X))

    # smoothed response of neuron across time
    sm_window = 5  # This should always be odd or there will be a frame shift
    assert sm_window % 2 == 1
    sm_shift = int(np.floor((sm_window - 1)/2) + sm_window*2)
    pad = np.zeros((np.shape(X)[0], sm_window*2, np.shape(X)[2]))
    smU_in = np.concatenate((pad, X, pad), axis=1)
    smU = bn.move_mean(smU_in, 5, axis=1)
    smU = smU[:, sm_shift:(np.shape(X)[1] + sm_shift), :]
    varex_smu = 1 - (bn.nanvar(X - smU)/bn.nanvar(X))

    # calculate trial concatenated PCA reconstruction of data, this is
    # the upper bound of performance we could expect
    if verbose:
        print('Calculating trial concatenated PCA control: ' + mouse)
    iX = deepcopy(X)
    iX[np.isnan(iX)] = bn.nanmean(iX[:])  # impute empties w/ mean of data
    sz = np.shape(iX)
    iX = iX.reshape(sz[0], sz[1]*sz[2])
    mu = bn.nanmean(iX, axis=0)
    catPCA = PCA()
    catPCA.fit(iX)
    nComp = len(V.results)
    Xhat = np.dot(catPCA.transform(iX)[:, :nComp],
                  catPCA.components_[:nComp, :])
    Xhat += mu
    varex_PCA = [1 - (bn.nanvar(X.reshape(sz[0], sz[1]*sz[2]) - Xhat)
                 / bn.nanvar(X.reshape(sz[0], sz[1]*sz[2])))][0]

    # make dataframe of data
    # create your index out of relevant variables
    index = pd.MultiIndex.from_arrays(
        [[mouse]*len(varex)],
        names=['mouse'])

    data = {'rank': rank,
            'iteration':  iteration,
            'variance_explained_tcamodel': varex,
            'variance_explained_smoothmodel': [varex_smu]*len(rank),
            'variance_explained_meanmodel': [varex_mu]*len(rank),
            'variance_explained_PCA': [varex_PCA]*len(rank)}

    dfvar = pd.DataFrame(data, index=index)

    return dfvar


@memoize(across='mouse', updated=190805, returns='other', large_output=True)
def groupday_varex_byday(
        mouse,
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        word=None,
        group_by='all',
        nan_thresh=0.85,
        rectified=True,
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
        sm_window = 5  # should always be odd or there will be a frame shift
        assert sm_window % 2 == 1
        sm_shift = int(np.floor((sm_window - 1)/2) + sm_window*2)
        pad = np.zeros((np.shape(X)[0], sm_window*2, np.shape(X)[2]))
        smU_in = np.concatenate((pad, X, pad), axis=1)
        smU = bn.move_mean(smU_in, 5, axis=1)
        smU = smU[:, sm_shift:(np.shape(X)[1] + sm_shift), :]
        # calculate variance explained per day
        for day in np.unique(dates):
            day_bool = dates.isin([day])
            bUd = bU[:, :, day_bool]
            mUd = mU[:, :, day_bool]
            smUd = smU[:, :, day_bool]
            bX = X[:, :, day_bool]
            rank.append(r)
            date.append(day)
            varex.append(1 - (bn.nanvar(bX - bUd)/bn.nanvar(bX)))
            varex_mu.append(1 - (bn.nanvar(bX - mUd)/bn.nanvar(bX)))
            varex_smu.append(1 - (bn.nanvar(bX - smUd)/bn.nanvar(bX)))

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


@memoize(across='mouse', updated=190805, returns='other', large_output=True)
def groupday_var_byday(
        mouse,
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        word=None,
        group_by='all',
        nan_thresh=0.85,
        score_threshold=0.8,
        rectified=True,
        verbose=False):
    """
    Plot total dataset variance across all whole groupday
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
    var = []
    var_model = []
    date = []
    rank = []
    for r in V.results:
        # model
        bU = V.results[r][0].factors.full()
        # calculate variance explained per day
        for day in np.unique(dates):
            day_bool = dates.isin([day])
            bUd = bU[:, :, day_bool]
            bX = X[:, :, day_bool]
            rank.append(r)
            date.append(day)
            var.append(bn.nanvar(bX))
            var_model.append(bn.nanvar(bUd))


    # make dataframe of data
    # create your index out of relevant variables
    index = pd.MultiIndex.from_arrays(
        [[mouse]*len(var)],
        names=['mouse'])

    data = {'rank': rank,
            'date':  date,
            'variance_tcamodel': var_model,
            'variance_input_tensor': var}

    dfvar = pd.DataFrame(data, index=index)

    return dfvar


@memoize(across='mouse', updated=190805, returns='other', large_output=True)
def groupday_varex_byday_bycomp(
        mouse,
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        word=None,
        group_by='all',
        nan_thresh=0.85,
        rectified=True,
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
                varex.append(1 - (bn.nanvar(bX - bUd)/bn.nanvar(bX)))

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


@memoize(across='mouse', updated=190805, returns='other', large_output=True)
def groupday_varex_byday_bycomp_bycell(
        mouse,
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        word=None,
        group_by='all',
        nan_thresh=0.85,
        rectified=True,
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
                for cell_num in range(np.shape(V.results[r][0].factors[0][:, :])[0]):
                    cell_identity = ids[cell_num]
                    bX = xxxx[cell_num, :, :]
                    bU = abcd[cell_num, :, :]
                    rank.append(r)
                    date.append(day)
                    component.append(fac_num+1)
                    cell_idx.append(cell_num)
                    cell_id.append(cell_identity)
                    varex.append(1 - (bn.nanvar(bX - bU)/bn.nanvar(bX)))

    # make dataframe of data
    # create your index out of relevant variables
    index = pd.MultiIndex.from_arrays(
        [[mouse]*len(varex)],
        names=['mouse'])

    data = {'rank': rank,
            'date':  date,
            'cell_num': cell_idx,
            'cell_id': cell_id,
            'component': component,
            'variance_explained_tcamodel': varex}

    dfvar = pd.DataFrame(data, index=index)

    return dfvar


@memoize(across='mouse', updated=190805, returns='other', large_output=True)
def groupday_varex_bycomp_bycell(
        mouse,
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        word=None,
        group_by='all',
        nan_thresh=0.85,
        rectified=True,
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
            # calculate variance explained per cell
            for cell_num in range(np.shape(V.results[r][0].factors[0][:, :])[0]):
                cell_identity = ids[cell_num]
                bX = X[cell_num, :, :]
                bU = abc[cell_num, :, :]
                rank.append(r)
                component.append(fac_num+1)
                cell_idx.append(cell_num)
                cell_id.append(cell_identity)
                varex.append(1 - (bn.nanvar(bX - bU)/bn.nanvar(bX)))

    # make dataframe of data
    # create your index out of relevant variables
    index = pd.MultiIndex.from_arrays(
        [[mouse]*len(varex)],
        names=['mouse'])

    data = {'rank': rank,
            'cell_num': cell_idx,
            'cell_id': cell_id,
            'component': component,
            'variance_explained_tcamodel': varex}

    dfvar = pd.DataFrame(data, index=index)

    return dfvar


@memoize(across='mouse', updated=190805, returns='other', large_output=True)
def groupday_varex_bycomp(
        mouse,
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        word=None,
        group_by='all',
        nan_thresh=0.85,
        rectified=True,
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
    rank = []
    component = []
    for r in V.results:
        for fac_num in range(np.shape(V.results[r][0].factors[0][:, :])[1]):
            # reconstruct single component model
            a = V.results[r][0].factors[0][:, fac_num]
            b = V.results[r][0].factors[1][:, fac_num]
            c = V.results[r][0].factors[2][:, fac_num]
            ab = a[:, None] @ b[None, :]
            bUd = ab[:, :, None] @ c[None, :]
            rank.append(r)
            component.append(fac_num+1)
            varex.append(1 - (bn.nanvar(X - bUd)/bn.nanvar(X)))

    # make dataframe of data
    # create your index out of relevant variables
    index = pd.MultiIndex.from_arrays(
        [[mouse]*len(varex)],
        names=['mouse'])

    data = {'rank': rank,
            'component': component,
            'variance_explained_tcamodel': varex}

    dfvar = pd.DataFrame(data, index=index)

    return dfvar

@memoize(across='mouse', updated=190805, returns='other', large_output=True)
def groupday_varex_byday_bycell(
        mouse,
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        word=None,
        group_by='all',
        nan_thresh=0.85,
        rectified=True,
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
    ids_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_group_ids_' + str(trace_type) + '.npy')
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
    dayframe = {}
    varex = []
    varex_smu = []
    varex_mu = []
    date = []
    rank = []
    cell_idx = []
    cell_id = []
    for r in V.results:
        # model
        bU = V.results[r][0].factors.full()
        # mean response of neuron across trials
        mU = np.nanmean(
            X, axis=2, keepdims=True) * np.ones((1, 1, np.shape(X)[2]))
        # smoothed response of neuron across time
        sm_window = 5  # should always be odd or there will be a frame shift
        assert sm_window % 2 == 1
        sm_shift = int(np.floor((sm_window - 1)/2) + sm_window*2)
        pad = np.zeros((np.shape(X)[0], sm_window*2, np.shape(X)[2]))
        smU_in = np.concatenate((pad, X, pad), axis=1)
        smU = bn.move_mean(smU_in, 5, axis=1)
        smU = smU[:, sm_shift:(np.shape(X)[1] + sm_shift), :]
        # calculate variance explained per day
        for day in np.unique(dates):
            day_bool = dates.isin([day])
            bUd = bU[:, :, day_bool]
            mUd = mU[:, :, day_bool]
            smUd = smU[:, :, day_bool]
            bX = X[:, :, day_bool]
            for cell_num in range(np.shape(X)[0]):
                cell_identity = ids[cell_num]
                cell_id.append(cell_identity)
                cell_idx.append(cell_num)
                rank.append(r)
                date.append(day)
                varex.append(
                    1 - (bn.nanvar(bX[cell_num, :, :] - bUd[cell_num, :, :])
                         / bn.nanvar(bX[cell_num, :, :])))
                varex_mu.append(
                    1 - (bn.nanvar(bX[cell_num, :, :] - mUd[cell_num, :, :])
                         / bn.nanvar(bX[cell_num, :, :])))
                varex_smu.append(
                    1 - (bn.nanvar(bX[cell_num, :, :] - smUd[cell_num, :, :])
                         / bn.nanvar(bX[cell_num, :, :])))

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

@memoize(across='mouse', updated=190805, returns='other', large_output=True)
def groupday_varex_bycell(
        mouse,
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        word=None,
        group_by='all',
        nan_thresh=0.85,
        rectified=True,
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
    ids_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_group_ids_' + str(trace_type) + '.npy')
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
    varex_smu = []
    varex_mu = []
    varex_daily_mu = []
    date = []
    rank = []
    cell_idx = []
    cell_id = []
    for r in V.results:
        # model
        bU = V.results[r][0].factors.full()
        # mean response of neuron across trials
        mU = np.nanmean(
            X, axis=2, keepdims=True) * np.ones((1, 1, np.shape(X)[2]))
        # smoothed response of neuron across time
        sm_window = 5  # should always be odd or there will be a frame shift
        assert sm_window % 2 == 1
        sm_shift = int(np.floor((sm_window - 1)/2) + sm_window*2)
        pad = np.zeros((np.shape(X)[0], sm_window*2, np.shape(X)[2]))
        smU_in = np.concatenate((pad, X, pad), axis=1)
        smU = bn.move_mean(smU_in, 5, axis=1)
        smU = smU[:, sm_shift:(np.shape(X)[1] + sm_shift), :]
        varex_smu = 1 - (bn.nanvar(X - smU)/bn.nanvar(X))
        # mean response of neurons per day recreating full tensor
        dmU = deepcopy(X)
        for day in np.unique(dates):
            day_bool = dates.isin([day])
            bX = (np.nanmean(X[:, :, day_bool], axis=2, keepdims=True)
                  * np.ones((1, 1, np.shape(X[:, :, day_bool])[2])))
            dmU[:, :, day_bool] = bX
        for cell_num in range(np.shape(X)[0]):
            cell_identity = ids[cell_num]
            cell_id.append(cell_identity)
            cell_idx.append(cell_num)
            rank.append(r)
            date.append(day)
            varex.append(
                1 - (bn.nanvar(X[cell_num, :, :] - bU[cell_num, :, :])
                     / bn.nanvar(X[cell_num, :, :])))
            varex_mu.append(
                1 - (bn.nanvar(X[cell_num, :, :] - mU[cell_num, :, :])
                     / bn.nanvar(X[cell_num, :, :])))
            varex_smu.append(
                1 - (bn.nanvar(X[cell_num, :, :] - smU[cell_num, :, :])
                     / bn.nanvar(X[cell_num, :, :])))
            varex_daily_mu.append(
                1 - (bn.nanvar(X[cell_num, :, :] - dmU[cell_num, :, :])
                     / bn.nanvar(X[cell_num, :, :])))

    # make dataframe of data
    # create your index out of relevant variables
    index = pd.MultiIndex.from_arrays(
        [[mouse]*len(varex)],
        names=['mouse'])

    data = {'rank': rank,
            'date':  date,
            'cell_num': cell_idx,
            'cell_id': cell_id,
            'variance_explained_tcamodel': varex,
            'variance_explained_smoothmodel': varex_smu,
            'variance_explained_meanmodel': varex_mu,
            'variance_explained_daily_meanmodel': varex_daily_mu}

    dfvar = pd.DataFrame(data, index=index)

    return dfvar

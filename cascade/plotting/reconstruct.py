""" Functions for visualizing TCA model reconstructions """
from .. import paths, tca
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def groupday_mean_responses(
        mouse,
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        word=None,
        group_by='all',
        nan_thresh=0.85,
        rectified=True,
        rank=18,
        verbose=True):
    """
    Plot reconstruction for whole groupday TCA decomposition ensemble.

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

    # save dir
    save_dir = paths.tca_plots(
        mouse, 'group', pars=pars, word=word, group_pars=group_pars)
    save_dir = os.path.join(save_dir, 'model reconstructions' + save_tag)
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    date_dir = os.path.join(save_dir, str(group_by) + ' ' + method)
    if not os.path.isdir(date_dir): os.mkdir(date_dir)

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

    # re-balance your factors
    if verbose:
        print('Re-balancing factors.')
    for r in ensemble[method].results:
        for i in range(len(ensemble[method].results[r])):
            ensemble[method].results[r][i].factors.rebalance()

    # sort cells according to which factor they respond to the most
    sort_ensemble, my_sorts = tca._sortfactors(ensemble[method])
    sorted_ids = ids[my_sorts]

    Xhat = V.results[rank][0].factors.full()

    rows = 1
    cols = 2
    fig, axes = plt.subplots(
        rows, cols, figsize=(17, rows),
        gridspec_kw={'width_ratios': [2, 2, 17]})

    # reshape for easier indexing
    ax = np.array(axes).reshape((rows, -1))
    ax[0, 0].set_title('Neuron factors')
    ax[0, 1].set_title('Temporal factors')
    ax[0, 2].set_title('Trial factors')



"""Script to run TCA, save outputs, and save simple plots of model results."""
import cascade as cas
import pandas as pd
import numpy as np
import tensortools as tt
from copy import deepcopy
import argparse
import os
import warnings

import matplotlib.pyplot as plt
import seaborn as sns

"""
1. This version excluded naive from consideration when calculating driven-ness.
2. This version also drops any cells that have more negative values than positive values 
    --> This should help exclude the few suppressed cells that some how sneak through the cracks
3. This version additionally drops cells with only a single stage of data.
4. Adds an additional model where data from each mouse is scaled by the mean(max(response per cell)) so 
that mice have more equivlaent contributions to the model. 
5. NEW ADDITION is to remove naive timepoint from fitting since there is tremendous diversity here 
and only n=4 mice have this timepoint. Instead models will be fit on mostly conserved time periods
and additional fitting or extrapolation can be done to analyze naive data. 
"""

def run_unwrapped_tca(thresh=4, iteration=10, force=False, verbose=False, debug=False, skip_tca=False):
    """
    Run script to build and save TCA inputs as well as TCA model outputs

    :param thresh: int
        -log10(p-value) threshold for calling a cell driven for any stage of learning.
    :param force: boolean
        Force a rerun even if output file exists.
    :param verbose: boolean
        View TCA progress in the terminal. 
    """

    # parameters
    # --------------------------------------------------------------------------------------------------

    # model naming
    mod_suffix = '_noT0'
    mod_date = 20210215

    # input data params
    mice = cas.lookups.mice['allFOV']
    words = ['respondent' if s in 'OA27' else 'computation' for s in mice]
    group_by = 'all3'
    with_model = False
    nan_thresh = 0.95

    # TCA params
    method = 'ncp_hals'
    replicates = iteration
    fit_options = {'tol': 0.0001, 'max_iter': 500, 'verbose': False}
    ranks = list(np.arange(1, 21, dtype=int))
    ranks.extend([40])
    ranks.extend([80])
    tca_ranks = [int(s) for s in ranks]

    # plot params
    hue_order = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']
    heatmap_rank = 12

    # Do not overwrite existing files
    if os.path.isfile(cas.paths.analysis_file(f'tca_ensemble_v{thresh}i{iteration}{mod_suffix}_{mod_date}.npy', 'tca_dfs')) and not force:
        return

    # load in a full size data
    # --------------------------------------------------------------------------------------------------
    tensor_list = []
    id_list = []
    bhv_list = []
    meta_list = []
    for mouse, word in zip(mice, words):

        # return   ids, tensor, meta, bhv
        out = cas.load.load_all_groupday(mouse, word=word, with_model=with_model,
                                        group_by=group_by, nan_thresh=nan_thresh)
        tensor_list.append(out[2])
        id_list.append(out[1])
        bhv_list.append(out[4])
        meta_list.append(cas.utils.add_stages_to_meta(out[3], 'parsed_11stage'))

    # load Oren's better offset classification
    # --------------------------------------------------------------------------------------------------
    off_df_all_mice = pd.read_pickle('/twophoton_analysis/Data/analysis/core_dfs/offsets_dfs.pkl')
    df_list = []
    for k, v in off_df_all_mice.items():
        v['mouse'] = k
        df_list.append(v)
    updated_off_df = pd.concat(df_list, axis=0)
    updated_off_df = updated_off_df.set_index(['mouse', 'cell_id'])
    updated_off_df.head()

    # build input for TCA, selecting cells driven any stage with a given negative log10 p-value 'thresh'
    # --------------------------------------------------------------------------------------------------
    # get data that has T0 naive removed
    data_dict = build_data_dict(
        meta_list, id_list, tensor_list, updated_off_df,
        thresh=thresh, iteration=iteration, debug=debug, mod_suffix=mod_suffix, add_suffix=''
    )
    # get data that has all stages 
    data_dict_allstages = build_data_dict(
        meta_list, id_list, tensor_list, updated_off_df,
        thresh=thresh, iteration=iteration, debug=debug, mod_suffix=mod_suffix, add_suffix='_allstages'
    )
    data_dict.update(data_dict_allstages)  # combine dicts

    # loop over keys and apply your T0 normalization to 'allstages' data 
    data_dict = renormalize_allstages_to_t0(
        data_dict, thresh=thresh, iteration=iteration, mod_suffix=mod_suffix, add_suffix='_allstages'
    )

    # filter out suppressed cells and cells with only a single stage of data (normed data used as benchmark)
    data_dict = filter_data_dict(
        data_dict, thresh=thresh, iteration=iteration, verbose=verbose, mod_suffix=mod_suffix
    )

    # add a scaled-by-mouse zscore tensor
    data_dict = add_scaled_zscore_to_data_dict(
        data_dict, thresh=thresh, iteration=iteration, verbose=verbose, mod_suffix=mod_suffix
    )

    # save data
    data_dict_path = cas.paths.analysis_file(f'input_data_v{thresh}i{iteration}{mod_suffix}_{mod_date}.npy', 'tca_dfs')
    np.save(data_dict_path, data_dict, allow_pickle=True)
    if verbose:
        print(f'Input data saved to:\n\t{data_dict_path}')

    # run TCA and save
    # --------------------------------------------------------------------------------------------------
    if skip_tca:
        ensemble = np.load(
            cas.paths.analysis_file(f'tca_ensemble_v{thresh}i{iteration}{mod_suffix}_{mod_date}.npy', 'tca_dfs'),
            allow_pickle=True
                           ).item()
    else:
        ensemble = {}

        for k, v in data_dict.items():
            if '_mouse_' in k or '_cell_' in k:
                continue
            if 'allstages' in k:
                continue
            mask = ~np.isnan(v)
            fit_options['mask'] = mask
            if verbose:
                print(f'TCA starting: {k} --> n={v.shape[0]} cells')
            ensemble[k] = tt.Ensemble(fit_method=method, fit_options=deepcopy(fit_options))
            ensemble[k].fit(v, ranks=tca_ranks, replicates=replicates, verbose=True)

        np.save(
            cas.paths.analysis_file(f'tca_ensemble_v{thresh}i{iteration}{mod_suffix}_{mod_date}.npy', 'tca_dfs'),
            ensemble,
            allow_pickle=True
        )

    # plot and save relevant results
    # --------------------------------------------------------------------------------------------------

    # plot model performance
    for k, v in ensemble.items():

        fig, ax = plt.subplots(2,2, figsize=(10,10), sharex='row', sharey='col')
        ax = ax.reshape([2,2])

        # full plot
        tt.visualization.plot_objective(v, ax=ax[0,0], line_kw={'color': 'red'}, scatter_kw={'facecolor': 'black', 'alpha': 0.5})
        tt.visualization.plot_similarity(v, ax=ax[0,1], line_kw={'color': 'blue'}, scatter_kw={'facecolor': 'black', 'alpha': 0.5})
        ax[0, 0].set_title(f'{k}: Objective function\ndata: cells x times-stages x cues')
        ax[0, 1].set_title(f'{k}: Model similarity\ndata: cells x times-stages x cues')
        ax[0, 0].axvline(-1, linestyle=':', color='grey')
        ax[0, 0].axvline(21, linestyle=':', color='grey')
        ax[0, 1].axvline(-1, linestyle=':', color='grey')
        ax[0, 1].axvline(21, linestyle=':', color='grey')

        # zoom in on 1-20
        tt.visualization.plot_objective(v, ax=ax[1,0], line_kw={'color': 'red'}, scatter_kw={'facecolor': 'black', 'alpha': 0.5})
        tt.visualization.plot_similarity(v, ax=ax[1,1], line_kw={'color': 'blue'}, scatter_kw={'facecolor': 'black', 'alpha': 0.5})
        ax[1, 1].set_xlim([-1, 21])
        ax[1, 0].set_title(f'Zoom: Objective function')
        ax[1, 1].set_title(f'Zoom: Model similarity')

        plt.savefig(cas.paths.analysis_file(f'{k}_obj_sim.png', 'tca_dfs/TCA_qc'), bbox_inches='tight')

    # plot factors after sorting
    sort_ensembles, sort_orders = {}, {}
    for k, v in ensemble.items():
        sort_ensembles[k], sort_orders[k] = cas.utils.sortfactors(v)

    for k, v in sort_ensembles.items():
        for rr in range(5, 20):
            fig, ax, _ = tt.visualization.plot_factors(v.results[rr][0].factors.rebalance(), plots=['scatter', 'line', 'line'],
                               scatter_kw=cas.lookups.tt_plot_options['ncp_hals']['scatter_kw'],
                               line_kw=cas.lookups.tt_plot_options['ncp_hals']['line_kw'],
                               bar_kw=cas.lookups.tt_plot_options['ncp_hals']['bar_kw']);

            cell_count = v.results[rr][0].factors[0].shape[0]
            for i in range(ax.shape[0]):
                ax[i, 0].set_ylabel(f'                 Component {i+1}', size=16, ha='right', rotation=0)
            ax[0, 1].set_title(f'{k}, rank {rr} (n = {cell_count})\n\n', size=20)

            plt.savefig(cas.paths.analysis_file(f'{k}_rank{rr}_facs.png', f'tca_dfs/TCA_factors/{k}'), bbox_inches='tight')

    # plot heatmap
    mod_heat = [f'v{thresh}i{iteration}_norm_on{mod_suffix}', f'v{thresh}i{iteration}_norm_off{mod_suffix}']
    mmod_heat = [f'v{thresh}i{iteration}_on_mouse{mod_suffix}', f'v{thresh}i{iteration}_off_mouse{mod_suffix}']
    for mod, mmod in zip(mod_heat, mmod_heat):

        mat2d_norm = data_dict[mod]
        mouse_dict = data_dict[mmod]
        mouse_mapper = {k: c for c, k in enumerate(np.unique(mouse_dict))}
        number_mouse_mat = np.array([mouse_mapper[s] for s in mouse_dict])

        # ensemble sort
        ensort = sort_orders[mod][heatmap_rank - 1]

        clabel = 'normalized \u0394F/F'
        # clabel = '\u0394F/F (z-score)'

        #sort
        mat2d_norm = mat2d_norm[ensort, :]
        number_mouse_mat = number_mouse_mat[ensort]

        ax = []
        fig = plt.figure(figsize=(30, 15))
        gs = fig.add_gridspec(100, 110)
        ax.append(fig.add_subplot(gs[:, 3:5]))
        ax.append(fig.add_subplot(gs[:, 10:38]))
        ax.append(fig.add_subplot(gs[:, 40:68]))
        ax.append(fig.add_subplot(gs[:, 70:98]))
        ax.append(fig.add_subplot(gs[:30, 105:108]))

        # plot "categorical" heatmap using defined color mappings
        sns.heatmap(number_mouse_mat[:, None], cmap='Set2', ax=ax[0], cbar=False)
        ax[0].set_xticklabels(['mouse'], rotation=45, ha='right', size=18)
        ax[0].set_yticklabels([])
        ax[0].set_ylabel('cell number', size=14)

        for i in range(1,4):
            if i == 3:
                g = sns.heatmap(mat2d_norm[:,:,i-1], ax=ax[i], center=0, vmax=1, vmin=-0.5, cmap='vlag',
                                cbar_ax=ax[4], cbar_kws={'label': clabel})
                cbar = g.collections[0].colorbar
                cbar.set_label(clabel, size=16)
            else:
                g = sns.heatmap(mat2d_norm[:,:,i-1], ax=ax[i], center=0, vmax=1, vmin=-0.5, cmap='vlag', cbar=False)
            g.set_facecolor('#c5c5c5')
            ax[i].set_title(f'initial cue: {hue_order[i-1]}\n', size=20)
            stage_labels = cas.lookups.staging['parsed_11stage_T'][1:]
            stim_starts = [15.5 + 47*s for s in np.arange(len(stage_labels))]
            stim_labels = [f'0\n\n{s}' if c%2 == 0 else f'0\n{s}' for c, s in enumerate(stage_labels)]
            ax[i].set_xticks(stim_starts)
            ax[i].set_xticklabels(stim_labels, rotation=0)
            if i == 1:
                ax[i].set_ylabel('cell number', size=18)
            ax[i].set_xlabel('\ntime from stimulus onset (sec)', size=18)

            if i > 1:
                ax[i].set_yticks([])

            plt.savefig(
                cas.paths.analysis_file(f'{mod}_rank{heatmap_rank}_heatmap.png', f'tca_dfs/TCA_heatmaps/v{thresh}i{iteration}{mod_suffix}'),
                bbox_inches='tight')


def build_data_dict(
    meta_list,
    id_list,
    tensor_list,
    updated_off_df,
    thresh=4,
    iteration=10,
    debug=False,
    mod_suffix='',
    add_suffix=''
):
    """
    :param add_suffix: str 
        Additional suffix string that can be added to call to build_datasets.keys() to specify things like "allstages"
    """

    driven_on_tensor_flat_run = []
    driven_off_tensor_flat_run = []
    driven_on_tensor_flat = []
    driven_off_tensor_flat = []
    off_mouse_vec_flat, on_mouse_vec_flat, off_cell_vec, on_cell_vec = [], [], [], []
    for meta, ids, tensor in zip(meta_list, id_list, tensor_list):

        # skip LM and seizure mouse
        if cas.utils.meta_mouse(meta) in ['AS20', 'AS23']:
            continue
        if cas.utils.meta_mouse(meta) in ['AS41', 'AS47', 'OA38']:
            continue
        if cas.utils.meta_mouse(meta) in cas.lookups.mice['lml5']:
            continue

        # calculate drivenness across stages, using Oren's offset boolean
        off_df = updated_off_df.loc[updated_off_df.reset_index().mouse.isin([cas.utils.meta_mouse(meta)]).values, ['offset_test']]
        off_df = off_df.reindex(ids, level=1)
        assert np.array_equal(off_df.reset_index().cell_id.values, ids)
        offset_bool = off_df.offset_test.values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            drive_df = cas.drive.multi_stat_drive(meta, ids, tensor, alternative='less', offset_bool=offset_bool, neg_log10_pv_thresh=thresh)

        # flatten tensor and unwrap it to look across stages
        flat_tensors, off_flat_tensors = build_datasets(meta, tensor, debug=debug)

        # get driven ids for different behaviors
        driven_onset, driven_offset = get_driven_cells(drive_df)
        driven_on_bool = np.isin(np.array(ids), driven_onset)
        driven_off_bool = np.isin(np.array(ids), driven_offset)
        on_cells_in_order = np.array(ids)[driven_on_bool]
        off_cells_in_order = np.array(ids)[driven_off_bool]

        # keep track of mice and cell ids
        on_mouse_vec_flat.append([cas.utils.meta_mouse(meta)] * len(on_cells_in_order))
        off_mouse_vec_flat.append([cas.utils.meta_mouse(meta)] * len(off_cells_in_order))
        on_cell_vec.append(on_cells_in_order)
        off_cell_vec.append(off_cells_in_order)

        # ensure that cells are not counted in both onset and offset groups
        assert all(~np.isin(driven_onset, driven_offset))

        # Onset cells
        ten_list = []
        for cc, cue in enumerate(['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']):
            invert_lookup = {v:k for k, v in cas.lookups.lookup_mm[cas.utils.meta_mouse(meta)].items()}
            ten_list.append(flat_tensors[invert_lookup[cue] + add_suffix][:,:,None])
        new_on_tensor_unwrap = np.dstack(ten_list)[driven_on_bool, :, :]
        driven_on_tensor_flat.append(new_on_tensor_unwrap)

        # Offset cells
        ten_list = []
        for cc, cue in enumerate(['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']):
            invert_lookup = {v:k for k, v in cas.lookups.lookup_mm[cas.utils.meta_mouse(meta)].items()}
            ten_list.append(off_flat_tensors[invert_lookup[cue] + add_suffix][:,:,None])
        new_off_tensor_unwrap = np.dstack(ten_list)[driven_off_bool, :, :]
        driven_off_tensor_flat.append(new_off_tensor_unwrap)

        # Onset cells with speed
        ten_list = []
        for speedi in ['_fast', '_slow']:
            for cue in ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']:
                invert_lookup = {v:k for k, v in cas.lookups.lookup_mm[cas.utils.meta_mouse(meta)].items()}
                ten_list.append(flat_tensors[invert_lookup[cue] + speedi + add_suffix][:,:,None])
        new_onset_tensor_unwrap = np.dstack(ten_list)[driven_on_bool, :, :]
        driven_on_tensor_flat_run.append(new_onset_tensor_unwrap)

        # Offset cells with speed
        ten_list = []
        for speedi in ['_fast', '_slow']:
            for cue in ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']:
                invert_lookup = {v:k for k, v in cas.lookups.lookup_mm[cas.utils.meta_mouse(meta)].items()}
                ten_list.append(off_flat_tensors[invert_lookup[cue] + speedi + add_suffix][:,:,None])
        new_offset_tensor_unwrap = np.dstack(ten_list)[driven_off_bool, :, :]
        driven_off_tensor_flat_run.append(new_offset_tensor_unwrap)

    # concatenate tensors
    on_mega_tensor_flat_run = np.vstack(driven_on_tensor_flat_run)
    off_mega_tensor_flat_run = np.vstack(driven_off_tensor_flat_run)
    on_mega_tensor_flat = np.vstack(driven_on_tensor_flat)
    off_mega_tensor_flat = np.vstack(driven_off_tensor_flat)

    # concatenate lists of mice and cells
    on_mega_mouse_flat = np.hstack(on_mouse_vec_flat)
    off_mega_mouse_flat = np.hstack(off_mouse_vec_flat)
    on_mega_cell_flat = np.hstack(on_cell_vec)
    off_mega_cell_flat = np.hstack(off_cell_vec)

    # make sure that you get the shapes you expected
    assert on_mega_tensor_flat_run.shape[0] == on_mega_tensor_flat.shape[0]
    assert off_mega_tensor_flat_run.shape[0] == off_mega_tensor_flat.shape[0]
    assert len(on_mega_mouse_flat) == len(on_mega_cell_flat)
    assert len(off_mega_mouse_flat) == len(off_mega_cell_flat)
    assert len(on_mega_cell_flat) == on_mega_tensor_flat.shape[0]
    assert len(off_mega_cell_flat) == off_mega_tensor_flat.shape[0]

    # Normalize per cell to max
    # TODO move this to a separate function like zscore scaling
    off_mega_tensor_flat_norm, scale_by_off_flat = _row_norm(off_mega_tensor_flat)
    on_mega_tensor_flat_norm, scale_by_on_flat = _row_norm(on_mega_tensor_flat)
    off_mega_tensor_flat_run_norm, scale_by_off_flat_run = _row_norm(off_mega_tensor_flat_run)
    on_mega_tensor_flat_run_norm, scale_by_on_flat_run = _row_norm(on_mega_tensor_flat_run)

    # initial restructure of data and save of input data
    # --------------------------------------------------------------------------------------------------
    data_dict = {
        f'v{thresh}i{iteration}_on{mod_suffix}{add_suffix}': on_mega_tensor_flat,
        f'v{thresh}i{iteration}_off{mod_suffix}{add_suffix}': off_mega_tensor_flat,
        f'v{thresh}i{iteration}_norm_on{mod_suffix}{add_suffix}': on_mega_tensor_flat_norm,
        f'v{thresh}i{iteration}_norm_off{mod_suffix}{add_suffix}': off_mega_tensor_flat_norm,
        f'v{thresh}i{iteration}_speed_on{mod_suffix}{add_suffix}': on_mega_tensor_flat_run,
        f'v{thresh}i{iteration}_speed_off{mod_suffix}{add_suffix}': off_mega_tensor_flat_run,
        f'v{thresh}i{iteration}_speed_norm_on{mod_suffix}{add_suffix}': on_mega_tensor_flat_run_norm,
        f'v{thresh}i{iteration}_speed_norm_off{mod_suffix}{add_suffix}': off_mega_tensor_flat_run_norm,

        f'v{thresh}i{iteration}_off_mouse{mod_suffix}{add_suffix}': off_mega_mouse_flat,
        f'v{thresh}i{iteration}_on_mouse{mod_suffix}{add_suffix}': on_mega_mouse_flat,
        f'v{thresh}i{iteration}_off_cell{mod_suffix}{add_suffix}': off_mega_cell_flat,
        f'v{thresh}i{iteration}_on_cell{mod_suffix}{add_suffix}': on_mega_cell_flat,

        f'v{thresh}i{iteration}_norm_on_cell_scale_factors{mod_suffix}{add_suffix}': scale_by_on_flat,
        f'v{thresh}i{iteration}_norm_off_cell_scale_factors{mod_suffix}{add_suffix}': scale_by_off_flat,
        f'v{thresh}i{iteration}_speed_norm_on_cell_scale_factors{mod_suffix}{add_suffix}': scale_by_on_flat_run,
        f'v{thresh}i{iteration}_speed_norm_off_cell_scale_factors{mod_suffix}{add_suffix}': scale_by_off_flat_run,
    }

    return data_dict


def renormalize_allstages_to_t0(data_dict, thresh=4, iteration=10, mod_suffix='', add_suffix='_allstages'):
    """Normalize _allstages data to T0 Naive so that data is matched T1: is matched.

    Args:
        data_dict (dict): datasets
        thresh (int, optional): Drivenness -log10(p-value) threshold. Defaults to 4.
        iteration (int, optional): Model iteration. Defaults to 10.
        mod_suffix (str, optional): Model name. Defaults to ''.
        add_suffix (str, optional): Additional suffix, defines the group to renormalize.
            Defaults to '_allstages'.

    Returns:
        dict: dict of datasets now with 4 newly noramlized key-value pairs. New 
        suffix name is '_allstagesT0norm'.
    """

    # loop over keys and apply your T0 normalization to 'allstages' data 
    for k, v in data_dict.items():
        if k == f'v{thresh}i{iteration}_on{mod_suffix}{add_suffix}':
            norm_key = f'v{thresh}i{iteration}_norm_on_cell_scale_factors{mod_suffix}'
            new_value = data_dict[k] / data_dict[norm_key]
            data_dict[k + 'T0norm'] = new_value
        elif k == f'v{thresh}i{iteration}_off{mod_suffix}{add_suffix}':
            norm_key = f'v{thresh}i{iteration}_norm_off_cell_scale_factors{mod_suffix}'
            new_value = data_dict[k] / data_dict[norm_key]
            data_dict[k + 'T0norm'] = new_value
        elif k == f'v{thresh}i{iteration}_speed_on{mod_suffix}{add_suffix}':
            norm_key = f'v{thresh}i{iteration}_speed_norm_on_cell_scale_factors{mod_suffix}'
            new_value = data_dict[k] / data_dict[norm_key]
            data_dict[k + 'T0norm'] = new_value
        elif k == f'v{thresh}i{iteration}_speed_off{mod_suffix}{add_suffix}':
            norm_key = f'v{thresh}i{iteration}_speed_norm_off_cell_scale_factors{mod_suffix}'
            new_value = data_dict[k] / data_dict[norm_key]
            data_dict[k + 'T0norm'] = new_value
        else:
            continue

    return data_dict       


def build_datasets(meta, tensor, debug=False):
    """
    Build possible datasets. Only use the first second before stim onset (offset) and two seconds post
    onset (offset).
    """

    # flatten tensor and unwrap it to look across stages
    flat_tensors = {}
    off_flat_tensors = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for cue in ['plus', 'minus', 'neutral']:
            meta_bool = meta.initial_condition.isin([cue]).values

            # get mean per stage per cue (also split on running speed)
            stage_mean_tensor_slow = cas.utils.balanced_mean_per_stage(meta, tensor, meta_bool=meta_bool, filter_running='low_pre_speed_only')
            stage_mean_tensor_fast = cas.utils.balanced_mean_per_stage(meta, tensor, meta_bool=meta_bool, filter_running='high_pre_speed_only')
            stage_mean_tensor = cas.utils.balanced_mean_per_stage(meta, tensor, meta_bool=meta_bool)

            # limit to 2 seconds across mice
            flat_tensors[cue + '_allstages'] = cas.utils.unwrap_tensor(stage_mean_tensor[:, :int(np.ceil(15.5*3)), :])
            flat_tensors[cue + '_slow_allstages'] = cas.utils.unwrap_tensor(stage_mean_tensor_slow[:, :int(np.ceil(15.5*3)), :])
            flat_tensors[cue + '_fast_allstages'] = cas.utils.unwrap_tensor(stage_mean_tensor_fast[:, :int(np.ceil(15.5*3)), :])
            # don't use first stage
            flat_tensors[cue] = cas.utils.unwrap_tensor(stage_mean_tensor[:, :int(np.ceil(15.5*3)), 1:])
            flat_tensors[cue + '_slow'] = cas.utils.unwrap_tensor(stage_mean_tensor_slow[:, :int(np.ceil(15.5*3)), 1:])
            flat_tensors[cue + '_fast'] = cas.utils.unwrap_tensor(stage_mean_tensor_fast[:, :int(np.ceil(15.5*3)), 1:])

            # for offset, use 1 second pre offset and 2 seconds post, assumes 15.5 Hz
            off_int = int(np.ceil(cas.lookups.stim_length[cas.utils.meta_mouse(meta)] + 1)*15.5)
            off_flat_tensors[cue + '_allstages'] = cas.utils.unwrap_tensor(stage_mean_tensor[:, (off_int - 16):(off_int + 31), :])
            off_flat_tensors[cue + '_slow_allstages'] = cas.utils.unwrap_tensor(stage_mean_tensor_slow[:, (off_int - 16):(off_int + 31), :])
            off_flat_tensors[cue + '_fast_allstages'] = cas.utils.unwrap_tensor(stage_mean_tensor_fast[:, (off_int - 16):(off_int + 31), :])
            # don't use first stage
            off_flat_tensors[cue] = cas.utils.unwrap_tensor(stage_mean_tensor[:, (off_int - 16):(off_int + 31), 1:])
            off_flat_tensors[cue + '_slow'] = cas.utils.unwrap_tensor(stage_mean_tensor_slow[:, (off_int - 16):(off_int + 31), 1:])
            off_flat_tensors[cue + '_fast'] = cas.utils.unwrap_tensor(stage_mean_tensor_fast[:, (off_int - 16):(off_int + 31), 1:])
            if debug:
                print('ONSET', flat_tensors[cue].shape,  'OFFSET', off_flat_tensors[cue].shape)

    return flat_tensors, off_flat_tensors


def get_driven_cells(drive_df):
    """
    Function to get driven onset and offset cells according to their cell_ids,
    checking cell_ids across stats returned as drive_df = cas.drive.multi_stat_drive().

    Cell is considered driven if it is driven on a single stage.
    """
    driven_onset = []
    driven_offset = []
    for cc, cue in enumerate(['plus', 'minus', 'neutral']):
        for c, stages in enumerate(cas.lookups.staging['parsed_11stage']):

            # skip naive when considering which cells are driven
            if stages in ['T0 naive', 'L0 naive']:
                continue

            # Onset cells
            on_cells = drive_df.loc[~drive_df.offset_cell & drive_df.driven]
            on_cells = on_cells.loc[on_cells.reset_index().parsed_11stage.isin([stages]).values, :]
            on_cells = on_cells.loc[on_cells.reset_index().initial_cue.isin([cue]).values, :]
            # make sure you can't double count cells
            assert on_cells.groupby(['mouse', 'cell_id']).nunique().gt(1).sum(axis=0).eq(0).all()
            id_vec = on_cells.cell_id.unique()
            driven_onset.extend(id_vec)

            # Offset cells
            off_cells = drive_df.loc[drive_df.offset_cell & drive_df.driven]
            off_cells = off_cells.loc[off_cells.reset_index().parsed_11stage.isin([stages]).values, :]
            off_cells = off_cells.loc[off_cells.reset_index().initial_cue.isin([cue]).values, :]
            # make sure you can't double count cells
            assert off_cells.groupby(['mouse', 'cell_id']).nunique().gt(1).sum(axis=0).eq(0).all()
            id_vec = off_cells.cell_id.unique()
            driven_offset.extend(id_vec)

    driven_onset = np.unique(driven_onset)
    driven_offset = np.unique(driven_offset)

    return driven_onset, driven_offset


def filter_data_dict(data_dict, thresh=4, iteration=10, verbose=False, mod_suffix=''):
    """
    Remove cells with only suppression or a single stage of data.
    
    Uses "norm"-ed data as your benchmark for all other keys in dict.
    """

    # remove cells with more negative than positive values (using max across cues)
    # this will basically only remove cells that are suppressed to all three cues
    # onset
    best_cue_resp = np.nanmax(data_dict[f'v{thresh}i{iteration}_norm_on{mod_suffix}'], axis=2)
    nneg = np.sum(best_cue_resp >= 0, axis=1)
    neg = np.sum(best_cue_resp < 0, axis=1)
    nneg_bool_on = np.greater(nneg, neg)
    if verbose:
        print(f'ONSET cells dropped for nneg: {np.sum(~nneg_bool_on)}')
    # offset
    best_cue_resp = np.nanmax(data_dict[f'v{thresh}i{iteration}_norm_off{mod_suffix}'], axis=2)
    nneg = np.sum(best_cue_resp >= 0, axis=1)
    neg = np.sum(best_cue_resp < 0, axis=1)
    nneg_bool_off = np.greater(nneg, neg)
    if verbose:
        print(f'OFFSET cells dropped for nneg: {np.sum(~nneg_bool_off)}')
    for k, v in data_dict.items():
        if '_on_' in k:
            data_dict[k] = v[nneg_bool_on]  # index 0 dim and first dimension equivlaently
        elif '_off_' in k:
            data_dict[k] = v[nneg_bool_off]
        else:
            raise ValueError

    # remove cells with only a single stage worth of data
    best_cue_resp = np.nanmax(data_dict[f'v{thresh}i{iteration}_norm_on{mod_suffix}'], axis=2)
    notnan = np.sum(~np.isnan(best_cue_resp), axis=1)
    stage_bool_on = notnan > np.ceil(15.5*3) # more than 3 seconds of data need to be finite per cell
    if verbose:
        print(f'ONSET cells dropped for only one stage: {np.sum(~stage_bool_on)}')
    # offset
    best_cue_resp = np.nanmax(data_dict[f'v{thresh}i{iteration}_norm_off{mod_suffix}'], axis=2)
    notnan = np.sum(~np.isnan(best_cue_resp), axis=1)
    stage_bool_off = notnan > np.ceil(15.5*3) # more than 3 seconds of data need to be finite per cell
    if verbose:
        print(f'OFFSET cells dropped for only one stage: {np.sum(~stage_bool_off)}')
    for k, v in data_dict.items():
        if '_on_' in k:
            data_dict[k] = v[stage_bool_on]  # index 0 dim and first dimension equivlaently
        elif '_off_' in k:
            data_dict[k] = v[stage_bool_off]
        else:
            raise ValueError

    return data_dict


def add_scaled_zscore_to_data_dict(data_dict, thresh=4, iteration=10, verbose=False, mod_suffix=''):
    """
    Function to scale zscore data by the mean of max values per cell per mouse so that mice are more 
    comparable. 

    NOTE: Scaling is only calulated on non-Naive data, then applied to everything for "allstages"
    """

    # rescale your z-score data for each mouse to match mouse statistics
    mdict_set = [data_dict[f'v{thresh}i{iteration}_off_mouse{mod_suffix}'], data_dict[f'v{thresh}i{iteration}_on_mouse{mod_suffix}']]
    ddict_set = [data_dict[f'v{thresh}i{iteration}_off{mod_suffix}'], data_dict[f'v{thresh}i{iteration}_on{mod_suffix}']]
    ddict_set_allstages = [data_dict[f'v{thresh}i{iteration}_off{mod_suffix}_allstages'],
                           data_dict[f'v{thresh}i{iteration}_on{mod_suffix}_allstages']]
    new_scaled_keys = [f'v{thresh}i{iteration}_scale_off{mod_suffix}', f'v{thresh}i{iteration}_scale_on{mod_suffix}']
    for mdict, ddict, ddictAS, k in zip(mdict_set, ddict_set, ddict_set_allstages, new_scaled_keys):
        new_data = []
        new_dataAS = []
        scale_factor = []
        scale_factor_expanded = []
        for m in np.unique(mdict):
            mouse_bool = mdict == m
            mouse_data = ddict[mouse_bool, :, :]
            mean_of_maxes = np.nanmean(np.nanmax(np.nanmax(mouse_data, axis=2), axis=1))
            new_data_scaled = mouse_data/mean_of_maxes
            if verbose:
                print(k)
                print(f'\tSCALED {m}: mean={np.nanmean(new_data_scaled)}, std={np.nanstd(new_data_scaled)}')
            new_data.append(new_data_scaled)
            new_dataAS.append(ddictAS[mouse_bool, :, :] / mean_of_maxes)
            scale_factor.append(mean_of_maxes)
            scale_factor_expanded.append([mean_of_maxes]*np.sum(mouse_bool))
        new_ten = np.vstack(new_data)
        new_tenAS = np.vstack(new_dataAS)
        assert new_ten.shape == ddict.shape
        assert new_tenAS.shape == ddictAS.shape
        data_dict[k] = new_ten
        data_dict[k + '_allstages'] = new_tenAS
        data_dict[k + '_mouse_scale_factors'] = np.hstack(scale_factor)
        data_dict[k + '_mouse_scale_factors_expanded'] = np.hstack(scale_factor_expanded)

    return data_dict


def _row_norm(any_mega_tensor_flat):
    """
    Normalize across a row.
    """
    cell_max = np.nanmax(np.nanmax(any_mega_tensor_flat, axis=1), axis=1)
    any_mega_tensor_flat_norm = any_mega_tensor_flat / cell_max[:, None, None]

    return any_mega_tensor_flat_norm, cell_max


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('thresh',
                        action='store',
                        nargs='?',
                        default=4,
                        help='Threshold.')
    parser.add_argument('iteration',
                        action='store',
                        nargs='?',
                        default=10,
                        help='Threshold.')
    parser.add_argument(
        '-f', '--force', action='store_true',
        help='Force rerun of TCA even when ensemble file exists.')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Increase verbosity of terminal.')
    parser.add_argument(
        '-db', '--debug', action='store_true',
        help='Add terminal outputs for debugging.')
    parser.add_argument(
        '-sk', '--skip', action='store_true',
        help='Skip running TCA, will save input data and plots if possible.')

    args = parser.parse_args()

    if args.verbose:
        print(f'Starting v{int(args.thresh)}i{int(args.iteration)}')

    run_unwrapped_tca(
        thresh=int(args.thresh),
        iteration=int(args.iteration),
        force=args.force,
        verbose=args.verbose,
        debug=args.debug,
        skip_tca=args.skip)

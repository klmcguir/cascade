"""Script to fit trial factors using factors from unwrapped models, USES AVG temporal factor."""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from copy import deepcopy
import os
import cascade as cas

import tensortools as tt
from tensortools.tensors import KTensor

from tqdm import tqdm

# parameters
# --------------------------------------------------------------------------------------------------

# input data params
mice = cas.lookups.mice['allFOV']
words = ['respondent' if s in 'OA27' else 'computation' for s in mice]
group_by = 'all3'
with_model = False
nan_thresh = 0.95

# TCA params
method = 'ncp_hals'
replicates = 3
fit_options = {'tol': 0.0001, 'max_iter': 500, 'verbose': False, 'skip_modes': [0, 1]}
models = ['v4i10_norm_on_noT0', 'v4i10_scale_on_noT0', 'v4i10_norm_off_noT0', 'v4i10_scale_off_noT0']
ranks = [9, 9, 8, 8]
iteration = 0

# plot params
hue_order = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']
plot_please = True

# save params
version = '_v2_avg_noT0'

# load in a full size data
# --------------------------------------------------------------------------------------------------
tensor_list = []
id_list = []
bhv_list = []
meta_list = []
for mouse, word in zip(mice, words):

    # return   ids, tensor, meta, bhv
    out = cas.load.load_all_groupday(mouse, word=word, with_model=with_model, group_by=group_by, nan_thresh=nan_thresh)
    tensor_list.append(out[2])
    id_list.append(out[1])
    bhv_list.append(out[4])
    meta_list.append(cas.utils.add_stages_to_meta(out[3], 'parsed_11stage'))

# load models
# --------------------------------------------------------------------------------------------------
ensemble = np.load(cas.paths.analysis_file('tca_ensemble_v4i10_noT0_20210215.npy', 'tca_dfs'), allow_pickle=True).item()
# print(ensemble)

data_dict = np.load(cas.paths.analysis_file('input_data_v4i10_noT0_20210215.npy', 'tca_dfs'), allow_pickle=True).item()
# print(data_dict.keys())

# run refitting
# --------------------------------------------------------------------------------------------------
for mod, rr in zip(models, ranks):
    # automatically pick your cell and mouse vec based on the mod
    if '_on' in mod:
        mod_mice = mod.replace('_norm_on', '_on_mouse').replace('_scale_on', '_on_mouse')
        mod_cells = mod.replace('_norm_on', '_on_cell').replace('_scale_on', '_on_cell')
    elif '_off' in mod:
        mod_mice = mod.replace('_norm_off', '_off_mouse').replace('_scale_off', '_off_mouse')
        mod_cells = mod.replace('_norm_off', '_off_cell').replace('_scale_off', '_off_cell')
    else:
        raise ValueError

    # Fit naive trace, but don't use it for temporal factor calculation
    naive_fit_result = cas.tca.refit_naive_tempfac_tca_unwrapped(ensemble, data_dict, mod=mod, chosen_rank=rr)
    # factors = naive_fit_result.results[rr][iteration].factors
    factors = ensemble.results[rr][iteration].factors

    mouse_vec = data_dict[mod_mice]
    cell_vec = data_dict[mod_cells]

    # rescale factors so most of the var is in cell_weights
    # factors = ensemble[mod].results[rr][0].factors
    temp_max = np.max(factors[1], axis=0)
    tune_max = np.max(factors[2], axis=0)
    scaled_cells = factors[0] * temp_max * tune_max
    scaled_traces = factors[1] / temp_max
    scaled_tune = factors[2] / tune_max
    scaled_factors = (scaled_cells, scaled_traces, scaled_tune)

    # take your scaled traces and break them back appart by stage
    stacker = []
    for i in range(11):
        dts = int(scaled_traces.shape[0] / 11)
        stacker.append(scaled_traces[(dts * i):(dts * (i + 1)), :].T)
    stage_scale_traces = np.dstack(stacker)

    mouse_tfac_list = []
    mouse_kt_list = []
    for mouse in np.unique(mouse_vec):

        # parse cells and mouse identifiers
        mouse_bool_mod = mouse_vec == mouse
        good_model_cell_ids = np.array(cell_vec[mouse_bool_mod])
        data_ind = int(np.where(np.array(mice) == mouse)[0][0])  # from data loading step in previous cell
        mouse_cell_factor = scaled_cells[mouse_bool_mod]

        # get "raw" zscore data
        ids = np.array(id_list[data_ind])
        data_ids_bool = np.isin(ids, good_model_cell_ids)
        data_ids_to_use = ids[data_ids_bool]
        if '_on' in mod:
            good_tensor = tensor_list[data_ind][data_ids_bool, :int(np.ceil(15.5 * 3)), :]
        elif '_off' in mod:
            off_int = int(np.ceil(cas.lookups.stim_length[mouse] + 1) * 15.5)
            good_tensor = tensor_list[data_ind][data_ids_bool, (off_int - 16):(off_int + 31), :]
        else:
            raise ValueError
        meta = meta_list[data_ind]

        # reorder data to match model sorting
        new_order = data_ids_to_use.argsort()[good_model_cell_ids.argsort()]
        assert np.array_equal(data_ids_to_use[new_order], good_model_cell_ids)
        sorted_ids = data_ids_to_use[new_order]
        sorted_tensor = good_tensor[new_order, :, :]

        # rescale or renormalize tensor to match _scale or _norm treatment of cells
        if '_scale' in mod:
            rescale_vec = data_dict[mod + '_mouse_scale_factors_expanded'][mouse_bool_mod]
            sorted_tensor = sorted_tensor / rescale_vec[:, None, None]
        elif '_norm' in mod:
            if '_on' in mod:
                renorm_vec = data_dict[mod.replace('_on', '_on_cell_scale_factors')][mouse_bool_mod]
            elif '_off' in mod:
                renorm_vec = data_dict[mod.replace('_off', '_off_cell_scale_factors')][mouse_bool_mod]
            sorted_tensor = sorted_tensor / renorm_vec[:, None, None]
        else:
            raise ValueError

        # preallocate a new "trial factor" for these cells
        output_trial_factors = np.zeros((sorted_tensor.shape[2], rr)) + np.nan

        # create your tensor per run and for only cells of interest
        stages = cas.lookups.staging['parsed_11stage']

        # loop over runs and build input tensor
        days = meta.reset_index().date.unique()
        for c, di in tqdm(enumerate(days), desc=f'{mouse}: fitting day #', total=len(days)):
            day_bool = meta.reset_index().date.isin([di]).values
            runs = meta.loc[day_bool, :].reset_index().run.unique()
            for c2, di2 in enumerate(runs):
                run_bool = meta.reset_index().run.isin([di2]).values
                this_run_bool = day_bool & run_bool

                # finish off parsing in time
                run_tensor = sorted_tensor[:, :, this_run_bool]
                run_stage = meta.loc[this_run_bool, :].parsed_11stage.unique()[0]
                assert isinstance(run_stage, str)

                # finish off parsing for cells
                missing_data = np.isnan(run_tensor[:, 0, 0])
                new_ids = sorted_ids[~missing_data]
                new_cell_factor = mouse_cell_factor[~missing_data, :]
                assert np.array_equal(good_model_cell_ids[~missing_data],
                                      new_ids)  # make sure youre cells are still in order and matched

                # for the current run get your set of new factors, data, and cells
                new_init_data = run_tensor[~missing_data, :, :]
                new_rand_trial_factor = np.random.rand(new_init_data.shape[2], rr)
                stage_ind = np.where(np.isin(stages, run_stage))[0][0]
                new_temporal_factors = np.nanmean(stage_scale_traces[:, :, :],
                                                  axis=2).T  # take the men shape then normalize to max
                new_temporal_factors_normed = new_temporal_factors / np.max(new_temporal_factors, axis=0)
                new_temporal_factors_normed[np.isnan(new_temporal_factors_normed)] = 0  # fix 0/0 = nan
                assert not np.isnan(new_init_data).any()
                assert new_temporal_factors.shape[1] == new_rand_trial_factor.shape[1]
                assert new_temporal_factors.shape[1] == new_cell_factor.shape[1]

                # run TCA on this small
                init_factors = KTensor((new_cell_factor, new_temporal_factors_normed, new_rand_trial_factor))
                temp_fit_options = deepcopy(fit_options)
                temp_fit_options['init'] = init_factors
                temp_ensemble = tt.Ensemble(fit_method=method, fit_options=temp_fit_options)
                temp_ensemble.fit(new_init_data, ranks=rr, replicates=replicates, verbose=False)

                # save trial factors into a single vector now concatenated together
                output_trial_factors[this_run_bool, :] = temp_ensemble.results[rr][0].factors[2]

        # hold onto model results
        new_KT = KTensor([mouse_cell_factor, scaled_traces, output_trial_factors])
        mouse_kt_list.append(new_KT)
        mouse_tfac_list.append(output_trial_factors)

        if plot_please:
            fig, ax, _ = tt.visualization.plot_factors(new_KT,
                                                       plots=['scatter', 'line', 'scatter'],
                                                       scatter_kw=cas.lookups.tt_plot_options['ncp_hals']['scatter_kw'],
                                                       line_kw=cas.lookups.tt_plot_options['ncp_hals']['line_kw'],
                                                       bar_kw=cas.lookups.tt_plot_options['ncp_hals']['bar_kw'])

            # cell_count = temp_ensemble.results[rr][0].factors[0].shape[0]
            cell_count = new_KT.shape[0]
            for i in range(ax.shape[0]):
                ax[i, 0].set_ylabel(f'                 Component {i+1}', size=16, ha='right', rotation=0)
            ax[0, 1].set_title(f'{mouse}: Trial factor fitting: {mod}, rank {rr} (n = {cell_count})\n\n', size=20)

            plt.savefig(cas.paths.analysis_file(f'{mouse}_{mod}_refittingTRIALs_rank{rr}_facs{version}.png',
                                                f'tca_dfs/TCA_factor_fitting{version}/{mod}'),
                        bbox_inches='tight')

    # save all results
    path = cas.paths.analysis_file(f'{mod}_fit_results_rank{rr}_facs{version}.npy',
                                   f'tca_dfs/TCA_factor_fitting{version}/{mod}')
    np.save(path, {
        'pseudo_ktensors': mouse_kt_list,
        'pseudo_trialfactors': mouse_tfac_list,
        'mice': np.unique(mouse_vec)
    },
            allow_pickle=True)

    if plot_please:
        # This is slow so added it to the end.

        # longform factors psuedocolored
        for kten, mouse in zip(mouse_kt_list, np.unique(mouse_vec)):
            data_ind = int(np.where(np.array(mice) == mouse)[0][0])  # from data loading step
            meta = meta_list[data_ind]

            # pass you naive fitting model factors so that a preferred tuning is chosen
            cas.plotting.tca_unwrapped.longform_factors_annotated(meta,
                                                                  kten,
                                                                  mod,
                                                                  unwrapped_ktensor=factors,
                                                                  folder_tag=version)

        # correlations of trial factor vectors
        sort_ensemble, _, tune_sort = cas.utils.sort_and_rescale_factors(naive_fit_result)
        assert len(tune_sort) == 1

        fig, ax = plt.subplots(len(mouse_kt_list), 3, figsize=(15, 5 * len(mouse_kt_list)))
        counter = 0
        for kt, mouse in zip(mouse_kt_list, np.unique(mouse_vec)):

            data_ind = int(np.where(np.array(mice) == mouse)[0][0])
            meta = meta_list[data_ind]
            inverted_lookup = {v: k for k, v in cas.lookups.lookup_mm[mouse].items()}

            trial_fac_sorted = kt[2][:, tune_sort[0]]

            for ci, cue in enumerate(hue_order):
                cue_boo = meta.initial_condition.isin([inverted_lookup[cue]])
                sns.heatmap(np.corrcoef(trial_fac_sorted[cue_boo, :].T),
                            cmap='vlag',
                            center=0,
                            ax=ax[counter, ci],
                            square=True,
                            vmin=-0.2,
                            vmax=1)
                ax[counter, ci].set_title(f'{mouse}: {cue} trials\nPearson corrcoef\n', size=12)
                if '_on' in mod:
                    ax[counter, ci].add_patch(
                        Rectangle((ci * 3, ci * 3), 3, 3, fill=False, edgecolor=cas.lookups.color_dict[cue], lw=3))
                ax[counter, ci].set_xticks(np.arange(rr) + 0.5)
                ax[counter, ci].set_xticklabels(labels=np.arange(rr) + 1)
                ax[counter, ci].set_yticks(np.arange(rr) + 0.5)
                ax[counter, ci].set_yticklabels(labels=np.arange(rr) + 1, rotation=0)
            counter += 1
        plt.suptitle(f'{mod}, trialfactor correlations, sorted factors', position=(0.5, 0.9), size=16)
        plt.savefig(cas.paths.analysis_file(f'{mod}_trialfactor_corr_rank{rr}{version}.png',
                                            f'tca_dfs/TCA_factor_fitting{version}/{mod}/trialfactor_corr'),
                    bbox_inches='tight')

        # plot the sorted factors to match the correlation plot
        fig, ax, _ = tt.visualization.plot_factors(sort_ensemble.results[rr][iteration].factors,
                                                   plots=['scatter', 'line', 'line'],
                                                   scatter_kw=cas.lookups.tt_plot_options['ncp_hals']['scatter_kw'],
                                                   line_kw=cas.lookups.tt_plot_options['ncp_hals']['line_kw'],
                                                   bar_kw=cas.lookups.tt_plot_options['ncp_hals']['bar_kw'])

        cell_count = sort_ensemble.results[rr][iteration].factors[0].shape[0]
        for i in range(ax.shape[0]):
            ax[i, 0].set_ylabel(f'                 Component {i+1}', size=16, ha='right', rotation=0)
        ax[0, 1].set_title(f'Sorted base model: {mod}, rank {rr} (n = {cell_count})\n\n', size=16)

        plt.savefig(cas.paths.analysis_file(f'{mod}_sortedT0factors_rank{rr}_facs{version}.png',
                                            f'tca_dfs/TCA_factor_fitting{version}/{mod}'),
                    bbox_inches='tight')
        plt.close('all')

        # plot the sorted factors to match the correlation plot
        rfactors = cas.utils.rescale_factors(ensemble.results[rr][iteration].factors)
        fig, ax, _ = tt.visualization.plot_factors(rfactors,
                                                   plots=['scatter', 'line', 'line'],
                                                   scatter_kw=cas.lookups.tt_plot_options['ncp_hals']['scatter_kw'],
                                                   line_kw=cas.lookups.tt_plot_options['ncp_hals']['line_kw'],
                                                   bar_kw=cas.lookups.tt_plot_options['ncp_hals']['bar_kw'])

        cell_count = rfactors[0].shape[0]
        for i in range(ax.shape[0]):
            ax[i, 0].set_ylabel(f'                 Component {i+1}', size=16, ha='right', rotation=0)
        ax[0, 1].set_title(f'Sorted base model: {mod}, rank {rr} (n = {cell_count})\n\n', size=16)

        plt.savefig(cas.paths.analysis_file(f'{mod}_T0factors_rank{rr}_facs{version}.png',
                                            f'tca_dfs/TCA_factor_fitting{version}/{mod}'),
                    bbox_inches='tight')
        plt.close('all')

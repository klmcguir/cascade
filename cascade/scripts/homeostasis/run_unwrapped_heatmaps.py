"""Script to run homeostasis population vector analysis (cosine dist, mean, etc.) and save plots."""

import numpy as np
import cascade as cas
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
import os

# ------------------------------------------------------------------------------------
# # load data
# ensemble = np.load(cas.paths.analysis_file('tca_ensemble_v4i10_noT0_20210215.npy', 'tca_dfs'), allow_pickle=True).item()
# # print(ensemble)
# data_dict = np.load(cas.paths.analysis_file('input_data_v4i10_noT0_20210215.npy', 'tca_dfs'), allow_pickle=True).item()
# # print(data_dict.keys())
ensemble = np.load(cas.paths.analysis_file('tca_ensemble_v4i10_noT0_20210307.npy', 'tca_dfs'), allow_pickle=True).item()
# print(ensemble)
data_dict = np.load(cas.paths.analysis_file('input_data_v4i10_noT0_20210307.npy', 'tca_dfs'), allow_pickle=True).item()
# print(data_dict.keys())

# get sort order for data
sort_ensembles = {}
cell_orders = {}
tune_orders = {}
for k, v in ensemble.items():
    sort_ensembles[k], cell_orders[k], tune_orders[k] = cas.utils.sort_and_rescale_factors(v)

# get all versions of models
models = ['v4i10_norm_on_noT0', 'v4i10_on_noT0', 'v4i10_norm_off_noT0', 'v4i10_off_noT0']
mod_w_naive = [s + '_allstages' for s in models]
models = models + mod_w_naive

# cues to use in order
cues = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']

# perform calculations
# ------------------------------------------------------------------------------------
for mod in tqdm(models, total=len(models), desc='Plotting unwrapped heatmaps'):
    for sort_by in ['tcafixed', 'cuepeakwbroad', 'cuesort']:  # 'mousecuesort', 'unwrappedTCAsort', 'mousesort'

        # set up save laocation
        save_folder = cas.paths.analysis_dir(f'tca_dfs/final_heatmaps/unwrapped_heatmaps')
        mat2ds = data_dict[mod]

        # set rank and TCA model to always be norm models for sorting
        if '_on' in mod:
            rr = 9
            factors = ensemble['v4i10_norm_on_noT0'].results[rr][0].factors
            cell_cats = cas.categorize.best_comp_cats(factors)
            cell_sorter = cell_orders['v4i10_norm_on_noT0'][rr - 1]
            mouse_vec = data_dict['v4i10_on_mouse_noT0']
            cell_vec = data_dict['v4i10_on_cell_noT0']
        elif '_off' in mod:
            rr = 8
            factors = ensemble['v4i10_norm_off_noT0'].results[rr][0].factors
            cell_cats = cas.categorize.best_comp_cats(factors)
            cell_sorter = cell_orders['v4i10_norm_off_noT0'][rr - 1]
            mouse_vec = data_dict['v4i10_off_mouse_noT0']
            cell_vec = data_dict['v4i10_off_mouse_noT0']
        else:
            raise ValueError

        if sort_by == 'tcafixed':
            cell_sorter = cas.sorters.pick_comp_order(cell_cats, cell_sorter)
        elif sort_by == 'cuepeakwbroad':
            cell_sorter = cas.sorters.sort_by_cue_peak_wbroad(mat2ds, mouse_vec) 
        elif sort_by == 'mouseunsort':
            cell_sorter = np.arange(len(cell_sorter), dtype=int)  # keep in order
        elif sort_by == 'mousecuesort':
            cell_sorter = cas.sorters.sort_by_cue_mouse(mat2ds, mouse_vec)
        elif sort_by == 'cuesort':
            cell_sorter = cas.sorters.sort_by_cue_peak(mat2ds, mouse_vec)

        # remap mouse vector for color axis
        mouse_mapper = {k: c for c, k in enumerate(np.unique(mouse_vec))}
        number_mouse_mat = np.array([mouse_mapper[s] for s in mouse_vec])
        number_comp_mat = np.array([s + len(np.unique(mouse_vec)) for s in cell_cats])
        cmap1 = sns.color_palette('muted', len(np.unique(mouse_vec)))
        cmap2 = sns.color_palette('Set3', rr)
        cmap = cmap1 + cmap2 
        
        # keep track of units for plotting
        if '_norm' in mod:
            clabel = 'normalized \u0394F/F'
        elif '_scale' in mod:
            clabel = '\u0394F/F (scaled z-score)'
        else:
            clabel = '\u0394F/F (z-score)'

        # pick your training stage
        if '_allstages' in mod:
            stages = cas.lookups.staging['parsed_11stage_label']
        else:
            stages = cas.lookups.staging['parsed_11stage_label'][1:]

        # plot heatmap 
        ax = []
        fig = plt.figure(figsize=(30, 15))
        gs = fig.add_gridspec(100, 110)
        ax.append(fig.add_subplot(gs[:, 3:6]))
        ax.append(fig.add_subplot(gs[:, 10:38]))
        ax.append(fig.add_subplot(gs[:, 40:68]))
        ax.append(fig.add_subplot(gs[:, 70:98]))
        ax.append(fig.add_subplot(gs[:30, 100:103]))

        # plot "categorical" heatmap using defined color mappings
        if sort_by == 'unwrappedTCAsort' or sort_by == 'tcafixed':
            color_vecs = np.concatenate([number_mouse_mat[cell_sorter, None], number_comp_mat[cell_sorter, None]], axis=1)
            sns.heatmap(color_vecs, cmap=cmap, ax=ax[0], cbar=False)
            ax[0].set_xticks([0.5, 1.5])
            ax[0].set_xticklabels(['mouse', 'component'], rotation=45, ha='right', size=18)
        else:
            sns.heatmap(number_mouse_mat[cell_sorter, None], cmap=cmap1, ax=ax[0], cbar=False)
            ax[0].set_xticklabels(['mouse'], rotation=45, ha='right', size=18)
        ax[0].set_yticklabels([])
        # ax[0].set_ylabel('cell number', size=14)

        if '_norm' in mod:
            vmax = 1
        else:
            vmax = None

        for i in range(1,4):
            forced_order = [0, 2, 1]  #switch the order of the cues
            if i == 3:
                g = sns.heatmap(mat2ds[cell_sorter,:,forced_order[i-1]], ax=ax[i], center=0, vmax=vmax, vmin=-0.5, cmap='vlag',
                                cbar_ax=ax[4], cbar_kws={'label': clabel})
                cbar = g.collections[0].colorbar
                cbar.set_label(clabel, size=16)
            else:
                g = sns.heatmap(mat2ds[cell_sorter,:,forced_order[i-1]], ax=ax[i], center=0, vmax=vmax, vmin=-0.5, cmap='vlag', cbar=False)
            g.set_facecolor('#c5c5c5')
            ax[i].set_title(f'initial cue: {cues[forced_order[i-1]]}\n', size=20)
            stim_starts = [15.5 + 47*s for s in np.arange(len(stages))]
            stim_labels = [f'0\n\n{s}' if c%2 == 0 else f'0\n{s}' for c, s in enumerate(stages)]
            stim_1s = [31 + 47*s for s in np.arange(len(stages))]
            stim_1_labels = ['1' for _ in np.arange(len(stages))]
            stim_starts = list(sum(zip(stim_starts, stim_1s), ()))
            stim_labels = list(sum(zip(stim_labels, stim_1_labels), ()))
            ax[i].set_xticks(stim_starts)
            ax[i].set_xticklabels(stim_labels, rotation=0)
            cell_counts = np.arange(
                100 if mat2ds.shape[0] > 1000 else 50,
                mat2ds.shape[0]+1,
                100 if mat2ds.shape[0] > 1000 else 50,
                dtype=int)
            ax[i].set_yticks(cell_counts-0.5)
            ax[i].set_yticklabels(cell_counts, rotation=0)
            if i == 1:
                ax[i].set_ylabel('cell number', size=18)
            ax[i].set_xlabel('\ntime from stimulus onset (sec)', size=18)
            if i > 1:
                ax[i].set_yticks([])
        # plt.savefig(os.path.join(save_folder, f'{mod}_{sort_by}_rank{rr}_heatmap.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(save_folder, f'{mod}_{sort_by}_rank{rr}_heatmap_highDPI.png'), bbox_inches='tight', dpi=300)
        plt.savefig(os.path.join(save_folder, f'{mod}_{sort_by}_rank{rr}_heatmap_lowDPI.png'), bbox_inches='tight')
        # plt.savefig(os.path.join(save_folder, f'{mod}_{sort_by}_rank{rr}_heatmap.pdf'), bbox_inches='tight')

        # if sort_by == 'tca_fixed':

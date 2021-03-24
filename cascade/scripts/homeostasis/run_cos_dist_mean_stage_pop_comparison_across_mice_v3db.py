"""Script to run homeostasis population vector analysis (cosine dist, mean, etc.) and save plots."""

import numpy as np
import cascade as cas
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
import logging
import os
import warnings


def create_logger(save_folder, name='my_logger'):
    """ Create a logger for watching model fitting, etc.
    """

    # Set up logger
    logger = logging.getLogger(name)
    # logger.setLevel(logging.INFO)
    log_path = os.path.join(save_folder, 'pop_vec.log')

    # Create handlers
    s_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_path)
    s_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    s_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    s_handler.setFormatter(s_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(s_handler)
    logger.addHandler(f_handler)

    return logger


# ------------------------------------------------------------------------------------
# load data
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
for mod in tqdm(models, total=len(models), desc='Running population vector calculations'):

    # set up save laocation
    save_folder = cas.paths.analysis_dir(f'tca_dfs/homeostasis/v3db_20210307/{mod}/population_vector_analysis_by_mouse')
    logger = create_logger(save_folder, name=mod)
    logger.info(f'Starting model: {mod}')
    mat2ds = data_dict[mod]

    # set rank and TCA model to always be norm models for sorting
    if '_on' in mod:
        rr = 9
        factors = ensemble['v4i10_norm_on_noT0'].results[rr][0].factors
        cell_cats = cas.categorize.best_comp_cats(factors)
        cell_sorter = cell_orders['v4i10_norm_on_noT0'][rr - 1]
        mouse_vec = data_dict['v4i10_on_mouse_noT0']
        cell_vec = data_dict['v4i10_on_cell_noT0']
        # drop_cells = np.isin(cell_cats, [0, 4, 7]) #0
        # resorted_drop = drop_cells[cell_orders['v4i10_norm_on_noT0'][rr-1]]
    elif '_off' in mod:
        rr = 8
        factors = ensemble['v4i10_norm_off_noT0'].results[rr][0].factors
        cell_cats = cas.categorize.best_comp_cats(factors)
        cell_sorter = cell_orders['v4i10_norm_off_noT0'][rr - 1]
        mouse_vec = data_dict['v4i10_off_mouse_noT0']
        cell_vec = data_dict['v4i10_off_mouse_noT0']
    else:
        raise ValueError

    # remap mouse vector for color axis
    mouse_mapper = {k: c for c, k in enumerate(np.unique(mouse_vec))}
    number_mouse_mat = np.array([mouse_mapper[s] for s in mouse_vec])
    
    # keep track of units for plotting
    if '_norm' in mod:
        clabel = 'normalized \u0394F/F'
    elif '_scale' in mod:
        clabel = '\u0394F/F (scaled z-score)'
    else:
        clabel = '\u0394F/F (z-score)'

    # pick your training stage
    if '_allstages' in mod:
        stages = cas.lookups.staging['parsed_11stage_T']
    else:
        stages = cas.lookups.staging['parsed_11stage_T'][1:]

    # take pairwise population vector calculations
    cos = np.zeros((len(stages), len(stages), len(cues), 7)) + np.nan
    norms = np.zeros((len(stages), len(stages), len(cues), 7)) + np.nan
    mdiff = np.zeros((len(stages), len(stages), len(cues), 7)) + np.nan
    mdiff2 = np.zeros((len(stages), len(stages), len(cues), 7)) + np.nan
    mdiff3 = np.zeros((len(stages), len(stages), len(cues), 7)) + np.nan
    mdiff_cells = np.zeros((len(stages), len(stages), len(cues), 7)) + np.nan
    mdiff_cells_frac = np.zeros((len(stages), len(stages), len(cues), 7)) + np.nan

    true_mean = mdiff_cells_frac = np.zeros((len(stages), len(cues), 7)) + np.nan
    for mc, m in enumerate(np.unique(mouse_vec)):
        for train_stage in stages:
            for test_stage in stages:

                train_ind = [c for c, s in enumerate(stages) if s == train_stage][0]
                test_ind = [c for c, s in enumerate(stages) if s == test_stage][0]

                sorted_mouse_bool = mouse_vec[cell_sorter] == m

                # get your traces, match and remove missing data
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    train_traces = np.nanmean(mat2ds[cell_sorter, 17 + 47 * train_ind:47 *
                                                     (train_ind + 1), :][sorted_mouse_bool, :, :],
                                              axis=1)
                    true_mean[train_ind, :, mc] = np.nanmean(train_traces, axis=0)
                    test_traces = np.nanmean(mat2ds[cell_sorter,
                                                    17 + 47 * test_ind:47 * (test_ind + 1), :][sorted_mouse_bool, :, :],
                                             axis=1)
                    missing = np.isnan(train_traces[:, 0]) | np.isnan(test_traces[:, 0])
                #         train_traces = train_traces[~missing & resorted_drop, :]
                #         test_traces = test_traces[~missing & resorted_drop, :]
                train_traces = train_traces[~missing, :]
                test_traces = test_traces[~missing, :]
                if test_traces.shape[0] > 0:
                    logger.info(f'{m} {train_stage}-{test_stage}: {test_traces.shape[0]} cells remain after matching')
                else:
                    logger.warning(f'{m} {train_stage}-{test_stage}: {test_traces.shape[0]} cells matched, skipping')
                    continue

                # take cosine distance
                for c, cue in enumerate(cues):
                    cos[train_ind, test_ind, c, mc] = cosine_distances(train_traces[:, c][None, :],
                                                                      test_traces[:, c][None, :])[0][0]
                    norms[train_ind, test_ind, c,
                          mc] = (np.linalg.norm(train_traces[:, c]) -
                                np.linalg.norm(test_traces[:, c])) / np.linalg.norm(train_traces[:, c])
                    mdiff[train_ind, test_ind, c,
                          mc] = (np.mean(train_traces[:, c]) - np.mean(test_traces[:, c])) / np.mean(train_traces[:, c])
                    mdiff2[train_ind, test_ind, c, mc] = np.mean(train_traces[:, c] - test_traces[:, c])
                    mdiff3[train_ind, test_ind, c, mc] = np.mean(np.abs(train_traces[:, c] - test_traces[:, c]))
                    mdiff_cells[train_ind, test_ind, c, mc] = np.mean(train_traces[:, c]) - np.mean(test_traces[:, c])
    logger.info(f'Finished model: {mod}\n')

    # plot heatmap 
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
    # ax[0].set_ylabel('cell number', size=14)

    if '_norm' in mod:
        vmax = 1
    else:
        vmax = None

    for i in range(1,4):
        if i == 3:
            g = sns.heatmap(mat2ds[:,:,i-1], ax=ax[i], center=0, vmax=vmax, vmin=-0.5, cmap='vlag',
                            cbar_ax=ax[4], cbar_kws={'label': clabel})
            cbar = g.collections[0].colorbar
            cbar.set_label(clabel, size=16)
        else:
            g = sns.heatmap(mat2ds[:,:,i-1], ax=ax[i], center=0, vmax=vmax, vmin=-0.5, cmap='vlag', cbar=False)
        g.set_facecolor('#c5c5c5')
        ax[i].set_title(f'initial cue: {cues[i-1]}\n', size=20)
        stim_starts = [15.5 + 47*s for s in np.arange(len(stages))]
        stim_labels = [f'0\n\n{s}' if c%2 == 0 else f'0\n{s}' for c, s in enumerate(stages)]
        ax[i].set_xticks(stim_starts)
        ax[i].set_xticklabels(stim_labels, rotation=0)
        if i == 1:
            ax[i].set_ylabel('cell number', size=18)
        ax[i].set_xlabel('\ntime from stimulus onset (sec)', size=18)
        if i > 1:
            ax[i].set_yticks([])
    plt.savefig(os.path.join(save_folder, f'{mod}_normsort_rank{rr}_heatmap.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(save_folder, f'{mod}_normsort_rank{rr}_heatmap.png'), bbox_inches='tight')

    # cosine distance
    for i in range(3):
        plt.figure()
        sns.heatmap(np.nanmean(cos, axis=3)[:, :, i], cbar_kws={'label': 'cosine distance'})
        plt.title(f'{cues[i]}\nCosine distance\nmean across mice\n{mod}\n', size=16)
        plt.ylabel('learning stage')
        plt.xlabel('learning stage')
        plt.xticks(np.arange(len(stages)) + 0.5, labels=stages, ha='right', rotation=45)
        plt.yticks(np.arange(len(stages)) + 0.5, labels=stages, rotation=0)
        plt.savefig(os.path.join(save_folder, f'cosine_dist_{mod}_{cues[i]}.png'), bbox_inches="tight")
        # plt.savefig(os.path.join(save_folder, f'cosine_dist_{mod}_{cues[i]}.pdf'), bbox_inches="tight")

    # cosine distance
    for i in range(3):
        plt.figure()
        vmin = np.nanmin(np.nanmean(cos, axis=3))
        vmax = np.nanmax(np.nanmean(cos, axis=3))
        sns.heatmap(np.nanmean(cos, axis=3)[:, :, i], cbar_kws={'label': 'cosine distance'}, vmin=vmin, vmax=vmax)
        plt.title(f'{cues[i]}\nCosine distance\nmean across mice\n{mod}\n', size=16)
        plt.ylabel('learning stage')
        plt.xlabel('learning stage')
        plt.xticks(np.arange(len(stages)) + 0.5, labels=stages, ha='right', rotation=45)
        plt.yticks(np.arange(len(stages)) + 0.5, labels=stages, rotation=0)
        plt.savefig(os.path.join(save_folder, f'cosine_dist_caxset_{mod}_{cues[i]}.png'), bbox_inches="tight")
        plt.savefig(os.path.join(save_folder, f'cosine_dist_caxset_{mod}_{cues[i]}.pdf'), bbox_inches="tight")

        # cosine distance
    for i in range(3):
        plt.figure()
        sns.heatmap(np.nanmean(cos, axis=3)[:, :, i], cbar_kws={'label': 'cosine distance'}, vmin=0, vmax=1)
        plt.title(f'{cues[i]}\nCosine distance\nmean across mice\n{mod}\n', size=16)
        plt.ylabel('learning stage')
        plt.xlabel('learning stage')
        plt.xticks(np.arange(len(stages)) + 0.5, labels=stages, ha='right', rotation=45)
        plt.yticks(np.arange(len(stages)) + 0.5, labels=stages, rotation=0)
        plt.savefig(os.path.join(save_folder, f'cosine_dist_caxset01_{mod}_{cues[i]}.png'), bbox_inches="tight")
        plt.savefig(os.path.join(save_folder, f'cosine_dist_caxset01_{mod}_{cues[i]}.pdf'), bbox_inches="tight")

    # change in magnitude mean across cells
    for i in range(3):
        plt.figure()
        sns.heatmap(np.nanmean(mdiff2, axis=3)[:, :, i], cbar_kws={'label': '\u0394 response magnitude'})  #vmax=0.06, vmin=-0.06
        plt.title(f'{cues[i]}\nChange in magnitude\nmean(s1 - s2)\nmean across mice\n{mod}\n',
                    size=16)
        plt.ylabel('learning stage')
        plt.xlabel('learning stage')
        plt.xticks(np.arange(len(stages)) + 0.5, labels=stages, ha='right', rotation=45)
        plt.yticks(np.arange(len(stages)) + 0.5, labels=stages, rotation=0)
        plt.savefig(os.path.join(save_folder, f'mean_dmag_v2_{mod}_{cues[i]}.png'), bbox_inches="tight")
        # plt.savefig(os.path.join(save_folder, f'mean_dmag_v2_{mod}_{cues[i]}.pdf'), bbox_inches="tight")

    # mean amplitude and residuals (with vmax/vmin set)
    for i in range(3):

        vmin = np.nanmin([np.nanmin(np.nanmean(mdiff2, axis=3)), np.nanmin(np.nanmean(true_mean, axis=2))])
        vmax = np.nanmax([np.nanmax(np.nanmean(mdiff2, axis=3)), np.nanmax(np.nanmean(true_mean, axis=2))])
        
        new_map = np.zeros((len(stages)+1, len(stages)+1)) + np.nan
        mean_vec = np.nanmean(true_mean, axis=2)[:, i]
        new_map[1:, 1:] = np.nanmean(mdiff2, axis=3)[:, :, i]
        new_map[1:, 0] = mean_vec
        new_map[0, 1:] = mean_vec

        plt.figure()
        sns.heatmap(new_map, cbar_kws={'label': '\u0394 response magnitude'}, vmax=vmax, vmin=vmin)
        plt.title(f'{cues[i]}\nAmplitude & residuals\nmean(s1) & mean(s1-s2)\nmean across mice\n{mod}\n',
                    size=16)
        plt.ylabel('learning stage')
        plt.xlabel('learning stage')
        plt.xticks(np.arange(len(stages) + 1) + 0.5, labels=['amplitude'] + stages, ha='right', rotation=45)
        plt.yticks(np.arange(len(stages) + 1) + 0.5, labels=['amplitude'] + stages, rotation=0)
        plt.savefig(os.path.join(save_folder, f'truemeanCAX_resid_dmag_v2_{mod}_{cues[i]}.png'), bbox_inches="tight")
        plt.savefig(os.path.join(save_folder, f'truemeanCAX_resid_dmag_v2_{mod}_{cues[i]}.pdf'), bbox_inches="tight")

    # mean amplitude and residuals
    for i in range(3):

        vmin = np.nanmin([np.nanmin(np.nanmean(mdiff2, axis=3)), np.nanmin(np.nanmean(true_mean, axis=2))])
        vmax = np.nanmin([np.nanmin(np.nanmean(mdiff2, axis=3)), np.nanmin(np.nanmean(true_mean, axis=2))])
        
        new_map = np.zeros((len(stages)+1, len(stages)+1)) + np.nan
        mean_vec = np.nanmean(true_mean, axis=2)[:, i]
        new_map[1:, 1:] = np.nanmean(mdiff2, axis=3)[:, :, i]
        new_map[1:, 0] = mean_vec
        new_map[0, 1:] = mean_vec

        plt.figure()
        sns.heatmap(new_map, cbar_kws={'label': '\u0394 response magnitude'})  #vmax=0.06, vmin=-0.06
        plt.title(f'{cues[i]}\nAmplitude & residuals\nmean(s1) & mean(s1-s2)\nmean across mice\n{mod}\n',
                    size=16)
        plt.ylabel('learning stage')
        plt.xlabel('learning stage')
        plt.xticks(np.arange(len(stages) + 1) + 0.5, labels=['amplitude'] + stages, ha='right', rotation=45)
        plt.yticks(np.arange(len(stages) + 1) + 0.5, labels=['amplitude'] + stages, rotation=0)
        plt.savefig(os.path.join(save_folder, f'truemean_resid_dmag_v2_{mod}_{cues[i]}.png'), bbox_inches="tight")
        plt.savefig(os.path.join(save_folder, f'truemean_resid_dmag_v2_{mod}_{cues[i]}.pdf'), bbox_inches="tight")

    # ABS magnitude changes: this double emphasizes the change in magnitude
    for i in range(3):
        plt.figure()
        sns.heatmap(np.nanmean(mdiff3, axis=3)[:, :, i], cbar_kws={'label': '|\u0394 response magnitude|'})
        plt.title(
            f'{cues[i]}\nAbs change in magnitude\nmean(abs(s1 - s2))\nmean across mice\n{mod}\n',
            size=16)
        plt.ylabel('learning stage')
        plt.xlabel('learning stage')
        plt.xticks(np.arange(len(stages)) + 0.5, labels=stages, ha='right', rotation=45)
        plt.yticks(np.arange(len(stages)) + 0.5, labels=stages, rotation=0)
        plt.savefig(os.path.join(save_folder, f'mean_dmag_abs_{mod}_{cues[i]}.png'), bbox_inches="tight")
        # plt.savefig(os.path.join(save_folder, f'mean_dmag_abs_{mod}_{cues[i]}.pdf'), bbox_inches="tight")

    # change in magnitude mean across cells MEAN ACROSS POP 1st
    for i in range(3):
        plt.figure()
        sns.heatmap(np.nanmean(mdiff_cells, axis=3)[:, :, i], cbar_kws={'label': '\u0394 response magnitude'})  #vmax=0.06, vmin=-0.06
        plt.title(f'{cues[i]}\nChange in magnitude\nmean(s1) - mean(s2)\nmean across mice\n{mod}\n',
                    size=16)
        plt.ylabel('learning stage')
        plt.xlabel('learning stage')
        plt.xticks(np.arange(len(stages)) + 0.5, labels=stages, ha='right', rotation=45)
        plt.yticks(np.arange(len(stages)) + 0.5, labels=stages, rotation=0)
        plt.savefig(os.path.join(save_folder, f'test_mean_dmag_v2_{mod}_{cues[i]}.png'), bbox_inches="tight")
        # plt.savefig(os.path.join(save_folder, f'mean_dmag_v2_{mod}_{cues[i]}.pdf'), bbox_inches="tight")

    # change in magnitude mean across cells MEAN ACROSS POP 1st
    for i in range(3):
        plt.figure()
        sns.heatmap(np.nanmean(mdiff, axis=3)[:, :, i], cbar_kws={'label': '\u0394 response magnitude'})  #vmax=0.06, vmin=-0.06
        plt.title(f'{cues[i]}\n% change in magnitude\n(mean(s1) - mean(s2))/mean(s1)\nmean across mice\n{mod}\n',
                    size=16)
        plt.ylabel('learning stage')
        plt.xlabel('learning stage')
        plt.xticks(np.arange(len(stages)) + 0.5, labels=stages, ha='right', rotation=45)
        plt.yticks(np.arange(len(stages)) + 0.5, labels=stages, rotation=0)
        plt.savefig(os.path.join(save_folder, f'testfrac_mean_dmag_v2_{mod}_{cues[i]}.png'), bbox_inches="tight")
        # plt.savefig(os.path.join(save_folder, f'mean_dmag_v2_{mod}_{cues[i]}.pdf'), bbox_inches="tight")

    plt.figure()
    for mc, m in enumerate(np.unique(mouse_vec)):
        for i in range(3):
            off_diag = []
            for k in range(10):
                try:
                    off_diag.append(cos[k,k-1,i, mc])
                except:
                    pass
            plt.plot(off_diag, color=cas.lookups.color_dict[cues[i]])
        plt.title(f'{cues[i]}\nOffdiag: Cosine distance\nmean across mice\n{mod}\n', size=16)
        plt.xticks(np.arange(len(stages)), labels=stages, ha='right', rotation=45)
        plt.xlabel('next learning stage')
        plt.ylabel('cosine distance')
    plt.savefig(os.path.join(save_folder, f'test_traces_cosdist_{mod}_{cues[i]}.png'), bbox_inches="tight")

    plt.figure()
    for mc, m in enumerate(np.unique(mouse_vec)):
        for i in range(3):
            off_diag = []
            for k in range(10):
                try:
                    off_diag.append(mdiff2[k,k-1,i, mc])
                except:
                    pass
            plt.plot(off_diag, color=cas.lookups.color_dict[cues[i]])
        plt.title(f'Offdiag: \u0394magnitude\nmean(s1 - s2)\nmean across mice\n{mod}\n',size=16)
        plt.xticks(np.arange(len(stages)), labels=stages, ha='right', rotation=45)
        plt.xlabel('next learning stage')
        plt.ylabel('\u0394 response magnitude')
    plt.savefig(os.path.join(save_folder, f'test_traces_mean_dmag_{mod}_{cues[i]}.png'), bbox_inches="tight")

    plt.figure()
    for mc, m in enumerate(np.unique(mouse_vec)):
        for i in range(3):
            off_diag = []
            for k in range(10):
                try:
                    off_diag.append(mdiff3[k,k-1,i, mc])
                except:
                    pass
            plt.plot(off_diag, color=cas.lookups.color_dict[cues[i]])
        plt.title(f'Offdiag: abs(\u0394magnitude)\nmean(abs(s1 - s2))\nmean across mice\n{mod}\n',size=16)
        plt.xticks(np.arange(len(stages)), labels=stages, ha='right', rotation=45)
        plt.xlabel('next learning stage')
        plt.ylabel('|\u0394 response magnitude|')
    plt.savefig(os.path.join(save_folder, f'test_traces_mean_dmag_abs_{mod}_{cues[i]}.png'), bbox_inches="tight")

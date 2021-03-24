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
    logger.setLevel(logging.INFO)
    log_path = os.path.join(save_folder, 'pop_vec.log')

    # Create handlers
    s_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_path)
    s_handler.setLevel(logging.WARNING)
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
    save_folder = cas.paths.analysis_dir(f'tca_dfs/homeostasis/v3db_20210307/{mod}/population_vector_analysis')
    logger = create_logger(save_folder, name=mod)
    logger.info(f'Starting model: {mod}')
    mat2ds = data_dict[mod]

    # set rank and TCA model to always be norm models
    if '_on' in mod:
        rr = 9
        factors = ensemble['v4i10_norm_on_noT0'].results[rr][0].factors
        cell_cats = cas.categorize.best_comp_cats(factors)
        cell_sorter = cell_orders['v4i10_norm_on_noT0'][rr-1]
        mouse_vec = data_dict['v4i10_on_mouse_noT0']
        cell_vec = data_dict['v4i10_on_cell_noT0']
        # drop_cells = np.isin(cell_cats, [0, 4, 7]) #0
        # resorted_drop = drop_cells[cell_orders['v4i10_norm_on_noT0'][rr-1]]
    elif '_off' in mod:
        rr = 8
        factors = ensemble['v4i10_norm_off_noT0'].results[rr][0].factors
        cell_cats = cas.categorize.best_comp_cats(factors)
        cell_sorter = cell_orders['v4i10_norm_off_noT0'][rr-1]
        mouse_vec = data_dict['v4i10_off_mouse_noT0']
        cell_vec = data_dict['v4i10_off_mouse_noT0']
    else:
        raise ValueError

    # pick your training stage
    if '_allstages' in mod:
        stages = cas.lookups.staging['parsed_11stage_T']
    else:
        stages = cas.lookups.staging['parsed_11stage_T'][1:]

    # take pairwise population vector calculations
    cos = np.zeros((len(stages), len(stages), len(cues))) + np.nan
    norms = np.zeros((len(stages), len(stages), len(cues))) + np.nan
    mdiff = np.zeros((len(stages), len(stages), len(cues))) + np.nan
    mdiff2 = np.zeros((len(stages), len(stages), len(cues))) + np.nan
    mdiff3 = np.zeros((len(stages), len(stages), len(cues))) + np.nan
    true_mean = np.nanmean(mat2ds, axis=0)
    for train_stage in stages:
        for test_stage in stages:

            train_ind = [c for c, s in enumerate(stages) if s == train_stage][0]
            test_ind = [c for c, s in enumerate(stages) if s == test_stage][0]
            
            # get your traces, match and remove missing data
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                train_traces = np.nanmean(mat2ds[cell_sorter, 17+47*train_ind:47*(train_ind+1), :], axis=1)
                test_traces = np.nanmean(mat2ds[cell_sorter, 17+47*test_ind:47*(test_ind+1), :], axis=1)
                missing = np.isnan(train_traces[:, 0]) |  np.isnan(test_traces[:, 0])
    #         train_traces = train_traces[~missing & resorted_drop, :]
    #         test_traces = test_traces[~missing & resorted_drop, :]
            train_traces = train_traces[~missing, :]
            test_traces = test_traces[~missing, :]
            if test_traces.shape[0] > 50:
                logger.info(f'{train_stage}-{test_stage}: {test_traces.shape[0]} cells remain after matching')
            else:
                logger.warning(f'{train_stage}-{test_stage}: {test_traces.shape[0]} cells remain after matching')

            
            # take cosine distance
            for c, cue in enumerate(cues):
                cos[train_ind, test_ind, c] = cosine_distances(train_traces[:, c][None, :], test_traces[:, c][None, :])[0][0]
                norms[train_ind, test_ind, c] = (np.linalg.norm(train_traces[:, c]) - np.linalg.norm(test_traces[:, c]))/np.linalg.norm(train_traces[:, c])
                mdiff[train_ind, test_ind, c] = (np.mean(train_traces[:, c]) - np.mean(test_traces[:, c]))/np.mean(train_traces[:, c])
                mdiff2[train_ind, test_ind, c] = np.mean(train_traces[:, c] - test_traces[:, c])
                mdiff3[train_ind, test_ind, c] = np.mean(np.abs(train_traces[:, c] - test_traces[:, c]))
    logger.info(f'Finished model: {mod}\n')

    # cosine distance
    for i in range(3):
        plt.figure()
        sns.heatmap(cos[:,:,i], cbar_kws={'label': 'cosine distance'})
        plt.title(f'{cues[i]}\nCosine distance\npopulation vector (pooling mice)\n{mod}\n', size=16)
        plt.ylabel('learning stage')
        plt.xlabel('learning stage')
        plt.xticks(np.arange(len(stages)) + 0.5, labels=stages, ha='right', rotation=45);
        plt.yticks(np.arange(len(stages)) + 0.5, labels=stages, rotation=0);
        plt.savefig(os.path.join(save_folder, f'cosine_dist_{mod}_{cues[i]}.png'), bbox_inches="tight")
        plt.savefig(os.path.join(save_folder, f'cosine_dist_{mod}_{cues[i]}.pdf'), bbox_inches="tight")

    # change in magnitude mean across cells
    for i in range(3):
        plt.figure()
        sns.heatmap(mdiff2[:,:,i], cbar_kws={'label': '\u0394 response magnitude'})  #vmax=0.06, vmin=-0.06
        plt.title(f'{cues[i]}\nChange in magnitude\nmean(s1 - s2)\npopulation vector (pooling mice)\n{mod}\n', size=16)
        plt.ylabel('learning stage')
        plt.xlabel('learning stage')
        plt.xticks(np.arange(len(stages)) + 0.5, labels=stages, ha='right', rotation=45);
        plt.yticks(np.arange(len(stages)) + 0.5, labels=stages, rotation=0);
        plt.savefig(os.path.join(save_folder, f'mean_dmag_v2_{mod}_{cues[i]}.png'), bbox_inches="tight")
        plt.savefig(os.path.join(save_folder, f'mean_dmag_v2_{mod}_{cues[i]}.pdf'), bbox_inches="tight")

    # ABS magnitude changes: this double emphasizes the change in magnitude 
    for i in range(3):
        plt.figure()
        sns.heatmap(mdiff3[:,:,i], cbar_kws={'label': '|\u0394 response magnitude|'})
        plt.title(f'{cues[i]}\nAbs change in magnitude\nmean(abs(s1 - s2))\npopulation vector (pooling mice)\n{mod}\n', size=16)
        plt.ylabel('learning stage')
        plt.xlabel('learning stage')
        plt.xticks(np.arange(len(stages)) + 0.5, labels=stages, ha='right', rotation=45);
        plt.yticks(np.arange(len(stages)) + 0.5, labels=stages, rotation=0);
        plt.savefig(os.path.join(save_folder, f'mean_dmag_abs_{mod}_{cues[i]}.png'), bbox_inches="tight")
        plt.savefig(os.path.join(save_folder, f'mean_dmag_abs_{mod}_{cues[i]}.pdf'), bbox_inches="tight")

    # change in magnitude mean across cells MEAN ACROSS POP 1st
    for i in range(3):
        plt.figure()
        sns.heatmap(mdiff[:, :, i], cbar_kws={'label': '\u0394 response magnitude'})  #vmax=0.06, vmin=-0.06
        plt.title(f'{cues[i]}\n% change in magnitude\n(mean(s1) - mean(s2))/mean(s1)\npopulation vector (pooling mice)\n{mod}\n',
                    size=16)
        plt.ylabel('learning stage')
        plt.xlabel('learning stage')
        plt.xticks(np.arange(len(stages)) + 0.5, labels=stages, ha='right', rotation=45)
        plt.yticks(np.arange(len(stages)) + 0.5, labels=stages, rotation=0)
        plt.savefig(os.path.join(save_folder, f'testfrac_mean_dmag_v2_{mod}_{cues[i]}.png'), bbox_inches="tight")
        # plt.savefig(os.path.join(save_folder, f'mean_dmag_v2_{mod}_{cues[i]}.pdf'), bbox_inches="tight")
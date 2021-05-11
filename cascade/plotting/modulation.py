""" Functions for plotting modulation (mostly figire 3-5 related) """
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import warnings
import numpy as np

from ..trialanalysis import build_any_mat, build_cue_mat
from .. import load, utils, lookups, paths, sorters, selectivity
import os
from copy import deepcopy


def make_all_plots():
    """Helper function to generate a shortlist of plots from this module
    """

    # heatmaps
    plot_modulation_heatmaps(match_to='onsets', index_on='pre_speed')
    plot_modulation_heatmaps(match_to='onsets', index_on='th')
    plot_modulation_heatmaps(match_to='onsets', index_on='gonogo')
    plot_modulation_heatmaps(match_to='offsets', index_on='pre_speed')
    plot_modulation_heatmaps(match_to='offsets', index_on='th')
    plot_modulation_heatmaps(match_to='offsets', index_on='gonogo')

    # modulation traces
    plot_modulation_vertical_traces(match_to='onsets', index_on='pre_speed')
    plot_modulation_vertical_traces(match_to='onsets', index_on='th')
    plot_modulation_vertical_traces(match_to='onsets', index_on='gonogo')
    plot_modulation_vertical_traces(match_to='offsets', index_on='pre_speed')
    plot_modulation_vertical_traces(match_to='offsets', index_on='th')
    plot_modulation_vertical_traces(match_to='offsets', index_on='gonogo')

    # vertical traces
    plot_modulation_vertical_bars(match_to='onsets', index_on='pre_speed')
    plot_modulation_vertical_bars(match_to='onsets', index_on='th')
    plot_modulation_vertical_bars(match_to='onsets', index_on='gonogo')
    plot_modulation_vertical_bars(match_to='offsets', index_on='pre_speed')
    plot_modulation_vertical_bars(match_to='offsets', index_on='th')
    plot_modulation_vertical_bars(match_to='offsets', index_on='gonogo')


def trial_count_modulation(match_to='onsets', trace_type='zscore_day'):
    """Scatterplot of mean response per stage vs trial number preceding that stage.

    Parameters
    ----------
    match_to : str, optional
        Onsests or offsets, by default 'onsets'
    trace_type : str, optional
        zscore_day, norm, or dff, by default 'zscore_day'

    Raises
    ------
    NotImplementedError
        Offsets need conditional handling for different types of tunning
    ValueError
        If you ask for a trace type that does not exist. 
    """

    # Not ready yet
    if match_to == 'offsets':
        raise NotImplementedError('Needs category specific averages for offset classes.')

    # load TCA data
    on_ens = load.core_tca_data(limit_to=None, match_to=match_to)
    mouse_vec = on_ens['mouse_vec']
    cell_vec = on_ens['cell_vec']
    cell_sorter = on_ens['cell_sorter']
    cell_cats = on_ens['cell_cats']

    # create a matrix of a given trace type
    if trace_type == 'zscore_day':
        z_cue_mat = build_cue_mat(
            mouse_vec,
            cell_vec,
            limit_tag='onsets',
            allstage=True,
            norm_please=False,
            no_disengaged=False,
        )
    elif trace_type == 'dff':
        z_cue_mat = build_cue_mat(
            mouse_vec,
            cell_vec,
            limit_tag='onsets',
            allstage=True,
            norm_please=False,
            no_disengaged=False,
            load_kws=dict(word_pair=('inspector', 'badge'), trace_type='dff')
        )
    elif trace_type == 'norm':
        z_cue_mat = build_cue_mat(
            mouse_vec,
            cell_vec,
            limit_tag='onsets',
            allstage=True,
            norm_please=True,
            no_disengaged=False,
        )
    else:
        raise ValueError

    # create matricies of trial counts, accounting for days with unbalanced trials

    early_trials = np.zeros((7, 3, 11))
    early_dp = np.zeros((7, 3, 11))
    early_hit_trials = np.zeros((7, 3, 11))
    early_error_trials = np.zeros((7, 3, 11))
    for mc, mouse in enumerate(lookups.mice['rev7']):

    #     word1 =  'march' if mouse == 'OA27' else 'frame' # unmatched allowed, no pavs
        word1 =  'metro' if mouse == 'OA27' else 'largely' # unmatched allowed, w pavs
        word2 =  'respondent' if mouse == 'OA27' else 'computation'
        long_meta = load.groupday_tca_meta(mouse=mouse,  word=word1, nan_thresh=0.95)
        true_meta = load.groupday_tca_meta(mouse=mouse,  word=word2, nan_thresh=0.95)

        # long_meta = utils.add_stages_to_meta(long_meta, 'parsed_11stage')
        true_meta = utils.add_stages_to_meta(true_meta, 'parsed_11stage')

        long_meta = utils.add_reversal_mismatch_condition_to_meta(long_meta)
        true_meta = utils.add_reversal_mismatch_condition_to_meta(true_meta)
        
        with_true_stages = long_meta.join(true_meta.loc[:, ['parsed_11stage', 'dprime_run']])
        complete_stages = with_true_stages.parsed_11stage.fillna(method='ffill').to_frame()

        stages = lookups.staging['parsed_11stage']
        for si, stagi in enumerate(stages):
                
            not_stages = [s for s in lookups.staging['parsed_11stage'] if s not in [stagi]] 
                
            early_uncounted_bool = (
                (complete_stages.parsed_11stage.isin([stagi]).values
                | with_true_stages.parsed_11stage.isna().values)
                & ~complete_stages.parsed_11stage.isin(not_stages)
            )
                        
    #     early_uncounted_bool = ((with_true_stages.parsed_11stage.isin(['L0 naive']).values
    #         | with_true_stages.parsed_11stage.isna().values)
    #         & ~(with_true_stages.learning_state.isin(['learning', 'reversal1']).values
    #             & ~with_true_stages.parsed_11stage.isna().values)
    #         & ~with_true_stages.learning_state.isin(['reversal1']).values
    #            )
                
            pre_T1_trials = long_meta.loc[early_uncounted_bool, :]
            trial_counts = pre_T1_trials.groupby('mismatch_condition').count().orientation
            if len(trial_counts) == 0:
                continue
            for c, cue in enumerate(['becomes_unrewarded', 'becomes_rewarded', 'remains_unrewarded']):
                try:
                    early_trials[mc, c, si] = trial_counts[cue]
                except:
                    pass
                
            pre_T1_trials = with_true_stages.loc[early_uncounted_bool, :]
            trial_dp = pre_T1_trials.groupby('mismatch_condition').mean().dprime_run
            if len(trial_counts) == 0:
                continue
            for c, cue in enumerate(['becomes_unrewarded', 'becomes_rewarded', 'remains_unrewarded']):
                try:
                    early_dp[mc, c, si] = trial_dp[cue]
                except:
                    pass
                
            stage_trials = with_true_stages.loc[early_uncounted_bool, :]
            trial_hit = pre_T1_trials.groupby(['mismatch_condition', 'trialerror']).count().orientation
            if len(trial_counts) == 0:
                continue
            for c, cue in enumerate(['becomes_unrewarded', 'becomes_rewarded', 'remains_unrewarded']):
                try:
                    errorboo = trial_hit[cue].keys().isin([1, 3, 5])
                    early_error_trials[mc, c, si] = trial_hit[cue][errorboo].sum()
                except:
                    pass
                try:
                    hitboo = trial_hit[cue].keys().isin([0])
                    early_hit_trials[mc, c, si] = trial_hit[cue][hitboo].sum()
                except:
                    pass

    # take mean for stages 
    early_responses = np.zeros((7, 3, 11))
    for mc, mouse in enumerate(lookups.mice['rev7']):
        mboo = mouse_vec == mouse
        for ci, i in enumerate([0, 2, 1]):

            mean_cue_resp = np.nanmean(
                np.nanmean(utils.wrap_tensor(z_cue_mat[mboo,:,i])[:, 17:, :], axis=1)
                , axis=0)
            T1_cue_resp= mean_cue_resp #[1]
            early_responses[mc, ci, :] = T1_cue_resp

    early_responses_adapt = np.zeros((7, 3, 11))
    for mc, mouse in enumerate(lookups.mice['rev7']):
        mboo = mouse_vec == mouse
        for ci, i in enumerate([0, 2, 1]):
            if i == 0:
                cboo = np.isin(cell_cats, [6])
            elif i == 2:
                cboo = np.isin(cell_cats, [2])
            elif i == 1:
                cboo = np.isin(cell_cats, [4])
            mean_cue_resp = np.nanmean(
                np.nanmean(utils.wrap_tensor(z_cue_mat[mboo & cboo,:,i])[:, 17:, :], axis=1)
                , axis=0)
            T1_cue_resp= mean_cue_resp #[1]
            early_responses_adapt[mc, ci, :] = T1_cue_resp

    early_responses_noadapt = np.zeros((7, 3, 11))
    for mc, mouse in enumerate(lookups.mice['rev7']):
        mboo = mouse_vec == mouse
        for ci, i in enumerate([0, 2, 1]):
            if i == 0:
                cboo = np.isin(cell_cats, [5, 0]) #[6])
            elif i == 2:
                cboo = np.isin(cell_cats, [3, 1]) #[2])
            elif i == 1:
                cboo = np.isin(cell_cats, [8])
            mean_cue_resp = np.nanmean(
                np.nanmean(utils.wrap_tensor(z_cue_mat[mboo & cboo,:,i])[:, 17:, :], axis=1)
                , axis=0)
            T1_cue_resp= mean_cue_resp #[1]
            early_responses_noadapt[mc, ci, :] = T1_cue_resp

    # plot scatters 
    with sns.axes_style('whitegrid'):
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True, sharex=True)

    flat_trials = np.cumsum(early_trials[:,:,:], axis=2).flatten()
    flat_resp = early_responses_adapt[:,:,:].flatten()
    flat_dp = early_dp[:,:,:].flatten()
    cue_color = ((['Initially rewarded']*11 + ['Becomes rewarded']*11 + ['Unrewarded']*11) * 7)
    g = sns.scatterplot(
    #         x=flat_trials, y=flat_resp, hue=cue_color, palette=lookups.color_dict, s=100,
        x=flat_trials, y=flat_resp, s=100, # hue=flat_trials,
        hue=cue_color, palette=lookups.color_dict,
        alpha=0.5, ax=ax[0]
    )
    g.axes.set_xscale('log')

    flat_resp = early_responses_noadapt[:,:,:].flatten()
    g = sns.scatterplot(
    #         x=flat_trials, y=flat_resp, hue=cue_color, palette=lookups.color_dict, s=100,
        x=flat_trials, y=flat_resp, s=100, # hue=flat_trials,
        hue=cue_color, palette=lookups.color_dict,
        alpha=0.5, ax=ax[1]
    )

    # add regression lines
    # sns.regplot(x=np.cumsum(early_trials, axis=2)[:,0,:6].flatten(),
    #             y=early_responses[:,0,:6].flatten(), scatter=False, logx=True,
    #             color=lookups.color_dict['Initially rewarded'], ci=False)
    # sns.regplot(x=np.cumsum(early_trials, axis=2)[:,1,:6].flatten(),
    #             y=early_responses[:,1,:6].flatten(), scatter=False, logx=True,
    #             color=lookups.color_dict['Becomes rewarded'], ci=False)
    # sns.regplot(x=np.cumsum(early_trials, axis=2)[:,2,:6].flatten(),
    #             y=early_responses[:,2,:6].flatten(), scatter=False, logx=True,
    #             color=lookups.color_dict['Unrewarded'], ci=False)

    # sns.despine()
    ax[0].set_xlabel('Trial count preceding learning stage', size=14)
    ax[1].set_xlabel('Trial count preceding learning stage', size=14)
    ax[0].set_title('Adapting class', size=14)
    ax[1].set_title('Non-adapting classes', size=14)
    # plt.xlabel("Performance (d')")
    ax[0].set_ylabel('Population response magnitude\n\u0394F/F (z-score)', size=14)
    plt.suptitle('Initial and reversal learning', position=(0.5, 1.05), size=20)


def plot_modulation_heatmaps(match_to='onsets',
                             index_on='pre_speed',
                             scale='norm',
                             group_or_comp_sort = 'group',
                             vmax=1,
                             allstage=False,
                             save_please=True):
    """Plot heatmaps sorted by cue groups or components for a given source of modulation.

    Parameters
    ----------
    match_to : str, optional
        Onsets or offsets, by default 'onsets'
    index_on : str, optional
        Type of modulation, by default 'pre_speed'
    scale : str, optional
        Norm or score color scale, by default 'norm'
    group_or_comp_sort : str, optional
        Group according to looks.tuning_groups or by components, by default 'group'
    vmax : int, optional
        Maximum color value, by default 1
    allstage : bool, optional
        Include naive, by default False
    save_please : bool, optional
        Choose to save, by default True
    """
    
    # text sizes
    xtl_size = 20
    ytl_size = 22
    t_size = 26
    xl_size = 24
    yl_size = 24
    
    # build or load necessary matrices and index dfs
    df, mat = selectivity.ab_index_df(match_to=match_to, index_on=index_on, return_matrix=True)
    ens = load.core_tca_data(match_to=match_to)
    mouse_vec = ens['mouse_vec']
    cell_cats = ens['cell_cats']
    cues =  ['Initially rewarded', 'Becomes rewarded', 'Unrewarded']
    
    # check that tca_data and index data match and if they do, create sorter
    assert np.array_equal(ens['cell_cats'], df.cell_cats.values)
    if group_or_comp_sort == 'comp':
        cell_sorter = sorters.pick_comp_order_plus_bhv_mod(ens['cell_cats'],
                                 ens['cell_sorter'],
                                 mod_vec=df['a-b'].values)
    elif group_or_comp_sort == 'group':
#         match_to = 'onsets'
#         bhv_baseline_or_stim = 'stim'
#         save_tag = 'on'
#         bhv_type = 'lick'
#         corr_df_2s = pd.read_pickle(
#             os.path.join(
#                 lookups.coreroot,
#                 f'{match_to}_{bhv_baseline_or_stim}_{save_tag + bhv_type}_corr_2s_df.pkl'))
#         test_vec = corr_df_2s.mean_onlick_corr_2s_11stage.values
        cell_sorter = sorters.pick_comp_order_plus_cuelevel_bhv_mod(ens['cell_cats'],
                                 ens['cell_sorter'],
#                                 mod_vec=test_vec,
                                 mod_vec=df['a-b'].values
                                )
    else:
        raise NotImplementedError
    
    # set save location
    save_folder = paths.analysis_dir(f'figures/figure_3/modulation_heatmaps/{match_to}_{index_on}_{group_or_comp_sort}')
    print(save_folder)

    # set number of componenets for plotting
    if match_to == 'onsets':
        rr = 9
    elif match_to == 'offsets':
        rr = 8
    else:
        raise ValueError

    # optionally remove naive data
    if allstage:
        mat2ds = deepcopy(mat)
    else:
        mat2ds = deepcopy(mat)
        mat2ds = mat2ds[:, 47:, :]
        assert mat2ds.shape[1] == 470

    # optionally peak normalize your data
    if scale == 'norm':
        max_vec = np.nanmax(np.nanmax(mat2ds, axis=2), axis=1)
        max_vec[max_vec <= 0] = np.nan
        mat2ds = mat2ds/max_vec[:, None, None]

    # remap mouse vector for color axis
    mouse_mapper = {k: c for c, k in enumerate(np.unique(mouse_vec))}
    number_mouse_mat = np.array([mouse_mapper[s] for s in mouse_vec])
    number_comp_mat = np.array([s + len(np.unique(mouse_vec)) for s in cell_cats])
    cmap1 = sns.color_palette('muted', len(np.unique(mouse_vec)))
    cmap2 = sns.color_palette('Set3', rr)
    cmap = cmap1 + cmap2

    # keep track of units for plotting
    if 'norm' in scale:
        clabel = 'normalized \u0394F/F'
    else:
        clabel = '\u0394F/F (z-score)'

    # pick your training stage
    if allstage:
        stages = lookups.staging['parsed_11stage_label_short']
    else:
        stages = lookups.staging['parsed_11stage_label_short'][1:]

    # plot heatmap
    ax = []
    fig = plt.figure(figsize=(30, 15))
    gs = fig.add_gridspec(100, 110)
    ax.append(fig.add_subplot(gs[:, 0:3]))
    ax.append(fig.add_subplot(gs[:, 10:38]))
    ax.append(fig.add_subplot(gs[:, 40:68]))
    ax.append(fig.add_subplot(gs[:, 70:98]))
    ax.append(fig.add_subplot(gs[:30, 100:103]))

    # plot "categorical" heatmap using defined color mappings
    color_vecs = np.concatenate([number_mouse_mat[cell_sorter, None], number_comp_mat[cell_sorter, None]], axis=1)
    fix_cmap = sns.color_palette('Set3', rr) #lookups.cmap_fixed_sort_rank9_onset
    just_comps = np.array(cell_cats[cell_sorter, None], dtype=float)
    just_comps[just_comps == -1] = np.nan
    sns.heatmap(just_comps, cmap=fix_cmap, ax=ax[0], cbar=False)
    ax[0].set_xticks([0.5])
    ax[0].set_xticklabels(['Component'], rotation=45, ha='right', size=yl_size)
    ax[0].set_yticks([])
    ax[0].set_yticklabels([])

    for ci, i in enumerate([0, 2, 1]):
        ci = ci + 1
        if i == 0:
            g = sns.heatmap(mat2ds[cell_sorter, :, i],
                            ax=ax[ci],
                            center=0,
                            vmax=vmax,
                            vmin=-0.5,
                            cmap='vlag',
                            cbar_ax=ax[4],
                            cbar_kws={'label': clabel})
            cbar = g.collections[0].colorbar
            cbar.set_label(clabel, size=yl_size)
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=ytl_size)
        else:
            g = sns.heatmap(mat2ds[cell_sorter, :, i],
                            ax=ax[ci],
                            center=0,
                            vmax=vmax,
                            vmin=-0.5,
                            cmap='vlag',
                            cbar=False)
        g.set_facecolor('#c5c5c5')
        ax[ci].set_title(f'{cues[ci-1]}\n', size=t_size, color=lookups.color_dict[cues[ci-1]])
        stim_starts = [15.5 + 47 * s for s in np.arange(len(stages))]
        stim_labels = [f'\n\n{s}' if c % 2 == 0 else f'\n{s}' for c, s in enumerate(stages)]
        ax[ci].set_xticks(stim_starts)
        ax[ci].set_xticklabels(stim_labels, rotation=0, size=xtl_size)
        if i == 0:
            ax[ci].set_ylabel('Cell number', size=yl_size)
            cell_counts = np.arange(
                    250 if mat2ds.shape[0] > 1000 else 50,
                    mat2ds.shape[0]+1,
                    250 if mat2ds.shape[0] > 1000 else 50,
                    dtype=int)
            ax[ci].set_yticks(cell_counts-0.5)
            ax[ci].set_yticklabels(cell_counts, rotation=0, size=ytl_size)
        ax[ci].set_xlabel('\nTime from stimulus onset (s)', size=xl_size)
        rect_height = mat2ds.shape[0]/100
        rec_list = [
            Rectangle((s, mat2ds.shape[0] - rect_height),31,rect_height)
            for s in np.arange(15.5, mat2ds.shape[1], 47)
        ]
        pc = PatchCollection(rec_list, facecolor='xkcd:medium gray', alpha=0.7, clip_on=False)
        ax[ci].add_collection(pc)  
        if ci > 1:
            ax[ci].set_yticks([])
    if save_please:
        plt.savefig(os.path.join(save_folder, f'{match_to}_{index_on}_{group_or_comp_sort}_high_heatmap.png'),
                bbox_inches='tight')


    # plot heatmap
    ax = []
    fig = plt.figure(figsize=(30, 15))
    gs = fig.add_gridspec(100, 110)
    ax.append(fig.add_subplot(gs[:, 0:3]))
    ax.append(fig.add_subplot(gs[:, 10:38]))
    ax.append(fig.add_subplot(gs[:, 40:68]))
    ax.append(fig.add_subplot(gs[:, 70:98]))
    ax.append(fig.add_subplot(gs[:30, 100:103]))

    # plot "categorical" heatmap using defined color mappings
    color_vecs = np.concatenate([number_mouse_mat[cell_sorter, None], number_comp_mat[cell_sorter, None]], axis=1)
    fix_cmap = sns.color_palette('Set3', rr) #lookups.cmap_fixed_sort_rank9_onset
    just_comps = np.array(cell_cats[cell_sorter, None], dtype=float)
    just_comps[just_comps == -1] = np.nan
    sns.heatmap(just_comps, cmap=fix_cmap, ax=ax[0], cbar=False)
    ax[0].set_xticks([0.5])
    ax[0].set_xticklabels(['Component'], rotation=45, ha='right', size=yl_size)
    ax[0].set_yticks([])
    ax[0].set_yticklabels([])

    for ci, ii in enumerate([6, 8, 7]):
        ci = ci + 1
        if ii == 7:
            g = sns.heatmap(mat2ds[cell_sorter, :, ii],
                            ax=ax[ci],
                            center=0,
                            vmax=vmax,
                            vmin=-0.5,
                            cmap='vlag',
                            cbar_ax=ax[4],
                            cbar_kws={'label': clabel})
            cbar = g.collections[0].colorbar
            cbar.set_label(clabel, size=yl_size)
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=ytl_size)
        else:
            g = sns.heatmap(mat2ds[cell_sorter, :, ii],
                            ax=ax[ci],
                            center=0,
                            vmax=vmax,
                            vmin=-0.5,
                            cmap='vlag',
                            cbar=False)
        g.set_facecolor('#c5c5c5')
        ax[ci].set_title(f'{cues[ci-1]}\n', size=t_size, color=lookups.color_dict[cues[ci-1]])
        stim_starts = [15.5 + 47 * s for s in np.arange(len(stages))]
        stim_labels = [f'\n\n{s}' if c % 2 == 0 else f'\n{s}' for c, s in enumerate(stages)]
        ax[ci].set_xticks(stim_starts)
        ax[ci].set_xticklabels(stim_labels, rotation=0, size=xtl_size)
        if ii == 6:
            ax[ci].set_ylabel('Cell number', size=yl_size)
            cell_counts = np.arange(
                    250 if mat2ds.shape[0] > 1000 else 50,
                    mat2ds.shape[0]+1,
                    250 if mat2ds.shape[0] > 1000 else 50,
                    dtype=int)
            ax[ci].set_yticks(cell_counts-0.5)
            ax[ci].set_yticklabels(cell_counts, rotation=0, size=ytl_size)
        ax[ci].set_xlabel('\nTime from stimulus onset (s)', size=xl_size)
        rect_height = mat2ds.shape[0]/100
        rec_list = [
            Rectangle((s, mat2ds.shape[0]- rect_height),31,rect_height)
            for s in np.arange(15.5, mat2ds.shape[1], 47)
        ]
        pc = PatchCollection(rec_list, facecolor='xkcd:medium gray', alpha=0.7, clip_on=False)
        ax[ci].add_collection(pc)  
        if ci > 1:
            ax[ci].set_yticks([])
    if save_please:
        plt.savefig(os.path.join(save_folder, f'{match_to}_{index_on}_{group_or_comp_sort}_low_heatmap.png'),
                bbox_inches='tight')

    # plot heatmap
    ax = []
    fig = plt.figure(figsize=(30, 15))
    gs = fig.add_gridspec(100, 110)
    ax.append(fig.add_subplot(gs[:, 0:3]))
    ax.append(fig.add_subplot(gs[:, 10:38]))
    ax.append(fig.add_subplot(gs[:, 40:68]))
    ax.append(fig.add_subplot(gs[:, 70:98]))
    ax.append(fig.add_subplot(gs[:30, 100:103]))

    # plot "categorical" heatmap using defined color mappings
    color_vecs = np.concatenate([number_mouse_mat[cell_sorter, None], number_comp_mat[cell_sorter, None]], axis=1)
    fix_cmap = sns.color_palette('Set3', rr) #lookups.cmap_fixed_sort_rank9_onset
    just_comps = np.array(cell_cats[cell_sorter, None], dtype=float)
    just_comps[just_comps == -1] = np.nan
    sns.heatmap(just_comps, cmap=fix_cmap, ax=ax[0], cbar=False)
    ax[0].set_xticks([0.5])
    ax[0].set_xticklabels(['Component'], rotation=45, ha='right', size=yl_size)
    ax[0].set_yticks([])
    ax[0].set_yticklabels([])

    for ci, ii in enumerate([6, 8, 7]):
        ci = ci + 1
        if ii == 7:
            g = sns.heatmap(mat2ds[cell_sorter, :, ii] - mat2ds[cell_sorter, :, ii-6],
                            ax=ax[ci],
                            center=0,
                            vmax=vmax,
                            vmin=-0.5,
                            cmap='vlag',
                            cbar_ax=ax[4],
                            cbar_kws={'label': clabel})
            cbar = g.collections[0].colorbar
            cbar.set_label(clabel, size=yl_size)
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=ytl_size)
        else:
            g = sns.heatmap(mat2ds[cell_sorter, :, ii] - mat2ds[cell_sorter, :, ii-6],
                            ax=ax[ci],
                            center=0,
                            vmax=vmax,
                            vmin=-0.5,
                            cmap='vlag',
                            cbar=False)
        g.set_facecolor('#c5c5c5')
        ax[ci].set_title(f'{cues[ci-1]}\n', size=t_size, color=lookups.color_dict[cues[ci-1]])
        stim_starts = [15.5 + 47 * s for s in np.arange(len(stages))]
        stim_labels = [f'\n\n{s}' if c % 2 == 0 else f'\n{s}' for c, s in enumerate(stages)]
        ax[ci].set_xticks(stim_starts)
        ax[ci].set_xticklabels(stim_labels, rotation=0, size=xtl_size)
        if ii == 6:
            ax[ci].set_ylabel('Cell number', size=yl_size)
            cell_counts = np.arange(
                    250 if mat2ds.shape[0] > 1000 else 50,
                    mat2ds.shape[0]+1,
                    250 if mat2ds.shape[0] > 1000 else 50,
                    dtype=int)
            ax[ci].set_yticks(cell_counts-0.5)
            ax[ci].set_yticklabels(cell_counts, rotation=0, size=ytl_size)
        ax[ci].set_xlabel('\nTime from stimulus onset (s)', size=xl_size)
        rect_height = mat2ds.shape[0]/100
        rec_list = [
            Rectangle((s, mat2ds.shape[0] - rect_height),31,rect_height)
            for s in np.arange(15.5, mat2ds.shape[1], 47)
        ]
        pc = PatchCollection(rec_list, facecolor='xkcd:medium gray', alpha=0.7, clip_on=False)
        ax[ci].add_collection(pc)  
        if ci > 1:
            ax[ci].set_yticks([])
    if save_please:
        plt.savefig(os.path.join(save_folder, f'{match_to}_{index_on}_{group_or_comp_sort}_diff_heatmap.png'),
                bbox_inches='tight')
        plt.close('all')


def plot_modulation_vertical_bars(match_to='onsets',
                                  index_on='pre_speed',
                                  with_diff=True,
                                  save_please=True):
    """Plot colorbar for component and barh plot for average normalized response per cell.

    Parameters
    ----------
    match_to : str, optional
        Onsets or offsets, by default 'onsets'
    index_on : str, optional
        Type of modulation to create an index on (pre_speed, th, gonogo), by default 'pre_speed'
    with_diff : bool, optional
        Add a blue line that is the diff of your high/low modualtion values, by default True
    save_please : bool, optional
        Save plot, by default True

    Raises
    ------
    NotImplementedError
        If you ask for an index_on type that has not been set up yet. 
    """

    # plot label params
    yl_size = 14
    xl_size = 14

    # set save folder
    save_folder = paths.analysis_dir(f'figures/figure_3/modulation_redblack/{match_to}_{index_on}')

    # build or load necessary matrices and index dfs
    df, mat = selectivity.ab_index_df(match_to=match_to, index_on=index_on, return_matrix=True)
    ens = load.core_tca_data(match_to=match_to)
    mouse_vec = ens['mouse_vec']
    cell_cats = ens['cell_cats']
    cues =  ['Initially rewarded', 'Becomes rewarded', 'Unrewarded']

    # check that tca_data and index data match and if they do, create sorter
    assert np.array_equal(ens['cell_cats'], df.cell_cats.values)
    cell_sorter2 = sorters.pick_comp_order_plus_bhv_mod(
        ens['cell_cats'],
        ens['cell_sorter'],
        mod_vec=df['a-b'].values
    )
    cell_sorter1 = sorters.pick_comp_order_plus_cuelevel_bhv_mod(
        ens['cell_cats'],
        ens['cell_sorter'],
        mod_vec=df['a-b'].values
    )

    # parse labels
    if index_on == 'pre_speed':
        label1 = 'Running'
        label2 = 'Stationary'
        label3 = 'Run. - Stat.'
    elif index_on == 'th':
        label1 = 'Previous cue different'
        label2 = 'Previous cue same'
        label3 = 'Different - Same'
    elif index_on == 'gonogo':
        label1 = 'Go'
        label2 = 'NoGo'
        label3 = 'Go - Nogo'
    else:
        raise NotImplementedError

    ax = []
    fig = plt.figure(figsize=(7, 15))
    gs = fig.add_gridspec(100, 54)
    ax.append(fig.add_subplot(gs[:, 0:3]))
    with sns.axes_style('whitegrid'):
        ax.append(fig.add_subplot(gs[:, 11:21]))
    ax.append(fig.add_subplot(gs[:, 33:36]))
    with sns.axes_style('whitegrid'):
        ax.append(fig.add_subplot(gs[:, 44:54]))


    for cax, cell_sorter in zip([ax[0], ax[2]], [cell_sorter1, cell_sorter2]):

        fix_cmap = sns.color_palette('Set3', 9) #lookups.cmap_fixed_sort_rank9_onset
        just_comps = np.array(cell_cats[cell_sorter, None], dtype=float)
        just_comps[just_comps == -1] = np.nan
        sns.heatmap(just_comps, cmap=fix_cmap, ax=cax, cbar=False)
        cax.set_xticks([0.5])
        cax.set_xticklabels(['Component'], rotation=45, ha='right', size=yl_size)
        cax.set_yticks([])
        cax.set_yticklabels([])

    for pax, cell_sorter in zip([ax[1], ax[3]], [cell_sorter1, cell_sorter2]):

        x1 = df.a.values[cell_sorter]
        x2 = df.b.values[cell_sorter]
        diff = df['a-b'].values[cell_sorter]
    #     x1[np.isnan(x1) | np.isnan(x2)] = np.nan
    #     x2[np.isnan(x1) | np.isnan(x2)] = np.nan
        y = np.arange(len(x1))[::-1]
        pax.barh(y, x1, color='red', label=label1)
        y = np.arange(len(x2))[::-1]
        pax.barh(y, x2*-1, color='black', alpha=0.7, label=label2)
        if with_diff:
            pax.plot(diff, y,  color='blue', label=label3)
        pax.set_ylim(0, len(x2))
        pax.set_ylabel('Cell number', size=yl_size)
        pax.set_xlabel('Response magnitude\n(norm. \u0394F/F)', size=xl_size)
        pax.set_xticks([-1, 0, 1])
        pax.set_xticklabels([f'1\n{label2}', '0', f'1\n{label1}'])

        # plot cells that are nan 
    #     x1 = df.a.values[cell_sorter]
    #     x2 = df.b.values[cell_sorter]
    #     x1[~np.isnan(df.b.values[cell_sorter])] = np.nan
    #     x2[~np.isnan(df.a.values[cell_sorter])] = np.nan
    #     y = np.arange(len(x1))[::-1]
    #     pax.plot(x1, y, 'o', markerfacecolor=[0, 0, 0, 0], markeredgecolor='red')
    #     y = np.arange(len(x2))[::-1]
    #     pax.plot(x2, y, 'o', markerfacecolor=[0, 0, 0, 0], markeredgecolor='black')

    pax.legend(bbox_to_anchor=(1.05, 1.0), loc=2)
    if with_diff:
        handles, labels = pax.get_legend_handles_labels()
        pax.legend(
            [handles[s] for s in [1, 2, 0]],
            [labels[s] for s in [1, 2, 0]],
            bbox_to_anchor=(1.05, 1.0),
            loc=2
        )

    if save_please:
        difftag = '_withdiff' if with_diff else ''
        plt.savefig(os.path.join(save_folder, f'{match_to}_{index_on}_avg_mod_bars{difftag}.png'),
                bbox_inches='tight')
        plt.savefig(os.path.join(save_folder, f'{match_to}_{index_on}_avg_mod_bars{difftag}.pdf'),
                bbox_inches='tight')
        plt.close('all')


def plot_modulation_vertical_traces(match_to='onsets',
                                    index_on='pre_speed',
                                    save_please=True):
    """Plot traces for component and barh plot for average normalized response per cell.
    NOTE: this is the less good version of plot_modulation_vertical_bars()

    Parameters
    ----------
    match_to : str, optional
        Onsets or offsets, by default 'onsets'
    index_on : str, optional
        Type of modulation to create an index on (pre_speed, th, gonogo), by default 'pre_speed'
    with_diff : bool, optional
        Add a blue line that is the diff of your high/low modualtion values, by default True
    save_please : bool, optional
        Save plot, by default True

    Raises
    ------
    NotImplementedError
        If you ask for an index_on type that has not been set up yet. 
    """
    # plot label params
    yl_size = 14
    xl_size = 14

    # set save folder
    save_folder = paths.analysis_dir(f'figures/figure_3/modulation_redblack/{match_to}_{index_on}')

    # build or load necessary matrices and index dfs
    df, mat = selectivity.ab_index_df(match_to=match_to, index_on=index_on, return_matrix=True)
    ens = load.core_tca_data(match_to=match_to)
    mouse_vec = ens['mouse_vec']
    cell_cats = ens['cell_cats']
    cues =  ['Initially rewarded', 'Becomes rewarded', 'Unrewarded']

    # check that tca_data and index data match and if they do, create sorter
    assert np.array_equal(ens['cell_cats'], df.cell_cats.values)
    cell_sorter2 = sorters.pick_comp_order_plus_bhv_mod(
        ens['cell_cats'],
        ens['cell_sorter'],
        mod_vec=df['a-b'].values
    )
    cell_sorter1 = sorters.pick_comp_order_plus_cuelevel_bhv_mod(
        ens['cell_cats'],
        ens['cell_sorter'],
        mod_vec=df['a-b'].values
    )

    # parse labels
    if index_on == 'pre_speed':
        label1 = 'Running'
        label2 = 'Stationary'
    elif index_on == 'th':
        label1 = 'Previous cue same'
        label2 = 'Previous cue different'
    elif index_on == 'gonogo':
        label1 = 'Go'
        label2 = 'NoGo'
    else:
        raise NotImplementedError

    ax = []
    fig = plt.figure(figsize=(7, 15))
    gs = fig.add_gridspec(100, 54)
    ax.append(fig.add_subplot(gs[:, 0:3]))
    with sns.axes_style('whitegrid'):
        ax.append(fig.add_subplot(gs[:, 11:21]))
    ax.append(fig.add_subplot(gs[:, 33:36]))
    with sns.axes_style('whitegrid'):
        ax.append(fig.add_subplot(gs[:, 44:54]))


    for cax, cell_sorter in zip([ax[0], ax[2]], [cell_sorter1, cell_sorter2]):

        fix_cmap = sns.color_palette('Set3', 9) #lookups.cmap_fixed_sort_rank9_onset
        just_comps = np.array(cell_cats[cell_sorter, None], dtype=float)
        just_comps[just_comps == -1] = np.nan
        sns.heatmap(just_comps, cmap=fix_cmap, ax=cax, cbar=False)
        cax.set_xticks([0.5])
        cax.set_xticklabels(['Component'], rotation=45, ha='right', size=yl_size)
        cax.set_yticks([])
        cax.set_yticklabels([])

    for pax, cell_sorter in zip([ax[1], ax[3]], [cell_sorter1, cell_sorter2]):

        x1 = df.a.values[cell_sorter]
        x2 = df.b.values[cell_sorter]
        x1[np.isnan(x1) | np.isnan(x2)] = np.nan
        x2[np.isnan(x1) | np.isnan(x2)] = np.nan
        y = np.arange(len(x1))[::-1]
        pax.plot(x1, y, color='red', label=label1)
        y = np.arange(len(x2))[::-1]
        pax.plot(x2, y, color='black', alpha=0.7, label=label2)
        pax.set_ylim(0, len(x2))
        pax.set_ylabel('Cell number', size=yl_size)
        pax.set_xlabel('Response magnitude\n(norm. \u0394F/F)', size=xl_size)
        
        # plot cells that are nan 
        x1 = df.a.values[cell_sorter]
        x2 = df.b.values[cell_sorter]
        x1[~np.isnan(df.b.values[cell_sorter])] = np.nan
        x2[~np.isnan(df.a.values[cell_sorter])] = np.nan
        y = np.arange(len(x1))[::-1]
        pax.plot(x1, y, 'o', markerfacecolor=[0, 0, 0, 0], markeredgecolor='red')
        y = np.arange(len(x2))[::-1]
        pax.plot(x2, y, 'o', markerfacecolor=[0, 0, 0, 0], markeredgecolor='black')

    pax.legend(bbox_to_anchor=(1.05, 1.0))

    if save_please:
        plt.savefig(os.path.join(save_folder, f'{match_to}_{index_on}_avg_mod_traces.png'),
                bbox_inches='tight')
        plt.savefig(os.path.join(save_folder, f'{match_to}_{index_on}_avg_mod_traces.pdf'),
                bbox_inches='tight')
        plt.close('all')
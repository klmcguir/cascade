""" Functions for plotting modulation (mostly figire 3-5 related) """
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import warnings
import numpy as np
import pandas as pd

from ..trialanalysis import build_any_mat, build_cue_mat
from .. import load, utils, lookups, paths, sorters, selectivity
from .plot_utils import heatmap_xticks
import os
from copy import deepcopy


def make_all_plots():
    """Helper function to generate a shortlist of plots from this module
    """

    # difference between learning stages T1R - T5L
    plot_horbar_stagediff_T5lT1r(match_to='onsets', limit_to='onsets', norm_please=True)
    plot_horbar_stagediff_T5lT1r(match_to='offsets', limit_to='offsets', norm_please=True)

    # difference between learning stages T5R - T5L
    plot_horbar_homeodiff(match_to='onsets', limit_to='onsets', norm_please=True)
    plot_horbar_homeodiff(match_to='offsets', limit_to='offsets', norm_please=True)

    # difference between learning stages T4L - T5L
    plot_horbar_stagediff_T5lT4l(match_to='onsets', limit_to='onsets', norm_please=True)
    plot_horbar_stagediff_T5lT4l(match_to='offsets', limit_to='offsets', norm_please=True)

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


def plot_modulation_summary_bars(match_to='onsets',
                                 index_on='pre_speed',
                                 with_pointplot=True,
                                 mouse_or_cells='mouse',
                                 save_please=True):

    # plot label params
    yl_size = 14
    xl_size = 14
    xtl_size = 14

    # set save folder
    save_folder = paths.analysis_dir(f'figures/figure_3/modulation_summary/{match_to}_{index_on}')

    # build or load necessary matrices and index dfs
    df, mat = selectivity.ab_index_df(match_to=match_to, index_on=index_on, return_matrix=True)
    ens = load.core_tca_data(match_to=match_to)
    mouse_vec = ens['mouse_vec']
    cell_cats = ens['cell_cats']
    cues = ['Initially rewarded', 'Becomes rewarded', 'Unrewarded']

    # check that tca_data and index data match and if they do, create sorter
    assert np.array_equal(ens['cell_cats'], df.cell_cats.values)

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
    elif index_on == 'disengaged':
        label1 = 'Engaged'
        label2 = 'Disengaged'
        label3 = 'Engaged - Disengaged'
    else:
        raise NotImplementedError

    # specify model
    model_spec = 'rank9_onset' if match_to == 'onsets' else 'rank8_offset'
    if match_to == 'onsets':
        cmap = lookups.cmap_fixed_sort_rank9_onset
    elif match_to == 'offsets':
        cmap = lookups.cmap_fixed_sort_rank8_offset

    # manually calc mean and sem over mice or cells
    if mouse_or_cells == 'mouse':
        mouse_means = (df.groupby(['mouse', 'cell_cats']).mean().reindex(lookups.fixed_component_sort[model_spec],
                                                                         level=1))
        mmean = mouse_means.groupby('cell_cats').mean()
        msem = mouse_means.groupby('cell_cats').sem()
        total_means = (df.groupby(['mouse']).mean())
        total_means['x'] = 0
    elif mouse_or_cells == 'cells':
        mouse_means = (df.groupby(['mouse', 'cell_id',
                                   'cell_cats']).mean().reindex(lookups.fixed_component_sort[model_spec], level=2))
        mmean = mouse_means.groupby('cell_cats').mean()
        msem = mouse_means.groupby('cell_cats').sem()
    else:
        raise ValueError
    single_mean = mouse_means.mean()
    single_sem = mouse_means.sem()

    # plot average across all whole population and per category
    fig = plt.figure(figsize=(6, 4))
    gs = fig.add_gridspec(100, 100)
    with sns.axes_style('whitegrid'):
        ax_mean = fig.add_subplot(gs[:, 0:15])
        ax_comps = fig.add_subplot(gs[:, 25:90], sharey=ax_mean)

    with sns.axes_style('whitegrid'):
        sns.barplot(x=[0.0], y=[single_mean['a-b']], ci=False, color='blue', zorder=10, ax=ax_mean)
        if mouse_or_cells == 'mouse':
            sns.stripplot(data=total_means.reset_index(),
                          x='x',
                          y='a-b',
                          color='blue',
                          linewidth=1,
                          edgecolor='xkcd:light gray',
                          zorder=11,
                          s=5,
                          jitter=True,
                          ax=ax_mean)
        ax_mean.errorbar(x=[0],
                         y=[single_mean['a-b']],
                         yerr=[single_sem['a-b']],
                         linewidth=0,
                         elinewidth=2,
                         color='black',
                         zorder=12)
        ax_mean.set_xlim(-1, 1)

    with sns.axes_style('whitegrid'):
        sns.barplot(data=mmean.reset_index(),
                    x='cell_cats',
                    y='a-b',
                    ci=False,
                    palette=cmap,
                    order=lookups.fixed_component_sort[model_spec],
                    zorder=10,
                    ax=ax_comps)
        if mouse_or_cells == 'mouse':
            sns.stripplot(data=mouse_means.reset_index(),
                          x='cell_cats',
                          y='a-b',
                          palette=cmap,
                          order=lookups.fixed_component_sort[model_spec],
                          linewidth=1,
                          edgecolor='xkcd:medium gray',
                          zorder=11,
                          s=5,
                          jitter=False,
                          ax=ax_comps)
        if with_pointplot and mouse_or_cells == 'mouse':
            for m in mouse_means.reset_index().mouse.unique():
                mboo = mouse_means.reset_index().mouse.isin([m]).values
                sns.pointplot(data=mouse_means.reset_index().loc[mboo, :],
                              x='cell_cats',
                              y='a-b',
                              color='xkcd:medium gray',
                              order=lookups.fixed_component_sort[model_spec],
                              zorder=0,
                              linewidth=1,
                              edgecolor='xkcd:medium gray',
                              scale=0.3,
                              ax=ax_comps)
        ax_comps.errorbar(x=np.arange(len(msem)),
                          y=mmean['a-b'].values,
                          yerr=msem['a-b'].values,
                          linewidth=0,
                          elinewidth=2,
                          color='black',
                          zorder=12)

    ax_mean.set_ylabel(f'Response magnitude\n{label3}\n(norm. \u0394F/F)', size=yl_size)
    ax_comps.set_xticks(ticks=np.arange(len(msem)))
    ax_comps.set_xticklabels(labels=lookups.comp_class_names[model_spec], rotation=90, size=xtl_size)
    ax_mean.set_xticks([0])
    ax_mean.set_xticklabels(['All cells'], rotation=90, size=xtl_size)
    #     ax_comps.set_yticks([])
    #     ax_comps.set_yticklabels([])
    ax_mean.set_xlabel('')
    ax_comps.set_xlabel('')
    ax_comps.set_ylabel('')
    if save_please:
        pointtag = '_withpoint' if with_pointplot else ''
        plt.savefig(os.path.join(save_folder, f'{match_to}_{index_on}_avg_sem_{mouse_or_cells}{pointtag}.png'),
                    bbox_inches='tight')
        plt.savefig(os.path.join(save_folder, f'{match_to}_{index_on}_avg_sem_{mouse_or_cells}{pointtag}.pdf'),
                    bbox_inches='tight')
        plt.close('all')


def plot_horbar_homeodiff(match_to='onsets', limit_to=None, norm_please=False, allstage=False, save_please=True, fix_xlim=True):
    """Plot horizontal barplot of difference between T5 reversal and T5 learning.

    Parameters
    ----------
    match_to : str, optional
        Onsets or offsets, by default 'onsets'
    limit_to : str, optional
        Trigger on offset or onset, by default None
    norm_please : bool, optional
        Normalize to peak over stages, by default False
    allstage : bool, optional
        Include naive, by default False
    save_please : bool, optional
        Save flag useful when testing new things, by default False

    Raises
    ------
    NotImplementedError
        You might be trying to use a dataset that doesn't exists, but really, who knows.
    """

    # set save location
    if norm_please:
        norm_tag = 'norm'
    else:
        norm_tag = 'zscore'
    if allstage:
        allstage_tag = '_allstage'
    else:
        allstage_tag = ''
    save_folder = paths.analysis_dir(f'figures/figure_2B/homeo_quant_bar/{match_to}/{match_to}_lim_{limit_to}_{norm_tag}_{allstage_tag}')

    # load tca reversal n=7 data
    tca_dict = load.core_tca_data(limit_to=None, match_to=match_to)
    mouse_vec = tca_dict['mouse_vec']
    cell_vec = tca_dict['cell_vec']
    cell_cats = tca_dict['cell_cats']

    # set groupings
    if match_to == 'onsets':
        cat_groups_pref_tuning = lookups.tuning_groups['rank9_onset']
        cmap = lookups.cmap_fixed_sort_rank9_onset
        comp_names = lookups.comp_class_names['rank9_onset']
    elif match_to == 'offsets':
        cat_groups_pref_tuning = lookups.tuning_groups['rank8_offset']
        cmap = lookups.cmap_fixed_sort_rank8_offset
        comp_names = lookups.comp_class_names['rank8_offset']
    else:
        raise ValueError

    # build your matrix
    if limit_to is None:
        limit_to = match_to
    tensor_stack = build_cue_mat(
        mouse_vec,
        cell_vec,
        limit_tag=limit_to,
        allstage=allstage,
        norm_please=norm_please,
        no_disengaged=False,
    )

    # also turn the tensor mat into a pref cue mat

    # loop over mice and make plot for each mouse
    df1_list = []
    df2_list = []
    for cuei, cue_name in enumerate(lookups.cue_names):

        for mouse in np.unique(mouse_vec):

            mboo = mouse_vec == mouse

            # take mean per comp
            mean_stack, sem_stack = utils.average_across_cats(
                tensor_stack[mboo, :, :], cell_cats[mboo], shared_cat=None, force_match_to=match_to
            )
            mean_stack_2s = utils.mean_2s_from_unwrapped_tensor(mean_stack)
            t1t1_diff = mean_stack_2s[:, 9, :] - mean_stack_2s[:, 4, :] # 9 x 3
            #     for

            # take mean over all cells
            total_mean = np.nanmean(tensor_stack[mboo, :, :], axis=0)[None, :, :]
            total_mean_2s = utils.mean_2s_from_unwrapped_tensor(total_mean)
            t1t1_diff_total = (total_mean_2s[:, 9, :] - total_mean_2s[:, 4, :]).flatten()

            df1 = pd.DataFrame(data={
                'mouse': [mouse] * len(t1t1_diff[:, cuei]),
                'component_name': comp_names,
                'component': np.arange(len(t1t1_diff[:, cuei])) + 1,
                'delta_T5': t1t1_diff[:, cuei],
                'cue': [cue_name] * len(t1t1_diff[:, cuei]),
            })
            df1_list.append(df1)

            df2 = pd.DataFrame(data={
                'mouse': [mouse],
                'component_name': ['All cells'],
                'component': [0],
                'delta_T5': [t1t1_diff_total[cuei]],
                'cue': [cue_name],
            })
            df2_list.append(df2)
    df1_all = pd.concat(df1_list, axis=0)
    df2_all = pd.concat(df2_list, axis=0)
    combo_df = pd.concat([df1_all, df2_all])

    # plot
    with sns.axes_style('whitegrid'):
        fig, ax = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for ci, cuen, in enumerate(lookups.cue_names_forced):
        plt_df = combo_df.loc[combo_df.cue.isin([cuen])]
        sns.barplot(
            data=plt_df, y='component', x='delta_T5', orient='h', ax=ax[ci],
            palette=['xkcd:blue'] + cmap, ci=False,
            alpha=0.7
        )
        sns.stripplot(
            data=plt_df, y='component', x='delta_T5', orient='h', ax=ax[ci],
            palette=['xkcd:powder blue'] + [np.array(s)*0.9 for s in cmap],
            edgecolor='xkcd:medium gray', linewidth=1,
        )

        # add sem over mice error bars
        meany = plt_df.groupby(['component', 'component_name']).mean()
        semy = plt_df.groupby('component').sem().delta_T5.values
        ax[ci].errorbar(
            x=meany.delta_T5.values, y=meany.reset_index().component.values, xerr=semy,
            linewidth=0, elinewidth=2, color='black',
            zorder=12
        )
        ax[ci].axhline(0.5, color='xkcd:light gray', linewidth=1)
        if norm_please:
            ax[ci].set_xlabel('T5 reversal - T5 learning\nNormalized \u0394F/F\n',size=14)
        else:
            ax[ci].set_xlabel('T5 reversal - T5 learning\n\u0394F/F (z-score)',size=14)
        ax[ci].set_ylabel('Component cluster', size=14)
        ax[ci].set_title(f'{cuen}\n', size=16, color=lookups.color_dict[cuen])
    ax[0].set_yticks(meany.reset_index().component.values)
    ax[0].set_yticklabels(meany.reset_index().component_name.to_list(), size=12)
    if fix_xlim:
        plt.xlim([-0.5, 0.5])

    if save_please:
        plt.savefig(os.path.join(save_folder, f'{match_to}_lim_{limit_to}_{norm_tag}_horbars{allstage_tag}.png'),
                bbox_inches='tight')
        plt.savefig(os.path.join(save_folder, f'{match_to}_lim_{limit_to}_{norm_tag}_horbars{allstage_tag}.pdf'),
                bbox_inches='tight')


def plot_horbar_stagediff_T5lT1r(match_to='onsets', limit_to=None, norm_please=False, allstage=False, save_please=True, fix_xlim=True):
    """Plot horizontal barplot of difference between T5 reversal and T5 learning.

    Parameters
    ----------
    match_to : str, optional
        Onsets or offsets, by default 'onsets'
    limit_to : str, optional
        Trigger on offset or onset, by default None
    norm_please : bool, optional
        Normalize to peak over stages, by default False
    allstage : bool, optional
        Include naive, by default False
    save_please : bool, optional
        Save flag useful when testing new things, by default False

    Raises
    ------
    NotImplementedError
        You might be trying to use a dataset that doesn't exists, but really, who knows.
    """

    # set save location
    if norm_please:
        norm_tag = 'norm'
    else:
        norm_tag = 'zscore'
    if allstage:
        allstage_tag = '_allstage'
    else:
        allstage_tag = ''
    save_folder = paths.analysis_dir(f'figures/figure_2B/homeo_quant_bar/{match_to}/{match_to}_lim_{limit_to}_{norm_tag}_{allstage_tag}')

    # load tca reversal n=7 data
    tca_dict = load.core_tca_data(limit_to=None, match_to=match_to)
    mouse_vec = tca_dict['mouse_vec']
    cell_vec = tca_dict['cell_vec']
    cell_cats = tca_dict['cell_cats']

    # set groupings
    if match_to == 'onsets':
        cat_groups_pref_tuning = lookups.tuning_groups['rank9_onset']
        cmap = lookups.cmap_fixed_sort_rank9_onset
        comp_names = lookups.comp_class_names['rank9_onset']
    elif match_to == 'offsets':
        cat_groups_pref_tuning = lookups.tuning_groups['rank8_offset']
        cmap = lookups.cmap_fixed_sort_rank8_offset
        comp_names = lookups.comp_class_names['rank8_offset']
    else:
        raise ValueError

    # build your matrix
    if limit_to is None:
        limit_to = match_to
    tensor_stack = build_cue_mat(
        mouse_vec,
        cell_vec,
        limit_tag=limit_to,
        allstage=allstage,
        norm_please=norm_please,
        no_disengaged=False,
    )

    # also turn the tensor mat into a pref cue mat

    # loop over mice and make plot for each mouse
    df1_list = []
    df2_list = []
    for cuei, cue_name in enumerate(lookups.cue_names):

        for mouse in np.unique(mouse_vec):

            mboo = mouse_vec == mouse

            # take mean per comp
            mean_stack, sem_stack = utils.average_across_cats(
                tensor_stack[mboo, :, :], cell_cats[mboo], shared_cat=None, force_match_to=match_to
            )
            mean_stack_2s = utils.mean_2s_from_unwrapped_tensor(mean_stack)
            t1t1_diff = mean_stack_2s[:, 5, :] - mean_stack_2s[:, 4, :] # 9 x 3
            #     for

            # take mean over all cells
            total_mean = np.nanmean(tensor_stack[mboo, :, :], axis=0)[None, :, :]
            total_mean_2s = utils.mean_2s_from_unwrapped_tensor(total_mean)
            t1t1_diff_total = (total_mean_2s[:, 5, :] - total_mean_2s[:, 4, :]).flatten()

            df1 = pd.DataFrame(data={
                'mouse': [mouse] * len(t1t1_diff[:, cuei]),
                'component_name': comp_names,
                'component': np.arange(len(t1t1_diff[:, cuei])) + 1,
                'delta_T5': t1t1_diff[:, cuei],
                'cue': [cue_name] * len(t1t1_diff[:, cuei]),
            })
            df1_list.append(df1)

            df2 = pd.DataFrame(data={
                'mouse': [mouse],
                'component_name': ['All cells'],
                'component': [0],
                'delta_T5': [t1t1_diff_total[cuei]],
                'cue': [cue_name],
            })
            df2_list.append(df2)
    df1_all = pd.concat(df1_list, axis=0)
    df2_all = pd.concat(df2_list, axis=0)
    combo_df = pd.concat([df1_all, df2_all])

    # plot
    with sns.axes_style('whitegrid'):
        fig, ax = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for ci, cuen, in enumerate(lookups.cue_names_forced):
        plt_df = combo_df.loc[combo_df.cue.isin([cuen])]
        sns.barplot(
            data=plt_df, y='component', x='delta_T5', orient='h', ax=ax[ci],
            palette=['xkcd:blue'] + cmap, ci=False,
            alpha=0.7
        )
        sns.stripplot(
            data=plt_df, y='component', x='delta_T5', orient='h', ax=ax[ci],
            palette=['xkcd:powder blue'] + [np.array(s)*0.9 for s in cmap],
            edgecolor='xkcd:medium gray', linewidth=1,
        )

        # add sem over mice error bars
        meany = plt_df.groupby(['component', 'component_name']).mean()
        semy = plt_df.groupby('component').sem().delta_T5.values
        ax[ci].errorbar(
            x=meany.delta_T5.values, y=meany.reset_index().component.values, xerr=semy,
            linewidth=0, elinewidth=2, color='black',
            zorder=12
        )
        ax[ci].axhline(0.5, color='xkcd:light gray', linewidth=1)
        if norm_please:
            ax[ci].set_xlabel('T1 reversal - T5 learning\nNormalized \u0394F/F\n',size=14)
        else:
            ax[ci].set_xlabel('T1 reversal - T5 learning\n\u0394F/F (z-score)',size=14)
        ax[ci].set_ylabel('Component cluster', size=14)
        ax[ci].set_title(f'{cuen}\n', size=16, color=lookups.color_dict[cuen])
    ax[0].set_yticks(meany.reset_index().component.values)
    ax[0].set_yticklabels(meany.reset_index().component_name.to_list(), size=12)
    if fix_xlim:
        plt.xlim([-0.5, 0.5])

    if save_please:
        plt.savefig(os.path.join(save_folder, f'{match_to}_lim_{limit_to}_{norm_tag}_horbars{allstage_tag}_T5lT1r.png'),
                bbox_inches='tight')
        plt.savefig(os.path.join(save_folder, f'{match_to}_lim_{limit_to}_{norm_tag}_horbars{allstage_tag}_T5lT1r.pdf'),
                bbox_inches='tight')


def plot_horbar_stagediff_T5lT4l(match_to='onsets', limit_to=None, norm_please=False, allstage=False, save_please=True, fix_xlim=True):
    """Plot horizontal barplot of difference between T4 learning and T5 learning.

    Parameters
    ----------
    match_to : str, optional
        Onsets or offsets, by default 'onsets'
    limit_to : str, optional
        Trigger on offset or onset, by default None
    norm_please : bool, optional
        Normalize to peak over stages, by default False
    allstage : bool, optional
        Include naive, by default False
    save_please : bool, optional
        Save flag useful when testing new things, by default False

    Raises
    ------
    NotImplementedError
        You might be trying to use a dataset that doesn't exists, but really, who knows.
    """

    # set save location
    if norm_please:
        norm_tag = 'norm'
    else:
        norm_tag = 'zscore'
    if allstage:
        allstage_tag = '_allstage'
    else:
        allstage_tag = ''
    save_folder = paths.analysis_dir(f'figures/figure_2B/homeo_quant_bar/{match_to}/{match_to}_lim_{limit_to}_{norm_tag}_{allstage_tag}')

    # load tca reversal n=7 data
    tca_dict = load.core_tca_data(limit_to=None, match_to=match_to)
    mouse_vec = tca_dict['mouse_vec']
    cell_vec = tca_dict['cell_vec']
    cell_cats = tca_dict['cell_cats']

    # set groupings
    if match_to == 'onsets':
        cat_groups_pref_tuning = lookups.tuning_groups['rank9_onset']
        cmap = lookups.cmap_fixed_sort_rank9_onset
        comp_names = lookups.comp_class_names['rank9_onset']
    elif match_to == 'offsets':
        cat_groups_pref_tuning = lookups.tuning_groups['rank8_offset']
        cmap = lookups.cmap_fixed_sort_rank8_offset
        comp_names = lookups.comp_class_names['rank8_offset']
    else:
        raise ValueError

    # build your matrix
    if limit_to is None:
        limit_to = match_to
    tensor_stack = build_cue_mat(
        mouse_vec,
        cell_vec,
        limit_tag=limit_to,
        allstage=allstage,
        norm_please=norm_please,
        no_disengaged=False,
    )

    # also turn the tensor mat into a pref cue mat

    # loop over mice and make plot for each mouse
    df1_list = []
    df2_list = []
    for cuei, cue_name in enumerate(lookups.cue_names):

        for mouse in np.unique(mouse_vec):

            mboo = mouse_vec == mouse

            # take mean per comp
            mean_stack, sem_stack = utils.average_across_cats(
                tensor_stack[mboo, :, :], cell_cats[mboo], shared_cat=None, force_match_to=match_to
            )
            mean_stack_2s = utils.mean_2s_from_unwrapped_tensor(mean_stack)
            t1t1_diff = mean_stack_2s[:, 3, :] - mean_stack_2s[:, 4, :] # 9 x 3
            #     for

            # take mean over all cells
            total_mean = np.nanmean(tensor_stack[mboo, :, :], axis=0)[None, :, :]
            total_mean_2s = utils.mean_2s_from_unwrapped_tensor(total_mean)
            t1t1_diff_total = (total_mean_2s[:, 3, :] - total_mean_2s[:, 4, :]).flatten()

            df1 = pd.DataFrame(data={
                'mouse': [mouse] * len(t1t1_diff[:, cuei]),
                'component_name': comp_names,
                'component': np.arange(len(t1t1_diff[:, cuei])) + 1,
                'delta_T5': t1t1_diff[:, cuei],
                'cue': [cue_name] * len(t1t1_diff[:, cuei]),
            })
            df1_list.append(df1)

            df2 = pd.DataFrame(data={
                'mouse': [mouse],
                'component_name': ['All cells'],
                'component': [0],
                'delta_T5': [t1t1_diff_total[cuei]],
                'cue': [cue_name],
            })
            df2_list.append(df2)
    df1_all = pd.concat(df1_list, axis=0)
    df2_all = pd.concat(df2_list, axis=0)
    combo_df = pd.concat([df1_all, df2_all])

    # plot
    with sns.axes_style('whitegrid'):
        fig, ax = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for ci, cuen, in enumerate(lookups.cue_names_forced):
        plt_df = combo_df.loc[combo_df.cue.isin([cuen])]
        sns.barplot(
            data=plt_df, y='component', x='delta_T5', orient='h', ax=ax[ci],
            palette=['xkcd:blue'] + cmap, ci=False,
            alpha=0.7
        )
        sns.stripplot(
            data=plt_df, y='component', x='delta_T5', orient='h', ax=ax[ci],
            palette=['xkcd:powder blue'] + [np.array(s)*0.9 for s in cmap],
            edgecolor='xkcd:medium gray', linewidth=1,
        )

        # add sem over mice error bars
        meany = plt_df.groupby(['component', 'component_name']).mean()
        semy = plt_df.groupby('component').sem().delta_T5.values
        ax[ci].errorbar(
            x=meany.delta_T5.values, y=meany.reset_index().component.values, xerr=semy,
            linewidth=0, elinewidth=2, color='black',
            zorder=12
        )
        ax[ci].axhline(0.5, color='xkcd:light gray', linewidth=1)
        if norm_please:
            ax[ci].set_xlabel('T4 learning - T5 learning\nNormalized \u0394F/F\n',size=14)
        else:
            ax[ci].set_xlabel('T4 learning - T5 learning\n\u0394F/F (z-score)',size=14)
        ax[ci].set_ylabel('Component cluster', size=14)
        ax[ci].set_title(f'{cuen}\n', size=16, color=lookups.color_dict[cuen])
    ax[0].set_yticks(meany.reset_index().component.values)
    ax[0].set_yticklabels(meany.reset_index().component_name.to_list(), size=12)
    if fix_xlim:
        plt.xlim([-0.5, 0.5])

    if save_please:
        plt.savefig(os.path.join(save_folder, f'{match_to}_lim_{limit_to}_{norm_tag}_horbars{allstage_tag}_T5lT4l.png'),
                bbox_inches='tight')
        plt.savefig(os.path.join(save_folder, f'{match_to}_lim_{limit_to}_{norm_tag}_horbars{allstage_tag}_T5lT4l.pdf'),
                bbox_inches='tight')


def plot_overlaid_stagetraces(match_to='onsets', limit_to=None, norm_please=False, allstage=False, save_please=True):
    """Plot traces averaged across groups of cells, overlaid. Homeostatis intro plot.

    Parameters
    ----------
    match_to : str, optional
        Onsets or offsets, by default 'onsets'
    limit_to : str, optional
        Trigger on offset or onset, by default None
    norm_please : bool, optional
        Normalize to peak over stages, by default False
    allstage : bool, optional
        Include naive, by default False
    save_please : bool, optional
        Save flag useful when testing new things, by default False

    Raises
    ------
    NotImplementedError
        You might be trying to use a dataset that doesn't exists, but really, who knows.
    """

    # set save location
    save_folder = paths.analysis_dir(f'figures/figure_3/homeo_traces/{match_to}')

    # load tca reversal n=7 data
    tca_dict = load.core_tca_data(limit_to=None, match_to=match_to)
    mouse_vec = tca_dict['mouse_vec']
    cell_vec = tca_dict['cell_vec']
    cell_cats = tca_dict['cell_cats']

    # build your matrix
    if limit_to is None:
        limit_to = match_to
    tensor_stack = build_cue_mat(
        mouse_vec,
        cell_vec,
        limit_tag=limit_to,
        allstage=allstage,
        norm_please=norm_please,
        no_disengaged=False,
    )

    mean_stack, sem_stack = utils.average_across_cats(
        tensor_stack, cell_cats, shared_cat=None, force_match_to=match_to)

    # reorder cues for preferred plot order
    plt_mat = deepcopy(mean_stack)[:, :, [0, 2, 1]]
    plt_mat[:, ::47, :] = np.nan
    plt_sem = deepcopy(sem_stack)[:, :, [0, 2, 1]]
    plt_sem[:, ::47, :] = np.nan

    # plot
    with sns.axes_style('whitegrid'):
        fig, ax = plt.subplots(2, 3, sharey='row', sharex='col', figsize=(15,4))

    # top row
    for i in range(plt_mat.shape[2]):
        for k in range(plt_mat.shape[0]):

            # try to move relevant traces for each subplot to top of order
            if match_to == 'onsets':
                cmap = lookups.cmap_fixed_sort_rank9_onset
                if i == 0 and k < 3:
                    zorder = 8
                elif i == 1 and (k < 6 & k > 3):
                    zorder = 8
                elif i == 2 and (k < 9 & k > 6):
                    zorder = 8
                elif k == 8:
                    zorder = 8
                else:
                    zorder = 1
            elif match_to == 'offsets':
                cmap = lookups.cmap_fixed_sort_rank8_offset
                if k == 0: # broad sharp
                    order = 6
                if k == 1 or k == 2:
                    order = 7
                if k == 3 or k == 4:
                    zorder = 5
                else:
                    zorder = 1
            else:
                raise NotImplementedError

            ax[0, i].plot(
                plt_mat[k,:,i],
                color=cmap[k],
                zorder=zorder,
                linewidth=1,
            )
            ax[0, i].fill_between(
                np.arange(len(plt_mat[k,:,i])),
                plt_mat[k,:,i] + plt_sem[k,:,i],
                plt_mat[k,:,i] - plt_sem[k,:,i],
                color=np.array(cmap[k])*0.8,
                zorder=zorder,
                alpha=0.4
            )

    # add stimulus bars now that the 1st axis is finished
    y2 = ax[0, i].get_ylim()[0]
    rect_height = (ax[0, i].get_ylim()[1] - ax[0, i].get_ylim()[0])/14
    for i in range(3):
        xt, xtl = heatmap_xticks(
            additional_pt=None, staging='parsed_11stage_label_short',
            drop_naive= False if allstage else True
        )
        ax[0, i].set_xticks(xt)
        ax[0, i].set_xticklabels(xtl)

        #         rec_list = [Rectangle((s, y2-rect_height), 31, rect_height) for s in np.arange(15.5, 470, 47)]
        #         pc = PatchCollection(rec_list, facecolor='xkcd:medium gray', alpha=0.7, clip_on=False)
        #         ax[0, i].add_collection(pc)

        if limit_to == 'onsets':
            ax[1, i].set_xlabel('\nTime from stimulus onset (s)', size=16)
            rec_list = [Rectangle((s, y2-rect_height), 31, rect_height) for s in np.arange(15.5, 470, 47)]
            pc = PatchCollection(rec_list, facecolor='xkcd:medium gray', alpha=0.7, clip_on=False)
            ax[0, i].add_collection(pc)
        elif limit_to == 'offsets':
            ax[1, i].set_xlabel('\nTime from stimulus offset (s)', size=16)
            rec_list = [Rectangle((s, y2-rect_height), 15.5, rect_height) for s in np.arange(0, 470, 47)]
            pc = PatchCollection(rec_list, facecolor='xkcd:medium gray', alpha=0.7, clip_on=False)
            ax[0, i].add_collection(pc)
        else:
            raise ValueError

    # for mean of all cells
    total_plt_mat = tensor_stack[:, :, [0, 2, 1]]
    total_plt_mat[:, ::47, :] = np.nan
    total_mean = np.nanmean(total_plt_mat, axis=0)
    total_sem = np.nanstd(total_plt_mat, axis=0) / np.sqrt(np.sum(~np.isnan(total_plt_mat), axis=0))

    for i in range(total_mean.shape[1]):
        ax[1, i].plot(
            total_mean[:,i],
            color='black',
            zorder=zorder,
            linewidth=1,
        )
        ax[1, i].fill_between(
            np.arange(len(total_mean[:,i])),
            total_mean[:,i] + total_sem[:,i],
            total_mean[:,i] - total_sem[:,i],
            color='black',
            zorder=zorder,
            alpha=0.4
        )

    # add stimulus bars now that the second axis is finished
    y2 = ax[1, i].get_ylim()[0]
    rect_height = (ax[1, i].get_ylim()[1] - ax[1, i].get_ylim()[0])/14
    for i in range(3):
        xt, xtl = heatmap_xticks(
            additional_pt=None, staging='parsed_11stage_label_short',
            drop_naive= False if allstage else True
        )

        # xticklabels
        ax[1, i].set_xticks(xt)
        ax[1, i].set_xticklabels(xtl)

        if limit_to == 'onsets':
            rec_list = [Rectangle((s, y2-rect_height), 31, rect_height) for s in np.arange(15.5, 470, 47)]
            pc = PatchCollection(rec_list, facecolor='xkcd:medium gray', alpha=0.7, clip_on=False)
            ax[1, i].add_collection(pc)
        elif limit_to == 'offsets':
            rec_list = [Rectangle((s, y2-rect_height), 15.5, rect_height) for s in np.arange(0, 470, 47)]
            pc = PatchCollection(rec_list, facecolor='xkcd:medium gray', alpha=0.7, clip_on=False)
            ax[1, i].add_collection(pc)
        else:
            raise ValueError

    if norm_please:
        ax[1, 0].set_ylabel('Normalized \u0394F/F\n', size=16, ha='left')
        norm_tag = 'norm'
    else:
        ax[1, 0].set_ylabel('\u0394F/F (z-score)\n', size=16, ha='left')
        norm_tag = 'zscore'

    if allstage:
        allstage_tag = '_allstage'
    else:
        allstage_tag = ''

    if save_please:
        plt.savefig(os.path.join(save_folder, f'{match_to}_lim_{limit_to}_{norm_tag}_trace_overlay{allstage_tag}.png'),
                bbox_inches='tight')
        plt.savefig(os.path.join(save_folder, f'{match_to}_lim_{limit_to}_{norm_tag}_trace_overlay{allstage_tag}.pdf'),
                bbox_inches='tight')
#         plt.close('all')


def plot_overlaid_stagetraces_bymouse(match_to='onsets', limit_to=None, norm_please=False, allstage=False, save_please=True):
    """Plot traces averaged across groups of cells, overlaid. Homeostatis intro plot.

    Parameters
    ----------
    match_to : str, optional
        Onsets or offsets, by default 'onsets'
    limit_to : str, optional
        Trigger on offset or onset, by default None
    norm_please : bool, optional
        Normalize to peak over stages, by default False
    allstage : bool, optional
        Include naive, by default False
    save_please : bool, optional
        Save flag useful when testing new things, by default False

    Raises
    ------
    NotImplementedError
        You might be trying to use a dataset that doesn't exists, but really, who knows.
    """

    # set save location
    if norm_please:
        norm_tag = 'norm'
    else:
        norm_tag = 'zscore'
    if allstage:
        allstage_tag = '_allstage'
    else:
        allstage_tag = ''
    save_folder = paths.analysis_dir(f'figures/figure_3/homeo_traces/{match_to}/{match_to}_lim_{limit_to}_{norm_tag}_{allstage_tag}')

    # load tca reversal n=7 data
    tca_dict = load.core_tca_data(limit_to=None, match_to=match_to)
    mouse_vec = tca_dict['mouse_vec']
    cell_vec = tca_dict['cell_vec']
    cell_cats = tca_dict['cell_cats']

    # build your matrix
    if limit_to is None:
        limit_to = match_to
    tensor_stack = build_cue_mat(
        mouse_vec,
        cell_vec,
        limit_tag=limit_to,
        allstage=allstage,
        norm_please=norm_please,
        no_disengaged=False,
    )

    # loop over mice and make plot for each mouse
    for mouse in np.unique(mouse_vec):
        mboo = mouse_vec == mouse
        mean_stack, sem_stack = utils.average_across_cats(
            tensor_stack[mboo, :, :], cell_cats[mboo], shared_cat=None, force_match_to=match_to)

        # reorder cues for preferred plot order
        plt_mat = deepcopy(mean_stack)[:, :, [0, 2, 1]]
        plt_mat[:, ::47, :] = np.nan
        plt_sem = deepcopy(sem_stack)[:, :, [0, 2, 1]]
        plt_sem[:, ::47, :] = np.nan

        # plot
        with sns.axes_style('whitegrid'):
            fig, ax = plt.subplots(2, 3, sharey='row', sharex='col', figsize=(15,4))

        # top row
        for i in range(plt_mat.shape[2]):
            for k in range(plt_mat.shape[0]):

                # try to move relevant traces for each subplot to top of order
                if match_to == 'onsets':
                    cmap = lookups.cmap_fixed_sort_rank9_onset
                    if i == 0 and k < 3:
                        zorder = 8
                    elif i == 1 and (k < 6 & k > 3):
                        zorder = 8
                    elif i == 2 and (k < 9 & k > 6):
                        zorder = 8
                    elif k == 8:
                        zorder = 8
                    else:
                        zorder = 1
                elif match_to == 'offsets':
                    cmap = lookups.cmap_fixed_sort_rank8_offset
                    if k == 0: # broad sharp
                        order = 6
                    if k == 1 or k == 2:
                        order = 7
                    if k == 3 or k == 4:
                        zorder = 5
                    else:
                        zorder = 1
                else:
                    raise NotImplementedError

                ax[0, i].plot(
                    plt_mat[k,:,i],
                    color=cmap[k],
                    zorder=zorder,
                    linewidth=1,
                )
                ax[0, i].fill_between(
                    np.arange(len(plt_mat[k,:,i])),
                    plt_mat[k,:,i] + plt_sem[k,:,i],
                    plt_mat[k,:,i] - plt_sem[k,:,i],
                    color=np.array(cmap[k])*0.8,
                    zorder=zorder,
                    alpha=0.4
                )

        # add stimulus bars now that the 1st axis is finished
        y2 = ax[0, i].get_ylim()[0]
        rect_height = (ax[0, i].get_ylim()[1] - ax[0, i].get_ylim()[0])/14
        for i in range(3):
            xt, xtl = heatmap_xticks(
                additional_pt=None, staging='parsed_11stage_label_short',
                drop_naive= False if allstage else True
            )
            ax[0, i].set_xticks(xt)
            ax[0, i].set_xticklabels(xtl)

            #         rec_list = [Rectangle((s, y2-rect_height), 31, rect_height) for s in np.arange(15.5, 470, 47)]
            #         pc = PatchCollection(rec_list, facecolor='xkcd:medium gray', alpha=0.7, clip_on=False)
            #         ax[0, i].add_collection(pc)

            if limit_to == 'onsets':
                ax[1, i].set_xlabel('\nTime from stimulus onset (s)', size=16)
                rec_list = [Rectangle((s, y2-rect_height), 31, rect_height) for s in np.arange(15.5, 470, 47)]
                pc = PatchCollection(rec_list, facecolor='xkcd:medium gray', alpha=0.7, clip_on=False)
                ax[0, i].add_collection(pc)
            elif limit_to == 'offsets':
                ax[1, i].set_xlabel('\nTime from stimulus offset (s)', size=16)
                rec_list = [Rectangle((s, y2-rect_height), 15.5, rect_height) for s in np.arange(0, 470, 47)]
                pc = PatchCollection(rec_list, facecolor='xkcd:medium gray', alpha=0.7, clip_on=False)
                ax[0, i].add_collection(pc)
            else:
                raise ValueError

        # for mean of all cells
        total_plt_mat = tensor_stack[:, :, [0, 2, 1]]
        total_plt_mat[:, ::47, :] = np.nan
        total_mean = np.nanmean(total_plt_mat[mboo, :, :], axis=0)
        total_sem = np.nanstd(total_plt_mat[mboo, :, :], axis=0) / np.sqrt(np.sum(~np.isnan(total_plt_mat[mboo, :, :]), axis=0))

        for i in range(total_mean.shape[1]):
            ax[1, i].plot(
                total_mean[:,i],
                color='black',
                zorder=zorder,
                linewidth=1,
            )
            ax[1, i].fill_between(
                np.arange(len(total_mean[:,i])),
                total_mean[:,i] + total_sem[:,i],
                total_mean[:,i] - total_sem[:,i],
                color='black',
                zorder=zorder,
                alpha=0.4
            )

        # add stimulus bars now that the second axis is finished
        y2 = ax[1, i].get_ylim()[0]
        rect_height = (ax[1, i].get_ylim()[1] - ax[1, i].get_ylim()[0])/14
        for i in range(3):
            xt, xtl = heatmap_xticks(
                additional_pt=None, staging='parsed_11stage_label_short',
                drop_naive= False if allstage else True
            )

            # xticklabels
            ax[1, i].set_xticks(xt)
            ax[1, i].set_xticklabels(xtl)

            if limit_to == 'onsets':
                rec_list = [Rectangle((s, y2-rect_height), 31, rect_height) for s in np.arange(15.5, 470, 47)]
                pc = PatchCollection(rec_list, facecolor='xkcd:medium gray', alpha=0.7, clip_on=False)
                ax[1, i].add_collection(pc)
            elif limit_to == 'offsets':
                rec_list = [Rectangle((s, y2-rect_height), 15.5, rect_height) for s in np.arange(0, 470, 47)]
                pc = PatchCollection(rec_list, facecolor='xkcd:medium gray', alpha=0.7, clip_on=False)
                ax[1, i].add_collection(pc)
            else:
                raise ValueError

        if norm_please:
            ax[1, 0].set_ylabel('Normalized \u0394F/F\n', size=16, ha='left')
        else:
            ax[1, 0].set_ylabel('\u0394F/F (z-score)\n', size=16, ha='left')

        if save_please:
            plt.savefig(os.path.join(save_folder, f'{mouse}_{match_to}_lim_{limit_to}_{norm_tag}_trace_overlay{allstage_tag}.png'),
                    bbox_inches='tight')
            plt.savefig(os.path.join(save_folder, f'{mouse}_{match_to}_lim_{limit_to}_{norm_tag}_trace_overlay{allstage_tag}.pdf'),
                    bbox_inches='tight')
    #         plt.close('all')


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


def trial_count_modulation(match_to='onsets', trace_type='zscore_day', save_please=True, with_th_mod=False):
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


    # set save location
    save_folder = paths.analysis_dir(f'figures/figure_3/modulation_scatter_trialn/')

    # load TCA data
    on_ens = load.core_tca_data(limit_to=None, match_to=match_to)
    mouse_vec = on_ens['mouse_vec']
    cell_vec = on_ens['cell_vec']
    cell_sorter = on_ens['cell_sorter']
    cell_cats = on_ens['cell_cats']

    # create a matrix of a given trace type
    if trace_type == 'zscore_day':
        if with_th_mod:
            df, mat = selectivity.ab_index_df(match_to=match_to, index_on='th', return_matrix=True)
            z_cue_mat = mat[:,:,[0, 1, 2]]
            z_cue_mat_prev_same = mat[:,:,[6, 7, 8]]
        else:
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

    # addtional data parsing and plotting for TH mod specific plots
    if with_th_mod:
        early_responses_adapt_prevsame = np.zeros((7, 3, 11))
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
                    np.nanmean(utils.wrap_tensor(z_cue_mat_prev_same[mboo & cboo,:,i])[:, 17:, :], axis=1)
                    , axis=0)
                T1_cue_resp= mean_cue_resp #[1]
                early_responses_adapt_prevsame[mc, ci, :] = T1_cue_resp

        early_responses_noadapt_prevsame= np.zeros((7, 3, 11))
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
                    np.nanmean(utils.wrap_tensor(z_cue_mat_prev_same[mboo & cboo,:,i])[:, 17:, :], axis=1)
                    , axis=0)
                T1_cue_resp= mean_cue_resp #[1]
                early_responses_noadapt_prevsame[mc, ci, :] = T1_cue_resp

        # plot scatters
        with sns.axes_style('whitegrid'):
            fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True, sharex=True)
        flat_trials = np.cumsum(early_trials[:,:,:-1], axis=2).flatten()
        flat_trials[flat_trials <= 5] = 0 # deal with tiny trial number at begining with pavs
        flat_resp = early_responses_adapt[:,:,1:].flatten()
        flat_resp_same = early_responses_adapt_prevsame[:,:,1:].flatten()
        cue_color = ((['Initially rewarded']*10 + ['Becomes rewarded']*10 + ['Unrewarded']*10) * 7)  # in the order of flatten
        g = sns.scatterplot(
            x=flat_trials+1, y=flat_resp, s=100, # hue=flat_trials,
            color='red',
            alpha=0.5, ax=ax[0]
        )
        g = sns.scatterplot(
            x=flat_trials+1, y=flat_resp_same, s=100, # hue=flat_trials,
            color='black',
            alpha=0.5, ax=ax[0]
        )
        g.axes.set_xscale('log')
        flat_resp = early_responses_noadapt[:,:,:-1].flatten()
        flat_resp_same = early_responses_noadapt_prevsame[:,:,:-1].flatten()
        g = sns.scatterplot(
            x=flat_trials+1, y=flat_resp, s=100, # hue=flat_trials,
            color='red',
            alpha=0.5, ax=ax[1]
        )
        g = sns.scatterplot(
            x=flat_trials+1, y=flat_resp_same, s=100, # hue=flat_trials,
            color='black',
            alpha=0.5, ax=ax[1]
        )

        # sns.despine()
        ax[0].set_xlabel('Trial count preceding learning stage', size=14)
        ax[1].set_xlabel('Trial count preceding learning stage', size=14)
        ax[0].set_title('Adapting class', size=14)
        ax[1].set_title('Non-adapting classes', size=14)
        # plt.xlabel("Performance (d')")
        if trace_type == 'zscore_day':
            ax[0].set_ylabel('Population response magnitude\n\u0394F/F (z-score)', size=14)
        elif trace_type == 'dff':
            ax[0].set_ylabel('Population response magnitude\n\u0394F/F', size=14)
        elif trace_type == 'norm':
            ax[0].set_ylabel('Population response magnitude\nNormalized \u0394F/F', size=14)
        else:
            raise ValueError
        plt.suptitle('Initial and reversal learning', position=(0.5, 1.05), size=20)

        if save_please:
            plt.savefig(os.path.join(save_folder, f'{match_to}_{trace_type}_trialn_vs_stagemean_scatter_TH.png'),
                    bbox_inches='tight')
            plt.savefig(os.path.join(save_folder, f'{match_to}_{trace_type}_trialn_vs_stagemean_scatter_TH.pdf'),
                    bbox_inches='tight')
            plt.close('all')


        # plot scatters
        with sns.axes_style('whitegrid'):
            fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True, sharex=True)

        flat_trials = early_trials[:,:,0].flatten()
        flat_trials[flat_trials <= 5] = 0
        flat_resp = early_responses_adapt[:,:,1].flatten()
        flat_resp_same = early_responses_adapt_prevsame[:,:,1].flatten()
        cue_color = ((['Initially rewarded']*1 + ['Becomes rewarded']*1 + ['Unrewarded']*1) * 7)  # rows then cols for 2d row-major
        g = sns.scatterplot(
            x=flat_trials+1, y=flat_resp, s=100, # hue=flat_trials,
            color='red',
            alpha=0.5, ax=ax[0]
        )
        g = sns.scatterplot(
            x=flat_trials+1, y=flat_resp_same, s=100, # hue=flat_trials,
            color='black',
            alpha=0.5, ax=ax[0]
        )
        g.axes.set_xscale('log')
        flat_resp = early_responses_noadapt[:,:,1].flatten()
        flat_resp_same = early_responses_noadapt_prevsame[:,:,1].flatten()
        g = sns.scatterplot(
            x=flat_trials+1, y=flat_resp, s=100, # hue=flat_trials,
            color='red',
            alpha=0.5, ax=ax[1]
        )
        g = sns.scatterplot(
            x=flat_trials+1, y=flat_resp_same, s=100, # hue=flat_trials,
            color='black',
            alpha=0.5, ax=ax[1]
        )

        # sns.despine()
        ax[0].set_xlabel('Trial count preceding learning stage', size=14)
        ax[1].set_xlabel('Trial count preceding learning stage', size=14)
        ax[0].set_title('Adapting class', size=14)
        ax[1].set_title('Non-adapting classes', size=14)
        # plt.xlabel("Performance (d')")
        if trace_type == 'zscore_day':
            ax[0].set_ylabel('Population response magnitude\n\u0394F/F (z-score)', size=14)
        elif trace_type == 'dff':
            ax[0].set_ylabel('Population response magnitude\n\u0394F/F', size=14)
        elif trace_type == 'norm':
            ax[0].set_ylabel('Population response magnitude\nNormalized \u0394F/F', size=14)
        else:
            raise ValueError
        plt.suptitle('T1 initial learning', position=(0.5, 1.05), size=20)

        if save_please:
            plt.savefig(os.path.join(save_folder, f'{match_to}_{trace_type}_trialn_vs_T1_scatter_TH.png'),
                    bbox_inches='tight')
            plt.savefig(os.path.join(save_folder, f'{match_to}_{trace_type}_trialn_vs_T1_scatter_TH.pdf'),
                    bbox_inches='tight')
            plt.close('all')
    else:
        # plot scatters
        with sns.axes_style('whitegrid'):
            fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True, sharex=True)

        flat_trials = np.cumsum(early_trials[:,:,:-1], axis=2).flatten()
        flat_trials[flat_trials <= 5] = 0 # deal with tiny trial number at begining with pavs
        flat_resp = early_responses_adapt[:,:,1:].flatten()
        flat_dp = early_dp[:,:,:].flatten()
        cue_color = ((['Initially rewarded']*10 + ['Becomes rewarded']*10 + ['Unrewarded']*10) * 7)  # in the order of flatten
        g = sns.scatterplot(
        #         x=flat_trials, y=flat_resp, hue=cue_color, palette=lookups.color_dict, s=100,
            x=flat_trials+1, y=flat_resp, s=100, # hue=flat_trials,
            hue=cue_color, palette=lookups.color_dict,
            alpha=0.5, ax=ax[0]
        )
        g.axes.set_xscale('log')

        flat_resp = early_responses_noadapt[:,:,:-1].flatten()
        g = sns.scatterplot(
        #         x=flat_trials, y=flat_resp, hue=cue_color, palette=lookups.color_dict, s=100,
            x=flat_trials+1, y=flat_resp, s=100, # hue=flat_trials,
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
        if trace_type == 'zscore_day':
            ax[0].set_ylabel('Population response magnitude\n\u0394F/F (z-score)', size=14)
        elif trace_type == 'dff':
            ax[0].set_ylabel('Population response magnitude\n\u0394F/F', size=14)
        elif trace_type == 'norm':
            ax[0].set_ylabel('Population response magnitude\nNormalized \u0394F/F', size=14)
        else:
            raise ValueError
        plt.suptitle('Initial and reversal learning', position=(0.5, 1.05), size=20)

        if save_please:
            plt.savefig(os.path.join(save_folder, f'{match_to}_{trace_type}_trialn_vs_stagemean_scatter.png'),
                    bbox_inches='tight')
            plt.savefig(os.path.join(save_folder, f'{match_to}_{trace_type}_trialn_vs_stagemean_scatter.pdf'),
                    bbox_inches='tight')
            plt.close('all')


        # plot scatters
        with sns.axes_style('whitegrid'):
            fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True, sharex=True)

        flat_trials = early_trials[:,:,0].flatten()
        flat_trials[flat_trials <= 5] = 0
        flat_resp = early_responses_adapt[:,:,1].flatten()
        # flat_dp = early_dp[:,:,:].flatten()
        cue_color = ((['Initially rewarded']*1 + ['Becomes rewarded']*1 + ['Unrewarded']*1) * 7)  # rows then cols for 2d row-major
        g = sns.scatterplot(
        #         x=flat_trials, y=flat_resp, hue=cue_color, palette=lookups.color_dict, s=100,
            x=flat_trials+1, y=flat_resp, s=100, # hue=flat_trials,
            hue=cue_color, palette=lookups.color_dict,
            alpha=0.5, ax=ax[0]
        )
        g.axes.set_xscale('log')

        flat_resp = early_responses_noadapt[:,:,1].flatten()
        g = sns.scatterplot(
        #         x=flat_trials, y=flat_resp, hue=cue_color, palette=lookups.color_dict, s=100,
            x=flat_trials+1, y=flat_resp, s=100, # hue=flat_trials,
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
        if trace_type == 'zscore_day':
            ax[0].set_ylabel('Population response magnitude\n\u0394F/F (z-score)', size=14)
        elif trace_type == 'dff':
            ax[0].set_ylabel('Population response magnitude\n\u0394F/F', size=14)
        elif trace_type == 'norm':
            ax[0].set_ylabel('Population response magnitude\nNormalized \u0394F/F', size=14)
        else:
            raise ValueError
        plt.suptitle('T1 initial learning', position=(0.5, 1.05), size=20)

        if save_please:
            plt.savefig(os.path.join(save_folder, f'{match_to}_{trace_type}_trialn_vs_T1_scatter.png'),
                    bbox_inches='tight')
            plt.savefig(os.path.join(save_folder, f'{match_to}_{trace_type}_trialn_vs_T1_scatter.pdf'),
                    bbox_inches='tight')
            plt.close('all')
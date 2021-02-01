import cascade as cas
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


# Load data that has been triggered, aligned, and had a TCA Model run on it
mice = cas.lookups.mice['rev9']
words = ['respondent' if s in 'OA27' else 'computation' for s in mice]
group_by = 'all3'

rank_level_sort = 15

# load in a full size tensor
model_list = []
tensor_list = []
id_list = []
bhv_list = []
meta_list = []
for mouse, word in zip(mice, words):

    # return   model, ids, tensor, meta, bhv
    out = cas.load.load_all_groupday(mouse, word=word, with_model=True, group_by=group_by, nan_thresh=0.95)
    sorted_model, sort_order = cas.utils.sortfactors(out[0])
    model_list.append(sorted_model)
    tensor_list.append(out[2][sort_order[rank_level_sort - 1], :, :])
    id_list.append(out[1][sort_order[rank_level_sort - 1]])
    bhv_list.append(out[4])
    meta_list.append(cas.utils.add_stages_to_meta(out[3], 'parsed_11stage'))


# load offset classification
off_df_all_mice = pd.read_pickle('/twophoton_analysis/Data/analysis/core_dfs/off_dfs_woren_calcs.pkl')
off_df_all_mice = off_df_all_mice.loc[off_df_all_mice.reset_index().mouse.isin(mice).values, :]

for vi in range(10):
    # testing on beastmode
    mm_df_list = []
    for meta, tensor, ids, offdf in zip(meta_list, tensor_list, id_list, cas.utils.df_split(off_df_all_mice)):

        # reorder you cells to match ids
        reordered_df = offdf.reindex(ids, level='cell_id')

        # must be using a matched df and ids must match
        assert cas.utils.meta_mouse(meta) == cas.utils.meta_mouse(offdf)
        assert np.array_equal(ids, reordered_df.reset_index().cell_id.values)

        offset_bool = reordered_df.offset_test.values

        # with oren
        df = cas.mismatch.mismatch_stat(meta, tensor, ids, search_epoch='L5 reversal1', offset_bool=offset_bool,
                                        stim_calc_start_s=0.2, stim_calc_end_s=0.700, off_calc_start_s=0.200, off_calc_end_s=0.700,
                                        plot_please=False, plot_w='heatmap', neg_log10_pv_thresh=4, alternative='less')
        mm_df_list.append(df)
    mm_dfs = pd.concat(mm_df_list, axis=0)
    mm_dfs.to_pickle(f'/twophoton_analysis/Data/analysis/core_dfs/mismatch_stat_df_allcues_vT5r1_newoff_v{vi}.pkl')
    mm_dfs.head()

thresh = 4
save_folder = '/twophoton_analysis/Data/analysis/mm_sus_versions_plots/'
for vi in range(10):
    mm_dfs = pd.read_pickle(f'/twophoton_analysis/Data/analysis/core_dfs/mismatch_stat_df_allcues_vT5r1_newoff_v{vi}.pkl')

    plus_df = mm_dfs.loc[~mm_dfs.offset_cell &
                         ((mm_dfs.mm_neglogpv_search.ge(thresh) | mm_dfs.mm_neglogpv_target.ge(thresh))
                          | mm_dfs.parsed_11stage.isin(['T5 reversal1']))]
    plus_df = plus_df.loc[~plus_df.reset_index().mouse.isin(['AS20', 'AS23']).values, :]
    ppp = plus_df

    plt.figure(figsize=(15, 4))
    text_shifter = 0
    for tun in ['becomes_rewarded', 'becomes_unrewarded', 'remains_unrewarded']:
        cue_err_df = ppp.loc[ppp.reset_index().mm_type.isin([tun]).values, :]
        x = np.arange(len(cas.lookups.staging['parsed_11stage_T']))
        y = cue_err_df.groupby(['parsed_11stage', 'mm_type'], sort=False).mean().mm_frac
        numer = cue_err_df.groupby(['parsed_11stage', 'mm_type'], sort=False).std().mm_frac
        denom = cue_err_df.groupby(['parsed_11stage', 'mm_type'], sort=False).count().apply(np.sqrt).mm_frac
        count = cue_err_df.groupby(['parsed_11stage', 'mm_type'], sort=False).count().mm_frac
        sem = numer / denom
        yerr = sem
        plt.errorbar(x, y, yerr, color=cas.lookups.color_dict[tun], label=tun)
        for xi, yi, ci in zip(x, y.values, count.values):
            plt.text(xi + 0.05, 0.2 - text_shifter, f'n={ci}', color=cas.lookups.color_dict[tun])
        text_shifter += 0.025

    plt.legend(title='Cue type',
               bbox_to_anchor=(1.01, 1), loc=2)
    plt.title(f'Fractional change in response, relative to T5 reversal1\n', size=16)
    plt.ylabel('Change in response (%)', size=14)
    plt.xlabel('\nstage of learning', size=14)
    plt.xticks(ticks=np.arange(len(cas.lookups.staging['parsed_11stage_T'])),
               labels=cas.lookups.staging['parsed_11stage_T'], rotation=45, ha='right')
    plt.axvline(5.5, linewidth=2, alpha=0.5, color='red')
    plt.savefig(os.path.join(save_folder, f'rev_v_{vi}_mm_frac_delta_T5r1.png'), bbox_inches='tight')


    plus_df = mm_dfs.loc[~mm_dfs.offset_cell &
                         ((mm_dfs.mm_neglogpv_target.ge(thresh) & mm_dfs.mm_neglogpv_search.ge(thresh))
                          | mm_dfs.parsed_11stage.isin(['T5 reversal1']))]
    plus_df = plus_df.loc[~plus_df.reset_index().mouse.isin(['AS20', 'AS23']).values, :]
    ppp = plus_df

    plt.figure(figsize=(15, 4))
    text_shifter = 0
    for tun in ['becomes_rewarded', 'becomes_unrewarded', 'remains_unrewarded']:
        cue_err_df = ppp.loc[ppp.reset_index().mm_type.isin([tun]).values, :]
        x = np.arange(len(cas.lookups.staging['parsed_11stage_T']))
        y = cue_err_df.groupby(['parsed_11stage', 'mm_type'], sort=False).mean().mm_amp
        numer = cue_err_df.groupby(['parsed_11stage', 'mm_type'], sort=False).std().mm_amp  # .mm_frac
        denom = cue_err_df.groupby(['parsed_11stage', 'mm_type'], sort=False).count().apply(np.sqrt).mm_amp
        count = cue_err_df.groupby(['parsed_11stage', 'mm_type'], sort=False).count().mm_amp
        sem = numer / denom
        yerr = sem

        plt.errorbar(x, y, yerr, color=cas.lookups.color_dict[tun], label=tun)
        for xi, yi, ci in zip(x, y.values, count.values):
            plt.text(xi + 0.05, 0.3 - text_shifter, f'n={ci}', color=cas.lookups.color_dict[tun])
        text_shifter += 0.045

    plt.legend(title='Cue type',
               bbox_to_anchor=(1.01, 1), loc=2)
    plt.title(f'Amplitude change in response, relative to T5 reversal1\n', size=16)
    plt.ylabel('Change in response (\u0394F/F, z-score)', size=14)
    plt.xlabel('\nstage of learning', size=14)
    plt.xticks(ticks=np.arange(len(cas.lookups.staging['parsed_11stage_T'])),
               labels=cas.lookups.staging['parsed_11stage_T'], rotation=45, ha='right')
    plt.axvline(5.5, linewidth=2, alpha=0.5, color='red')
    plt.savefig(os.path.join(save_folder, f'rev_v_{vi}_mm_amp_delta_T5r1.png'), bbox_inches='tight')


    plus_df = mm_dfs.loc[~mm_dfs.offset_cell &
                         ((mm_dfs.mm_neglogpv_search.ge(thresh) | mm_dfs.mm_neglogpv_target.ge(thresh))
                          | mm_dfs.parsed_11stage.isin(['T5 reversal1']))]
    plus_df = plus_df.loc[~plus_df.reset_index().mouse.isin(['AS20', 'AS23']).values, :]
    ppp = plus_df

    plt.figure(figsize=(15, 4))
    text_shifter = 0
    for tun in ['becomes_rewarded', 'becomes_unrewarded', 'remains_unrewarded']:
        cue_err_df = ppp.loc[ppp.reset_index().mm_type.isin([tun]).values, :]
        x = np.arange(len(cas.lookups.staging['parsed_11stage_T']))
        y = cue_err_df.groupby(['parsed_11stage', 'mm_type'], sort=False).mean().delta_frac_sustainedness
        numer = cue_err_df.groupby(['parsed_11stage', 'mm_type'], sort=False).std().delta_frac_sustainedness  # .mm_frac
        denom = cue_err_df.groupby(['parsed_11stage', 'mm_type'], sort=False).count().apply(
            np.sqrt).delta_frac_sustainedness
        count = cue_err_df.groupby(['parsed_11stage', 'mm_type'], sort=False).count().delta_frac_sustainedness
        sem = numer / denom
        yerr = sem
        plt.errorbar(x, y, yerr, color=cas.lookups.color_dict[tun], label=tun)
        for xi, yi, ci in zip(x, y.values, count.values):
            plt.text(xi + 0.05, 0.2 - text_shifter, f'n={ci}', color=cas.lookups.color_dict[tun])
        text_shifter += 0.025

    plt.legend(title='Cue type',
               bbox_to_anchor=(1.01, 1), loc=2)
    plt.title(f'Fractional change in sustainedness, relative to T5 reversal1\n', size=16)
    plt.ylabel('Change SI (% delta SI)', size=14)
    plt.xlabel('\nstage of learning', size=14)
    plt.xticks(ticks=np.arange(len(cas.lookups.staging['parsed_11stage_T'])),
               labels=cas.lookups.staging['parsed_11stage_T'], rotation=45, ha='right')
    plt.axvline(5.5, linewidth=2, alpha=0.5, color='red')
    plt.savefig(os.path.join(save_folder, f'rev_v_{vi}_sus_frac_delta_T5r1.png'), bbox_inches='tight')


    plus_df = mm_dfs.loc[~mm_dfs.offset_cell &
                         ((mm_dfs.mm_neglogpv_search.ge(thresh) | mm_dfs.mm_neglogpv_target.ge(thresh))
                          | mm_dfs.parsed_11stage.isin(['T5 reversal1']))]
    plus_df = plus_df.loc[~plus_df.reset_index().mouse.isin(['AS20', 'AS23']).values, :]
    ppp = plus_df

    plt.figure(figsize=(15, 4))
    text_shifter = 0
    for tun in ['becomes_rewarded', 'becomes_unrewarded', 'remains_unrewarded']:
        cue_err_df = ppp.loc[ppp.reset_index().mm_type.isin([tun]).values, :]
        x = np.arange(len(cas.lookups.staging['parsed_11stage_T']))
        y = cue_err_df.groupby(['parsed_11stage', 'mm_type'], sort=False).mean().delta_sustainedness
        numer = cue_err_df.groupby(['parsed_11stage', 'mm_type'], sort=False).std().delta_sustainedness
        denom = cue_err_df.groupby(['parsed_11stage', 'mm_type'], sort=False).count().apply(np.sqrt).delta_sustainedness
        count = cue_err_df.groupby(['parsed_11stage', 'mm_type'], sort=False).count().delta_sustainedness
        sem = numer / denom
        yerr = sem
        plt.errorbar(x, y, yerr, color=cas.lookups.color_dict[tun], label=tun)
        for xi, yi, ci in zip(x, y.values, count.values):
            plt.text(xi + 0.05, 0.1 - text_shifter, f'n={ci}', color=cas.lookups.color_dict[tun])
        text_shifter += 0.015

    plt.legend(title='Cue type',
               bbox_to_anchor=(1.01, 1), loc=2)
    plt.title(f'Change in sustainedness, relative to T5 reversal1\n', size=16)
    plt.ylabel('Change in SI (SI units)', size=14)
    plt.xlabel('\nstage of learning', size=14)
    plt.xticks(ticks=np.arange(len(cas.lookups.staging['parsed_11stage_T'])),
               labels=cas.lookups.staging['parsed_11stage_T'], rotation=45, ha='right')
    plt.axvline(5.5, linewidth=2, alpha=0.5, color='red')
    plt.savefig(os.path.join(save_folder, f'rev_v_{vi}_sus_amp_delta_T5r1.png'), bbox_inches='tight')

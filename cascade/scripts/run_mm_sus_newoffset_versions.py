
off_df_all_mice = pd.read_pickle('/twophoton_analysis/Data/analysis/core_dfs/off_dfs_woren_calcs.pkl')

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
    df = cas.mismatch.mismatch_stat(meta, tensor, ids, search_epoch='L5 learning', offset_bool=offset_bool,
                                    stim_calc_start_s=0.2, stim_calc_end_s=0.700, off_calc_start_s=0.200, off_calc_end_s=0.700,
                                    plot_please=False, plot_w='heatmap', neg_log10_pv_thresh=4, alternative='less')
    mm_df_list.append(df)
mm_dfs = pd.concat(mm_df_list, axis=0)
mm_dfs.to_pickle(f'/twophoton_analysis/Data/analysis/core_dfs/mismatch_stat_df_allcues_vT5l_newoff_v{vi}.pkl')
mm_dfs.head()
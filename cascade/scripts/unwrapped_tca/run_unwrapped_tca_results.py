import cascade as cas
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensortools as tt


thresh=4
iteration=10
mod_suffix = '_noT0'
heatmap_rank = 9

for date in [20210307, 20210215]:

    ensemble = np.load(cas.paths.analysis_file(f'tca_ensemble_v4i10_noT0_{date}.npy', 'tca_dfs'),
                       allow_pickle=True).item()
    # print(ensemble)

    data_dict = np.load(cas.paths.analysis_file(f'input_data_v4i10_noT0_{date}.npy', 'tca_dfs'),
                        allow_pickle=True).item()
    # print(data_dict.keys())

    for mod in ['v4i10_scale_on_noT0', 'v4i10_scale_off_noT0', 'v4i10_norm_on_noT0', 'v4i10_norm_off_noT0']:
        # mod = 'v4i10_norm_on_noT0'

        for bot, top in zip([9, 9, 6, 8, 7, 6], [15, 9, 20, 8, 7, 12]):
            # bot = 9
            # top = 9
            add_tuning = True
            iter_to_plot = 10

            hue_order = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']
            fac_stack = []
            hue_stack = []
            rank_stack = []
            iter_stack = []
            iter_num = []
            tune_stack = []
            cmap = sns.color_palette('Purples', 21)
            cmap2 = sns.color_palette('binary', iter_to_plot)
            for rr in range(bot, top + 1):
                for it in range(iter_to_plot):
                    factors = ensemble[mod].results[rr][it].factors

                    if np.isnan(factors[0]).any():
                        print(f'skipped {rr}')
                        continue
                    temp_max = np.max(factors[1], axis=0)
                    tune_max = np.max(factors[2], axis=0)
                    scaled_cells = factors[0] * temp_max * tune_max
                    scaled_traces = factors[1] / temp_max
                    scaled_tune = factors[2] / tune_max
                    factors = (scaled_cells, scaled_traces, scaled_tune)
                    fac_stack.append(scaled_traces)
                    tune_stack.append(scaled_tune)

                    max_tune = np.argmax(scaled_tune.T, axis=1)
                    max_tune = [hue_order[s] for s in max_tune]
                    hue_stack.append(max_tune)
                    rank_stack.append([cmap[rr]] * rr)
                    iter_stack.append([cmap2[it]] * rr)
                    iter_num.append([it] * rr)

            test = np.hstack(fac_stack)
            test_hue = np.hstack(hue_stack)
            rank_hue = np.vstack(rank_stack)
            iter_hue = np.vstack(iter_stack)
            iter_num = np.hstack(iter_num)
            tune_vecs = np.hstack(tune_stack)
            cue_hue = [cas.lookups.color_dict[s] for s in test_hue]

            # rare but sometime rescaling creates nans
            test[np.isnan(test)] = 0  # fix 0/0 nans

            # plot total clustering of pooled models
            hues = [rank_hue, iter_hue, cue_hue]
            if add_tuning:
                cmap = sns.color_palette("rocket", 10)
                sns.palplot(cmap)
                cmap_inds = np.digitize(tune_vecs, np.arange(0, 1, 0.1)) - 1
                for i in range(3):
                    hues.append([cmap[int(s)] for s in cmap_inds[i, :]])
            g2 = sns.clustermap(test[:, :].T, col_cluster=False, row_cluster=True, row_colors=hues, method='ward')
            sorter2 = g2.dendrogram_row.reordered_ind

            plt.suptitle(f'{mod}, rank {bot}-{top}, temporal factors\nhierarchical clustering',
                         size=20,
                         position=(0.5, 0.95))
            plt.savefig(cas.paths.analysis_file(f'{mod}_rank{bot}-{top}_hierclusTemp_resc.png', f'tca_dfs/TCA_results/{mod}/{date}/clustering_factors/'),
                        bbox_inches='tight')

            # break things up by model iteration
            # hues = [rank_hue, iter_hue, cue_hue]
            sort_hues = []
            for h in hues:
                sort_hues.append(np.array(h)[sorter2])

            for i in range(iter_to_plot):
                iter_boo = np.isin(iter_num, i)[sorter2]

                iter_hues = []
                for h in sort_hues:
                    iter_hues.append(np.array(h)[iter_boo])

                try:
                    sns.clustermap(test[:, sorter2][:, iter_boo].T,
                                   col_cluster=False,
                                   row_cluster=False,
                                   row_colors=iter_hues,
                                   method='ward')

                    plt.suptitle(f'{mod}, rank {bot}-{top}, iteration {i+1}\ntemporal factors, hierarchical clustering',
                                 size=20,
                                 position=(0.5, 0.95))
                    plt.savefig(cas.paths.analysis_file(f'{mod}_rank{bot}-{top}_iter{i+1}_hierclusTemp_resc.png',
                                                        f'tca_dfs/TCA_results/{mod}/{date}/clustering_factors/'),
                                bbox_inches='tight')
                except:
                    print(f'skipped iteration {i}')
        plt.close('all')
        print('made it here')
        # plot and save relevant results
        # --------------------------------------------------------------------------------------------------

        # plot model performance
        for k, v in ensemble.items():

            if not k in mod:
                continue

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

            plt.savefig(cas.paths.analysis_file(f'{k}_obj_sim.png', f'tca_dfs/TCA_results/{mod}/{date}/TCA_qc'), bbox_inches='tight')
        plt.close('all')

        # plot factors after sorting
        sort_ensembles, sort_orders = {}, {}
        for k, v in ensemble.items():
            sort_ensembles[k], sort_orders[k] = cas.utils.sortfactors(v)

        for k, v in sort_ensembles.items():
            
            if not k in mod:
                continue

            for rr in range(5, 20):
                fig, ax, _ = tt.visualization.plot_factors(v.results[rr][0].factors.rebalance(), plots=['scatter', 'line', 'line'],
                                scatter_kw=cas.lookups.tt_plot_options['ncp_hals']['scatter_kw'],
                                line_kw=cas.lookups.tt_plot_options['ncp_hals']['line_kw'],
                                bar_kw=cas.lookups.tt_plot_options['ncp_hals']['bar_kw']);

                cell_count = v.results[rr][0].factors[0].shape[0]
                for i in range(ax.shape[0]):
                    ax[i, 0].set_ylabel(f'                 Component {i+1}', size=16, ha='right', rotation=0)
                ax[0, 1].set_title(f'{k}, rank {rr} (n = {cell_count})\n\n', size=20)

                plt.savefig(cas.paths.analysis_file(f'{k}_rank{rr}_facs.png', f'tca_dfs/TCA_results/{mod}/{date}/TCA_factors/'), bbox_inches='tight')
            plt.close('all')


        # plot heatmap
        if '_on' in mod:
            mmod = f'v{thresh}i{iteration}_on_mouse{mod_suffix}'
        elif '_off' in mod:
            mmod = f'v{thresh}i{iteration}_off_mouse{mod_suffix}'

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
                cas.paths.analysis_file(f'{mod}_rank{heatmap_rank}_heatmap.png', f'tca_dfs/TCA_results/{mod}/{date}/TCA_heatmaps/'),
                bbox_inches='tight')
            plt.close('all')
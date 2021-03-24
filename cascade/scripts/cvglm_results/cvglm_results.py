import os
import cascade as cas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def main():

    # TCA params
    models = ['v4i10_norm_on_noT0', 'v4i10_norm_off_noT0', 'v4i10_scale_on_noT0', 'v4i10_scale_off_noT0']
    ranks = [9, 8, 9, 8]
    glm_tag = '_v3_gng'
    versions = ['_v3_cellrect2_noT0', '_v3_avg_noT0']  #_v2_avg_noT0
    stage_order = ['early_learning', 'late_learning', 'early_reversal', 'late_reversal']

    for version in versions:
        for mod, rr in zip(models, ranks):

            # load data
            save_folder = cas.paths.analysis_dir(
                f'tca_dfs/TCA_factor_fitting{version}/{mod}/cvglm_stages_poisson_exp{glm_tag}')
            all_mddev_df = pd.read_pickle(os.path.join(save_folder, f'{mod}_rank{rr}_cvGLM_delta_deviance_df.pkl'))
            all_mfdev_df = pd.read_pickle(os.path.join(save_folder, f'{mod}_rank{rr}_cvGLM_fractional_deviance_df.pkl'))
            all_mdev_df = pd.read_pickle(os.path.join(save_folder, f'{mod}_rank{rr}_cvGLM_model_performance_df.pkl'))
            # save_folder = cas.paths.analysis_dir(
            #     f'tca_dfs/TCA_factor_fitting{version}/{mod}/cvglm_stages_poisson_exp{glm_tag}_beta_w0')

            added_col_df = all_mfdev_df.merge(all_mdev_df['total_model_devex_test'],
                                            how='left',
                                            left_index=True,
                                            right_index=True)
            sorter = all_mdev_df['total_model_devex_test'].argsort()
            beta_df = pd.DataFrame(data=np.stack(all_mdev_df['beta_w'].values, axis=0),
                                columns=[s for s in all_mfdev_df.columns if '_agg' not in s], # deal with agg cols
                                index=added_col_df.index)
            # beta_df['beta_w0'] = all_mdev_df['beta_w0'].apply(lambda x: x[0]).values

            # save color mappers for mice and components
            plt.figure();
            sns.palplot(sns.color_palette('muted', 7));
            plt.xticks(np.arange(7), labels=beta_df.reset_index().mouse.unique(), size=16);
            plt.title('Mice', size=20)
            plt.savefig(os.path.join(save_folder, 'mice_colors.pdf'), bbox_inches="tight")
            plt.figure();
            sns.palplot(sns.color_palette('Set3', rr));
            plt.xticks(np.arange(rr), labels=np.arange(rr)+1, size=16);
            plt.title('Components', size=20)
            plt.savefig(os.path.join(save_folder, 'comp_colors.pdf'), bbox_inches="tight")

            # clustermap summary
            thresh = all_mdev_df['total_model_devex_test'].gt(.10)
            mapper1 = {k:v for k,v in zip(beta_df.reset_index().mouse.unique(), sns.color_palette('muted', 7))}
            mapper2 = {k:v for k,v in zip(beta_df.reset_index().component.unique(), sns.color_palette('Set3', rr))}
            m_col = beta_df.reset_index().mouse.map(mapper1).values[thresh]
            comp_col = beta_df.reset_index().component.map(mapper2).values[thresh]
            row_colors = [m_col, comp_col]
            sns.clustermap(beta_df.loc[thresh], cmap='vlag', center=0, xticklabels=True, method='ward',
                        figsize=(15,15), col_cluster=True, row_colors=row_colors)
            plt.savefig(os.path.join(save_folder, 'clustering_betas_gt10_test.png'), bbox_inches="tight")

            # component summary
            thresh = all_mdev_df['total_model_devex_test'].gt(.10).swaplevel(0, 2).sort_index()
            beta_df2 = beta_df.swaplevel(0, 2).sort_index()
            mapper1 = {k:v for k,v in zip(beta_df2.reset_index().mouse.unique(), sns.color_palette('muted', 7))}
            mapper2 = {k:v for k,v in zip(beta_df2.reset_index().component.unique(), sns.color_palette('Set3', rr))}
            m_col = beta_df2.reset_index().mouse.map(mapper1).values[thresh]
            comp_col = beta_df2.reset_index().component.map(mapper2).values[thresh]
            row_colors = [m_col, comp_col]
            hue_order = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']
            cols = beta_df2.columns
            end_cols = [s for s in cols if [ss in s for ss in hue_order]]
            sns.clustermap(beta_df2.loc[thresh], cmap='vlag', center=0, xticklabels=True,
                            figsize=(15,15), row_cluster=False, col_cluster=False, row_colors=row_colors)
            plt.savefig(os.path.join(save_folder, 'clustering_betas_bycomp_gt10.png'), bbox_inches="tight")
            plt.close('all')

            # barplots
            thresh = 0.1
            for name, mini_beta in beta_df.groupby(['component', 'stage']):

                col_lbls, mini_beta = rename_cols(mini_beta, stage=name[1])

                added_beta = mini_beta.merge(all_mdev_df['total_model_devex_test'], how='left',
                                            left_index=True, right_index=True)
                mini_beta = added_beta.loc[added_beta.total_model_devex_test.gt(thresh)].drop(columns=['total_model_devex_test'])

                plt.figure(figsize=(15, 4))
                x = np.arange(len(mini_beta.columns))
                y = np.nanmean(mini_beta, axis=0)
                sem = np.nanstd(mini_beta, axis=0) / np.sqrt(np.nansum(~mini_beta.isna(), axis=0))
                colors = sns.color_palette('husl', len(x))
                plt.bar(x, y, yerr=sem, alpha=0.5, color=colors)
                for i in range(mini_beta.shape[0]):
                    plt.scatter(x, mini_beta.values[i, :], color=colors, alpha=0.9)
                plt.xticks(x, labels=col_lbls, rotation=45, ha='right')
                plt.title(
                    f'Betas across mice, component {name[0]}, {name[1]}\nerrorbars = SEM, dots = mice (n={mini_beta.shape[0]})\n',
                    size=18)
                plt.ylabel('Beta coefficient', size=16)
                plt.xlabel('Filter name', size=16)
                sns.despine()

                save_folder2 = cas.paths.analysis_dir(os.path.join(save_folder, f'betas_gt{int(thresh*100)}'))
                plt.savefig(os.path.join(
                    save_folder2, f'beta_comp{name[0]}_{name[1]}_{mod}{version}{glm_tag}_gt{int(thresh*100)}.png'),
                            bbox_inches="tight")
                plt.close('all')

                # deviance explained by stage
                for test in col_parser(added_col_df):
    
                    test.fillna(0, inplace=True)
                    cat = test.reset_index().stage.unique()[0]
                    
                    m_col = test.reset_index().mouse.map(mapper1).values
                    comp_col = test.reset_index().component.map(mapper2).values
                    row_colors = [m_col, comp_col]

                    sns.clustermap(test, cmap='rocket', center=0, xticklabels=True, method='ward', yticklabels=False,
                                figsize=(10,10), col_cluster=False, row_cluster=False, vmax=1.2, vmin=-.2, row_colors=row_colors)
                    plt.suptitle(f'deviance explained {cat}\n{mod}', size=20, position=(0.5, 1.05))
                    plt.savefig(os.path.join(save_folder, f'clustering_devex_test_{cat}_structcols.png'), bbox_inches="tight")
                    plt.close('all')

                # ZSCORE betas by stage
                for test in col_parser(beta_df):
    
                    test.fillna(0, inplace=True)
                    cat = test.reset_index().stage.unique()[0]
                    
                    m_col = test.reset_index().mouse.map(mapper1).values
                    comp_col = test.reset_index().component.map(mapper2).values
                    row_colors = [m_col, comp_col]

                    sns.clustermap(test, cmap='vlag', center=0, xticklabels=True, yticklabels=False, method='ward', z_score=0,
                                figsize=(10,10), col_cluster=False, row_cluster=False, vmax=5, vmin=-5, row_colors=row_colors)
                    plt.suptitle(f'betas {cat}\n{mod}\nrow_z_score', size=20, position=(0.5, 1.05))
                    plt.savefig(os.path.join(save_folder, f'clustering_betas_test_{cat}_ZSCstructcols.png'), bbox_inches="tight")
                    plt.close('all')

                # betas by stage
                for test in col_parser(beta_df):
    
                    test.fillna(0, inplace=True)
                    cat = test.reset_index().stage.unique()[0]
                    
                    m_col = test.reset_index().mouse.map(mapper1).values
                    comp_col = test.reset_index().component.map(mapper2).values
                    row_colors = [m_col, comp_col]

                    sns.clustermap(test, cmap='vlag', center=0, xticklabels=True, yticklabels=False, method='ward',
                                figsize=(10,10), col_cluster=False, row_cluster=False, vmax=1.2, vmin=-1.2, row_colors=row_colors)
                    plt.suptitle(f'betas {cat}\n{mod}', size=20, position=(0.5, 1.05))
                    plt.savefig(os.path.join(save_folder, f'clustering_betas_test_{cat}_structcols.png'), bbox_inches="tight")
                    plt.close('all')
                
                # devex stage sumamry
                test = added_col_df.loc[:, ['total_model_devex_test']].swaplevel(0,2).sort_index()
                test = test.unstack(1)
                test.columns = test.columns.swaplevel().sortlevel()[0]
                test = test.fillna(-0.25)
                test = test.loc[:, stage_order]

                m_col = test.reset_index().mouse.map(mapper1).values
                comp_col = test.reset_index().component.map(mapper2).values
                row_colors = [m_col, comp_col]

                mapper3 = {k:v for k,v in zip(stage_order, sns.color_palette('bone', 4))}
                stage_cols = [mapper3[s] for s in [s[0] for s in test.columns.values]]

                sns.clustermap(test, cmap='icefire', center=0, xticklabels=True, yticklabels=False, method='ward',
                            figsize=(6,10), col_cluster=False, row_cluster=False, #vmax=1.2, vmin=-1.2,
                            col_colors=stage_cols, row_colors=row_colors)
                plt.suptitle('deviance explained across stages, blue = nan ', size=20, position=(0.5, 1.05))
                plt.savefig(os.path.join(save_folder, f'devex_stages_summary.png'), bbox_inches="tight")
                plt.close('all')

            for z_please in [False, True]:

                df_list = []
                lbl_list = []
                stage_label = []

                match_index = col_parser(added_col_df)[0].reset_index().set_index(['component', 'mouse']).index
                for sc, test in enumerate(col_parser(added_col_df)):
                # match_index = col_parser(beta_df)[0].reset_index().set_index(['component', 'mouse']).index
                # for sc, test in enumerate(col_parser(beta_df)):
                    
                    cat = test.reset_index().stage.unique()[0]
                    
                    if sc == 0:
                        m_col = test.reset_index().mouse.map(mapper1).values
                        comp_col = test.reset_index().component.map(mapper2).values
                        row_colors = [m_col, comp_col]
                    
                    df_list.append(test
                                .reset_index()
                                .set_index(['component', 'mouse'])
                                .reindex(match_index)
                                .reset_index()
                                .set_index(['component', 'stage', 'mouse']).values)
                    lbl_list.extend(test.columns)
                    
                    mapper3 = {k:v for k,v in zip(stage_order, sns.color_palette('bone', 4))}
                    stage_label.extend([mapper3[cat]] * len(test.columns))

                new_test = np.concatenate(df_list, axis=1)


                g = sns.clustermap(new_test, cmap='rocket', center=0, xticklabels=True, yticklabels=False, method='ward',
                                figsize=(32,10), col_cluster=False, row_cluster=False, vmin=-1.2, vmax=1.2,
                                row_colors=row_colors, col_colors=stage_label, z_score=0 if z_please else None,
                                colors_ratio=[0.01, 0.03], cbar_pos=(0.15, 0.6, 0.02, 0.18))

                g.ax_heatmap.set_xticks(ticks=np.arange(new_test.shape[1])+0.5);
                g.ax_heatmap.set_xticklabels(labels=np.array(lbl_list));
                ztag1 = 'row-z-score' if z_please else ''
                ztag2 = '_row_z_score' if z_please else ''
                plt.suptitle(f'{mod} devex over stages\n{version}{glm_tag}\n{ztag1}', size=20, position=(0.63, 0.93));
                plt.savefig(os.path.join(save_folder, f'devex_over_stage_{mod}{version}{glm_tag}_structcols{ztag2}.png'), bbox_inches="tight")
                plt.close('all')
                
                df_list = []
                lbl_list = []
                stage_label = []

                match_index = col_parser(beta_df)[0].reset_index().set_index(['component', 'mouse']).index
                for sc, test in enumerate(col_parser(beta_df)):
                    
                    cat = test.reset_index().stage.unique()[0]
                    
                    if sc == 0:
                        m_col = test.reset_index().mouse.map(mapper1).values
                        comp_col = test.reset_index().component.map(mapper2).values
                        row_colors = [m_col, comp_col]
                    
                    df_list.append(test
                                .reset_index()
                                .set_index(['component', 'mouse'])
                                .reindex(match_index)
                                .reset_index()
                                .set_index(['component', 'stage', 'mouse']).values)
                    lbl_list.extend(test.columns)
                    
                    mapper3 = {k:v for k,v in zip(stage_order, sns.color_palette('bone', 4))}
                    stage_label.extend([mapper3[cat]] * len(test.columns))

                new_test = np.concatenate(df_list, axis=1)

                g = sns.clustermap(new_test, cmap='vlag', center=0, xticklabels=True, yticklabels=False, method='ward',
                                figsize=(32,10), col_cluster=False, row_cluster=False, vmin=None, vmax=None,
                                row_colors=row_colors, col_colors=stage_label, z_score=0 if z_please else None,
                                colors_ratio=[0.01, 0.03], cbar_pos=(0.15, 0.6, 0.02, 0.18))

                g.ax_heatmap.set_xticks(ticks=np.arange(new_test.shape[1])+0.5);
                g.ax_heatmap.set_xticklabels(labels=np.array(lbl_list));
                ztag1 = 'row-z-score' if z_please else ''
                ztag2 = '_row_z_score' if z_please else ''
                plt.suptitle(f'{mod} betas over stages\n{version}{glm_tag}\n{ztag1}', size=20, position=(0.63, 0.93));
                plt.savefig(os.path.join(save_folder, f'betas_over_stage_{mod}{version}{glm_tag}_structcols{ztag2}.png'), bbox_inches="tight")
                plt.close('all')


def col_parser(df):
    """Rename columns using names like 'hit' and 'miss' instead of interaction col names. Also
    resort your column order moving cue specific vaiables to front. 

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame of betas or deviance explained across mice and stages. 

    Returns
    -------
    list of pandas.DataFrames
        Return a list of your newly sorted and col-renamed dfs in a list, one per stage.
    """
    hue_order = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']
    cols = df.columns
    end_cols = [s for s in cols if not any([ss in s for ss in hue_order])]
    cue_cols = [[s for s in cols if h in s] for h in hue_order]
    cue_cols = [item for sublist in cue_cols for item in sublist]
    resort_list = cue_cols + end_cols

    sorted_df = df.loc[:, resort_list]

    stage_dfs = []
    for stage, group_df in sorted_df.groupby('stage'):
        col_list = []
        for stage_col in group_df.columns:
            if 'learning' in stage:
                if 'nogo_' in stage_col and 'becomes_unrewarded' in stage_col:
                    col_list.append('miss')
                elif 'nogo_' in stage_col:
                    col_list.append('CR')
                elif 'go_' in stage_col and 'becomes_unrewarded' in stage_col:
                    col_list.append('hit')
                elif 'go_' in stage_col:
                    col_list.append('FA')
                else:
                    col_list.append(stage_col)
            if 'reversal' in stage:
                if 'nogo_' in stage_col and 'becomes_rewarded' in stage_col:
                    col_list.append('miss')
                elif 'nogo_' in stage_col:
                    col_list.append('CR')
                elif 'go_' in stage_col and 'becomes_rewarded' in stage_col:
                    col_list.append('hit')
                elif 'go_' in stage_col:
                    col_list.append('FA')
                else:
                    col_list.append(stage_col)
        new_df = pd.DataFrame(data=group_df.values, columns=col_list, index=group_df.index)
        stage_dfs.append(new_df.swaplevel(0, 2).sort_index())

    # back to temporal order
    stage_dfs = [stage_dfs[0], stage_dfs[2], stage_dfs[1], stage_dfs[3]]

    return stage_dfs


def rename_cols(group_df, stage=None):
    """Rename columns using names like 'hit' and 'miss' instead of interaction col names. Also
    resort your column order moving cue specific vaiables to front. 

    Parameters
    ----------
    group_df : pandas.DataFrame
        DataFrame of betas or deviance explained.
    stage : str, optional
        Stage name that the data comes from, changes the meaning of 'go' and 'nogo', by default None

    Returns
    -------
    list and pandas.DataFrame
        Return a list of your new columns as well as a df with new sort order
    """

    hue_order = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']
    cols = group_df.columns
    end_cols = [s for s in cols if not any([ss in s for ss in hue_order])]
    cue_cols = [[s for s in cols if h in s] for h in hue_order]
    cue_cols = [item for sublist in cue_cols for item in sublist]
    resort_list = cue_cols + end_cols

    sorted_df = group_df.copy().loc[:, resort_list]

    col_list = []
    for stage_col in sorted_df.columns:
        if 'learning' in stage:
            if 'nogo_' in stage_col and 'becomes_unrewarded' in stage_col:
                col_list.append('miss')
            elif 'nogo_' in stage_col:
                col_list.append('CR')
            elif 'go_' in stage_col and 'becomes_unrewarded' in stage_col:
                col_list.append('hit')
            elif 'go_' in stage_col:
                col_list.append('FA')
            else:
                col_list.append(stage_col)
        if 'reversal' in stage:
            if 'nogo_' in stage_col and 'becomes_rewarded' in stage_col:
                col_list.append('miss')
            elif 'nogo_' in stage_col:
                col_list.append('CR')
            elif 'go_' in stage_col and 'becomes_rewarded' in stage_col:
                col_list.append('hit')
            elif 'go_' in stage_col:
                col_list.append('FA')
            else:
                col_list.append(stage_col)

    return col_list, sorted_df


if __name__ == '__main__':
    main()
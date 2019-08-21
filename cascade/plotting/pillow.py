"""
Functions for plotting psytrack behavioral modeling from the
Pillow lab. Also includes functions for comparing to tensortools TCA results.
"""
import flow
import pool
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from .. import paths, load
from scipy.stats import pearsonr


def correlate_pillow_tca(
        mouse,
        word='already',
        group_by='all'):

    # LOADING

    # Load or run your psytracker behavioral model
    ms = flow.Mouse(mouse)
    psy = ms.psytracker(verbose=True)
    dateRuns = psy.data['dateRuns']
    trialRuns = psy.data['runLength']

    # create your trial indices per day and run
    trial_idx = []
    for i in trialRuns:
        trial_idx.extend(range(i))

    # get date and run vectors
    date_vec = []
    run_vec = []
    for c, i in enumerate(dateRuns):
        date_vec.extend([i[0]]*trialRuns[c])
        run_vec.extend([i[1]]*trialRuns[c])

    # create your data dict, transform from log odds to odds ratio
    data = {}
    for c, i in enumerate(psy.weight_labels):
        # adding multiplication step here with binary vector
        data[i] = np.exp(psy.fits[c, :])*psy.inputs[:, c].T
    ori_0_in = [i[0] for i in psy.data['inputs']['ori_0']]
    ori_135_in = [i[0] for i in psy.data['inputs']['ori_135']]
    ori_270_in = [i[0] for i in psy.data['inputs']['ori_270']]
    blank_in = [0 if i == 1 else 1 for i in
                np.sum((ori_0_in, ori_135_in, ori_270_in), axis=0)]

    # create a single list of orientations to match format of meta
    ori_order = [0, 135, 270, -1]
    data['orientation'] = [
        ori_order[np.where(np.isin(i, 1))[0][0]]
        for i in zip(ori_0_in, ori_135_in, ori_270_in, blank_in)]

    # create your index out of relevant variables
    index = pd.MultiIndex.from_arrays([
                [mouse]*len(trial_idx),
                date_vec,
                run_vec,
                trial_idx
                ],
                names=['mouse', 'date', 'run', 'trial_idx'])

    dfr = pd.DataFrame(data, index=index)

    # Load TCA results
    tensor, ids, clus, meta = load.groupday_tca(
        mouse, word=word, group_by=group_by)
    savepath = paths.tca_plots(
        mouse, 'group', word=word, group_pars={'group_by': group_by})
    savepath = os.path.join(savepath, 'psytrack-vs-tca')
    if not os.path.isdir(savepath): os.mkdir(savepath)

    # Load smooth dprime
    dfr['dprime'] = pool.calc.psytrack.dprime(flow.Mouse(mouse))


    # CHECK THAT TRIAL INDICES ARE MATCHED AND HAVE MATCHED ORIS

    # filter out blank trials
    psy_df = dfr.loc[(dfr['orientation'] >= 0), :]

    # check that all runs have matched trial orienations
    new_psy_df_list = []
    new_meta_df_list = []
    dates = meta.reset_index()['date'].unique()
    for d in dates:
        psy_day_bool = psy_df.reset_index()['date'].isin([d]).values
        meta_day_bool = meta.reset_index()['date'].isin([d]).values
        psy_day_df = psy_df.iloc[psy_day_bool, :]
        meta_day_df = meta.iloc[meta_day_bool, :]
        runs = meta_day_df.reset_index()['run'].unique()
        for r in runs:
            psy_run_bool = psy_day_df.reset_index()['run'].isin([r]).values
            meta_run_bool = meta_day_df.reset_index()['run'].isin([r]).values
            psy_run_df = psy_day_df.iloc[psy_run_bool, :]
            meta_run_df = meta_day_df.iloc[meta_run_bool, :]
            psy_run_idx = psy_run_df.reset_index()['trial_idx'].values
            meta_run_idx = meta_run_df.reset_index()['trial_idx'].values

            # drop extra trials from trace2P that don't have associated imaging 
            max_trials = np.min([len(psy_run_idx), len(meta_run_idx)])

            # get just your orientations for checking that trials are matched
            meta_ori = meta_run_df['orientation'].iloc[:max_trials]
            psy_ori = psy_run_df['orientation'].iloc[:max_trials]

            # make sure all oris match between vectors of the same length each day
            assert np.all(psy_ori.values == meta_ori.values)

            # if everything looks good, copy meta index into psy
            meta_new = meta_run_df.iloc[:max_trials]
            psy_new = psy_run_df.iloc[:max_trials]
            data = {}
            for i in psy_new.columns:
                data[i] = psy_new[i].values
            new_psy_df_list.append(pd.DataFrame(data=data, index=meta_new.index))
            new_meta_df_list.append(meta_new)

    meta1 = pd.concat(new_meta_df_list, axis=0)
    psy1 = pd.concat(new_psy_df_list, axis=0)

    # NOW TAKE TCA TRIAL FACTORS AND TRY CORRELATING FOR WITH PILLOW
    # put factors for a given rank into a dataframe

    ori = 'all'
    save_pls = True
    iteration = 0
    for rank in tensor.results:
        data = {}
        for i in range(rank):
            fac = tensor.results[rank][iteration].factors[2][:,i]
            data['factor_' + str(i+1)] = fac
        fac_df = pd.DataFrame(data=data, index=meta1.index)

        # loop over single oris
        single_ori = pd.concat([psy1, fac_df], axis=1).drop(columns='orientation')
        corr = np.corrcoef(single_ori.values.T)

        single_data ={}
        for c, i in enumerate(single_ori.columns):
            single_data[i] = corr[:, c]
        corr_plt = pd.DataFrame(data=single_data, index=single_ori.columns)

        num_corr = np.shape(single_ori)[1]
        corrmat = np.zeros((num_corr, num_corr))
        pmat = np.zeros((num_corr, num_corr))
        for i in range(num_corr):
            for k in range(num_corr):
                corA, corP = pearsonr(single_ori.values[:, i], single_ori.values[:, k])
                corrmat[i, k] = corA
                pmat[i, k] = corP

        labels = single_ori.columns
        annot = True
        figsize = (16, 16)

        # create your path for saving
        rankpath = os.path.join(savepath, 'rank ' + str(rank))
        if not os.path.isdir(rankpath): os.mkdir(rankpath)
        var_path_prefix = os.path.join(
            rankpath, mouse + '_psytrack-vs-tca_ori-' + str(ori) +
            '_rank-' + str(rank))

        plt.figure(figsize=figsize)
        sns.heatmap(corrmat, annot=annot, xticklabels=labels,
                    yticklabels=labels,
                    square=True, cbar_kws={'label': 'correlation (R)'})
        plt.xticks(rotation=45, ha='right')
        plt.title('Pearson-R corrcoef: rank ' + str(rank))
        if save_pls:
            plt.savefig(var_path_prefix + '_corr.pdf', bbox_inches='tight')

        plt.figure(figsize=figsize)
        sns.heatmap(pmat, annot=annot, xticklabels=labels, yticklabels=labels,
                    square=True, cbar_kws={'label': 'p-value'})
        plt.xticks(rotation=45, ha='right')
        plt.title('Pearson-R p-values: rank ' + str(rank))
        if save_pls:
            plt.savefig(var_path_prefix + '_pvals.pdf', bbox_inches='tight')

        plt.figure(figsize=figsize)
        logger = np.log10(pmat).flatten()
        vmin = np.nanmin(logger[np.isfinite(logger)])
        vmax = 0
        sns.heatmap(np.log10(pmat), annot=annot, xticklabels=labels,
                    yticklabels=labels, vmin=vmin, vmax=vmax,
                    square=True, cbar_kws={'label': 'log$_{10}$(p-value)'})
        plt.xticks(rotation=45, ha='right')
        plt.title('Pearson-R log$_{10}$(p-values): rank ' + str(rank))
        if save_pls:
            plt.savefig(var_path_prefix + '_log10pvals.pdf',
                        bbox_inches='tight')

        # close plots after saving to save memory
        plt.close('all')


def groupmouse_correlate_pillow_tca(
        mice=['OA27', 'OA32', 'OA34', 'CC175', 'OA36', 'OA26', 'OA67',
              'VF226'],
        words=['orlando', 'already', 'already', 'already', 'already',
               'already', 'already', 'already'],
        group_by='all'):

    # preallocate
    corr_list = []
    pmat_list = []
    x_labels = []

    # LOADING
    for mouse, word in zip(mice, words):
        # Load or run your psytracker behavioral model
        ms = flow.Mouse(mouse)
        psy = ms.psytracker(verbose=True)
        dateRuns = psy.data['dateRuns']
        trialRuns = psy.data['runLength']

        # create your trial indices per day and run
        trial_idx = []
        for i in trialRuns:
            trial_idx.extend(range(i))

        # get date and run vectors
        date_vec = []
        run_vec = []
        for c, i in enumerate(dateRuns):
            date_vec.extend([i[0]]*trialRuns[c])
            run_vec.extend([i[1]]*trialRuns[c])

        # create your data dict, transform from log odds to odds ratio
        data = {}
        for c, i in enumerate(psy.weight_labels):
            # adding multiplication step here with binary vector
            data[i] = np.exp(psy.fits[c, :])*psy.inputs[:, c].T
        ori_0_in = [i[0] for i in psy.data['inputs']['ori_0']]
        ori_135_in = [i[0] for i in psy.data['inputs']['ori_135']]
        ori_270_in = [i[0] for i in psy.data['inputs']['ori_270']]
        blank_in = [0 if i == 1 else 1 for i in
                    np.sum((ori_0_in, ori_135_in, ori_270_in), axis=0)]

        # create a single list of orientations to match format of meta
        ori_order = [0, 135, 270, -1]
        data['orientation'] = [
            ori_order[np.where(np.isin(i, 1))[0][0]]
            for i in zip(ori_0_in, ori_135_in, ori_270_in, blank_in)]

        # create your index out of relevant variables
        index = pd.MultiIndex.from_arrays([
                    [mouse]*len(trial_idx),
                    date_vec,
                    run_vec,
                    trial_idx
                    ],
                    names=['mouse', 'date', 'run', 'trial_idx'])

        dfr = pd.DataFrame(data, index=index)

        # Load TCA results
        tensor, ids, clus, meta = load.groupday_tca(
            mouse, word=word, group_by=group_by)
        savepath = paths.tca_plots(
            mouse, 'group', word=word, group_pars={'group_by': group_by})
        savepath = os.path.join(savepath, 'psytrack-vs-tca')
        if not os.path.isdir(savepath): os.mkdir(savepath)

        # Load smooth dprime
        dfr['dprime'] = pool.calc.psytrack.dprime(flow.Mouse(mouse))


        # CHECK THAT TRIAL INDICES ARE MATCHED AND HAVE MATCHED ORIS

        # filter out blank trials
        psy_df = dfr.loc[(dfr['orientation'] >= 0), :]

        # check that all runs have matched trial orienations
        new_psy_df_list = []
        new_meta_df_list = []
        dates = meta.reset_index()['date'].unique()
        for d in dates:
            psy_day_bool = psy_df.reset_index()['date'].isin([d]).values
            meta_day_bool = meta.reset_index()['date'].isin([d]).values
            psy_day_df = psy_df.iloc[psy_day_bool, :]
            meta_day_df = meta.iloc[meta_day_bool, :]
            runs = meta_day_df.reset_index()['run'].unique()
            for r in runs:
                psy_run_bool = psy_day_df.reset_index()['run'].isin([r]).values
                meta_run_bool = meta_day_df.reset_index()['run'].isin([r]).values
                psy_run_df = psy_day_df.iloc[psy_run_bool, :]
                meta_run_df = meta_day_df.iloc[meta_run_bool, :]
                psy_run_idx = psy_run_df.reset_index()['trial_idx'].values
                meta_run_idx = meta_run_df.reset_index()['trial_idx'].values

                # drop extra trials from trace2P that don't have associated imaging 
                max_trials = np.min([len(psy_run_idx), len(meta_run_idx)])

                # get just your orientations for checking that trials are matched
                meta_ori = meta_run_df['orientation'].iloc[:max_trials]
                psy_ori = psy_run_df['orientation'].iloc[:max_trials]

                # make sure all oris match between vectors of the same length each day
                assert np.all(psy_ori.values == meta_ori.values)

                # if everything looks good, copy meta index into psy
                meta_new = meta_run_df.iloc[:max_trials]
                psy_new = psy_run_df.iloc[:max_trials]
                data = {}
                for i in psy_new.columns:
                    data[i] = psy_new[i].values
                new_psy_df_list.append(pd.DataFrame(data=data, index=meta_new.index))
                new_meta_df_list.append(meta_new)

        meta1 = pd.concat(new_meta_df_list, axis=0)
        psy1 = pd.concat(new_psy_df_list, axis=0)

        # NOW TAKE TCA TRIAL FACTORS AND TRY CORRELATING FOR WITH PILLOW
        # put factors for a given rank into a dataframe

        ori = 'all'
        save_pls = False
        iteration = 0
        for rank in [18]:  # tensor.results
            data = {}
            for i in range(rank):
                fac = tensor.results[rank][iteration].factors[2][:,i]
                data['factor_' + str(i+1)] = fac
            fac_df = pd.DataFrame(data=data, index=meta1.index)

            # loop over single oris
            single_ori = pd.concat([psy1, fac_df], axis=1).drop(columns='orientation')
            corr = np.corrcoef(single_ori.values.T)

            single_data ={}
            for c, i in enumerate(single_ori.columns):
                single_data[i] = corr[:, c]
            corr_plt = pd.DataFrame(data=single_data, index=single_ori.columns)

            num_corr = np.shape(single_ori)[1]
            corrmat = np.zeros((num_corr, num_corr))
            pmat = np.zeros((num_corr, num_corr))
            for i in range(num_corr):
                for k in range(num_corr):
                    corA, corP = pearsonr(single_ori.values[:, i], single_ori.values[:, k])
                    corrmat[i, k] = corA
                    pmat[i, k] = corP

            if mouse == mice[0]:
                y_label = single_ori.columns

        # stick chunks of corr matrix together
        x_labels.append([mouse + ' ' + s for s in single_ori.columns[0:7]])
        corr_list.append(corrmat[:, 0:7])
        pmat_list.append(pmat[:, 0:7])

    # concatenate final matrix together
    corrmat = np.concatenate(corr_list, axis=1)
    pmat = np.concatenate(pmat_list, axis=1)
    annot = True
    figsize = (16, 16)

    # create your path for saving
    rankpath = os.path.join(savepath, 'rank ' + str(rank))
    if not os.path.isdir(rankpath): os.mkdir(rankpath)
    var_path_prefix = os.path.join(
        rankpath, mouse + '_psytrack-vs-tca_ori-' + str(ori) +
        '_rank-' + str(rank))

    plt.figure(figsize=figsize)
    sns.heatmap(corrmat, annot=annot, xticklabels=x_labels,
                yticklabels=y_label,
                square=True, cbar_kws={'label': 'correlation (R)'})
    plt.xticks(rotation=45, ha='right')
    plt.title('Pearson-R corrcoef: rank ' + str(rank))
    if save_pls:
        plt.savefig(var_path_prefix + '_corr.pdf', bbox_inches='tight')

    plt.figure(figsize=figsize)
    sns.heatmap(pmat, annot=annot, xticklabels=x_labels, yticklabels=y_label,
                square=True, cbar_kws={'label': 'p-value'})
    plt.xticks(rotation=45, ha='right')
    plt.title('Pearson-R p-values: rank ' + str(rank))
    if save_pls:
        plt.savefig(var_path_prefix + '_pvals.pdf', bbox_inches='tight')

    plt.figure(figsize=figsize)
    logger = np.log10(pmat).flatten()
    vmin = np.nanmin(logger[np.isfinite(logger)])
    vmax = 0
    sns.heatmap(np.log10(pmat), annot=annot, xticklabels=y_labels,
                yticklabels=x_labels, vmin=vmin, vmax=vmax,
                square=True, cbar_kws={'label': 'log$_{10}$(p-value)'})
    plt.xticks(rotation=45, ha='right')
    plt.title('Pearson-R log$_{10}$(p-values): rank ' + str(rank))
    if save_pls:
        plt.savefig(var_path_prefix + '_log10pvals.pdf',
                    bbox_inches='tight')

    # close plots after saving to save memory
    plt.close('all')

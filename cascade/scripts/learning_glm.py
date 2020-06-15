import cascade as cas
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.model_selection import KFold
from statsmodels.distributions.empirical_distribution import ECDF

# set parameters for running GLM
mice = cas.lookups.mice['all15']
words = ['facilitate'] * len(mice)
trace_type = 'zscore_day'
folder_name = 'cvGLM trial history no_p'
group_by = 'learning'
nan_thresh = 0.95
score_threshold = 0.8

# set rank
for rank_num in [10, 15]:

    # save folder creation
    save_folder = cas.paths.save_dir_groupmouse(
        mice,
        folder_name,
        method='ncp_hals',
        nan_thresh=nan_thresh,
        score_threshold=score_threshold,
        pars=None,
        words=words,
        rank_num=rank_num,
        grouping='group',
        group_pars={'group_by': group_by})

    df_list_fdev = []
    df_list_ddev = []
    total_model_devex = []
    for m, wi, in zip(mice, words):

        # breathing room
        print(' \n')

        # get X and y inputs for GLM
        pars = cas.psytrack.train_factor.default_pars
        psy1_df, meta1_df, fac1_df, _ = cas.psytrack.train_factor.sync_tca_pillow(
            m, word=wi, nan_thresh=nan_thresh, score_threshold=score_threshold, group_by=group_by,
            rank_num=rank_num, trace_type=trace_type, **pars)

        # drop ['prev_choice', 'prev_punish', 'prev_reward'] from psy1 so it can be redefined
        psy1_df = psy1_df.drop(columns=['prev_choice', 'prev_punish', 'prev_reward'])

        # add a binary column for possible category/semantic representations
        new_meta_df1, p_cols, i_cols, cs_cols = cas.glm.learning_columns_df(m, meta1_df)

        # add pillow interaction terms
        new_meta_df1, pillow_cols = cas.glm.add_pillow_interactions(m, new_meta_df1, psy1_df)

        # add engagement from HMM
        new_meta_df1 = cas.glm._add_hmm_to_design_mat(new_meta_df1, meta1_df)

        # start design matrix
        design_X = pd.concat([new_meta_df1.loc[:, p_cols],
                              new_meta_df1.loc[:, cs_cols],
                              new_meta_df1.loc[:, ['choice', 'reward']],
                              new_meta_df1.loc[:, ['hmm_engaged']],
                              new_meta_df1.loc[:, pillow_cols],
                              new_meta_df1.loc[:, i_cols]
                              ], axis=1)

        # add acceleration and change in lick rate to model
        design_X['speed_delta'] = meta1_df['speed'] - meta1_df['pre_speed']
        design_X['lick_delta'] = meta1_df['anticipatory_licks' ] /cas.lookups.stim_length[m] - meta1_df['pre_licks']  # prestim is 1 sec

        # make sure to remove later reversals from fits so fits are only calculated over a single reversal
        ls_bool = meta1_df['learning_state'].isin(['learning', 'reversal1'])
        psy1_df = psy1_df.loc[ls_bool, :]
        meta1_df = meta1_df.loc[ls_bool, :]
        fac1_df = fac1_df.loc[ls_bool, :]
        design_X = design_X.loc[ls_bool, :]

        # create design matrix, X
        X = pd.concat([design_X, meta1_df.loc[:, ['anticipatory_licks', 'speed', 'firstlickbout']]], axis=1)

        # warn user against the use of pupil in models
        if 'pupil' in X.columns:
            print('    !!!!!!!')
            print('    Heads up, Arthurs animals mostly dont have pupil tracking!')
            print('    !!!!!!!')

        # missing licks are set as nans right now. This interaction term needs a value otherwise it will blow things up.
        # Close of reward period is 93 for 3 sec, and 77.5 for 2 sec stim.
        # Set values above this to an arbitrary point in the ITI 2 seconds later
        if m not in ['OA27', 'OA26', 'OA67', 'VF226', 'CC175']:
            response_closes = 93
        else:
            response_closes = 77.5
        late_or_none = X['firstlickbout'].gt(response_closes) | X['firstlickbout'].isna()
        X.loc[late_or_none, 'firstlickbout'] = response_closes + 15.5 * 2

        # add in interaction terms for all three cues
        new_meta = {}
        base_cols = ['anticipatory_licks', 'speed', 'firstlickbout']
        interaction_cols = ['initial_plus', 'initial_minus', 'initial_neutral', 'cs_plus', 'cs_minus', 'cs_neutral']
        inter_cols = []
        for rep in base_cols:
            for inter in interaction_cols:
                new_meta['{}_x_{}'.format(rep, inter)] = X[rep].values * X[inter].values
                inter_cols.append('{}_x_{}'.format(rep, inter))
        new_meta_df = pd.DataFrame(data=new_meta, index=X.index)
        X = pd.concat([X, new_meta_df], axis=1)

        # drop 'initial_plus', 'initial_minus', 'initial_neutral'
        X.drop(columns=i_cols, inplace=True)
        X.drop(columns=cs_cols, inplace=True)
        X.drop(columns=pillow_cols, inplace=True)

        # z-score each column
        X = X.transform(lambda x: (x - x.mean()) / x.std())
        col_bool = np.sum(X.isna().values, axis=0) < X.shape[0]
        col_start = deepcopy(X.columns)
        X = X.iloc[:, col_bool]
        kept_cols = X.columns
        X_bool = (~X.isna().any(axis=1))
        X = X.loc[X_bool].values
        Y = fac1_df.loc[X_bool].values

        # make sure you aren't losing all rows
        assert np.sum(X_bool) > 0
        print('{}/{} trials in X after removing nans.'.format(np.sum(X_bool), len(X_bool)))
        print(f'    fitting: {col_start}')

        # train-validation-test split
        frac_train = 0.75  # 75% train, 12.5% validation, 12.5% test
        np.random.seed(seed = 42) # seed random state
        nVTs = X.shape[0]
        nSelected = np.round(nVTs *(1.-frac_train)).astype(int)
        selectedInd = np.random.choice(np.arange(nVTs), nSelected, replace=False)
        nTest = np.round(nSelected /2).astype(int)
        testInd = selectedInd[:nTest]
        valInd = selectedInd[nTest:]
        trainInd = np.array(list(set(np.arange(nVTs) ) -set(selectedInd)))
        X_train = X[trainInd ,:]
        Y_train = Y[trainInd ,:]
        X_test = X[testInd ,:]
        Y_test = Y[testInd ,:]
        X_val = X[valInd ,:]
        Y_val = Y[valInd ,:]

        # fit with CV
        n_folds = 5
        fit_param_dict = {'loss_type': 'poisson', 'activation': 'exp',
                          'lambda_series' :10.0 ** np.linspace(-1, -7, 25),
                          'regularization': 'elastic_net', 'l1_ratio': 0.5,
                          'learning_rate': 1e-3, 'device': '/cpu:0'}

        # train-vtest split
        frac_train = 0.8 # 80% train, 20% test
        np.random.seed(seed = 42) # seed random state
        nVTs = X.shape[0]
        nTest = np.round(nVTs *(1.-frac_train)).astype(int)
        testInd = np.random.choice(np.arange(nVTs), nTest, replace=False)
        fitInd = np.array(list(set(np.arange(nVTs))-set(testInd)))
        X_fit = X[fitInd ,:]
        Y_fit = Y[fitInd ,:]
        X_test = X[testInd ,:]
        Y_test = Y[testInd ,:]

        # CV setting on fit data
        kf = KFold(n_splits = n_folds, shuffle = True, random_state = 42)

        # get train and validation indices
        train_ind = {n_folds:np.arange(Y_fit.shape[0])}  # store frames for fit data as the n-th fold
        val_ind = {}
        for n_fold, (trainInd, valInd) in enumerate(kf.split(fitInd)):
            # find train vs. test frames and split data
            train_ind[n_fold] = trainInd
            val_ind[n_fold] = valInd

        # fit GLM
        w_series_dict, lambda_series, loss_trace_dict, lambda_trace_dict, all_prediction, all_deviance = cas.cvglm.fit_glm_cv(
            Y_fit, X_fit, [], n_folds, train_ind, val_ind, **fit_param_dict)

        # get fit quality on CV validation data (pooled across n folds)
        all_fit_qual_cv, _, _ = cas.cvglm.calculate_fit_quality_cv(lambda_series, all_prediction, Y_fit,
                                                                   loss_type = fit_param_dict['loss_type'],
                                                                   activation = fit_param_dict['activation'],
                                                                   make_fig = True)

        # get fit quality on training data of the full model
        all_fit_qual_full, _, _, _ = cas.cvglm.calculate_fit_quality(w_series_dict[n_folds], lambda_series, X_fit, Y_fit,
                                                                     loss_type = fit_param_dict['loss_type'],
                                                                     activation = fit_param_dict['activation'], make_fig = True)

        # get fit quality on test data using the weights from the full model
        all_fit_qual_test, _, _, _ = cas.cvglm.calculate_fit_quality(w_series_dict[n_folds], lambda_series, X_test, Y_test,
                                                                     loss_type = fit_param_dict['loss_type'],
                                                                     activation = fit_param_dict['activation'], make_fig = True)

        #### SelectWeight Part Table: Model selection
        se_fraction = 1
        all_dev, all_w, all_w0, all_lambda, all_lambda_ind = cas.cvglm.select_model_cv(w_series_dict, lambda_series,
                                                                                       all_deviance, n_folds, se_fraction,
                                                                                       all_fit_qual_cv, make_fig = False)
        selected_fit_qual_full = [all_fit_qual_full[lam_idx, idx] for idx, lam_idx in enumerate(all_lambda_ind)]
        selected_fit_qual_test = [all_fit_qual_test[lam_idx, idx] for idx, lam_idx in enumerate(all_lambda_ind)]

        # grab the last figure created by CV and save
        plt.title(f'{m}')
        plt.savefig(os.path.join(save_folder, f'{m}_cv_model_selection.png'), bbox_inches='tight')
        plt.close('all')

        # plot cumulative deviance explained by models
        ecdf = ECDF(selected_fit_qual_test)
        x = np.arange(0 ,1 ,0.01)
        this_ecdf = ecdf(x)
        print('Mean deviance explained =', np.mean(selected_fit_qual_test))

        # save a fraction of deviance explained summary
        fig, axes = plt.subplots(1 ,1, figsize = (5 ,5))
        axes.plot(x ,this_ecdf)
        axes.set_xlabel('Fraction deviance explained')
        axes.set_ylabel('Cumulative density');
        axes.set_title(f'{m}')
        plt.savefig(os.path.join(save_folder, f'{m}_deviance_explained_best_cv_model.png'), bbox_inches='tight')
        plt.close('all')

        # calculate deviance explained from zeroed/ablated reconstructions (predictions) for each w and w0
        ddev_per_y = []
        fdev_per_y = []
        full_dev = []
        full_w = []
        for n_factor in range(Y_test.shape[1]):
            y = Y_test[:, n_factor]
            w = all_w[n_factor]
            w0 = all_w0[n_factor]
            fdev_per_y_w = []
            ddev_per_y_w =[]
            for each_ablation in range(X_test.shape[1]):
                X_ablated = deepcopy(X_test)
                X_ablated[:, each_ablation] = 0
                mu_full = cas.cvglm.make_prediction(X_test, w, w0)
                mu = cas.cvglm.make_prediction(X_ablated, w, w0)  # this should be done on test data
                ddev_per_y_w.append(cas.cvglm.deviance(mu_full, y)[0] - cas.cvglm.deviance(mu, y)[0])
                fdev_per_y_w.append(1 - cas.cvglm.deviance(mu, y)[0] / cas.cvglm.deviance(mu_full, y)[0])
            ddev_per_y.append(ddev_per_y_w)
            fdev_per_y.append(fdev_per_y_w)
            full_dev.append(cas.cvglm.deviance(mu_full, y)[0])
            full_w.append(w)

        # reshape into array
        fac_dev_breakdown = np.stack(ddev_per_y, axis=1)

        # create your index out of relevant variables
        index = pd.MultiIndex.from_arrays([
            [m] * len(full_w),
            np.arange(1, len(full_w) + 1)
        ],
            names=['mouse', 'component'])
        data = {k: fac_dev_breakdown[c, :] for c, k in enumerate(kept_cols)}
        devex_df = pd.DataFrame(data=data, index=index)
        df_list_ddev.append(devex_df)

        # reshape into array
        fac_dev_breakdown = np.stack(fdev_per_y, axis=1)

        # create your index out of relevant variables
        index = pd.MultiIndex.from_arrays([
            [m] * len(full_w),
            np.arange(1, len(full_w) + 1)
        ],
            names=['mouse', 'component'])
        data = {k: fac_dev_breakdown[c, :] for c, k in enumerate(kept_cols)}
        devex_df = pd.DataFrame(data=data, index=index)
        df_list_fdev.append(devex_df)
        total_model_devex.append(pd.DataFrame(data={
            'beta_w0': all_w0,
            'beta_w': all_w,
            'total_model_devex_test': selected_fit_qual_test,
            'total_model_devex_full': selected_fit_qual_full}, index=index))

    all_mddev_df = pd.concat(df_list_ddev, axis=0, sort=False)
    all_mfdev_df = pd.concat(df_list_fdev, axis=0, sort=False)
    all_mdev_df = pd.concat(total_model_devex, axis=0, sort=False)

    # save
    all_mddev_df.to_pickle(os.path.join(save_folder, 'cvGLM_delta_deviance_df.npy'))
    all_mfdev_df.to_pickle(os.path.join(save_folder, 'cvGLM_fractional_deviance_df.npy'))
    all_mdev_df.to_pickle(os.path.join(save_folder, 'cvGLM_model_performance_df.npy'))

    print('done!')
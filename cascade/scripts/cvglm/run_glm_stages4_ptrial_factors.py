from seaborn.axisgrid import jointplot
import cascade as cas
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.model_selection import KFold
from statsmodels.distributions.empirical_distribution import ECDF

from tqdm import tqdm
import argparse
import logging
from datetime import datetime


# run as a big ol' function
def main(version='_v2_avg', loss_type='poisson', activation='exp', force=False, verbose=False, test=False):
    # parameters
    # --------------------------------------------------------------------------------------------------

    # input data params
    group_by = 'all3'
    nan_thresh = 0.95

    # TCA params
    models = ['v4i10_norm_on_noT0', 'v4i10_scale_on_noT0', 'v4i10_norm_off_noT0', 'v4i10_scale_off_noT0']
    ranks = [9, 9, 8, 8]
    # if testing only run a single model
    if test:
        models = ['v4i10_norm_on_noT0']
        ranks = [9]
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d_%H%M")
        glm_tag = f'_test_{loss_type}_{activation}_{dt_string}'
    else:
        glm_tag = f'_{loss_type}_{activation}'

    # GLM params
    n_folds = 5
    fit_param_dict = {
        'loss_type': loss_type,
        'activation': activation,
        'lambda_series': 10.0**np.linspace(-1, -5, 25),
        'regularization': 'elastic_net',
        'l1_ratio': 0.5,
        'learning_rate': 1e-3,
        'device': '/cpu:0'
    }
    frac_train = 0.8  # 80% train, 20% test # train-test split

    # load in trial factors
    # --------------------------------------------------------------------------------------------------
    model_list = []
    for mod, rr in zip(models, ranks):
        # save all results
        path = cas.paths.analysis_file(f'{mod}_fit_results_rank{rr}_facs{version}.npy',
                                       f'tca_dfs/TCA_factor_fitting{version}/{mod}')

        # {
        #     'pseudo_ktensors': mouse_kt_list,
        #     'pseudo_trialfactors': mouse_tfac_list,
        #     'mice': np.unique(mouse_vec)
        # },
        model_dict = np.load(path, allow_pickle=True).item()
        model_list.append(model_dict)

    # load in a full size data
    # --------------------------------------------------------------------------------------------------
    meta_list = []
    mice = model_list[0]['mice']
    words = ['respondent' if s in 'OA27' else 'computation' for s in mice]
    for mouse, word in zip(mice, words):

        # return   ids, tensor, meta, bhv
        out = cas.load.groupday_tca_meta(mouse, word=word, group_by=group_by, nan_thresh=nan_thresh)
        meta_list.append(cas.utils.add_stages_to_meta(out, 'parsed_11stage'))

    # make your design matrix
    # --------------------------------------------------------------------------------------------------
    design_mat_list = []
    for meta in meta_list:

        design_df = cas.glm.design_matrix_df(meta)

        # add in interaction terms for the three cues
        new_meta = {}
        interaction_cols = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']
        base_cols = [s for s in design_df.columns if s not in interaction_cols]
        base_cols = [s for s in base_cols if 'prev_same' not in s]  # previous same only makes sense with its given cue
        base_cols = [s for s in base_cols if 'prev_diff' not in s]

        inter_cols = []
        for rep in base_cols:
            for inter in interaction_cols:
                new_meta[f'{rep}_x_{inter}'] = design_df[rep].values * design_df[inter].values
                inter_cols.append(f'{rep}_x_{inter}')
        new_meta_df = pd.DataFrame(data=new_meta, index=design_df.index)
        X = pd.concat([design_df, new_meta_df], axis=1)
        X.drop(columns=interaction_cols, inplace=True)
        assert not X.isna().any().any()
        design_mat_list.append(X)

    # reorganize rescale (semi-zscore) your model trial factors to create your Y matrix
    # --------------------------------------------------------------------------------------------------
    y_mat_mod_list = []
    for mod in model_list:  # for your 4 models
        y_mat_list = []
        for meta, mouse_mod in zip(meta_list, mod['pseudo_trialfactors']):  # for your 7 mice
            new_y = {}
            for compi in range(mouse_mod.shape[1]):
                # semi-zscore --> divide by std (but don't center data, keep nonnegative)
                new_y[f'component_{int(compi + 1)}'] = mouse_mod[:, compi] / np.nanstd(mouse_mod[:, compi])
            y_df = pd.DataFrame(data=new_y, index=meta.index)
            assert not y_df.isna().any().any()
            y_mat_list.append(y_df)
        y_mat_mod_list.append(y_mat_list)

    # add staging/binning column to meta, check the patency of your design mat, drop empty filters
    # --------------------------------------------------------------------------------------------------
    bad_col_agg = []
    for meta, desi in zip(meta_list, design_mat_list):
        new_stage_vec = np.zeros(len(meta)) + np.nan
        new_names = ['early_learning', 'late_learning', 'early_reversal', 'late_reversal']
        stage_sets = [
            ['L1 learning', 'L2 learning', 'L3 learning'],
            ['L4 learning', 'L5 learning'],
            ['L1 reversal1', 'L2 reversal1', 'L3 reversal1'],
            ['L4 reversal1', 'L5 reversal1'],
        ]
        mouse = cas.utils.meta_mouse(meta)
        for c, sset in enumerate(stage_sets):
            meta_bool = meta.parsed_11stage.isin(sset)
            new_stage_vec[meta_bool] = c
            col_sums = desi.loc[meta_bool].sum(axis=0)
            bad_cols = col_sums[col_sums == 0].keys().to_list()
            # only count a bad column if you have trials for that stage in the first place
            if len(bad_cols) > 0 and meta_bool.sum() > 0:
                bad_col_agg.extend(bad_cols)
        new_stage_col = [new_names[int(s)] if not np.isnan(s) else 'stageless' for s in new_stage_vec]
        meta['parsed_4stage'] = new_stage_col

    # drop columns that are missing for a stage (but only when that stage had trials)
    new_design_mat_list = []
    for desi in design_mat_list:
        new_X = desi.drop(columns=bad_col_agg)
        new_design_mat_list.append(new_X)
    design_mat_list = new_design_mat_list  # overwrite original list

    # fit GLM
    # --------------------------------------------------------------------------------------------------
    for mod, rr, y_mat_list in zip(models, ranks, y_mat_mod_list):

        df_list_fdev = []
        df_list_ddev = []
        total_model_devex = []
        save_folder = cas.paths.analysis_dir(f'tca_dfs/TCA_factor_fitting{version}/{mod}/cvglm_stages{glm_tag}')

        # check if the model has already been run
        check_file = os.path.join(save_folder, f'{mod}_rank{rr}_cvGLM_delta_deviance_df.pkl')
        if os.path.isfile(check_file) and not force:
            if verbose:
                print(f'{check_file} exists. Skipping GLM.')
            continue

        logger = create_logger(save_folder)
        logger.info(f'Starting fitting for {mod} rank {rr} {version} {loss_type} {activation}')
        if len(bad_col_agg) > 0:
            [logger.error(f'Dropped {s}, column patency') for s in bad_col_agg]

        for Xdf, Ydf, meta in tqdm(zip(design_mat_list, y_mat_list, meta_list), desc=f'{mod} GLM', total=len(y_mat_list)):

            kept_cols = Xdf.columns

            # set mouse
            mouse = cas.utils.meta_mouse(Xdf)
            if test:
                # only run on a single mouse for testing
                if mouse not in ['OA27']:
                    continue

            for stagi in ['early_learning', 'late_learning', 'early_reversal', 'late_reversal']:
                
                meta_bool = meta.parsed_4stage.isin([stagi])
                if meta_bool.sum() == 0:
                    logger.error(f'{mouse}: skipped {stagi}, no trials')
                    continue
                else:
                    logger.info(f'{mouse}: {stagi}, {meta_bool.sum()} trials')
                Xdf_sub = Xdf.loc[meta_bool]
                Ydf_sub = Ydf.loc[meta_bool]

                # set data for run
                kept_cols = Xdf.columns
                assert Xdf_sub.sum().gt(0).all()  # columns shouldn't be empty!
                X = Xdf_sub.values
                Y = Ydf_sub.values

                # train-validation-test split
                np.random.seed(seed=42)  # seed random state
                nVTs = X.shape[0]
                nTest = np.round(nVTs * (1. - frac_train)).astype(int)
                testInd = np.random.choice(np.arange(nVTs), nTest, replace=False)
                fitInd = np.array(list(set(np.arange(nVTs)) - set(testInd)))
                X_fit = X[fitInd, :]
                Y_fit = Y[fitInd, :]
                X_test = X[testInd, :]
                Y_test = Y[testInd, :]

                # CV setting on fit data
                kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

                # get train and validation indices
                train_ind = {n_folds: np.arange(Y_fit.shape[0])}  # store frames for fit data as the n-th fold
                val_ind = {}
                for n_fold, (trainInd, valInd) in enumerate(kf.split(fitInd)):
                    # find train vs. test frames and split data
                    train_ind[n_fold] = trainInd
                    val_ind[n_fold] = valInd

                # fit GLM
                w_series_dict, lambda_series, loss_trace_dict, lambda_trace_dict, all_prediction, all_deviance = cas.cvglm.fit_glm_cv(
                    Y_fit, X_fit, [], n_folds, train_ind, val_ind, **fit_param_dict)

                # get fit quality on CV validation data (pooled across n folds)
                all_fit_qual_cv, _, _ = cas.cvglm.calculate_fit_quality_cv(lambda_series,
                                                                        all_prediction,
                                                                        Y_fit,
                                                                        loss_type=fit_param_dict['loss_type'],
                                                                        activation=fit_param_dict['activation'],
                                                                        make_fig=True)

                # get fit quality on training data of the full model
                all_fit_qual_full, _, _, _ = cas.cvglm.calculate_fit_quality(w_series_dict[n_folds],
                                                                            lambda_series,
                                                                            X_fit,
                                                                            Y_fit,
                                                                            loss_type=fit_param_dict['loss_type'],
                                                                            activation=fit_param_dict['activation'],
                                                                            make_fig=True)

                # get fit quality on test data using the weights from the full model
                all_fit_qual_test, _, _, _ = cas.cvglm.calculate_fit_quality(w_series_dict[n_folds],
                                                                            lambda_series,
                                                                            X_test,
                                                                            Y_test,
                                                                            loss_type=fit_param_dict['loss_type'],
                                                                            activation=fit_param_dict['activation'],
                                                                            make_fig=True)

                #### SelectWeight Part Table: Model selection
                se_fraction = 1
                all_dev, all_w, all_w0, all_lambda, all_lambda_ind = cas.cvglm.select_model_cv(w_series_dict,
                                                                                            lambda_series,
                                                                                            all_deviance,
                                                                                            n_folds,
                                                                                            se_fraction,
                                                                                            all_fit_qual_cv,
                                                                                            make_fig=False)
                selected_fit_qual_full = [all_fit_qual_full[lam_idx, idx] for idx, lam_idx in enumerate(all_lambda_ind)]
                selected_fit_qual_test = [all_fit_qual_test[lam_idx, idx] for idx, lam_idx in enumerate(all_lambda_ind)]

                # grab the last figure created by CV and save
                plt.title(f'{mouse}')
                plt.savefig(os.path.join(save_folder, f'{mod}_rank{rr}_{mouse}_{stagi}_cv_model_selection.png'),
                            bbox_inches='tight')
                plt.close('all')

                # plot cumulative deviance explained by models
                ecdf = ECDF(selected_fit_qual_test)
                x = np.arange(0, 1, 0.01)
                this_ecdf = ecdf(x)
                logger.info(f'{mouse}: {stagi}: Test set: mean deviance explained = {np.mean(selected_fit_qual_test)}')
                [
                    logger.info(f'{mouse}: {stagi}: Test set: component {compi + 1}: deviance explained = {s}')
                    for compi, s in enumerate(selected_fit_qual_test)
                ]

                # save a fraction of deviance explained summary
                fig, axes = plt.subplots(1, 1, figsize=(5, 5))
                axes.plot(x, this_ecdf)
                axes.set_xlabel('Fraction deviance explained')
                axes.set_ylabel('Cumulative density')
                axes.set_title(f'{mouse}')
                plt.savefig(os.path.join(save_folder, f'{mod}_rank{rr}_{mouse}_{stagi}_deviance_explained_best_cv_model.png'),
                            bbox_inches='tight')
                plt.close('all')

                # calculate deviance explained from zeroed/ablated reconstructions (predictions) for each w and w0
                ddev_per_y = []
                fdev_per_y = []
                full_dev = []
                full_w = []
                if test:
                    _, ax = plt.subplots(Y_test.shape[1], 1, figsize=(15, Y_test.shape[1]*3), sharex=True, sharey=True)
                for n_factor in range(Y_test.shape[1]):
                    y = Y_test[:, n_factor]
                    w = all_w[n_factor]
                    w0 = all_w0[n_factor]
                    fdev_per_y_w = []
                    ddev_per_y_w = []
                    for each_ablation in range(X_test.shape[1]):
                        X_ablated = deepcopy(X_test)
                        X_ablated[:, each_ablation] = 0
                        mu_full = cas.cvglm.make_prediction(X_test, w, w0)
                        mu = cas.cvglm.make_prediction(X_ablated, w, w0)  # this should be done on test data
                        ddev_per_y_w.append(cas.cvglm.deviance(mu_full, y)[0] - cas.cvglm.deviance(mu, y)[0])
                        fdev_per_y_w.append(1 - cas.cvglm.deviance(mu, y)[0] / cas.cvglm.deviance(mu_full, y)[0])
                    if test:
                        if n_factor == 0:
                            ax[n_factor].plot(y, label='data', alpha=0.5)
                            ax[n_factor].plot(mu_full, label='GLM fit (test set)', alpha=0.5)
                            ax[n_factor].legend(loc=2, bbox_to_anchor=(1.05, 1.0))
                        else:
                            ax[n_factor].plot(y, alpha=0.5)
                            ax[n_factor].plot(mu_full, alpha=0.5)
                        if n_factor + 1 == Y_test.shape[1]:
                            ax[n_factor].set_xlabel('Trial number (test set)', size=16)
                        ax[n_factor].set_ylabel(f'Component {n_factor + 1}        ', size=16, rotation=0, ha='right')
                        mu_full = cas.cvglm.make_prediction(X_test, w, w0)
                        ax[n_factor].set_title(f'{stagi}Full model deviance explained: {cas.cvglm.deviance(mu_full, y)[0]}', size=16)
                        plt.suptitle(f'{stagi}\n{mouse} {mod} rank {rr} {version} {loss_type} {activation}', size=20, position=(0.5, 0.9))
                        plt.savefig(os.path.join(save_folder,
                                                f'{mouse}_{mod}_rank{rr}{version}_{loss_type}_{activation}_traces_{stagi}.png'),
                                    bbox_inches='tight')
                    ddev_per_y.append(ddev_per_y_w)
                    fdev_per_y.append(fdev_per_y_w)
                    full_dev.append(cas.cvglm.deviance(mu_full, y)[0])
                    full_w.append(w)

                # create index for all dfs
                index = pd.MultiIndex.from_arrays(
                    [[mouse] * len(full_w), [stagi] * len(full_w), np.arange(1,len(full_w) + 1)],
                    names=['mouse', 'stage', 'component'])
                # create your index out of relevant variables
                fac_dev_breakdown = np.stack(ddev_per_y, axis=1)
                data = {k: fac_dev_breakdown[c, :] for c, k in enumerate(kept_cols)}
                devex_df = pd.DataFrame(data=data, index=index)
                df_list_ddev.append(devex_df)

                # create your index out of relevant variables
                fac_dev_breakdown = np.stack(fdev_per_y, axis=1)
                data = {k: fac_dev_breakdown[c, :] for c, k in enumerate(kept_cols)}
                devex_df = pd.DataFrame(data=data, index=index)
                df_list_fdev.append(devex_df)

                # hold onto other values using the same index
                total_model_devex.append(
                    pd.DataFrame(data={
                        'beta_w0': all_w0,
                        'beta_w': all_w,
                        'total_model_devex_test': selected_fit_qual_test,
                        'total_model_devex_full': selected_fit_qual_full
                    },
                                index=index))

        all_mddev_df = pd.concat(df_list_ddev, axis=0, sort=False)
        all_mfdev_df = pd.concat(df_list_fdev, axis=0, sort=False)
        all_mdev_df = pd.concat(total_model_devex, axis=0, sort=False)

        # save
        all_mddev_df.to_pickle(os.path.join(save_folder, f'{mod}_rank{rr}_cvGLM_delta_deviance_df.pkl'))
        all_mfdev_df.to_pickle(os.path.join(save_folder, f'{mod}_rank{rr}_cvGLM_fractional_deviance_df.pkl'))
        all_mdev_df.to_pickle(os.path.join(save_folder, f'{mod}_rank{rr}_cvGLM_model_performance_df.pkl'))

        # log
        logger.info(f'Done: {mod} rank {rr} {version} {loss_type} {activation}\n')

        # plot heatamps of devex and betas
        added_col_df = all_mfdev_df.merge(all_mdev_df['total_model_devex_test'], how='left',
                                  left_index=True, right_index=True)
        sorter = all_mdev_df['total_model_devex_test'].argsort()
        beta_df = pd.DataFrame(data=np.stack(all_mdev_df['beta_w'].values, axis=0),
                            columns=all_mfdev_df.columns, index=added_col_df.index)
        _, ax = plt.subplots(1,2, figsize=(30,10))
        sns.heatmap(added_col_df.iloc[sorter, :], vmin=None, vmax=None, ax=ax[0], xticklabels=True)
        ax[0].set_title('Deviance explained per filter', size=20)
        sns.heatmap(beta_df.iloc[sorter, :], center=0, cmap='RdBu_r', ax=ax[1], xticklabels=True)
        ax[1].set_title('Betas per filter', size=20)
        plt.suptitle(f'{mod} rank {rr} {version} {loss_type} {activation}', size=20)
        plt.savefig(os.path.join(save_folder,
                                 f'{mod}_rank{rr}{version}_{loss_type}_{activation}_devex_betas_heatmap.png'),
                    bbox_inches='tight')
        _, ax = plt.subplots(1,2, figsize=(30,10))
        sns.heatmap(added_col_df.iloc[sorter, :], vmin=1, vmax=-0.1, ax=ax[0], xticklabels=True)
        ax[0].set_title('Deviance explained per filter', size=20)
        sns.heatmap(beta_df.iloc[sorter, :], center=0, cmap='RdBu_r', ax=ax[1], xticklabels=True)
        ax[1].set_title('Betas per filter', size=20)
        plt.suptitle(f'{mod} rank {rr} {version} {loss_type} {activation}', size=20)
        plt.savefig(os.path.join(save_folder,
                                 f'{mod}_rank{rr}{version}_{loss_type}_{activation}_devex_betas_heatmapVLIMS.png'),
                    bbox_inches='tight')


def create_logger(save_folder):
    """ Create a logger for watching model fitting, etc.
    """

    # Set up logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_path = os.path.join(save_folder, 'glm.log')

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('version',
                        action='store',
                        nargs='?',
                        default="_v2_avg",
                        help='TCA model version. i.e., "_v1", "_v2_avg", "_v3_sq", "_v4_sqsm"')
    parser.add_argument('loss_type',
                        action='store',
                        nargs='?',
                        default="poisson",
                        help='Loss type for GLM fitting. Can be "poisson" or "exponential".')
    parser.add_argument('activation',
                        action='store',
                        nargs='?',
                        default="exp",
                        help='Activation function for GLM fitting. Can be "exp", "softplus" or "relu".')
    parser.add_argument('-f', '--force', action='store_true', help='Force rerun of GLM even when file exists.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Increase verbosity of terminal.')
    parser.add_argument('-t', '--test', action='store_true', help='Run on a single model as a test run.')

    args = parser.parse_args()

    if args.test:
        main(version=args.version,
             loss_type=args.loss_type,
             activation=args.activation,
             force=args.force,
             verbose=args.verbose,
             test=True)
    else:
        main(version=args.version,
             loss_type=args.loss_type,
             activation=args.activation,
             force=args.force,
             verbose=args.verbose)

import logging
import numpy as np
from numpy.core.defchararray import replace
from numpy.lib.index_tricks import IndexExpression
from sklearn.model_selection import train_test_split
from sklearn import svm

from tqdm import tqdm
import logging
import os

from . import utils, lookups, paths


"""
TODO 
1. Could train an anticipatory lick number or lick bout onset classifier?
2. Think about stratification in outcome models.
"""

def ablate_cells_from_comp(clf, test_data, test_y, ids, id_categories):
    raise NotImplementedError


def balance_classes(class_vec, inds):
    """Balance the number of each class across their indices. 

    NOTE: Sampling with no replacement.

    Parameters
    ----------
    class_vec : numpy.ndarray
        array of class type (i.e. trial types for some set of trials)
    inds : numpy.ndarray
        set of indices (relative to all trial metadata) that match your class_vec

    Returns
    -------
    numpy.ndarray
        array of indices ensuring that you have an equal number of classes in those indices
    """

    n_classes = np.unique(class_vec)
    class_counts = [np.sum(class_vec == s) for s in n_classes]
    max_samples = np.min(class_counts)
    balanced_samples = [np.random.choice(inds[class_vec == s], size=max_samples, replace=False) for s in n_classes]
    balanced_inds = np.sort(np.hstack(balanced_samples))

    return balanced_inds


def svm_from_dict(load_dict, staging='parsed_4stage', save_please=True):
    """Run SVM on each stage-day, train 80% of trials, 20% test. Predict each timepoint 
    of a trial (which cue it belongs to).

    Parameters
    ----------
    load_dict : dict
        Dict of "raw" data.
    staging : str, optional
        Binning method, by default 'parsed_4stage'

    Returns
    -------
    dict
        dict of SVM results and test data.
    """

    cue_model = {'probs': [], 'y_test': [], 'data_test': [], 'stage_arr': [], 'id_arr': []}
    logger = _create_logger(paths.analysis_dir('decoder'))

    for meta, tensor, ids in zip(load_dict['meta_list'], load_dict['tensor_list'], load_dict['id_list']):

        # add reversal condition to be used as your cue
        meta = utils.add_reversal_mismatch_condition_to_meta(meta)

        # get average response per stage
        stages = lookups.staging[staging]
        new_tensor = np.zeros((tensor.shape[0], tensor.shape[1], len(stages)))
        new_tensor[:] = np.nan
        sta, probs, ys, kept_ids, y_data = [], [], [], [], []
        for c, di in enumerate(stages):
            stage_boo = meta[staging].isin([di]).values
            stage_days = meta.loc[stage_boo, :].reset_index()['date'].unique()
            day_means = np.zeros((tensor.shape[0], tensor.shape[1], len(stage_days)))
            day_means[:] = np.nan
            for c2, di2 in tqdm(enumerate(stage_days),
                                desc=f'{utils.meta_mouse(meta)}: {di}, cue SVM',
                                total=len(stage_days)):
                day_boo = meta.reset_index()['date'].isin([di2]).values
                all_inds = np.where(stage_boo & day_boo)[0]
                class_vec = meta.iloc[all_inds].mismatch_condition.values
                balanced_inds = balance_classes(class_vec, all_inds)
                if len(balanced_inds) < 100:
                    logger.warning(f'{utils.meta_mouse(meta)}: {di} {di2} too few trials, total trials: {len(balanced_inds)}')
                    continue
                # stratify test and training sets matching balanced portions of y (classes)
                strat = meta.iloc[balanced_inds].mismatch_condition.values
                train_inds, test_inds = train_test_split(balanced_inds, test_size=0.2, random_state=42, stratify=strat)

                no_nans = ~np.isnan(tensor[:, 0, balanced_inds[0]])
                y_test = np.hstack([[s]*tensor.shape[1] for s in meta.iloc[test_inds].mismatch_condition])
                y_train = np.hstack([[s]*tensor.shape[1] for s in meta.iloc[train_inds].mismatch_condition])
                train_flat_tensor = utils.unwrap_tensor(tensor[:, :, train_inds][no_nans, :, :])
                test_flat_tensor = utils.unwrap_tensor(tensor[:, :, test_inds][no_nans, :, :])

                clf = svm.SVC(C=1, probability=True, gamma='scale')
                clf.fit(train_flat_tensor.T, y_train)
                allX_prob = clf.predict_proba(test_flat_tensor.T)

                probs.append(allX_prob[:,:])
                ys.append(y_test)
                sta.append(di)
                kept_ids.append(ids[no_nans])
                y_data.append(y_test)

        cue_model['probs'].append(probs)
        cue_model['y_test'].append(ys)
        cue_model['data_test'].append(y_data)
        cue_model['stage_arr'].append(sta)
        cue_model['id_arr'].append(kept_ids)

    # optionally save
    if save_please:
        np.save(paths.analysis_file('svm_cue.npy', 'decoder'), cue_model, allow_pickle=True)

    return cue_model


def svm_from_dict_go(load_dict, staging='parsed_4stage', save_please=True):
    """Run SVM on each stage-day, train 80% of trials, 20% test. Predict each timepoint 
    of a trial (which outcome it belongs to).

    Parameters
    ----------
    load_dict : dict
        Dict of "raw" data.
    staging : str, optional
        Binning method, by default 'parsed_4stage'

    Returns
    -------
    dict
        dict of SVM results and test data.
    """

    cue_model = {'probs': [], 'y_test': [], 'data_test': [], 'stage_arr': [], 'id_arr': []}
    logger = _create_logger(paths.analysis_dir('decoder'), logger_name='SVM_logger2', logger_file='svm_gonogo.log')

    for meta, tensor, ids in zip(load_dict['meta_list'], load_dict['tensor_list'], load_dict['id_list']):

        # add reversal condition to be used as your cue
        meta = utils.add_reversal_mismatch_condition_to_meta(meta)

        # get average response per stage
        stages = lookups.staging[staging]
        new_tensor = np.zeros((tensor.shape[0], tensor.shape[1], len(stages)))
        new_tensor[:] = np.nan
        sta, probs, ys, kept_ids, y_data = [], [], [], [], []
        for c, di in enumerate(stages):
            stage_boo = meta[staging].isin([di]).values
            stage_days = meta.loc[stage_boo, :].reset_index()['date'].unique()
            day_means = np.zeros((tensor.shape[0], tensor.shape[1], len(stage_days)))
            day_means[:] = np.nan
            for c2, di2 in tqdm(enumerate(stage_days),
                                desc=f'{utils.meta_mouse(meta)}: {di}, go/nogo SVM',
                                total=len(stage_days)):
                day_boo = meta.reset_index()['date'].isin([di2]).values
                all_inds = np.where(stage_boo & day_boo)[0]
                class_vec = meta.iloc[all_inds].trialerror.isin([0, 3, 5]).values
                balanced_inds = balance_classes(class_vec, all_inds)
                if len(balanced_inds) < 100:
                    logger.warning(f'{utils.meta_mouse(meta)}: {di} {di2} too few trials, total trials: {len(balanced_inds)}')
                    continue
                # stratify test and training sets matching balanced portions of y (classes)
                strat = meta.iloc[balanced_inds].mismatch_condition.values
                train_inds, test_inds = train_test_split(balanced_inds, test_size=0.2, random_state=42, stratify=strat)

                # go True
                go = meta.trialerror.isin([0, 3, 5]).values

                no_nans = ~np.isnan(tensor[:, 0, balanced_inds[0]])
                y_test = np.hstack([[s]*tensor.shape[1] for s in go[test_inds]])
                y_train = np.hstack([[s]*tensor.shape[1] for s in go[train_inds]])
                train_flat_tensor = utils.unwrap_tensor(tensor[:, :, train_inds][no_nans, :, :])
                test_flat_tensor = utils.unwrap_tensor(tensor[:, :, test_inds][no_nans, :, :])

                clf = svm.SVC(C=1, probability=True, gamma='scale')
                clf.fit(train_flat_tensor.T, y_train)
                allX_prob = clf.predict_proba(test_flat_tensor.T)

                probs.append(allX_prob[:,:])
                ys.append(y_test)
                sta.append(di)
                kept_ids.append(ids[no_nans])
                y_data.append(y_test)

        cue_model['probs'].append(probs)
        cue_model['y_test'].append(ys)
        cue_model['data_test'].append(y_data)
        cue_model['stage_arr'].append(sta)
        cue_model['id_arr'].append(kept_ids)

    # optionally save
    if save_please:
        np.save(paths.analysis_file('svm_gonogo.npy', 'decoder'), cue_model, allow_pickle=True)

    return cue_model


def svm_from_dict_trialerror(load_dict, staging='parsed_4stage', save_please=True):
    """Run SVM on each stage-day, train 80% of trials, 20% test. Predict each timepoint 
    of a trial (which trialerror it belongs to).

    Parameters
    ----------
    load_dict : dict
        Dict of "raw" data.
    staging : str, optional
        Binning method, by default 'parsed_4stage'

    Returns
    -------
    dict
        dict of SVM results and test data.
    """

    cue_model = {'probs': [], 'y_test': [], 'data_test': [], 'stage_arr': [], 'id_arr': []}
    logger = _create_logger(paths.analysis_dir('decoder'), logger_name='SVM_logger_TE', logger_file='svm_trialerror.log')

    for meta, tensor, ids in zip(load_dict['meta_list'], load_dict['tensor_list'], load_dict['id_list']):

        # add reversal condition to be used as your cue
        meta = utils.add_reversal_mismatch_condition_to_meta(meta)

        # get average response per stage
        stages = lookups.staging[staging]
        new_tensor = np.zeros((tensor.shape[0], tensor.shape[1], len(stages)))
        new_tensor[:] = np.nan
        sta, probs, ys, kept_ids, y_data = [], [], [], [], []
        for c, di in enumerate(stages):
            stage_boo = meta[staging].isin([di]).values
            stage_days = meta.loc[stage_boo, :].reset_index()['date'].unique()
            day_means = np.zeros((tensor.shape[0], tensor.shape[1], len(stage_days)))
            day_means[:] = np.nan
            for c2, di2 in tqdm(enumerate(stage_days),
                                desc=f'{utils.meta_mouse(meta)}: {di}, trialerror SVM',
                                total=len(stage_days)):
                day_boo = meta.reset_index()['date'].isin([di2]).values
                all_inds = np.where(stage_boo & day_boo)[0]
                class_vec = meta.iloc[all_inds].trialerror.values
                balanced_inds = balance_classes(class_vec, all_inds)
                if len(balanced_inds) < 50:
                    logger.warning(f'{utils.meta_mouse(meta)}: {di} {di2} too few trials, total trials: {len(balanced_inds)}')
                    continue
                # stratify test and training sets matching balanced portions of y (classes)
                strat = meta.iloc[balanced_inds].trialerror.values
                train_inds, test_inds = train_test_split(balanced_inds, test_size=0.2, random_state=42, stratify=strat)

                no_nans = ~np.isnan(tensor[:, 0, balanced_inds[0]])
                y_test = np.hstack([[s]*tensor.shape[1] for s in meta.iloc[test_inds].trialerror])
                y_train = np.hstack([[s]*tensor.shape[1] for s in meta.iloc[train_inds].trialerror])
                train_flat_tensor = utils.unwrap_tensor(tensor[:, :, train_inds][no_nans, :, :])
                test_flat_tensor = utils.unwrap_tensor(tensor[:, :, test_inds][no_nans, :, :])

                clf = svm.SVC(C=1, probability=True, gamma='scale')
                clf.fit(train_flat_tensor.T, y_train)
                allX_prob = clf.predict_proba(test_flat_tensor.T)

                probs.append(allX_prob[:,:])
                ys.append(y_test)
                sta.append(di)
                kept_ids.append(ids[no_nans])
                y_data.append(y_test)

        cue_model['probs'].append(probs)
        cue_model['y_test'].append(ys)
        cue_model['data_test'].append(y_data)
        cue_model['stage_arr'].append(sta)
        cue_model['id_arr'].append(kept_ids)

    # optionally save
    if save_please:
        np.save(paths.analysis_file('svm_trialerror.npy', 'decoder'), cue_model, allow_pickle=True)

    return cue_model


def svm_from_dict_hitmiss(load_dict, staging='parsed_4stage', save_please=True):
    """Run SVM on each stage-day, train 80% of trials, 20% test. Predict each timepoint 
    of a trial (which trialerror it belongs to).

    Parameters
    ----------
    load_dict : dict
        Dict of "raw" data.
    staging : str, optional
        Binning method, by default 'parsed_4stage'

    Returns
    -------
    dict
        dict of SVM results and test data.
    """

    cue_model = {'probs': [], 'y_test': [], 'data_test': [], 'stage_arr': [], 'id_arr': []}
    logger = _create_logger(paths.analysis_dir('decoder'), logger_name='SVM_logger_teHM', logger_file='svm_hitmiss.log')

    for meta, tensor, ids in zip(load_dict['meta_list'], load_dict['tensor_list'], load_dict['id_list']):

        # add reversal condition to be used as your cue
        meta = utils.add_reversal_mismatch_condition_to_meta(meta)

        # get average response per stage
        stages = lookups.staging[staging]
        new_tensor = np.zeros((tensor.shape[0], tensor.shape[1], len(stages)))
        new_tensor[:] = np.nan
        sta, probs, ys, kept_ids, y_data = [], [], [], [], []
        for c, di in enumerate(stages):
            stage_boo = meta[staging].isin([di]).values
            stage_days = meta.loc[stage_boo, :].reset_index()['date'].unique()
            day_means = np.zeros((tensor.shape[0], tensor.shape[1], len(stage_days)))
            day_means[:] = np.nan
            for c2, di2 in tqdm(enumerate(stage_days),
                                desc=f'{utils.meta_mouse(meta)}: {di}, hit-miss SVM',
                                total=len(stage_days)):
                day_boo = meta.reset_index()['date'].isin([di2]).values
                hit_miss_boo = meta.trialerror.isin([0, 1]).values
                all_inds = np.where(stage_boo & day_boo & hit_miss_boo)[0]
                class_vec = meta.iloc[all_inds].trialerror.values
                if len(np.unique(class_vec)) < 2:
                    logger.error(f'{utils.meta_mouse(meta)}: {di} {di2} missing class type: {np.unique(class_vec)}')
                    continue
                balanced_inds = balance_classes(class_vec, all_inds)
                if len(balanced_inds) < 50:
                    logger.warning(f'{utils.meta_mouse(meta)}: {di} {di2} too few trials, total trials: {len(balanced_inds)}')
                    continue
                # stratify test and training sets matching balanced portions of y (classes)
                strat = meta.iloc[balanced_inds].trialerror.values
                train_inds, test_inds = train_test_split(balanced_inds, test_size=0.2, random_state=42, stratify=strat)

                no_nans = ~np.isnan(tensor[:, 0, balanced_inds[0]])
                y_test = np.hstack([[s]*tensor.shape[1] for s in meta.iloc[test_inds].trialerror])
                y_train = np.hstack([[s]*tensor.shape[1] for s in meta.iloc[train_inds].trialerror])
                train_flat_tensor = utils.unwrap_tensor(tensor[:, :, train_inds][no_nans, :, :])
                test_flat_tensor = utils.unwrap_tensor(tensor[:, :, test_inds][no_nans, :, :])

                clf = svm.SVC(C=1, probability=True, gamma='scale')
                clf.fit(train_flat_tensor.T, y_train)
                allX_prob = clf.predict_proba(test_flat_tensor.T)

                probs.append(allX_prob[:,:])
                ys.append(y_test)
                sta.append(di)
                kept_ids.append(ids[no_nans])
                y_data.append(y_test)

        cue_model['probs'].append(probs)
        cue_model['y_test'].append(ys)
        cue_model['data_test'].append(y_data)
        cue_model['stage_arr'].append(sta)
        cue_model['id_arr'].append(kept_ids)

    # optionally save
    if save_please:
        np.save(paths.analysis_file('svm_hitmiss.npy', 'decoder'), cue_model, allow_pickle=True)

    return cue_model


def _create_logger(save_folder, logger_name='SVM_logger', logger_file='svm.log'):
    """ Create a logger for watching model fitting, etc.
    """

    # Set up logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    log_path = os.path.join(save_folder, logger_file)

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

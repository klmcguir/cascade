from . import lookups, utils, drive, load, paths
import numpy as np
import pandas as pd
import warnings
import os
from tqdm import tqdm


def comp_corr(match_to='onsets', driven_only=False, save_please=False, apply_norm=True):
    """Generate the average correlations of cells averaged by component for stages 
    of learning. Calculates correlations for each cell with bhv signal for each stage-day
    then takes the average over 11-stage- and 4-stage-binning.
    
    Cells use a mean over 2s stim or response window for vector for correlation.

    Parameters
    ----------
    match_to : str, optional
        onsets or offsets, by default 'onsets'
    driven_only : boolean, optional
        Nan undriven trials per cell before taking average across components
    save_please : boolean, optional
        Save output as a pickled pandas.DataFrame. 
    apply_norm : boolean, optional
        Apply normalization scale factor used to peak-stage-normalize zscore traces from saved from TCA fitting.
    """

    # run/load drivenness matrix
    drive_mat_list = drive.drive_day_mat_from_core_reversal_data(match_to=match_to)
    data_dict = np.load(paths.analysis_file('input_data_v4i10_noT0_20210307.npy', 'tca_dfs'), allow_pickle=True).item()

    # load and parse raw inputs
    ondict = load.core_reversal_data(limit_to='onsets', match_to='onsets')
    if match_to == 'onsets':
        save_tag = 'on'
        all_cats = lookups.fixed_component_sort['rank9_onset']
        meta_list = ondict['meta_list']
        tensor_list = ondict['tensor_list']
        id_list = ondict['id_list']
    elif match_to == 'offsets':
        save_tag = 'off'
        all_cats = lookups.fixed_component_sort['rank8_offset']
        offdict = load.core_reversal_data(limit_to='offsets', match_to='offsets')
        meta_list = ondict['meta_list']
        tensor_list = offdict['tensor_list']
        id_list = offdict['id_list']
    if driven_only:
        save_tag += 'driven'

    # load tca data
    on_ens = load.core_tca_data(match_to=match_to)

    mod_dfs = []
    mod_dfs_4stage = []
    mean_corr_stageday_list = []
    # for meta, tensor, bhv, ids, dr in zip(ldict['meta_list'], offdict['tensor_list'], ldict['bhv_list'], offdict['id_list'], drive_mat_list_off):
    for meta, tensor, ids, dr in tqdm(zip(meta_list, tensor_list, id_list, drive_mat_list),
                                           desc=f'Component correlation',
                                           total=len(meta_list)):

        # add mm condition
        meta = utils.add_reversal_mismatch_condition_to_meta(meta)
        meta = utils.add_4stagesv2_to_meta(meta)

        # get your cell category vector checking that ids are in correct order
        mouse_cats = on_ens['cell_cats'][on_ens['mouse_vec'] == utils.meta_mouse(meta)]
        mouse_ids = on_ens['cell_vec'][on_ens['mouse_vec'] == utils.meta_mouse(meta)]
        assert np.array_equal(mouse_ids, ids)

        # get average trace per cell over all time
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            # optionally ablate undriven days (trials for each cue for that day)
            if driven_only:
                unf_cues = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']
                for c, cue in enumerate(unf_cues):
                    cue_trials = meta.mismatch_condition.isin([cue])
                    dr_mat = dr[:, :, c]
                    for celli in range(tensor.shape[0]):
                        tensor[celli, :, (dr_mat[celli] == 0) & cue_trials.values] = np.nan
            if apply_norm:
                norm_factors = data_dict['v4i10_norm_on_cell_scale_factors_noT0_allstages']
                norm_factors_mouse = norm_factors[on_ens['mouse_vec'] == utils.meta_mouse(meta)]
                tensor = tensor/norm_factors_mouse[:, None, None]
            cat_mean, _ = utils.average_across_cats(tensor, mouse_cats, force_match_to=match_to)
            trial_mean = np.nanmean(cat_mean[:, -16:, :], axis=1) # last sec only

        # preallocate your funny unknown number of stage days
        stage_day_n = meta.groupby(['parsed_4stagev2', 'parsed_11stage', 'date']).count().shape[0]

        # get all unique stage day combos in order
        stage_day_df = (meta.groupby(['parsed_4stagev2', 'parsed_11stage',
                                      'date']).count().reindex(lookups.staging['parsed_4stagev3'],
                                                               level=0).reindex(lookups.staging['parsed_11stage'],
                                                                                level=1))

        # preallocate
        stageday_corr_mat = np.zeros((len(all_cats), len(all_cats), stage_day_n, 3)) + np.nan

        # use index from row iterator to get stage-day boolean
        for c, (ind, _) in enumerate(stage_day_df.iterrows()):

            # boolean
            stage_boo = meta.parsed_11stage.isin([ind[1]]).values
            day_boo = meta.reset_index().date.isin([ind[2]]).values
            sdb = stage_boo & day_boo

            # Subset chunks of data to only use NaN free sets
            trial_chunk = trial_mean[:, sdb]
            meta_chunk = meta.loc[sdb]

            # skip if you have removed all trials for a day-stage
            if trial_chunk.shape[1] == 0:
                continue

            # only use cells that exist for corr
            existing_data = np.isfinite(trial_chunk[:, 0])

            # correlate each cell only for it's preferred cue
            cues = ['becomes_unrewarded', 'becomes_rewarded', 'remains_unrewarded']
            for ci, cue in enumerate(cues):
                cue_trials = meta_chunk.mismatch_condition.isin([cue])
                # cue_trials = meta_chunk.mismatch_condition.isin(cues)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    corr = np.corrcoef(trial_chunk[existing_data, :][:, cue_trials])
                if all(existing_data):
                    stageday_corr_mat[:, :, c, ci] = corr
                else:
                    for ck, k in enumerate(np.where(existing_data)[0]):
                        stageday_corr_mat[existing_data, k, c, ci] = corr[:, ck]


        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            # take mean over stages
            stagemean_corr_mat = np.zeros((len(all_cats), len(all_cats), 11, 3)) + np.nan
            for sc, stage in enumerate(lookups.staging['parsed_11stage']):
                stage_boo = stage_day_df.reset_index().parsed_11stage.isin([stage])
                for ci, cue in enumerate(cues):
                    stagemean_corr_mat[:, :, sc, ci] = np.nanmean(stageday_corr_mat[:, :, stage_boo, ci], axis=2)

            # take mean over 4 stages too
            stagemean_corr_mat2 = np.zeros((len(all_cats), len(all_cats), 4, 3)) + np.nan
            for sc, stage in enumerate(lookups.staging['parsed_4stagev2']):
                stage_boo = stage_day_df.reset_index().parsed_4stagev2.isin([stage])
                for ci, cue in enumerate(cues):
                    stagemean_corr_mat2[:, :, sc, ci] = np.nanmean(stageday_corr_mat[:, :, stage_boo, ci], axis=2)

            mean_corr_stageday_list.append(stageday_corr_mat)
            mod_dfs.append(stagemean_corr_mat)
            mod_dfs_4stage.append(stagemean_corr_mat2)

    # corr_avg = np.concatenate(mean_corr_stageday_list, axis=0)
    # corr_avg4 = np.concatenate(mod_dfs_4stage, axis=0)
    # corr_avg11 = np.concatenate(mod_dfs, axis=0)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        corr_avg4 = np.nanmean(mod_dfs_4stage, axis=0)
        corr_avg11 = np.nanmean(mod_dfs, axis=0)

    # data = {
    #     'mouse': on_ens['mouse_vec'],
    #     'cell_id': on_ens['cell_vec'],
    #     'cell_cats': on_ens['cell_cats'],
    #     f'mean_{save_tag}_corr_2s_stageday': corr_avg,
    #     f'mean_{save_tag}_corr_2s_11stage': single_avg11,
    #     f'mean_{save_tag}_corr_2s_4stage': single_avg4,
    # }
    # corr_df_2s = pd.DataFrame(data=data).set_index(['mouse', 'cell_id'])
    # if save_please:
    #     corr_df_2s.to_pickle(
    #         os.path.join(lookups.coreroot, f'{match_to}_{save_tag}_compcorr_2s_df.pkl'))

    return corr_avg4, corr_avg11, mean_corr_stageday_list


def comp_corr_day(match_to='onsets', driven_only=False, do_diff=False, apply_norm=True, save_please=False):
    """Generate the average correlations of cells averaged by component for stages 
    of learning. Calculates correlations for each cell with bhv signal for each stage-day
    then takes the average over 11-stage- and 4-stage-binning.
    
    Cells use a mean over 2s stim or response window for vector for correlation.

    Parameters
    ----------
    match_to : str, optional
        onsets or offsets, by default 'onsets'
    driven_only : boolean, optional
        Nan undriven trials per cell before taking average across components
    save_please : boolean, optional
        Save output as a pickled pandas.DataFrame. 
    """

    # run/load drivenness matrix
    drive_mat_list = drive.drive_day_mat_from_core_reversal_data(match_to=match_to)
    data_dict = np.load(paths.analysis_file('input_data_v4i10_noT0_20210307.npy', 'tca_dfs'), allow_pickle=True).item()

    # load and parse raw inputs
    ondict = load.core_reversal_data(limit_to='onsets', match_to='onsets')
    if match_to == 'onsets':
        save_tag = 'on'
        all_cats = lookups.fixed_component_sort['rank9_onset']
        meta_list = ondict['meta_list']
        tensor_list = ondict['tensor_list']
        id_list = ondict['id_list']
    elif match_to == 'offsets':
        save_tag = 'off'
        all_cats = lookups.fixed_component_sort['rank8_offset']
        offdict = load.core_reversal_data(limit_to='offsets', match_to='offsets')
        meta_list = ondict['meta_list']
        tensor_list = offdict['tensor_list']
        id_list = offdict['id_list']
    if driven_only:
        save_tag += 'driven'

    # load tca data
    on_ens = load.core_tca_data(match_to=match_to)

    stageday_avg_list = []
    mean_corr_stageday_list = []
    # for meta, tensor, bhv, ids, dr in zip(ldict['meta_list'], offdict['tensor_list'], ldict['bhv_list'], offdict['id_list'], drive_mat_list_off):
    for meta, tensor, ids, dr in tqdm(zip(meta_list, tensor_list, id_list, drive_mat_list),
                                           desc=f'Component correlation',
                                           total=len(meta_list)):

        # add mm condition
        meta = utils.add_reversal_mismatch_condition_to_meta(meta)
        meta = utils.add_4stagesv2_to_meta(meta)

        # get your cell category vector checking that ids are in correct order
        mouse_cats = on_ens['cell_cats'][on_ens['mouse_vec'] == utils.meta_mouse(meta)]
        mouse_ids = on_ens['cell_vec'][on_ens['mouse_vec'] == utils.meta_mouse(meta)]
        assert np.array_equal(mouse_ids, ids)

        # get average trace per cell over all time
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            # optionally ablate undriven days (trials for each cue for that day)
            if driven_only:
                unf_cues = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']
                for c, cue in enumerate(unf_cues):
                    cue_trials = meta.mismatch_condition.isin([cue])
                    dr_mat = dr[:, :, c]
                    for celli in range(tensor.shape[0]):
                        tensor[celli, :, (dr_mat[celli] == 0) & cue_trials.values] = np.nan
            if apply_norm:
                norm_factors = data_dict['v4i10_norm_on_cell_scale_factors_noT0_allstages']
                norm_factors_mouse = norm_factors[on_ens['mouse_vec'] == utils.meta_mouse(meta)]
                tensor = tensor/norm_factors_mouse[:, None, None]
            cat_mean, _ = utils.average_across_cats(tensor, mouse_cats, force_match_to=match_to)
            # trial_mean = np.nanmean(cat_mean[:, 17:, :], axis=1)
            trial_mean = np.nanmean(cat_mean[:, -16:, :], axis=1)

        # preallocate your funny unknown number of stage days
        # stage_day_n = meta.groupby(['parsed_4stagev2', 'parsed_11stage', 'date']).count().shape[0]
        stage_day_n = meta.groupby(['date']).count().shape[0]

        # get all unique stage day combos in order
        # stage_day_df = (meta.groupby(['parsed_4stagev2', 'parsed_11stage',
        #                               'date']).count().reindex(lookups.staging['parsed_4stagev3'],
        #                                                        level=0).reindex(lookups.staging['parsed_11stage'],
        #                                                                         level=1))
        stage_day_df = meta.groupby(['date']).count()

        # preallocate
        stageday_mat = np.zeros((len(all_cats), stage_day_n, 3)) + np.nan
        # use index from row iterator to get stage-day boolean
        for c, (ind, _) in enumerate(stage_day_df.iterrows()):

            # boolean
            # stage_boo = meta.parsed_11stage.isin([ind[1]]).values
            # day_boo = meta.reset_index().date.isin([ind[2]]).values
            # sdb = stage_boo & day_boo
            sdb = meta.reset_index().date.isin([ind]).values

            # Subset chunks of data to only use NaN free sets
            trial_chunk = trial_mean[:, sdb]
            meta_chunk = meta.loc[sdb]

            # skip if you have removed all trials for a day-stage
            if trial_chunk.shape[1] == 0:
                continue

            # only use cells that exist for corr
            existing_data = np.isfinite(trial_chunk[:, 0])

            # correlate each cell only for it's preferred cue
            cues = ['becomes_unrewarded', 'becomes_rewarded', 'remains_unrewarded']
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                for ci, cue in enumerate(cues):
                    cue_trials = meta_chunk.mismatch_condition.isin([cue])
                    prev_same = (meta_chunk.prev_same_plus | meta_chunk.prev_same_minus | meta_chunk.prev_same_neutral)
                    cue_trials = cue_trials & ~prev_same #meta_chunk.pre_speed.gt(10)
                    # cue_trials = cue_trials & meta_chunk.pre_speed.lt(5) #meta_chunk.pre_speed.gt(10)
                    stageday_mat[:, c, ci] = np.nanmean(trial_chunk[:, cue_trials], axis=1)
        
        # take diff across days
        if do_diff:
            stageday_mat = np.diff(stageday_mat, axis=1)

        # take the correlation of traces of each day for each comp over time. 
        cues = ['becomes_unrewarded', 'becomes_rewarded', 'remains_unrewarded']
        stageday_corr_mat = np.zeros((len(all_cats), len(all_cats), 3)) + np.nan
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            for ci, cue in enumerate(cues):
                existing_days = ~np.any(np.isnan(stageday_mat[:,:,ci]), axis=0)
                stageday_corr_mat[:,:, ci] = np.corrcoef(stageday_mat[:, existing_days, ci])

        # with warnings.catch_warnings():
        #     warnings.filterwarnings('ignore')
        #     # take mean over stages
        #     stagemean_corr_mat = np.zeros((len(all_cats), len(all_cats), 11, 3)) + np.nan
        #     for sc, stage in enumerate(lookups.staging['parsed_11stage']):
        #         stage_boo = stage_day_df.reset_index().parsed_11stage.isin([stage])
        #         for ci, cue in enumerate(cues):
        #             stagemean_corr_mat[:, :, sc, ci] = np.nanmean(stageday_corr_mat[:, :, stage_boo, ci], axis=2)

        #     # take mean over 4 stages too
        #     stagemean_corr_mat2 = np.zeros((len(all_cats), len(all_cats), 4, 3)) + np.nan
        #     for sc, stage in enumerate(lookups.staging['parsed_4stagev2']):
        #         stage_boo = stage_day_df.reset_index().parsed_4stagev2.isin([stage])
        #         for ci, cue in enumerate(cues):
        #             stagemean_corr_mat2[:, :, sc, ci] = np.nanmean(stageday_corr_mat[:, :, stage_boo, ci], axis=2)

            mean_corr_stageday_list.append(stageday_corr_mat)
            stageday_avg_list.append(stageday_mat)
            # mod_dfs_4stage.append(stagemean_corr_mat2)

    # corr_avg = np.concatenate(mean_corr_stageday_list, axis=0)
    # corr_avg4 = np.concatenate(mod_dfs_4stage, axis=0)
    # corr_avg11 = np.concatenate(mod_dfs, axis=0)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        # corr_avg4 = np.nanmean(mod_dfs_4stage, axis=0)
        avg_corr = np.nanmean(mean_corr_stageday_list, axis=0)

    # data = {
    #     'mouse': on_ens['mouse_vec'],
    #     'cell_id': on_ens['cell_vec'],
    #     'cell_cats': on_ens['cell_cats'],
    #     f'mean_{save_tag}_corr_2s_stageday': corr_avg,
    #     f'mean_{save_tag}_corr_2s_11stage': single_avg11,
    #     f'mean_{save_tag}_corr_2s_4stage': single_avg4,
    # }
    # corr_df_2s = pd.DataFrame(data=data).set_index(['mouse', 'cell_id'])
    # if save_please:
    #     corr_df_2s.to_pickle(
    #         os.path.join(lookups.coreroot, f'{match_to}_{save_tag}_compcorr_2s_df.pkl'))

    return avg_corr, stageday_avg_list, mean_corr_stageday_list


def bhv_corr(match_to='onsets', bhv_type='speed', bhv_baseline_or_stim='stim', fix_onset_bhv=True, save_please=False):
    """Generate the average correlations of cells with different bhv signals for stages 
    of learning. Calculates correlations for each cell with bhv signal for each stage-day
    then takes the average over 11-stage- and 4-stage-binning.
    
    Cells use a mean over 2s stim or response window for vecttor for correlation. bhv traces
    use either the baseline 1s window or a 2s window for stim/response period.

    Parameters
    ----------
    match_to : str, optional
        onsets or offsets, by default 'onsets'
    bhv_type : str, optional
        Type of behavioral trace, can be int (ind in bhv matrix) or str, by default 'speed'
    bhv_baseline_or_stim : str, optional
        Use baseline or stim period for behavioral vectors for correlation, by default 'stim'
    fix_onset_bhv : boolean, optional
        Force the bhv used in correlation to come from the onset triggered period. i.e. if you want to correlate
        offsets with the 1s baseline running and licking before the trial (rather than the 1s before offset).
    save_please : boolean, optional
        Save output as a pickled pandas.DataFrame. 

    Raises
    ------
    ValueError
        If you specifiy a non-existent bhv_type.
    """

    # run/load drivenness matrix
    drive_mat_list = drive.preferred_drive_day_mat(match_to=match_to)

    # load and parse raw inputs
    ondict = load.core_reversal_data(limit_to='onsets', match_to='onsets')
    if match_to == 'onsets':
        save_tag = 'on'
        cat_groups_pref_tuning = lookups.tuning_groups["rank9_onset"]
        meta_list = ondict['meta_list']
        tensor_list = ondict['tensor_list']
        bhv_list = ondict['bhv_list']
        id_list = ondict['id_list']
    elif match_to == 'offsets':
        cat_groups_pref_tuning = lookups.tuning_groups["rank8_offset"]
        offdict = load.core_reversal_data(limit_to='offsets', match_to='offsets')
        meta_list = ondict['meta_list']
        tensor_list = offdict['tensor_list']
        if fix_onset_bhv:
            bhv_list = ondict['bhv_list']
            save_tag = 'on'
        else:
            bhv_list = offdict['bhv_list']
            save_tag = 'off'
        id_list = offdict['id_list']

    # load tca data
    on_ens = load.core_tca_data(match_to=match_to)

    # parse bhv input
    # (pupil_traces, dpupil_traces, lick_traces, dlick_traces,
    #  speed_traces, dspeed_traces, neuropil_traces, dneuropil_traces)
    if bhv_type.lower() == 'speed' or bhv_type == 4:
        bhv_num = 4
        bhv_type = 'speed'
    elif bhv_type.lower() == 'lick' or bhv_type == 2:
        bhv_num = 2
        bhv_type = 'lick'
    elif bhv_type.lower() == 'pupil' or bhv_type == 0:
        bhv_num = 0
        bhv_type = 'pupil'
    elif bhv_type.lower() == 'neuropil' or bhv_type == 6:
        bhv_num = 6
        bhv_type = 'neuropil'
    else:
        raise ValueError

    mod_dfs = []
    mod_dfs_4stage = []
    mean_corr_stageday_list = []
    # for meta, tensor, bhv, ids, dr in zip(ldict['meta_list'], offdict['tensor_list'], ldict['bhv_list'], offdict['id_list'], drive_mat_list_off):
    for meta, tensor, bhv, ids, dr in tqdm(zip(meta_list, tensor_list, bhv_list, id_list, drive_mat_list),
                                           desc=f'BHV: {bhv_type} correlation',
                                           total=len(meta_list)):

        # add mm condition
        meta = utils.add_reversal_mismatch_condition_to_meta(meta)
        meta = utils.add_4stagesv2_to_meta(meta)

        # if pupul, normalize pupil to max area per day
        if bhv_num == 0:
            norm_tensor = np.zeros(bhv.shape) + np.nan
            days = pd.unique(meta.reset_index().date)
            for c, di in enumerate(days):
                date_boo = meta.reset_index().date.isin([di]).values
                norm_tensor[:, :, date_boo] = bhv[:, :, date_boo] / np.nanmax(bhv[:, :, date_boo])
            bhv = norm_tensor

        # get behavioral trace over all time
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            trial_mean = np.nanmean(tensor[:, 17:, :], axis=1)
            if bhv_baseline_or_stim == 'baseline':
                bhv_mean = np.nanmean(bhv[bhv_num, :17, :], axis=0)
            elif bhv_baseline_or_stim == 'stim':
                bhv_mean = np.nanmean(bhv[bhv_num, 17:, :], axis=0)

        # get your cell category vector checking that ids are in correct order
        mouse_cats = on_ens['cell_cats'][on_ens['mouse_vec'] == utils.meta_mouse(meta)]
        mouse_ids = on_ens['cell_vec'][on_ens['mouse_vec'] == utils.meta_mouse(meta)]
        assert np.array_equal(mouse_ids, ids)

        # preallocate your funny unknown number of stage days
        stage_day_n = meta.groupby(['parsed_4stagev2', 'parsed_11stage', 'date']).count().shape[0]

        # get all unique stage day combos in order
        stage_day_df = (meta.groupby(['parsed_4stagev2', 'parsed_11stage',
                                      'date']).count().reindex(lookups.staging['parsed_4stagev3'],
                                                               level=0).reindex(lookups.staging['parsed_11stage'],
                                                                                level=1))

        # preallocate
        stageday_corr_mat = np.zeros((tensor.shape[0], stage_day_n)) + np.nan

        # use index from row iterator to get stage-day boolean
        for c, (ind, _) in enumerate(stage_day_df.iterrows()):
            #         print(c, ind)

            # boolean
            stage_boo = meta.parsed_11stage.isin([ind[1]]).values
            day_boo = meta.reset_index().date.isin([ind[2]]).values
            sdb = stage_boo & day_boo

            # Subset chunks of data to only use NaN free sets
            trial_chunk = trial_mean[:, sdb]
            bhv_chunk = bhv_mean[sdb]
            meta_chunk = meta.loc[sdb]
            bhv_exist = np.isfinite(bhv_chunk)
            bhv_chunk = bhv_chunk[bhv_exist]
            trial_chunk = trial_chunk[:, bhv_exist]
            meta_chunk = meta_chunk.loc[bhv_exist]

            # skip if you have removed all trials for a day-stage
            if trial_chunk.shape[1] == 0:
                # print('empty trial chunk')
                continue

            # only use cells that exist for corr
            existing_data = np.isfinite(trial_chunk[:, 0])

            # correlate each cell only for it's preferred cue
            cues = ['becomes_unrewarded', 'becomes_rewarded', 'remains_unrewarded']
            for cue in cues:
                cue_cells = np.isin(mouse_cats, cat_groups_pref_tuning[cue])
                if (existing_data & cue_cells).sum() == 0:
                    continue
                cue_trials = meta_chunk.mismatch_condition.isin([cue])
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    corr = np.corrcoef(
                        np.concatenate(
                            [trial_chunk[existing_data & cue_cells, :][:, cue_trials], bhv_chunk[None, cue_trials]],
                            axis=0))
                corr_with_bhv = corr[:-1, -1]
                stageday_corr_mat[existing_data & cue_cells, c] = corr_with_bhv

            # deal with special tuning cases
            cue_cells = np.isin(mouse_cats, cat_groups_pref_tuning['broad'])
            if (existing_data & cue_cells).sum() == 0:
                continue
            cue_trials = meta_chunk.mismatch_condition.isin(cues)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                corr = np.corrcoef(
                    np.concatenate([trial_chunk[existing_data & cue_cells, :][:, cue_trials], bhv_chunk[None, cue_trials]],
                                axis=0))
            corr_with_bhv = corr[:-1, -1]
            stageday_corr_mat[existing_data & cue_cells, c] = corr_with_bhv

            # deal with special tuning cases
            if any(['joint' in s for s in cat_groups_pref_tuning.keys()]):
                jcues = [s for s in cat_groups_pref_tuning.keys() if 'joint' in s]
                for cue in jcues:
                    cue_cells = np.isin(mouse_cats, cat_groups_pref_tuning[cue])
                    if (existing_data & cue_cells).sum() == 0:
                        continue
                    cue_trials = meta_chunk.mismatch_condition.isin([s for s in cues if s in cue])
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore')
                        corr = np.corrcoef(
                            np.concatenate(
                                [trial_chunk[existing_data & cue_cells, :][:, cue_trials], bhv_chunk[None, cue_trials]],
                                axis=0))
                    corr_with_bhv = corr[:-1, -1]
                    stageday_corr_mat[existing_data & cue_cells, c] = corr_with_bhv

        # remove undriven days from calc
        stageday_corr_mat[dr == 0] = np.nan

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            # take mean over stages
            stagemean_corr_mat = np.zeros((tensor.shape[0], 11)) + np.nan
            for sc, stage in enumerate(lookups.staging['parsed_11stage']):
                stage_boo = stage_day_df.reset_index().parsed_11stage.isin([stage])
                stagemean_corr_mat[:, sc] = np.nanmean(stageday_corr_mat[:, stage_boo], axis=1)

            # take mean over 4 stages too
            stagemean_corr_mat2 = np.zeros((tensor.shape[0], 4)) + np.nan
            for sc, stage in enumerate(lookups.staging['parsed_4stagev2']):
                stage_boo = stage_day_df.reset_index().parsed_4stagev2.isin([stage])
                stagemean_corr_mat2[:, sc] = np.nanmean(stageday_corr_mat[:, stage_boo], axis=1)

            mean_corr_stageday_list.append(np.nanmean(stageday_corr_mat, axis=1))
            mod_dfs.append(stagemean_corr_mat)
            mod_dfs_4stage.append(stagemean_corr_mat2)

    corr_avg = np.concatenate(mean_corr_stageday_list, axis=0)
    corr_avg4 = np.concatenate(mod_dfs_4stage, axis=0)
    corr_avg11 = np.concatenate(mod_dfs, axis=0)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        single_avg11 = np.nanmean(corr_avg11, axis=1)
        single_avg4 = np.nanmean(corr_avg4, axis=1)

    data = {
        'mouse': on_ens['mouse_vec'],
        'cell_id': on_ens['cell_vec'],
        'cell_cats': on_ens['cell_cats'],
        f'mean_{save_tag + bhv_type}_corr_2s_stageday': corr_avg,
        f'mean_{save_tag + bhv_type}_corr_2s_11stage': single_avg11,
        f'mean_{save_tag + bhv_type}_corr_2s_4stage': single_avg4,
    }
    corr_df_2s = pd.DataFrame(data=data).set_index(['mouse', 'cell_id'])
    if save_please:
        corr_df_2s.to_pickle(
            os.path.join(lookups.coreroot, f'{match_to}_{bhv_baseline_or_stim}_{save_tag + bhv_type}_corr_2s_df.pkl'))

    return corr_df_2s


def meta_or_bhv_index(on='speed',
                      match_to='onsets',
                      bhv_baseline_or_stim='stim',
                      fix_onset_bhv=True,
                      driven_only=False,
                      save_please=False):
    """Generate the average modulation index of cells with different bhv signals for stages 
    of learning or for existing trial metrics from meta. bhv signals for each stage-day us 1s baselines
    or 2s stim (or response) periods

    Starting with averaging over stage-days, then stages for each condition --> then uses 2 values
    a and b to calcualte a - b / a + b index.
    
    Cells use a mean over 2s stim or response window for vecttor for correlation. bhv traces
    use either the baseline 1s window or a 2s window for stim/response period.

    Parameters
    ----------
    match_to : str, optional
        onsets or offsets, by default 'onsets'
    bhv_type : str, optional
        Type of behavioral trace, can be int (ind in bhv matrix) or str, by default 'speed'
    bhv_baseline_or_stim : str, optional
        Use baseline or stim period for behavioral vectors for correlation, by default 'stim'
    fix_onset_bhv : boolean, optional
        Force the bhv used in correlation to come from the onset triggered period. i.e. if you want to correlate
        offsets with the 1s baseline running and licking before the trial (rather than the 1s before offset).
    save_please : boolean, optional
        Save output as a pickled pandas.DataFrame. 

    Raises
    ------
    ValueError
        If you specifiy a non-existent bhv_type.
    """

    # run/load drivenness matrix
    drive_mat_list = drive.preferred_drive_day_mat(match_to=match_to)

    # load and parse raw inputs
    ondict = load.core_reversal_data(limit_to='onsets', match_to='onsets')
    if match_to == 'onsets':
        save_tag = 'on'
        cat_groups_pref_tuning = lookups.tuning_groups["rank9_onset"]
        meta_list = ondict['meta_list']
        tensor_list = ondict['tensor_list']
        bhv_list = ondict['bhv_list']
        id_list = ondict['id_list']
    elif match_to == 'offsets':
        cat_groups_pref_tuning = lookups.tuning_groups["rank8_offset"]
        offdict = load.core_reversal_data(limit_to='offsets', match_to='offsets')
        meta_list = ondict['meta_list']
        tensor_list = offdict['tensor_list']
        if fix_onset_bhv:
            bhv_list = ondict['bhv_list']
            save_tag = 'on'
        else:
            bhv_list = offdict['bhv_list']
            save_tag = 'off'
        id_list = offdict['id_list']

    # load tca data
    on_ens = load.core_tca_data(match_to=match_to)

    df_list = []
    # for meta, tensor, bhv, ids, dr in zip(ldict['meta_list'], offdict['tensor_list'], ldict['bhv_list'], offdict['id_list'], drive_mat_list_off):
    for meta, tensor, bhv, ids, dr in tqdm(zip(meta_list, tensor_list, bhv_list, id_list, drive_mat_list),
                                           desc=f'BHV: {on} index',
                                           total=len(meta_list)):

        # add mm condition
        meta = utils.add_reversal_mismatch_condition_to_meta(meta)
        meta = utils.add_4stagesv2_to_meta(meta)

        # parse index type define trials you calc index on
        if on == 'go':
            hit_boo = meta.trialerror.isin([0, 3, 5]) #3,5
            cr_boo = meta.trialerror.isin([1, 2, 4]) #2, 4
        elif on == 'correct':
            hit_boo = meta.trialerror.isin([0, 2, 4])
            cr_boo = meta.trialerror.isin([1, 3, 5])
        elif on == 'th':
            prev_same = (meta.prev_same_plus | meta.prev_same_neutral | meta.prev_same_minus)
            hit_boo = ~prev_same
            cr_boo = prev_same
        elif on == 'rh' or on == 'reward':
            # reward history, focusing only on FC-FC trials
            hit_boo = meta.prev_reward & (meta.prev_same_plus | meta.prev_same_neutral | meta.prev_same_minus)
            cr_boo = ~meta.prev_reward & (meta.prev_same_plus | meta.prev_same_neutral | meta.prev_same_minus)
        elif on == 'ph' or on == 'punishment':
            # punishment history, focusing only on QC-QC trials
            hit_boo = meta.prev_punish & (meta.prev_same_plus | meta.prev_same_neutral | meta.prev_same_minus)
            cr_boo = ~meta.prev_punish & (meta.prev_same_plus | meta.prev_same_neutral | meta.prev_same_minus)
        elif on == 'rh2' or on == 'reward2':
            # reward history, focusing only on FC-FC trials
            hit_boo = meta.prev_reward #& meta.trialerror.isin([0])
            cr_boo = ~meta.prev_reward
        elif on in  ['pupil', 'speed', 'lick']:
            # if pupul, normalize pupil to max area per day
            if on == 'pupil':
                raise NotImplementedError # you need thresholds 
                bhv_num = 0
                bhv_type = 'pupil'
                norm_tensor = np.zeros(bhv.shape) + np.nan
                days = pd.unique(meta.reset_index().date)
                for c, di in enumerate(days):
                    date_boo = meta.reset_index().date.isin([di]).values
                    norm_tensor[:, :, date_boo] = bhv[:, :, date_boo] / np.nanmax(bhv[:, :, date_boo])
                bhv = norm_tensor
                if bhv_baseline_or_stim == 'baseline':
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore')
                        bhv_mean = np.nanmean(bhv[bhv_num, :17, :], axis=0)
                    hit_boo = bhv_mean >= 10 
                    cr_boo = bhv_mean <= 5 
                elif bhv_baseline_or_stim == 'stim':
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore')
                        bhv_mean = np.nanmean(bhv[bhv_num, 17:, :], axis=0)
                    hit_boo = bhv_mean >= 10 
                    cr_boo = bhv_mean <= 5 
                else:
                    raise ValueError
            elif on == 'speed':
                bhv_num = 4
                bhv_type = 'speed'
                if bhv_baseline_or_stim == 'baseline':
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore')
                        bhv_mean = np.nanmean(bhv[bhv_num, :17, :], axis=0)
                    hit_boo = bhv_mean >= 10 
                    cr_boo = bhv_mean <= 5 
                elif bhv_baseline_or_stim == 'stim':
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore')
                        bhv_mean = np.nanmean(bhv[bhv_num, 17:, :], axis=0)
                    hit_boo = bhv_mean >= 10 
                    cr_boo = bhv_mean <= 5 
                else:
                    raise ValueError
            elif on == 'lick':
                raise NotImplementedError # you need thresholds 
                bhv_num = 2
                bhv_type = 'lick'
                if bhv_baseline_or_stim == 'baseline':
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore')
                        bhv_mean = np.nanmean(bhv[bhv_num, :17, :], axis=0)
                    hit_boo = bhv_mean >= 10 
                    cr_boo = bhv_mean <= 5 
                elif bhv_baseline_or_stim == 'stim':
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore')
                        bhv_mean = np.nanmean(bhv[bhv_num, 17:, :], axis=0)
                    hit_boo = bhv_mean >= 10 
                    cr_boo = bhv_mean <= 5 
                else:
                    raise ValueError
            else:
                raise ValueError

        # get cell mean during 2s stim or response window
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            trial_mean = np.nanmean(tensor[:, 17:, :], axis=1)

        # get your cell category vector checking that ids are in correct order
        mouse_cats = on_ens['cell_cats'][on_ens['mouse_vec'] == utils.meta_mouse(meta)]
        mouse_ids = on_ens['cell_vec'][on_ens['mouse_vec'] == utils.meta_mouse(meta)]
        mouse_mouse = on_ens['mouse_vec'][on_ens['mouse_vec'] == utils.meta_mouse(meta)]
        assert np.array_equal(mouse_ids, ids)

        # preallocate your funny unknown number of stage days
        stage_day_n = meta.groupby(['parsed_4stagev2', 'parsed_11stage', 'date']).count().shape[0]

        # get all unique stage day combos in order
        stage_day_df = (meta.groupby(['parsed_4stagev2', 'parsed_11stage',
                                      'date']).count().reindex(lookups.staging['parsed_4stagev3'],
                                                               level=0).reindex(lookups.staging['parsed_11stage'],
                                                                                level=1))

        # preallocate
        stageday_hit_mat = np.zeros((tensor.shape[0], stage_day_n)) + np.nan
        stageday_cr_mat = np.zeros((tensor.shape[0], stage_day_n)) + np.nan

        # use index from row iterator to get stage-day boolean
        for c, (ind, _) in enumerate(stage_day_df.iterrows()):

            # boolean
            stage_boo = meta.parsed_11stage.isin([ind[1]]).values
            day_boo = meta.reset_index().date.isin([ind[2]]).values
            sdb = stage_boo & day_boo

            # index each cell only for it's preferred cue
            cues = ['becomes_unrewarded', 'becomes_rewarded', 'remains_unrewarded']
            for cue in cues:
                cue_cells = np.isin(mouse_cats, cat_groups_pref_tuning[cue])
                if cue_cells.sum() == 0:
                    continue
                cue_trials = meta.mismatch_condition.isin([cue])
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    pref_cue_hit_trials = sdb & hit_boo & cue_trials
                    pref_cue_cr_trials = sdb & cr_boo & cue_trials
                    stageday_hit_mat[cue_cells, c] = np.nanmean(trial_mean[cue_cells, :][:, pref_cue_hit_trials], axis=1)
                    stageday_cr_mat[cue_cells, c] = np.nanmean(trial_mean[cue_cells, :][:, pref_cue_cr_trials], axis=1)

            # deal with special tuning cases --> broad
            if any(['broad' in s for s in cat_groups_pref_tuning.keys()]):
                cue_cells = np.isin(mouse_cats, cat_groups_pref_tuning['broad'])
                if cue_cells.sum() == 0:
                    continue
                cue_trials = meta.mismatch_condition.isin(cues)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    pref_cue_hit_trials = sdb & hit_boo & cue_trials
                    pref_cue_cr_trials = sdb & cr_boo & cue_trials
                    stageday_hit_mat[cue_cells, c] = np.nanmean(trial_mean[cue_cells, :][:, pref_cue_hit_trials], axis=1)
                    stageday_cr_mat[cue_cells, c] = np.nanmean(trial_mean[cue_cells, :][:, pref_cue_cr_trials], axis=1)

            # deal with special tuning cases --> joint-tuned
            if any(['joint' in s for s in cat_groups_pref_tuning.keys()]):
                jcues = [s for s in cat_groups_pref_tuning.keys() if 'joint' in s]
                for cue in jcues:
                    cue_cells = np.isin(mouse_cats, cat_groups_pref_tuning[cue])
                    if cue_cells.sum() == 0:
                        continue
                    cue_trials = meta.mismatch_condition.isin([s for s in cues if s in cue])
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore')
                        pref_cue_hit_trials = sdb & hit_boo & cue_trials
                        pref_cue_cr_trials = sdb & cr_boo & cue_trials
                        stageday_hit_mat[cue_cells, c] = np.nanmean(trial_mean[cue_cells, :][:, pref_cue_hit_trials], axis=1)
                        stageday_cr_mat[cue_cells, c] = np.nanmean(trial_mean[cue_cells, :][:, pref_cue_cr_trials], axis=1)

        # remove undriven days from calc
        if driven_only:
            stageday_hit_mat[dr == 0] = np.nan
            stageday_cr_mat[dr == 0] = np.nan

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')

            # take index averaging days within stages, then across all stages for hit/cr sets
            # index is then calculated off of a single value for hit/cr for each cell
            # NOTE hit being larger wwill make the index positive (so consider that assigning groups)
            mouse_11_4_index_list = []
            mouse_vals_cr = []
            mouse_vals_hit = []
            for stage_type in ['parsed_11stage', 'parsed_4stagev2']:
                hit_cr = []
                for mat in [stageday_hit_mat, stageday_cr_mat]:
                    # take mean over stages
                    new_mat = np.zeros((tensor.shape[0], 11)) + np.nan
                    for sc, stage in enumerate(lookups.staging[stage_type]):
                        stage_boo = stage_day_df.reset_index()[stage_type].isin([stage])
                        new_mat[:, sc] = np.nanmean(mat[:, stage_boo], axis=1)
                    # normalize to max response over stages
                    # new_mat = new_mat/np.nanmax(new_mat, axis=1)[:, None]
                    hit_cr.append(np.nanmean(new_mat, axis=1))
                
                hit_vec = hit_cr[0]
                cr_vec = hit_cr[1]
                # if index_type == 'ab_ab':
                # removes cells that are negative in both vecs
                double_neg = (hit_vec < 0) & (cr_vec < 0)
                hit_vec[double_neg] = np.nan
                cr_vec[double_neg] = np.nan
                # rectify cells that are negative in either vec
                hit_vec[hit_vec < 0] = 0 #.00000001
                cr_vec[cr_vec < 0] = 0 #.00000001
                # # if cells have a zero value in one vec, force other value to be >= 0.00999
                # any_zero_cr = (hit_vec == 0) & (cr_vec < 0.01)
                # any_zero_hit = (cr_vec == 0) & (hit_vec < 0.01)
                # hit_vec[any_zero_hit] = 0 #.00000001
                # cr_vec[any_zero_cr] = 0 #.00000001

                index_vec = (hit_vec - cr_vec) / (hit_vec + cr_vec)
                # index_vec = (hit_vec - cr_vec) / ((hit_vec + cr_vec)/2)
                # index_vec = hit_vec - cr_vec
                mouse_11_4_index_list.append(index_vec)
                mouse_vals_hit.append(hit_vec)
                mouse_vals_cr.append(cr_vec)

            # also calcualte the unweightted, simple average across days
            hit_vec_simp = np.nanmean(stageday_hit_mat, axis=1)
            cr_vec_simp = np.nanmean(stageday_cr_mat, axis=1)
            index_vec_simp = (hit_vec_simp - cr_vec_simp) / (hit_vec_simp + cr_vec_simp)

        data = {
            'mouse': mouse_mouse,
            'cell_id': mouse_ids,
            'cell_cats': mouse_cats,
            f'mean_bhv{save_tag}_{on}_index_2s_stageday': index_vec_simp,
            f'mean_bhv{save_tag}_{on}_index_2s_11stage': mouse_11_4_index_list[0],
            f'mean_bhv{save_tag}_{on}_index_2s_4stage': mouse_11_4_index_list[1],
            'cr_raw11': mouse_vals_cr[0],
            'hit_raw11': mouse_vals_hit[0],
        }
        index_df_2s = pd.DataFrame(data=data).set_index(['mouse', 'cell_id'])
        df_list.append(index_df_2s)

    index_df_all = pd.concat(df_list, axis=0)
    if save_please:
        index_df_all.to_pickle(
            os.path.join(lookups.coreroot, f'{match_to}_{bhv_baseline_or_stim}_bhv{save_tag}_{on}_index_2s_df.pkl'))

    return index_df_all

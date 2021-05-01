from . import lookups, utils, drive, load
import numpy as np
import pandas as pd
import warnings
import os


def bhv_corr(match_to='onsets', bhv_type='speed', bhv_baseline_or_stim='stim', save_please=False):
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
    save_please : boolean
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
        bhv_list = ondict['bhv_list']
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
    # for meta, tensor, bhv, ids, dr in zip(ldict['meta_list'], offdict['tensor_list'], ldict['bhv_list'], offdict['id_list'], drive_mat_list_off):
    for meta, tensor, bhv, ids, dr in zip(meta_list, tensor_list, bhv_list, id_list, drive_mat_list):
        
        # add mm condition 
        meta = utils.add_reversal_mismatch_condition_to_meta(meta)
        meta = utils.add_4stagesv2_to_meta(meta)

        # if pupul, normalize pupil to max area per day
        if bhv_num == 0:
            norm_tensor = np.zeros(tensor.shape) + np.nan
            days = pd.unique(meta.reset_index().date)
            for c, di in enumerate(days):
                date_boo = meta.reset_index().date.isin([di]).values
                norm_tensor[:, :, date_boo] = tensor[:, :, date_boo] / np.nanmax(tensor[:, :, date_boo])
            tensor = norm_tensor
        
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
        stage_day_df = (
            meta
            .groupby(['parsed_4stagev2', 'parsed_11stage', 'date'])
            .count()
            .reindex(lookups.staging['parsed_4stagev3'], level=0)
            .reindex(lookups.staging['parsed_11stage'], level=1)
        )
        
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
                print('empty trial chunk')
                continue
            
            # only use cells that exist for corr
            existing_data = np.isfinite(trial_chunk[:, 0])
            
            # correlate each cell only for it's preferred cue
            ## TODO ADD JOINT HANDLING
            cues = ['becomes_unrewarded', 'becomes_rewarded', 'remains_unrewarded']
            for cue in cues:
                cue_cells = np.isin(mouse_cats, cat_groups_pref_tuning[cue])
                if (existing_data & cue_cells).sum() == 0:
                    continue
                cue_trials = meta_chunk.mismatch_condition.isin([cue])
                corr = np.corrcoef(
                    np.concatenate(
                        [trial_chunk[existing_data & cue_cells, :][:, cue_trials], bhv_chunk[None, cue_trials]],
                        axis=0
                    )
                )
                corr_with_bhv = corr[:-1, -1]
                stageday_corr_mat[existing_data & cue_cells, c] = corr_with_bhv
                
            # deal with special tuning cases
            cue_cells = np.isin(mouse_cats, cat_groups_pref_tuning['broad'])
            if (existing_data & cue_cells).sum() == 0:
                continue
            cue_trials = meta_chunk.mismatch_condition.isin(cues)
            corr = np.corrcoef(
                np.concatenate(
                    [trial_chunk[existing_data & cue_cells, :][:, cue_trials], bhv_chunk[None, cue_trials]],
                    axis=0
                )
            )
            corr_with_bhv = corr[:-1, -1]
            stageday_corr_mat[existing_data & cue_cells, c] = corr_with_bhv
    #         print('hitt this')
            
        # remove undriven days from calc
        stageday_corr_mat[drive_mat_list == 0] = np.nan
        
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
                
        mod_dfs.append(stagemean_corr_mat)
        mod_dfs_4stage.append(stagemean_corr_mat2)

    corr_avg4 = np.concatenate(mod_dfs_4stage, axis=0)
    corr_avg11 = np.concatenate(mod_dfs, axis=0)

    data = {
        'mouse': on_ens['mouse_vec'],
        'cell_id': on_ens['cell_vec'],
        'cell_cats': on_ens['cell_cats'],
        f'mean_{bhv_type}_corr_2s_11stage': np.nanmean(corr_avg11, axis=1),
        f'mean_{bhv_type}_corr_2s_4stage': np.nanmean(corr_avg4, axis=1),
    }
    corr_df_2s = pd.DataFrame(data=data).set_index(['mouse', 'cell_id'])
    if save_please:
        corr_df_2s.to_pickle(os.path.join(lookups.coreroot, f'{match_to}_{bhv_baseline_or_stim}_{bhv_type}_corr_2s_df.pkl'))
    
    return corr_df_2s       
                
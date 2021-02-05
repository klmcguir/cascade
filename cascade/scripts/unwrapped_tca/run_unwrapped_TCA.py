"""Script to run TCA, save outputs, and save simple plots of model results."""
import cascade as cas
import pandas as pd
import numpy as np
import tensortools as tt
from copy import deepcopy


# parameters
# --------------------------------------------------------------------------------------------------
# drive params
thresh = 4

# input data params
mice = cas.lookups.mice['allFOV']
words = ['respondent' if s in 'OA27' else 'computation' for s in mice]
group_by = 'all3'
with_model = False
nan_thresh = 0.95

# TCA params
method = 'ncp_hals'
replicates = 3
fit_options = {'tol': 0.0001, 'max_iter': 500, 'verbose': False}
ranks = list(np.arange(1, 21, dtype=int))
ranks.extend([40])
ranks.extend([80])
tca_ranks = [int(s) for s in ranks]

# load in a full size data
# --------------------------------------------------------------------------------------------------
tensor_list = []
id_list = []
bhv_list = []
meta_list = []
for mouse, word in zip(mice, words):

    # return   ids, tensor, meta, bhv
    out = cas.load.load_all_groupday(mouse, word=word, with_model=with_model,
                                     group_by=group_by, nan_thresh=nan_thresh)
    tensor_list.append(out[2])
    id_list.append(out[1])
    bhv_list.append(out[4])
    meta_list.append(cas.utils.add_stages_to_meta(out[3], 'parsed_11stage'))

# load Oren's better offset classification
# --------------------------------------------------------------------------------------------------
off_df_all_mice = pd.read_pickle('/twophoton_analysis/Data/analysis/core_dfs/offsets_dfs.pkl')
df_list = []
for k, v in off_df_all_mice.items():
    v['mouse'] = k
    df_list.append(v)
updated_off_df = pd.concat(df_list, axis=0)
updated_off_df = updated_off_df.set_index(['mouse', 'cell_id'])
updated_off_df.head()

# build input for TCA, selecting cells driven any stage with a given negative log10 p-value 'thresh'
# --------------------------------------------------------------------------------------------------
driven_on_tensor_flat_run = []
driven_off_tensor_flat_run = []
driven_on_tensor_flat = []
driven_off_tensor_flat = []
off_mouse_vec_flat, on_mouse_vec_flat, off_cell_vec, on_cell_vec = [], [], [], []
for meta, ids, tensor in zip(meta_list, id_list, tensor_list):

    # skip LM and seizure mouse
    if cas.utils.meta_mouse(meta) in ['AS20', 'AS23']:
        continue
    if cas.utils.meta_mouse(meta) in ['AS41', 'AS47', 'OA38']:
        continue
    if cas.utils.meta_mouse(meta) in cas.lookups.mice['lml5']:
        continue

    # calculate drivenness across stages, using Oren's offset boolean
    off_df = updated_off_df.loc[updated_off_df.reset_index().mouse.isin([cas.utils.meta_mouse(meta)]).values, ['offset_test']]
    off_df = off_df.reindex(ids, level=1)
    assert np.array_equal(off_df.reset_index().cell_id.values, ids)
    offset_bool = off_df.offset_test.values
    drive_df = cas.drive.multi_stat_drive(meta, ids, tensor, alternative='less', offset_bool=offset_bool, neg_log10_pv_thresh=4)

    # flatten tensor and unwrap it to look across stages
    flat_tensors = {}
    off_flat_tensors = {}
    for cue in ['plus', 'minus', 'neutral']:
        meta_bool = meta.initial_condition.isin([cue]).values

        # get mean per stage per cue (also split on running speed)
        stage_mean_tensor_slow = cas.utils.balanced_mean_per_stage(meta, tensor, meta_bool=meta_bool, filter_running='low_pre_speed_only')
        stage_mean_tensor_fast = cas.utils.balanced_mean_per_stage(meta, tensor, meta_bool=meta_bool, filter_running='high_pre_speed_only')
        stage_mean_tensor = cas.utils.balanced_mean_per_stage(meta, tensor, meta_bool=meta_bool)

        # limit to 2 seconds across mice
        flat_tensors[cue] = cas.utils.unwrap_tensor(stage_mean_tensor[:, :int(np.ceil(15.5*3)), :])
        flat_tensors[cue + '_slow'] = cas.utils.unwrap_tensor(stage_mean_tensor_slow[:, :int(np.ceil(15.5*3)), :])
        flat_tensors[cue + '_fast'] = cas.utils.unwrap_tensor(stage_mean_tensor_fast[:, :int(np.ceil(15.5*3)), :])

        # for offset, use 1 second pre offset and 2 seconds post, assumes 15.5 Hz
        off_int = int(np.ceil(cas.lookups.stim_length[cas.utils.meta_mouse(meta)] + 1)*15.5)
        off_flat_tensors[cue] = cas.utils.unwrap_tensor(stage_mean_tensor[:, (off_int - 15):(off_int + 42), :])
        off_flat_tensors[cue + '_slow'] = cas.utils.unwrap_tensor(stage_mean_tensor_slow[:, (off_int - 15):(off_int + 42), :])
        off_flat_tensors[cue + '_fast'] = cas.utils.unwrap_tensor(stage_mean_tensor_fast[:, (off_int - 15):(off_int + 42), :])

    # get driven ids for different behaviors
    driven_onset = []
    driven_offset = []
    for cc, cue in enumerate(['plus', 'minus', 'neutral']):
        for c, stages in enumerate(cas.lookups.staging['parsed_11stage']):

            # Onset cells
            on_cells = drive_df.loc[~drive_df.offset_cell & drive_df.driven]
            on_cells = on_cells.loc[on_cells.reset_index().parsed_11stage.isin([stages]).values, :]
            on_cells = on_cells.loc[on_cells.reset_index().initial_cue.isin([cue]).values, :]
            # make sure you can't double count cells
            assert on_cells.groupby(['mouse', 'cell_id']).nunique().gt(1).sum(axis=0).eq(0).all()
            id_vec = on_cells.cell_id.unique()
            driven_onset.extend(id_vec)

            # Offset cells
            off_cells = drive_df.loc[drive_df.offset_cell & drive_df.driven]
            off_cells = off_cells.loc[off_cells.reset_index().parsed_11stage.isin([stages]).values, :]
            off_cells = off_cells.loc[off_cells.reset_index().initial_cue.isin([cue]).values, :]
            # make sure you can't double count cells
            assert off_cells.groupby(['mouse', 'cell_id']).nunique().gt(1).sum(axis=0).eq(0).all()
            id_vec = off_cells.cell_id.unique()
            driven_offset.extend(id_vec)

    driven_onset = np.unique(driven_onset)
    driven_offset = np.unique(driven_offset)
    driven_on_bool = np.isin(np.array(ids), driven_onset)
    driven_off_bool = np.isin(np.array(ids), driven_offset)
    on_cells_in_order = np.array(ids)[driven_on_bool]
    off_cells_in_order = np.array(ids)[driven_off_bool]

    # keep track of mice and cell ids
    on_mouse_vec_flat.append([cas.utils.meta_mouse(meta)] * len(on_cells_in_order))
    off_mouse_vec_flat.append([cas.utils.meta_mouse(meta)] * len(off_cells_in_order))
    on_cell_vec.append(on_cells_in_order)
    off_cell_vec.append(off_cells_in_order)

    # ensure that cells are not counted in both onset and offset groups
    assert all(~np.isin(driven_onset, driven_offset))

    # Onset cells
    ten_list = []
    for cc, cue in enumerate(['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']):
        invert_lookup = {v:k for k, v in cas.lookups.lookup_mm[cas.utils.meta_mouse(meta)].items()}
        ten_list.append(flat_tensors[invert_lookup[cue]][:,:,None])
    new_on_tensor_unwrap = np.dstack(ten_list)[driven_on_bool, :, :]
    driven_on_tensor_flat.append(new_on_tensor_unwrap)

    # Offset cells
    ten_list = []
    for cc, cue in enumerate(['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']):
        invert_lookup = {v:k for k, v in cas.lookups.lookup_mm[cas.utils.meta_mouse(meta)].items()}
        ten_list.append(off_flat_tensors[invert_lookup[cue]][:,:,None])
    new_off_tensor_unwrap = np.dstack(ten_list)[driven_off_bool, :, :]
    driven_off_tensor_flat.append(new_off_tensor_unwrap)

    # Onset cells with speed
    ten_list = []
    for speedi in ['_fast', '_slow']:
        for cue in ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']:
            invert_lookup = {v:k for k, v in cas.lookups.lookup_mm[cas.utils.meta_mouse(meta)].items()}
            ten_list.append(flat_tensors[invert_lookup[cue] + speedi][:,:,None])
    new_onset_tensor_unwrap = np.dstack(ten_list)[driven_on_bool, :, :]
    driven_on_tensor_flat_run.append(new_onset_tensor_unwrap)

    # Offset cells with speed
    ten_list = []
    for speedi in ['_fast', '_slow']:
        for cue in ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']:
            invert_lookup = {v:k for k, v in cas.lookups.lookup_mm[cas.utils.meta_mouse(meta)].items()}
            ten_list.append(off_flat_tensors[invert_lookup[cue] + speedi][:,:,None])
    new_offset_tensor_unwrap = np.dstack(ten_list)[driven_off_bool, :, :]
    driven_off_tensor_flat_run.append(new_offset_tensor_unwrap)

# concatenate tensors
on_mega_tensor_flat_run = np.vstack(driven_on_tensor_flat_run)
off_mega_tensor_flat_run = np.vstack(driven_off_tensor_flat_run)
on_mega_tensor_flat = np.vstack(driven_on_tensor_flat)
off_mega_tensor_flat = np.vstack(driven_off_tensor_flat)

# concatenate lists of mice and cells
on_mega_mouse_flat = np.hstack(on_mouse_vec_flat)
off_mega_mouse_flat = np.hstack(off_mouse_vec_flat)
on_mega_cell_flat = np.hstack(on_cell_vec)
off_mega_cell_flat = np.hstack(off_cell_vec)

# make sure that you get the shapes you expected
assert on_mega_tensor_flat_run.shape[0] == on_mega_tensor_flat.shape[0]
assert off_mega_tensor_flat_run.shape[0] == off_mega_tensor_flat.shape[0]
assert len(on_mega_mouse_flat) == len(on_mega_cell_flat)
assert len(off_mega_mouse_flat) == len(off_mega_cell_flat)
assert len(on_mega_cell_flat) == on_mega_tensor_flat.shape[0]
assert len(off_mega_cell_flat) == off_mega_tensor_flat.shape[0]

# Normalize per cell to max
cell_max = np.nanmax(np.nanmax(off_mega_tensor_flat, axis=1), axis=1)
off_mega_tensor_flat_norm = off_mega_tensor_flat / cell_max[:, None, None]
cell_max = np.nanmax(np.nanmax(on_mega_tensor_flat, axis=1), axis=1)
on_mega_tensor_flat_norm = on_mega_tensor_flat / cell_max[:, None, None]
cell_max = np.nanmax(np.nanmax(off_mega_tensor_flat_run, axis=1), axis=1)
off_mega_tensor_flat_run_norm = off_mega_tensor_flat_run / cell_max[:, None, None]
cell_max = np.nanmax(np.nanmax(on_mega_tensor_flat_run, axis=1), axis=1)
on_mega_tensor_flat_run_norm = on_mega_tensor_flat_run / cell_max[:, None, None]

# run TCA
# --------------------------------------------------------------------------------------------------
data_dict = {
    'v4_on': on_mega_tensor_flat,
    'v4_off': off_mega_tensor_flat,
    'v4_norm_on': on_mega_tensor_flat_norm,
    'v4_norm_off': off_mega_tensor_flat_norm,
    'v4_speed_on': on_mega_tensor_flat_run,
    'v4_speed_off': off_mega_tensor_flat_run,
    'v4_speed_norm_on': on_mega_tensor_flat_run_norm,
    'v4_speed_norm_off': off_mega_tensor_flat_run_norm
}
ensemble = {}

for k, v in data_dict.items():
    mask = ~np.isnan(v)
    fit_options['mask'] = mask  
    ensemble[k] = tt.Ensemble(fit_method=method, fit_options=deepcopy(fit_options))
    ensemble[k].fit(v, ranks=tca_ranks, replicates=3, verbose=True)

# save ensembe and input data
# --------------------------------------------------------------------------------------------------
np.save(cas.paths.analysis_file('tca_ensemble_v4_20210205.npy', 'tca_dfs'), ensemble, allow_pickle=True)

# add mouse and cell data to dict then save
data_dict['v4_off_mouse'] = off_mega_mouse_flat
data_dict['v4_on_mouse'] = on_mega_mouse_flat
data_dict['v4_off_cell'] = off_mega_cell_flat
data_dict['v4_on_cell'] = on_mega_cell_flat
np.save(cas.paths.analysis_file('input_data_v4_20210205.npy', 'tca_dfs'), data_dict, allow_pickle=True)

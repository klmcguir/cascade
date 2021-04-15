import numpy as np
import pandas as pd
from . import lookups


def pick_comp_order(cell_cats, cell_sorter):
    """Move cells (sorted by componenets) into a user defined order.

    Parameters
    ----------
    cell_cats : numpy.ndarray
        Cell categorization vector, cell categories as integers.
    cell_sorter : numpy.ndarray
        Sort order defined from sort based on "best" component.
        See:  cascade.utils.sort_and_rescale_factors()

    Raises
    ------
    NotImplementedError
        Offsets do not yet have a defined order.
    ValueError
        This assumes we are using a rank 8 model for offset and rank 9 for onsets. 
    """

    # parse the type of data based on number of ranks
    all_comps = np.unique(cell_cats)
    no_supp_cells = all_comps[all_comps >= 0]
    n_comps = len(no_supp_cells)
    if n_comps == 8:
        dataset = 'offsets'
        # order assumed from pd.unique(cell_cats[cell_sorter])
        assumed_order = [ 3,  7,  4,  5,  6,  1,  2,  0, -1]
        fixed_order = lookups.fixed_component_sort['rank8_offset'] + [-1]
    elif n_comps == 9:
        dataset = 'onsets'
        # order assumed from pd.unique(cell_cats[cell_sorter])
        assumed_order = [ 0,  5,  6,  4,  8,  1,  2,  3,  7, -1]
        # fixed_order = [6, 5, 0, 4, 8, 2, 1, 3, 7, -1]
        fixed_order = lookups.fixed_component_sort['rank9_onset'] + [-1]
    else:
        raise ValueError
    
    # sort your components into order in chunks
    orig_desired_order = cell_cats[cell_sorter]
    existing_order = pd.unique(orig_desired_order)
    assert np.array_equal(assumed_order, existing_order)
    reordered_vec = []
    for fi in fixed_order:
        cat_chunk = orig_desired_order == fi
        reordered_vec.append(cell_sorter[cat_chunk])
    reordered_vec = np.hstack(reordered_vec)

    return reordered_vec


def sort_by_cue_mouse(mat, mouse_vec):
    """Sort an unwrapped trace matrix by mouse and within mouse by cue.

    Parameters
    ----------
    mat : numpy array
        cells x times-stages x cues
    mouse_vec : list or array
        vector of mouse names or unique identifiers

    Returns
    -------
    numpy array
        return the argsort order for this specific type of sort
    """

    best_cue_vec = np.argmax(np.nanmax(mat, axis=1), axis=1)
    peaktime = np.nanargmax(np.nanmean(np.nanmax(mat, axis=2).reshape([mat.shape[0], -1, 47]), axis=1), axis=1)
    peaktime[peaktime<17] = 47

    sort_vec = []
    for mouse in np.unique(mouse_vec):

        # first sort by cue
        mouse_inds = np.where(mouse_vec == mouse)[0]
        bcue_vec = best_cue_vec[mouse_vec == mouse]
        peak_vec = peaktime[mouse_vec == mouse]
        cue_sort = np.argsort(bcue_vec)
        mouse_cue_sort = mouse_inds[cue_sort]
        peak_vec_sort = peak_vec[cue_sort]
        cue_vec_sort = bcue_vec[cue_sort]

        # second sort by average peak time
        for a in np.unique(bcue_vec):
            cue_peaks = peak_vec_sort[cue_vec_sort == a]
            new_sort = mouse_cue_sort[cue_vec_sort == a][np.argsort(cue_peaks)]
            sort_vec.extend(new_sort)
    
    return np.array(sort_vec, dtype=int)


def sort_by_cue_peak(mat, mouse_vec):
    """Sort an unwrapped trace matrix by cue and peak.

    Parameters
    ----------
    mat : numpy array
        cells x times-stages x cues
    mouse_vec : list or array
        vector of mouse names or unique identifiers

    Returns
    -------
    numpy array
        return the argsort order for this specific type of sort
    """

    best_cue_vec = np.argmax(np.nanmax(mat, axis=1), axis=1)
    best_cue_vec[best_cue_vec == 1] = 4 ## MOVE the remains unrewarded group manually to the end 
    peaktime = np.nanargmax(np.nanmean(np.nanmax(mat, axis=2).reshape([mat.shape[0], -1, 47]), axis=1), axis=1)
    peaktime[peaktime<17] = 47

    # first sort by cue
    mouse_inds = np.arange(len(mouse_vec), dtype=int)
    cue_sort = np.argsort(best_cue_vec)
    mouse_cue_sort = mouse_inds[cue_sort]
    peak_vec_sort = peaktime[cue_sort]
    cue_vec_sort = best_cue_vec[cue_sort]

    # second sort by average peak time
    sort_vec = []
    for a in np.unique(best_cue_vec):
        cue_peaks = peak_vec_sort[cue_vec_sort == a]
        new_sort = mouse_cue_sort[cue_vec_sort == a][np.argsort(cue_peaks)]
        sort_vec.extend(new_sort)
    
    return np.array(sort_vec, dtype=int)


def sort_by_cue_peak_wbroad(mat, mouse_vec):
    """Sort an unwrapped trace matrix by cue and peak. Put broad at bottom. 

    Parameters
    ----------
    mat : numpy array
        cells x times-stages x cues
    mouse_vec : list or array
        vector of mouse names or unique identifiers

    Returns
    -------
    numpy array
        return the argsort order for this specific type of sort
    """

    best_cue_vec = np.argmax(np.nanmax(mat, axis=1), axis=1)
    best_cue_vec[best_cue_vec == 1] = 4 ## MOVE the remains unrewarded group manually to the end 
    # define cells whose max cue responses were within 75% of maximum (to all cues) as broadly tuned
    broad_true = np.all(np.nanmax(mat, axis=1)/np.nanmax(np.nanmax(mat, axis=1), axis=1)[:, None] > 0.75, axis=1)
    best_cue_vec[broad_true] = 5
    peaktime = np.nanargmax(np.nanmean(np.nanmax(mat, axis=2).reshape([mat.shape[0], -1, 47]), axis=1), axis=1)
    peaktime[peaktime<17] = 47

    # first sort by cue
    mouse_inds = np.arange(len(mouse_vec), dtype=int)
    cue_sort = np.argsort(best_cue_vec)
    mouse_cue_sort = mouse_inds[cue_sort]
    peak_vec_sort = peaktime[cue_sort]
    cue_vec_sort = best_cue_vec[cue_sort]

    # second sort by average peak time
    sort_vec = []
    for a in np.unique(best_cue_vec):
        cue_peaks = peak_vec_sort[cue_vec_sort == a]
        new_sort = mouse_cue_sort[cue_vec_sort == a][np.argsort(cue_peaks)]
        sort_vec.extend(new_sort)
    
    return np.array(sort_vec, dtype=int)


def run_corr_sort(mouse_vec, cell_vec, data_dict, mod, stim_or_baseline_corr='stim'):
    """Sort an unwrapped trace matrix by cue and (stimulus or baseline) running correlation.

    Parameters
    ----------
    mat : numpy array
        cells x times-stages x cues
    mouse_vec : list or array
        vector of mouse names or unique identifiers

    Returns
    -------
    numpy array
        return the argsort order for this specific type of sort
    """
    
    # this will force the best cue preference to be done on the mean across all trial data
    if '_on' in mod:
        best_mod = 'v4i10_norm_on_noT0'
    elif '_off' in mod:
        best_mod = 'v4i10_norm_off_noT0'
    best_cue_vec = np.argmax(np.nanmax(data_dict[best_mod], axis=1), axis=1)
    
    if stim_or_baseline_corr == 'stim':
        corr_df = pd.read_pickle(f'/twophoton_analysis/Data/analysis/Group-attractive/cell_stim_runcorr.pkl')
    else:
        corr_df = pd.read_pickle(f'/twophoton_analysis/Data/analysis/Group-attractive/cell_baseline_runcorr.pkl')
    
    corr_dfs = []
    for mi in np.unique(mouse_vec):
        good_cells = cell_vec[mouse_vec == mi]
        df = corr_df.loc[corr_df.reset_index().mouse.isin([mi]).values & corr_df.reset_index().cell_id.isin(good_cells).values, :]
        df = df.groupby(['mouse', 'cell_id']).mean().drop(columns=['cell_n', 'L5_run_corr', 'R5_run_corr'])
        assert len(df.reset_index().cell_id) == len(good_cells)
        assert np.array_equal(df.reset_index().cell_id.values, good_cells)
    #     print(np.nansum(df.reset_index().cell_id.values - good_cells))
        corr_dfs.append(df)
    matched_corr_df = pd.concat(corr_dfs, axis=0)
    assert np.array_equal(matched_corr_df.reset_index().cell_id.values, cell_vec)
    corr_vec = matched_corr_df.mean_run_corr.values

    # first sort by cue
    mouse_inds = np.arange(len(mouse_vec), dtype=int)
    cue_sort = np.argsort(best_cue_vec)
    mouse_cue_sort = mouse_inds[cue_sort]
    peak_vec_sort = corr_vec[cue_sort]
    cue_vec_sort = best_cue_vec[cue_sort]

    # second sort by average peak time
    sort_vec = []
    for a in np.unique(best_cue_vec):
        cue_peaks = peak_vec_sort[cue_vec_sort == a]
        new_sort = mouse_cue_sort[cue_vec_sort == a][np.argsort(cue_peaks)]
        sort_vec.extend(new_sort)
    
    return np.array(sort_vec, dtype=int)


def run_corr_sort_nobroad(mouse_vec, cell_vec, cell_cats, tca_cell_sorter,
                          data_dict, mod, stim_or_baseline_corr='stim'):
    """Sort an unwrapped trace matrix by cue and (stimulus or baseline) running correlation.

    Parameters
    ----------
    mat : numpy array
        cells x times-stages x cues
    mouse_vec : list or array
        vector of mouse names or unique identifiers

    Returns
    -------
    numpy array
        return the argsort order for this specific type of sort
    """
    
    # this will force the best cue preference to be done on the mean across all trial data
    if '_on' in mod:
        best_mod = 'v4i10_norm_on_noT0'
    elif '_off' in mod:
        raise ValueError
#         best_mod = 'v4i10_norm_off_noT0'
    best_cue_vec = np.argmax(np.nanmax(data_dict[best_mod], axis=1), axis=1)
    
    if stim_or_baseline_corr == 'stim':
        corr_df = pd.read_pickle(f'/twophoton_analysis/Data/analysis/Group-attractive/cell_stim_runcorr.pkl')
    else:
        corr_df = pd.read_pickle(f'/twophoton_analysis/Data/analysis/Group-attractive/cell_baseline_runcorr.pkl')
    
    corr_dfs = []
    for mi in np.unique(mouse_vec):
        good_cells = cell_vec[mouse_vec == mi]
        df = corr_df.loc[corr_df.reset_index().mouse.isin([mi]).values & corr_df.reset_index().cell_id.isin(good_cells).values, :]
        df = df.groupby(['mouse', 'cell_id']).mean().drop(columns=['cell_n', 'L5_run_corr', 'R5_run_corr'])
        assert len(df.reset_index().cell_id) == len(good_cells)
        assert np.array_equal(df.reset_index().cell_id.values, good_cells)
    #     print(np.nansum(df.reset_index().cell_id.values - good_cells))
        corr_dfs.append(df)
    matched_corr_df = pd.concat(corr_dfs, axis=0)
    assert np.array_equal(matched_corr_df.reset_index().cell_id.values, cell_vec)
    corr_vec = matched_corr_df.mean_run_corr.values
    
    # first define the chunk that will be left at the end
    if'_on' in mod:
        sorted_cats = cell_cats[tca_cell_sorter]
        # CELL CATS are in an arbitrary order! 
        trans_and_suppress = (sorted_cats == -1) | (sorted_cats == 7)
        end_chunk = tca_cell_sorter[trans_and_suppress]
        needs_sort = sorted(tca_cell_sorter[~trans_and_suppress])
    else:
        raise ValueError
    
    # first sort by cue
    mouse_inds = np.array(needs_sort, dtype=int)
    cue_sort = np.argsort(best_cue_vec[needs_sort])
    mouse_cue_sort = mouse_inds[cue_sort]
    peak_vec_sort = corr_vec[needs_sort][cue_sort]
    cue_vec_sort = best_cue_vec[needs_sort][cue_sort]

    # second sort by average peak time
    sort_vec = []
    for a in np.unique(best_cue_vec):
        cue_peaks = peak_vec_sort[cue_vec_sort == a]
        new_sort = mouse_cue_sort[cue_vec_sort == a][np.argsort(cue_peaks)]
        sort_vec.extend(new_sort)
    sort_vec.extend(end_chunk)
    sort_vec = np.array(sort_vec, dtype=int)
    
    assert len(sort_vec) == len(mouse_vec)
    
    return sort_vec
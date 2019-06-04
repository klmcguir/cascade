"""Functions for calculating FC bias."""
import pandas as pd
import numpy as np
import flow
import pool
import os
import matplotlib.pyplot as plt
import seaborn as sns
from . import paths
from . import tca
from . import utils

def build_tensor(
        mouse,
        tags=None,

        # grouping params
        group_by='all',
        up_or_down='up',
        use_dprime=False,
        dprime_threshold=2,

        # tensor params
        trace_type='zscore_day',
        cs='',
        downsample=True,
        start_time=-1,
        end_time=6,
        clean_artifacts=None,
        thresh=20,
        warp=False,
        smooth=True,
        smooth_win=5,
        nan_trial_threshold=None,
        verbose=True,

        # filtering params
        exclude_tags=('disengaged', 'orientation_mapping', 'contrast', 'retinotopy', 'sated'),
        exclude_conds=('blank', 'blank_reward', 'pavlovian'),
        driven=True,
        drive_css=('0', '135', '270'),
        drive_threshold=15,
        drive_type='visual'):

    """
    Perform tensor component analysis (TCA) on data aligned
    across a group of days. Builds one large tensor.

    Algorithms from https://github.com/ahwillia/tensortools.

    Parameters
    -------
    methods, tuple of str
        'cp_als', fits CP Decomposition using Alternating
            Least Squares (ALS).
        'ncp_bcd', fits nonnegative CP Decomposition using
            the Block Coordinate Descent (BCD) Method.
        'ncp_hals', fits nonnegative CP Decomposition using
            the Hierarchical Alternating Least Squares
            (HALS) Method.
        'mcp_als', fits CP Decomposition with missing data using
            Alternating Least Squares (ALS).

    rank, int
        number of components you wish to fit

    replicates, int
        number of initializations/iterations fitting for each rank

    Returns
    -------

    """

    # set grouping parameters
    if group_by.lower() == 'naive':
        tags = 'naive'
        use_dprime = False
        exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                        'retinotopy', 'sated', 'learning_start')

    elif group_by.lower() == 'high_dprime_learning':
        use_dprime = True
        up_or_down = 'up'
        tags = 'learning'
        exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                        'retinotopy', 'sated', 'learning_start',
                        'reversal1_start')

    elif group_by.lower() == 'low_dprime_leanrning':
        use_dprime = True
        up_or_down = 'down'
        tags = 'learning'
        exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                        'retinotopy', 'sated', 'learning_start')

    elif group_by.lower() == 'high_dprime_reversal1':
        use_dprime = True
        up_or_down = 'up'
        tags = 'reversal1'
        exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                        'retinotopy', 'sated', 'reversal2_start')

    elif group_by.lower() == 'low_dprime_reversal1':
        use_dprime = True
        up_or_down = 'down'
        tags = 'reversal1'
        exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                        'retinotopy', 'sated')

    elif group_by.lower() == 'high_dprime_reversal2':
        use_dprime = True
        up_or_down = 'up'
        tags = 'reversal2'
        exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                        'retinotopy', 'sated')

    elif group_by.lower() == 'low_dprime_reversal2':
        use_dprime = True
        up_or_down = 'down'
        tags = 'reversal2'
        exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                        'retinotopy', 'sated')

    elif group_by.lower() == 'naive_vs_high_dprime':
        use_dprime = True
        up_or_down = 'up'
        tags = None
        days = flow.DateSorter.frommeta(mice=[mouse], tags='naive')
        days.extend(flow.DateSorter.frommeta(mice=[mouse], tags='learning'))
        dates = set(days)
        exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                        'retinotopy', 'sated', 'learning_start',
                        'reversal1_start')

    elif group_by.lower() == 'l_vs_r1':  # high dprime
        use_dprime = True
        up_or_down = 'up'
        tags = None
        days = flow.DateSorter.frommeta(mice=[mouse], tags='learning')
        days.extend(flow.DateSorter.frommeta(mice=[mouse], tags='reversal1'))
        dates = set(days)
        exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                        'retinotopy', 'sated', 'learning_start',
                        'reversal1_start')

    elif group_by.lower() == 'all':
        tags = None
        use_dprime = False
        if mouse == 'OA27':
            exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                            'retinotopy', 'sated', 'learning_start',
                            'reversal1_start', 'reversal2_start')
        else:
            exclude_tags = ('disengaged', 'orientation_mapping', 'contrast',
                            'retinotopy', 'sated')


    else:
        print('Using input parameters without modification by group_by=...')

    # create folder structure and save dir
    pars = {'tags': tags,
            'trace_type': trace_type, 'cs': cs, 'downsample': downsample,
            'start_time': start_time, 'end_time': end_time,
            'clean_artifacts': clean_artifacts, 'thresh': thresh,
            'warp': warp, 'smooth': smooth, 'smooth_win': smooth_win,
            'exclude_tags': exclude_tags, 'exclude_conds': exclude_conds,
            'driven': driven, 'drive_css': drive_css,
            'drive_threshold': drive_threshold}
    group_pars = {'group_by': group_by, 'up_or_down': up_or_down,
                  'use_dprime': use_dprime,
                  'dprime_threshold': dprime_threshold}
    # save_dir = paths.tca_path(mouse, 'group', pars=pars, group_pars=group_pars)

    # get DateSorter object
    if np.isin(group_by.lower(), ['naive_vs_high_dprime', 'l_vs_r1']):
        days = flow.DateSorter(dates=dates)
    else:
        days = flow.DateSorter.frommeta(mice=[mouse], tags=tags)

    # filter DateSorter object if you are filtering on dprime
    if use_dprime:
        dprime = []
        for day1 in days:
            # for comparison with naive make sure dprime keeps naive days
            if np.isin('naive', day1.tags):
                if up_or_down.lower() == 'up':
                    dprime.append(np.inf)
                else:
                    dprime.append(-np.inf)
            else:
                dprime.append(pool.calc.performance.dprime(day1))
        if up_or_down.lower() == 'up':
            days = [d for c, d in enumerate(days) if dprime[c]
                    > dprime_threshold]
        elif up_or_down.lower() == 'down':
            days = [d for c, d in enumerate(days) if dprime[c]
                    <= dprime_threshold]

    # preallocate for looping over a group of days/runs
    meta_list = []
    tensor_list = []
    id_list = []
    for c, day1 in enumerate(days, 0):

        # get cell_ids
        d1_ids = flow.xday._read_crossday_ids(day1.mouse, day1.date)
        d1_ids = np.array([int(s) for s in d1_ids])

        # filter cells based on visual/trial drive across all cs, prevent
        # breaking when only pavs are shown
        if driven:
            good_ids = tca._group_drive_ids(
                days, drive_css, drive_threshold, drive_type=drive_type)
            d1_ids_bool = np.isin(d1_ids, good_ids)
            d1_sorter = np.argsort(d1_ids[d1_ids_bool])
        else:
            d1_ids_bool = np.ones(np.shape(d1_ids)) > 0
            d1_sorter = np.argsort(d1_ids[d1_ids_bool])
        ids = d1_ids[d1_ids_bool][d1_sorter]

        # TODO add in additional filter for being able to check for quality of xday alignment

        # get all runs for both days
        d1_runs = day1.runs()

        # filter for only runs without certain tags
        d1_runs = [run for run in d1_runs if not
                   any(np.isin(run.tags, exclude_tags))]

        # build tensors for all correct runs and trials after filtering
        if d1_runs:
            d1_tensor_list = []
            d1_meta = []
            for run in d1_runs:
                t2p = run.trace2p()
                # trigger all trials around stimulus onsets
                run_traces = utils.getcstraces(
                    run, cs=cs, trace_type=trace_type,
                    start_time=start_time, end_time=end_time,
                    downsample=True, clean_artifacts=clean_artifacts,
                    thresh=thresh, warp=warp, smooth=smooth,
                    smooth_win=smooth_win)
                # filter and sort
                run_traces = run_traces[d1_ids_bool, :, :][d1_sorter, :, :]
                # get matched trial metadata/variables
                dfr = _trialmetafromrun(run)
                # subselect metadata if you are only running certain cs
                if cs != '':
                    if cs == 'plus' or cs == 'minus' or cs == 'neutral':
                        dfr = dfr.loc[(dfr['condition'].isin([cs])), :]
                    elif cs == '0' or cs == '135' or cs == '270':
                        dfr = dfr.loc[(dfr['orientation'].isin([cs])), :]
                    else:
                        print('ERROR: cs called - "' + cs + '" - is not\
                              a valid option.')

                # subselect metadata to remove certain conditions
                if len(exclude_conds) > 0:
                    run_traces = run_traces[:, :, (~dfr['condition'].isin(exclude_conds))]
                    dfr = dfr.loc[(~dfr['condition'].isin(exclude_conds)), :]

                # drop trials with nans and add to lists
                keep = np.sum(np.sum(np.isnan(run_traces), axis=0,
                              keepdims=True),
                              axis=1, keepdims=True).flatten() == 0
                dfr = dfr.iloc[keep, :]
                d1_tensor_list.append(run_traces[:, :, keep])
                d1_meta.append(dfr)

            # concatenate matched cells across trials 3rd dim (aka, 2)
            tensor = np.concatenate(d1_tensor_list, axis=2)

            # concatenate all trial metadata in pd dataframe
            meta = pd.concat(d1_meta, axis=0)

            meta_list.append(meta)
            tensor_list.append(tensor)
            id_list.append(ids)

    # get total trial number across all days/runs
    meta = pd.concat(meta_list, axis=0)
    trial_num = len(meta.reset_index()['trial_idx'])

    # get union of ids. Use these for indexing and splicing tensors together
    id_union = np.unique(np.concatenate(id_list, axis=0))
    cell_num = len(id_union)

    # build a single large tensor leaving zeros where cell is not found
    trial_start = 0
    trial_end = 0
    group_tensor = np.zeros((cell_num, np.shape(tensor_list[0])[1], trial_num))
    group_tensor[:] = np.nan
    for i in range(len(tensor_list)):
        trial_end += np.shape(tensor_list[i])[2]
        for c, k in enumerate(id_list[i]):
            celln_all_trials = tensor_list[i][c, :, :]
            group_tensor[(id_union == k), :, trial_start:trial_end] = celln_all_trials
        trial_start += np.shape(tensor_list[i])[2]

    # allow for cells with low number of trials to be dropped
    if nan_trial_threshold:
        # update saving tag
        nt_tag = '_nantrial' + str(nan_trial_threshold)
        # remove cells with too many nan trials
        ntrials = np.shape(group_tensor)[2]
        nbadtrials = np.sum(np.isnan(group_tensor[:, 0, :]), 1)
        badtrialratio = nbadtrials/ntrials
        badcell_indexer = badtrialratio < nan_trial_threshold
        group_tensor = group_tensor[badcell_indexer, :, :]
        if verbose:
            print('Removed ' + str(np.sum(~badcell_indexer)) +
                  ' cells from tensor:' + ' badtrialratio < ' +
                  str(nan_trial_threshold))
            print('Kept ' + str(np.sum(badcell_indexer)) +
                  ' cells from tensor:' + ' badtrialratio < ' +
                  str(nan_trial_threshold))
    else:
        nt_tag = ''

    # just so you have a clue how big the tensor is
    if verbose:
        print('Tensor built: tensor shape = ' + str(np.shape(group_tensor)))

    return group_tensor, meta, id_union


def bias_df(
        mice=['OA27', 'OA26', 'VF226', 'OA67', 'CC175'],

        # trace params
        trace_type='zscore_day',
        cs='',
        downsample=True,
        start_time=-1,
        end_time=6,
        clean_artifacts=None,
        thresh=20,
        warp=False,
        smooth=True,
        smooth_win=5,

        # drive params
        driven=True,
        drive_css=['plus', 'minus', 'neutral'],
        stim_offset=1,
        drive_thresh=5):

    pars = {'trace_type': trace_type, 'cs': cs, 'downsample': downsample,
            'start_time': start_time, 'end_time': end_time,
            'clean_artifacts': clean_artifacts, 'thresh': thresh,
            'warp': warp, 'smooth': smooth, 'smooth_win': smooth_win}

    xmouse_bias = []
    xmouse_dprime = []
    xmouse_lstate = []
    for mouse in mice:

        """
        Get mean response per cell.
        """

        # get all days for a mouse
        learning_state = []
        xday_bias = []
        xday_norm_response = []
        xday_dprime = []
        days = flow.DateSorter.frommeta(mice=[mouse])
        for DaySorter in days:

            # get all cell ids
            d1_ids = flow.xday._read_crossday_ids(DaySorter.mouse,
                                                  DaySorter.date)
            d1_ids = np.array([int(s) for s in d1_ids])

            # filter cells based on visual drive across all cs, prevent
            # breaking when only pavs are shown
            if driven:
                d1_drive = []
                for dcs in drive_css:
                    try:
                        d1_drive.append(
                            pool.calc.driven.visually(DaySorter, dcs))
                    except KeyError:
                        print(
                            str(DaySorter) + ' requested ' + dcs +
                            ': no match to what was shown (probably pav only)')
                d1_drive = np.max(d1_drive, axis=0) > drive_thresh
                cells = d1_ids[d1_drive]
            else:
                cells = d1_ids

            # get traces for the day
            dft = _singleday(DaySorter, pars)
            dft = dft.reset_index(level=['cell_idx', 'timestamp'])

            # filter out cells which are not driven
            cell_indexer = dft['cell_idx'].isin(cells)
            dft = dft.loc[cell_indexer, :]

            # keep only times when stim is on the screen
            time_indexer = dft['timestamp'].between(
                0, stim_offset, inclusive=False)
            dft = dft.loc[time_indexer, :]

            # get metadata for the day
            save_dir = os.path.join(flow.paths.outd, str(DaySorter.mouse))
            meta_path = os.path.join(save_dir, str(DaySorter.mouse) +
                                     '_df_trialmeta.pkl')
            dfm = pd.read_pickle(meta_path)

            # ensure that
            dfm = _update_naive_conditions(dfm)

            # filter metadata trials before merging
            responses = []
            for dcs in drive_css:
                trial_indexer = (
                                ((dfm.orientation == 0) |
                                 (dfm.orientation == 135) |
                                 (dfm.orientation == 270))
                                &
                                ((dfm.learning_state == 'naive') |
                                 (dfm.learning_state == 'learning_start') |
                                 (dfm.learning_state == 'learning') |
                                 (dfm.learning_state == 'reversal1') |
                                 (dfm.learning_state == 'reversal1_start') |
                                 (dfm.learning_state == 'reversal2') |
                                 (dfm.learning_state == 'reversal2_start'))
                                &
                                ((dfm.condition == dcs))
                                &
                                ((dfm.tag == 'standard'))
                                &
                                (dfm.hunger == 'hungry'))
                dfcs = dfm.loc[trial_indexer, :]

                # merge on filtered trials
                dff = pd.merge(
                    dft, dfcs, on=['mouse', 'date', 'run', 'trial_idx'],
                    how='inner')

                # check that df is not empty, skip dfs that filtering empties
                if dff.empty:
                    responses = [[np.nan, np.nan, np.nan],
                                 [np.nan, np.nan, np.nan],
                                 [np.nan, np.nan, np.nan]]
                    print('Day: ' + str(DaySorter.date) +
                          ': skipped: empty dataframe after merge.')
                    break

                # smooth signal with rolling 3 unit window
                # if smooth:
                #     dff['trace'] = dff['trace'].rolling(3).mean()

                trial_mean = dff.pivot_table(
                    index=['cell_idx', 'trial_idx'],
                    columns='timestamp',
                    values='trace').mean(axis=1).to_frame()
                cell_mean = trial_mean.pivot_table(
                    index=['cell_idx'],
                    columns=['trial_idx']).mean(axis=1).tolist()
                cell_mean = np.array(cell_mean)
                cell_mean[cell_mean < 0] = np.nan  # 0 original
                responses.append(cell_mean)

# CHECK ON THIS KEY ERROR! all days that make it here have the three stimuli!
            try:
                xday_dprime.append(pool.calc.behavior.dprime(DaySorter, hmm_engaged=False))
            except KeyError:
                xday_dprime.append(np.nan)

            ressy = np.array(responses)
            FC = ressy[0, :]/(ressy[0, :] + ressy[1, :] + ressy[2, :])
            QC = ressy[1, :]/(ressy[0, :] + ressy[1, :] + ressy[2, :])
            NC = ressy[2, :]/(ressy[0, :] + ressy[1, :] + ressy[2, :])
            print('new bias: FC-', np.nanmean(FC), ' QC-',
                  np.nanmean(QC), ' NC-', np.nanmean(NC))
            xday_bias.append(np.nanmean(FC))
            learning_state.append(np.unique(dff['learning_state']))

        xmouse_lstate.append(learning_state)
        xmouse_bias.append(xday_bias)
        xmouse_dprime.append(xday_dprime)

    early = []
    late = []
    during = []
    post = []
    during2 = []
    post2 = []
    std_naive = []
    std_early = []
    std_late = []
    std_during = []
    std_post = []
    std_during2 = []
    std_post2 = []
    dprime_thresh = 2
    dprime_thresh2 = 2
    for mouse in range(len(mice)):
        dprime = xmouse_dprime[mouse]
        lstate = xmouse_lstate[mouse]
        bias = xmouse_bias[mouse]
        xdp = []
        xls = []
        xbias =[]
        for c, dp, ls in enumerate(zip(dprime, lstate)):
            xbias.append(bias[c])
            xdp.append(dp)
            xls.append(ls)
        xbias = np.array(xbias)
        xbias[xbias > 4] = np.nan
        xdp = np.array(xdp)
        test = []
        for s in range(len(xls)):
            if len(xls[s]) == 0:
                test.append(np.nan)
            else:
                test.append(xls[s][0])
        xls = test

        naive.append(np.nanmean(xbias[np.isin(xls, 'naive').flatten()]))
        early.append(
            np.nanmean(xbias[
                (np.isin(xls, 'learning').flatten() |
                 np.isin(xls, 'learning_start').flatten()) &
                (xdp < dprime_thresh)]))
        late.append(
            np.nanmean(xbias[
                np.isin(xls, 'learning').flatten() & (xdp > dprime_thresh2)]))
        during.append(
            np.nanmean(xbias[
                (np.isin(xls, 'reversal1_start').flatten() |
                 np.isin(xls, 'reversal1').flatten()) &
                (xdp < dprime_thresh)]))
        post.append(
            np.nanmean(xbias[
                np.isin(xls, 'reversal1').flatten() & (xdp > dprime_thresh2)]))
        during2.append(
            np.nanmean(xbias[
                (np.isin(xls, 'reversal2_start').flatten() |
                 np.isin(xls, 'reversal2').flatten()) &
                (xdp < dprime_thresh)]))
        post2.append(
            np.nanmean(xbias[
                np.isin(xls, 'reversal2').flatten() & (xdp > dprime_thresh2)]))

        std_naive.append(np.nanstd(xbias[np.isin(xls, 'naive').flatten()]))
        std_early.append(
            np.nanstd(xbias[
                (np.isin(xls, 'learning').flatten() |
                 np.isin(xls, 'learning_start').flatten()) &
                (xdp < dprime_thresh)]))
        std_late.append(
            np.nanstd(xbias[
                np.isin(xls, 'learning').flatten() & (xdp > dprime_thresh2)]))
        std_during.append(
            np.nanstd(xbias[
                (np.isin(xls, 'reversal1_start').flatten() |
                 np.isin(xls, 'reversal1').flatten()) &
                (xdp < dprime_thresh)]))
        std_post.append(
            np.nanstd(xbias[
                np.isin(xls, 'reversal1').flatten() & (xdp > dprime_thresh2)]))
        std_during2.append(
            np.nanstd(xbias[
                (np.isin(xls, 'reversal2_start').flatten() |
                 np.isin(xls, 'reversal2').flatten()) &
                (xdp < dprime_thresh)]))
        std_post2.append(
            np.nanstd(xbias[
                np.isin(xls, 'reversal2').flatten() & (xdp > dprime_thresh2)]))

    fig = plt.figure()
    cmap = sns.color_palette("hls", 7)
    # cmap = sns.color_palette("cubehelix", 7)
    for c, mouse in enumerate(zip(naive, early, late, during, post, during2, post2)):
        yerr = np.array([std_naive[c], std_early[c], std_late[c], std_during[c], std_post[c], std_during2[c], std_post2[c]])
    #     plt.errorbar((0,1,2,3,4,5,6), mouse, '-o', yerr=yerr, label=mice[c])
    #     sns.lineplot(x=(0,1,2,3,4,5,6), y=mouse, label=mice[c])
        plt.plot(mouse, '-o', label=mice[c], color=cmap[c])
        plt.title('FC bias')
        plt.ylabel('FC bias')
        plt.xlabel('Learning stage')
        plt.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
    ax = fig.axes
    x = plt.xlim()
    ax[0].set_xticks(range(0,7))
    ax[0].set_xticklabels(['naive', 'early', 'late', 'during', 'post', 'during2', 'post2'])
    plt.plot(x, (0.33, 0.33), ':k')
    figpath = os.path.join(flow.paths.graphd, 'FC bias')
    if not os.path.isdir(figpath): os.mkdir(figpath)
    figpath = os.path.join(figpath, 'FCbias.eps')
    # plt.savefig(figpath, bbox_inches='tight')

    # plot with seaborn
    org_data = [naive, early, late, during, post, during2, post2]
    my_x = []
    my_y = []
    for c, i in enumerate(org_data):
        my_x.extend([c]*len(i))
        my_y.extend(list(i))

    fig2 = plt.figure()
    sns.lineplot(x=my_x, y=my_y, palette="muted")
    # plt.plot(mouse, '-o', label=mice[c])
    plt.title('FC bias')
    plt.ylabel('FC bias')
    plt.xlabel('Learning stage')
    # plt.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)

    figpath = os.path.join(flow.paths.graphd, 'FC bias')
    if not os.path.isdir(figpath): os.mkdir(figpath)
    figpath = os.path.join(figpath, 'FCbias_mean_across_mice.pdf')
    # plt.savefig(figpath, bbox_inches='tight')

    ax1 = fig2.axes
    x = plt.xlim()
    ax1[0].set_xticks(range(0, 7))
    ax1[0].set_xticklabels([
        'naive', 'early', 'late', 'during', 'post', 'during2', 'post2'])
    plt.plot(x, (0.33, 0.33), ':k')
    # plt.savefig(figpath, bbox_inches='tight')


def _singleday(DaySorter, pars):
    """
    Build df for a single day loading/indexing efficiently.

    Parameters
    ----------
    DaySorter : obj
    pars : dict

    Returns
    -------
    pandas df
        df of all traces over all days for a cell
    """

    # assign folder structure for loading and load
    save_dir = paths.df_path(DaySorter.mouse, pars=pars)
    path = os.path.join(
        save_dir, str(DaySorter.mouse) + '_' + str(DaySorter.date) + '_df_' +
        pars['trace_type'] + '.pkl')
    dft = pd.read_pickle(path)

    # slice out your day of interest
    day_indexer = dft.index.get_level_values('date') == DaySorter.date
    dft = dft.loc[day_indexer, :]

    return dft


def _update_naive_conditions(dfm):
    """
    Function to ensure that standard training naive runs match the
    conditions of the learning learning_state. This ensure that slicing
    on condition rather than orientation will still make sense. Treat
    naive "pavlovian" as "plus".

    Parameters:
    -----------
    dfm : pandas dataframe
        Dataframe of cross-day trial metadata for a mouse.

    Returns:
    --------
    dfm : pandas dataframe
        Updated dataframe with matched naive/learning standard training
        conditions.
    """

    plus = np.unique(
        dfm.loc[
            ((dfm.condition == 'plus') &
             (dfm.learning_state == 'learning')), ['orientation']].values)
    neutral = np.unique(
        dfm.loc[
            ((dfm.condition == 'neutral') &
             (dfm.learning_state == 'learning')), ['orientation']].values)
    minus = np.unique(
        dfm.loc[
            ((dfm.condition == 'minus') &
             (dfm.learning_state == 'learning')), ['orientation']].values)
    pav = np.unique(
        dfm.loc[
            ((dfm.condition == 'pavlovian') &
             (dfm.learning_state == 'learning')), ['orientation']].values)
    oris = np.array([plus, minus, neutral, pav])
    oris = [int(s) for s in oris]
    conds = list(['plus', 'minus', 'neutral', 'plus'])
    # last entry, pavlovian treated as plus above
    for c, ori in enumerate(oris):
        dfm.loc[((dfm.orientation == ori) &
                 (dfm.learning_state == 'naive') &
                 (dfm.tag == 'standard')), 'condition'] = conds[c]

    return dfm

"""Functions for calculating FC bias."""
import pandas as pd
import numpy as np
import flow
import pool
import os
import matplotlib.pyplot as plt
import seaborn as sns


def bias_df(
        mice=['OA27', 'OA26', 'VF226', 'OA67', 'CC175'],
        trace_type='dff',
        driven=True,
        drive_css=['plus', 'minus', 'neutral'],
        stim_offset=1,
        smooth=False,
        drive_thresh=5):

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
            dft = _singleday(DaySorter, trace_type=trace_type)
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
                                 [np.nan, np.nan, np.nan]]
                    print('Day: ' + str(DaySorter.date) +
                          ': skipped: empty dataframe after merge.')
                    break

                # smooth signal with rolling 3 unit window
                if smooth:
                    dff['trace'] = dff['trace'].rolling(3).mean()

                trial_mean = dff.pivot_table(
                    index=['cell_idx', 'trial_idx'],
                    columns='timestamp',
                    values='trace').mean(axis=1).to_frame()
                cell_mean = trial_mean.pivot_table(
                    index=['cell_idx'],
                    columns=['trial_idx']).mean(axis=1).tolist()
                cell_mean = np.array(cell_mean)
                cell_mean[cell_mean < 0] = 0
                responses.append(cell_mean)

# CHECK ON THIS KEY ERROR! all days that make it here have the three stimuli!
            try:
                xday_dprime.append(pool.calc.behavior.dprime(DaySorter, hmm_engaged=False))
            except KeyError:
                xday_dprime.append(np.nan)

            max_response = np.array([np.nanmax(s) for s in zip(*responses)])
            min_response = 0
            norm_response = [
                (np.array(s)-min_response)/max_response for s in responses]
            bias = np.array(norm_response[0])/np.array([
                np.nansum(cell) for cell in zip(*norm_response)])
            print(np.nanmean(bias))
            xday_bias.append(np.nanmean(bias))
            learning_state.append(np.unique(dff['learning_state']))

        xmouse_lstate.append(learning_state)
        xmouse_bias.append(xday_bias)
        xmouse_norm_response.append(xday_norm_response)
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


def _singleday(DaySorter, trace_type='dff'):
    """
    Build df for a single day loading/indexing efficiently.

    Parameters
    ----------
    mouse : str, mouse
    date : str, date
    trace_type : str, {'dff', 'zscore', 'deconvolved'}

    Returns
    -------
    pandas df
        df of all traces over all days for a cell
    """

    # assign folder structure for loading and load
    save_dir = os.path.join(
        flow.paths.outd, str(DaySorter.mouse), 'dfs ' + str(trace_type))
    path = os.path.join(
        save_dir, str(DaySorter.mouse) + '_' + str(DaySorter.date) + '_df_' +
        trace_type + '.pkl')
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

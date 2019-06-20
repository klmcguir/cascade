"""Functions for plotting xday dataframes."""
import os
import flow
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
from .. import df
from .. import tca
from .. import paths


def mean_response(
        mouse,
        tags=None,

        # drive params
        drive_type='trial',

        # trace params
        trace_type='zscore_day',
        word=None,
        drive_threshold=15,  # usually 15
        drive_css=('0', '135', '270'),
        smooth=False,

        verbose=True):
    """
    Create line plot of each cell's mean response to each
    simulus aligned across time.

    Parameters:
    -----------
    mouse : str
        mouse name
    drive_type : str
        trial, visual, inbibition
    tags : [str]
        list of metadata tags

    Returns:
    --------
    Saves figures to .../analysis folder
    """

    # use seaborn defaults for matplotlib
    sns.set()

    # get drive across all days
    days = flow.DateSorter.frommeta(mice=[mouse], tags=tags)
    all_driven_trial = tca._group_drive_ids(
        days, drive_css, drive_threshold, drive_type='trial')
    all_driven_vis = tca._group_drive_ids(
        days, drive_css, drive_threshold, drive_type='visual')
    all_driven_inhib = tca._group_drive_ids(
        days, drive_css, drive_threshold, drive_type='inhib')
    drivers = {'trial': all_driven_trial,
               'visual': all_driven_vis,
               'inhibition': all_driven_inhib}

    # load metadata
    load_dir = os.path.join(flow.paths.outd, str(mouse))
    meta_path = os.path.join(load_dir, str(mouse) + '_df_trialmeta.pkl')
    dfm = pd.read_pickle(meta_path)
    xmap = df.get_xdaymap(mouse)

    # oris
    oris = np.array([0, 135, 270])

    for cell_idx in drivers[drive_type]:
        dft = df.singlecell(mouse, trace_type, cell_idx, xmap=xmap, word=word)
        dft = dft.reset_index(level=['cell_idx', 'timestamp'])

        # filter metadata trials before merging
        trial_indexer = (
            ((dfm.orientation == 0) | (dfm.orientation == 135)
             | (dfm.orientation == 270)) &
            ((dfm.tag == 'standard') | (dfm.tag == 'learning_start')
             | (dfm.tag == 'reversal1_start') | (dfm.tag == 'reversal2_start')) &
            ((dfm.condition == 'plus') | (dfm.condition == 'minus')
             | (dfm.condition == 'neutral')) &
            (dfm.hunger == 'hungry'))
        dfm = dfm.loc[trial_indexer, :]

        # merge on filtered trials
        dff = pd.merge(dft, dfm, on=['mouse', 'date', 'run', 'trial_idx'],
                       how='inner')

        # smooth signal with rolling 3 unit window
        if smooth:
            dff['trace'] = dff['trace'].rolling(3).mean()

        # get timestamp info for plotting lines
        times = np.unique(dff['timestamp'])
        zero_sec = np.where(times <= 0)[0][-1]
        three_sec = np.where(times <= 3)[0][-1]

        # get pivot table for slicing
        toplot = dff.pivot_table(
            index=['date', 'run', 'trial_idx', 'orientation', 'learning_state'],
            columns='timestamp', values='trace')

        # create mean response per day per orientation for each cell
        mean_list = []
        for d in np.unique(toplot.reset_index()['date']):
            for o in oris:
                indexer = (np.where((toplot.reset_index()['orientation'] == o)
                           & (toplot.reset_index()['date'] == d))[0])
                mean_list.append(
                    toplot.iloc[indexer, :].mean(level=['date', 'orientation',
                                                        'learning_state']))
        mean_df = pd.concat(mean_list, axis=0)

        # PLOTTING

        # subplot params
        rows = 1
        cols = 3

        # get plot with most lines for figure legend
        count = []
        for col in range(cols):
            count.append(
                np.nansum(mean_df.reset_index()['orientation'] == oris[col]))
        legend_ind = np.argmax(count)
        color_ind = np.array(
            [np.where(np.unique(mean_df.reset_index()['date']) == i)[0][0]
                for i in mean_df.reset_index()['date']])

        # colormap
        day_num = len(np.unique(mean_df.reset_index()['date']))
        a = sns.color_palette("Greys", int(np.ceil(day_num*1.5)))[-day_num:]
        b = sns.color_palette("Greens", day_num)
        c = sns.color_palette("Reds", day_num)
        d = sns.color_palette("Purples", day_num)
        colors = {'naive': a, 'learning': b, 'reversal1': c, 'reversal2': d}

        # preallocate for legend
        labels = []

        # get driven tag for title
        ttag = 'trial, ' if np.isin(cell_idx, all_driven_trial) else ''
        vtag = 'visually, ' if np.isin(cell_idx, all_driven_vis) else ''
        itag = 'inhibition' if np.isin(cell_idx, all_driven_inhib) else ''
        title_tage = ttag + vtag + itag

        # plot responses for each orientation on each day
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4))
        fig.suptitle('Cell #' + str(cell_idx) + ' - driven: '
                     + title_tage, fontsize=16, y=1.05, weight='bold')
        for col in range(cols):
            indexer = np.where(
                mean_df.reset_index()['orientation'] == oris[col])[0]
            traces = mean_df.iloc[indexer, :].values
            dates = mean_df.reset_index()['date'].iloc[indexer]
            state = mean_df.reset_index()['learning_state'].iloc[indexer]
            c_ind = color_ind[indexer]
            axes[col].set_title('Orientation = ' + str(oris[col]))
            if col == 0:
                axes[col].set_ylabel(r'$\Delta$' + 'F/F (z-score)')
            axes[col].set_xlabel('Time from stimulus onset (sec)')
            for l in range(np.shape(traces)[0]):
                axes[col].plot(
                    times, traces[l, :], color=colors[state.iloc[l]][c_ind[l]],
                    label=(str(dates.iloc[l]) + '-' + state.iloc[l]))
                if col == legend_ind:
                    labels.append(str(dates.iloc[l]) + '-' + state.iloc[l])
        axes[-1].legend(
            axes[legend_ind].lines, labels, loc='upper left',
            bbox_to_anchor=(1.02, 1.03), title='Days')

        # match y-limits
        ys = []
        for col in range(cols):
            ys.extend(axes[col].get_ylim())
        maxy = np.max(ys)
        miny = np.min(ys)
        for col in range(cols):
            axes[col].set_ylim((miny, maxy))

        # add a line for stim onset and offset
        # NOTE: assumes downsample, 1 sec before onset, 3 sec stim
        for col in range(cols):
            y_lim = axes[col].get_ylim()
            ons = 0
            offs = 3
            axes[col].plot([ons, ons], y_lim, ':', color='#525252')
            axes[col].plot([offs, offs], y_lim, ':', color='#525252')

        # save
        save_dir = paths.df_plots(
            mouse, pars={'trace_type': trace_type},
            word=word, plot_type='traces')
        save_dir = os.path.join(save_dir, 'driven ' + drive_type
                                + ' ' + str(drive_threshold))
        if not os.path.isdir(save_dir): os.mkdir(save_dir)
        path = os.path.join(
            save_dir, str(mouse) + '_cell_' + str(cell_idx) + '_'
            + trace_type + '_' + drive_type + str(drive_threshold) + '.pdf')
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        if verbose:
            print('Cell: ' + str(cell_idx) + ': done.')


def heatmap(
        mouse,
        cell_id=None,
        trace_type='dff',
        cs_bar=True,
        day_bar=True,
        day_line=True,
        run_line=False,
        match_clim=True,
        quinine_ticks=False,
        ensure_ticks=False,
        lick_ticks=False,
        label_cbar=True,
        vmin=None,
        vmax=None,
        smooth=False,
        save_tag='',
        word=None,
        verbose=False):
    """
    Create heatmap of each cell aligned across time.

    Parameters:
    -----------
    mouse : str; mouse name
    trace_type : str; dff, zscore, deconvolved
    cs_bar : logical; add a bar to match cs on edge of plot
    day_bar : logical; add checkerboard to match days on cs bar
    day_line : logical; add a lines between days
    run_line : logical; ddd lines between runs

    Returns:
    --------
    Saves figures to .../analysis folder
    """

    # arthur's predetermined hex colors
    colors = {
        'orange': '#E86E0A',
        'red': '#D61E21',
        'gray': '#7C7C7C',
        'black': '#000000',
        'green': '#75D977',
        'mint': '#47D1A8',
        'purple': '#C880D1',
        'indigo': '#5E5AE6',
        'blue': '#47AEED',  # previously 4087DD
        'yellow': '#F2E205',
    }

    # cs to color mapping
    cs_colors = {
        'plus': 'mint',
        'minus': 'red',
        'neutral': 'blue',
        'pavlovian': 'mint',
        'naive': 'gray'
    }

    # checkerboard overlay or day_bar
    day_colors = {
         'A': '#FDFEFE',
         'B': '#7B7D7D'
    }

    # red=high, white=middle, blue=low colormap
    cmap = sns.diverging_palette(220, 10, sep=30, as_cmap=True)

    # load metadata
    save_dir = os.path.join(flow.paths.outd, str(mouse))
    meta_path = os.path.join(save_dir, str(mouse) + '_df_trialmeta.pkl')
    dfm = pd.read_pickle(meta_path)

    # create a binary map of every day a cell is present
    xmap = df.get_xdaymap(mouse)

    # set up range conditonal on input
    if cell_id is None:
        cell_range = range(1, np.shape(xmap)[0]+1)
    else:
        cell_range = cell_id

    # loop through cells and plot
    for cell_idx in cell_range:

        # create single cell df
        dft = df.singlecell(mouse, trace_type, cell_idx, xmap=xmap, word=word)
        dft = dft.reset_index(level=['cell_idx', 'timestamp'])

        # filter metadata trials before merging
        trial_indexer = (
            ((dfm.orientation == 0) | (dfm.orientation == 135)
             | (dfm.orientation == 270)) &
            ((dfm.tag == 'standard') | (dfm.tag == 'learning_start')
             | (dfm.tag == 'reversal1_start') | (dfm.tag == 'reversal2_start')) &
            ((dfm.condition == 'plus') | (dfm.condition == 'minus')
             | (dfm.condition == 'neutral')) &
            (dfm.hunger == 'hungry'))
        dfm = dfm.loc[trial_indexer, :]

        # merge on filtered trials
        dff = pd.merge(dft, dfm, on=['mouse', 'date', 'run', 'trial_idx'],
                       how='inner')

        # check that df is not empty, skip dfs that filtering empties
        if dff.empty:
            print('Cell: ' + str(cell_idx) + ': skipped: empty dataframe after merge.')
            continue

        # smooth signal with rolling 3 unit window
        if smooth:
            dff['trace'] = dff['trace'].rolling(3).mean()

        # get timestamp info for plotting lines
        times = np.unique(dff['timestamp'])
        zero_sec = np.where(times <= 0)[0][-1]
        three_sec = np.where(times <= 3)[0][-1]

        # oris
        oris = np.array([0, 135, 270])

        # cbarlabel
        if label_cbar:
            cbarlabel = r'$\DeltaF/F$'
        else:
            cbarlabel = None

        # plot main figure
        sns.set_context('talk')
        toplot = dff.pivot_table(
            index=['date', 'run', 'trial_idx', 'orientation'],
            columns='timestamp', values='trace')
        g = sns.FacetGrid(
            toplot.reset_index('orientation'), col='orientation',
            height=8, aspect=1, sharey=False, dropna=False)
        g.map_dataframe(
            _myheatmap, vmax=vmax, vmin=vmin, center=0,
            xticklabels=31, cmap=cmap, cbarlabel=cbarlabel)
        g.fig.suptitle('Cell ' + str(cell_idx), x=0.98)

        # loop through axes and plot relevant metadata on top
        count = 0
        cmin = []
        cmax = []
        ccmap = []
        for ax in g.axes[0]:

            # match vmin and max across plots (first check values)
            if match_clim and vmax is None:
                _vmin, _vmax = ax.collections[0].get_clim()
                cmin.append(_vmin)
                cmax.append(_vmax)
                ccmap.append(ax.collections[0].get_cmap())

            # get metadata for this orientation/set of trials
            meta = dff.loc[dff['orientation'] == oris[count], ['condition',
                           'ensure', 'quinine', 'firstlick', 'learning_state']]
            meta = meta.reset_index()
            meta = meta.drop_duplicates()
            ensure = np.array(meta['ensure'])
            quinine = np.array(meta['quinine'])
            firstlick = np.array(meta['firstlick'])
            css = meta['condition']
            learning_state = meta['learning_state']

            ori_inds = np.array(toplot.index.get_level_values('orientation'))
            ori_inds = ori_inds == oris[count]

            # set labels
            if count == 0:
                ax.set_ylabel('Trials')
            ax.set_xlabel('Time (sec)')

            # plot cs color bar/line
            if cs_bar:
                css[learning_state == 'naive'] = 'naive'
                for cs in np.unique(css):
                    cs_line_color = colors[cs_colors[cs]]
                    cs_y = np.where(css == cs)[0]
                    cs_y = [cs_y[0], cs_y[-1]+1]
                    ax.plot((2, 2), cs_y, color=cs_line_color, ls='-',
                            lw=15, alpha=0.8, solid_capstyle='butt')

            # find days where learning or reversal start
            if day_bar:
                days = np.array(toplot.index.get_level_values('date'))
                days = days[ori_inds]
                runs = np.array(toplot.index.get_level_values('run'))
                runs = runs[ori_inds]
                count_d = 0
                for day in np.unique(days):
                    day_y = np.where(days == day)[0]
                    day_y = [day_y[0], day_y[-1]+1]
                    day_bar_color = day_colors[sorted(day_colors.keys())[count_d%2]]
                    ax.plot((3.5, 3.5), day_y, color=day_bar_color, ls='-',
                            lw=6, alpha=0.4, solid_capstyle='butt')
                    count_d = count_d + 1

            # get limits for plotting
            y_lim = ax.get_ylim()
            x_lim = ax.get_xlim()

            # plot lines between days
            if day_line:
                days = np.array(toplot.index.get_level_values('date'))
                days = days[ori_inds]
                days = np.diff(days)
                day_ind = np.where(days > 0)[0]
                for y in day_ind:
                    day_y = [y+1, y+1]
                    ax.plot(x_lim, day_y, color='#8e8e8e', ls='-', lw=1, alpha=0.8)

            # plot lines between runs
            if run_line:
                runs = np.array(toplot.index.get_level_values('run'))
                runs = runs[ori_inds]
                runs = np.diff(runs)
                run_ind = np.where(runs > 0)[0]
                for y in run_ind:
                    run_y = [y+1, y+1]
                    ax.plot(x_lim, run_y, color='#bababa', ls='-', lw=1,  alpha=0.8)

            # plot onset/offest lines
            ax.plot((zero_sec, zero_sec), y_lim, color='#8e8e8e', ls='-', lw=2, alpha=0.8)
            ax.plot((three_sec, three_sec), y_lim, color='#8e8e8e', ls='-', lw=2, alpha=0.8)

            # if you allow caxis to scale automatically, add alpha to ticks
            if not vmin and not vmax:
                tick_alpha = 0.5
            else:
                tick_alpha = 1

            # plot quinine
            if quinine_ticks:
                for l in range(len(quinine)):
                    if np.isfinite(quinine[l]):
                        x = [quinine[l], quinine[l]]
                        y = [l+0.5, l+0.5]
                        ax.plot(x, y, color='#0fffc3', ls='',
                                marker='.', markersize=2, alpha=tick_alpha)

            # plot ensure
            if ensure_ticks:
                for l in range(len(ensure)):
                    if np.isfinite(ensure[l]):
                        x = [ensure[l], ensure[l]]
                        y = [l+0.5, l+0.5]
                        ax.plot(x, y, color='#ffb30f', ls='',
                                marker='.', markersize=2, alpha=tick_alpha)

            # plot licks
            if lick_ticks:
                for l in range(len(firstlick)):
                    if np.isfinite(firstlick[l]):
                        x = [firstlick[l], firstlick[l]]
                        y = [l+0.5, l+0.5]
                        ax.plot(x, y, color='#7237f2', ls='',
                                marker='.', markersize=2, alpha=tick_alpha)

            # reset yticklabels
            if y_lim[0] < 100:
                step = 10
            elif y_lim[0] < 200:
                step = 20
            elif y_lim[0] < 500:
                step = 50
            elif y_lim[0] < 5000:
                step = 500
            elif y_lim[0] < 10000:
                step = 1000
            elif y_lim[0] >= 10000:
                step = 5000
            base_yticks = range(int(y_lim[-1]), int(y_lim[0]), int(step))
            base_yticks = [s for s in base_yticks]
            base_ylabels = [str(s) for s in base_yticks]

            dates = np.array(toplot.index.get_level_values('date'))
            dates = dates[ori_inds]
            date_yticks = []
            date_label = []

            date_rel = flow.DateSorter.frommeta(mice=[mouse])
            date_rel = [s.date for s in date_rel]

            for day in np.unique(dates):

                # find number of inds needed to shift labels to put in middle of date block
                last_ind = np.where(dates == day)[0][-1]
                first_ind = np.where(dates == day)[0][0]
                shifter = np.round((last_ind - first_ind)/2)
                label_ind = last_ind - shifter

                # get your relative day number
                day_val = np.where(date_rel == day)[0][0] + 1  # add one to make it one-indexed

                # add a pad to keep labels left-justified
                if day_val < 10:
                    pad = '  '
                else:
                    pad = ''

                # if the date label and trial label inds are exactly the same
                # force the label info onto one line of text
                # label days with imaging day number
                if np.isin(label_ind, base_yticks):
                    # remove the existing ind and add a special label to end
                    good_tick = ~np.isin(base_yticks, label_ind)
                    base_yticks = [base_yticks[s] for s in range(len(good_tick))
                                   if good_tick[s]]
                    base_ylabels = [base_ylabels[s] for s in range(len(good_tick))
                                   if good_tick[s]]
                    dpad = '          '
                    dpad = dpad[0:(len('          ') - len(str(label_ind)*2))]
                    base_ylabels.append('Day ' + str(day_val) + dpad + str(label_ind))
                else:
                    base_ylabels.append('Day ' + str(day_val) + '          ' + pad)
                base_yticks.append(label_ind)

            ax.set_yticks(base_yticks)
            ax.set_yticklabels(base_ylabels)

            # reset xticklabels
            xticklabels = np.array([-1, 0, 1, 2, 3, 4, 5, 6])
            xticklabels = xticklabels[(xticklabels > times[0]) & (xticklabels < times[-1])]
            xticks = [np.where(times <= s)[0][-1] for s in xticklabels]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation='horizontal')

            # update count through loops
            count = count + 1

        # match vmin and max across plots (using cmax to choose cmap and cmin)
        scale_by = 0.7
        if match_clim and vmax is None:
            max_ind = np.nanargmax(cmax)
            cmin = cmin[max_ind]*scale_by
            cmax = cmax[max_ind]*scale_by
            ccmap = ccmap[max_ind]

            for ax in g.axes[0]:
                ax.collections[0].set_clim(vmax=cmax, vmin=cmin)
                ax.collections[0].set_cmap(ccmap)

        # save figures into folder structure
        save_dir = paths.df_plots(mouse, pars={'trace_type': trace_type},
                                  word=word)
        save_dir = os.path.join(save_dir, 'basic heatmaps' + save_tag)
        if not os.path.isdir(save_dir): os.mkdir(save_dir)
        path = os.path.join(save_dir, str(mouse) + '_cell_' + str(cell_idx) + '_' + trace_type + '.png')
        print('Cell: ' + str(cell_idx) + ': done.')
        g.savefig(path)

        if not verbose:
            plt.close()


def _myheatmap(data, **kwargs):
    """ Helper function for FacetGrid heatmap."""

    mydata = data.set_index('orientation', append=True)
    sns.heatmap(mydata, **kwargs)

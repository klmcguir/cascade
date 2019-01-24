"""Functions for plotting dataframes."""
import os
import flow
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .. import df


def xdayheatmap(mouse, cell_id=None, trace_type='dff', cs_bar=True, day_bar=True,
                day_line=True, run_line=False, match_clim=True,
                vmin=None, vmax=None):
    """ Create heatmap of each cell aligned across time.

    Parameters:
    -----------
    mouse : str; mouse name
    trace_type : str; dff, zscore, deconvolved
    cs_bar : logical; add a bar to match cs on edge of plot
    day_bar : logical; add checkerboard to match days on cs bar
    day_line : logical; add a lines between days
    run_line : logical; ddd lines between runs
    Returns:
    ________
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
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # load metadata
    meta_path = os.path.join(flow.paths.outd, mouse + '_df_' + trace_type + '_trialmeta.pkl')
    dfm = pd.read_pickle(meta_path)

    # create a binary map of every day a cell is present
    xmap = df.get_xdaymap('OA27')

    # set up range conditonal on input
    if cell_id is None:
        cell_range = range(1, np.shape(xmap)[0]+1)
    else:
        cell_range = cell_id

    # loop through cells and plot
    for cell_idx in cell_range:

        # create single cell df
        dft = df.singlecell(mouse, trace_type, cell_idx, xmap=xmap)
        dft = dft.reset_index(level=['cell_idx', 'timestamp'])

        # filter metadata trials before merging
        trial_indexer = (((dfm.orientation == 0) | (dfm.orientation == 135) | (dfm.orientation == 270))
                         & ((dfm.tag == 'standard') | (dfm.tag == 'learning_start') | (dfm.tag == 'reversal1_start')
                         | (dfm.tag == 'reversal2_start'))
                         & ((dfm.condition == 'plus') | (dfm.condition == 'minus') | (dfm.condition == 'neutral'))
                         & (dfm.hunger == 'hungry'))
        dfm = dfm.loc[trial_indexer, :]

        # merge on filtered trials
        dff = pd.merge(dft, dfm, on=['mouse', 'date', 'run', 'trial_idx'], how='inner')

        # get timestamp info for plotting lines
        times = dff['timestamp']
        zero_sec = np.where(np.unique(times) >= 0)[0][0]
        three_sec = np.where(np.unique(times) >= 3)[0][0]

        # oris
        oris = np.array([0, 135, 270])

        # plot main figure
        toplot = dff.pivot_table(index=['date', 'run','trial_idx','orientation'],
                                columns='timestamp', values='trace')
        g = sns.FacetGrid(toplot.reset_index('orientation'), col='orientation',
                          height=6, sharey=False, dropna=False)
        g.map_dataframe(_myheatmap, vmax=vmax, vmin=vmin, center=0, xticklabels=31, cmap=cmap)
        g.fig.suptitle('Cell ' + str(cell_idx), x=0.98)

        # loop through axes and plot relevant metadata on top
        count = 0
        cmin = []
        cmax = []
        ccmap = []
        for ax in g.axes[0]:

            # match vmin and max across plots (first check values)
            if match_clim and vmax is None:
                vmin, vmax = ax.collections[0].get_clim()
                cmin.append(vmin)
                cmax.append(vmax)
                ccmap.append(ax.collections[0].get_cmap())

            # get metadata for this orientation/set of trials
            meta = dff.loc[dff['orientation'] == oris[count], ['condition',
                          'ensure', 'quinine', 'firstlick', 'run_type']]
            meta = meta.reset_index()
            meta = meta.drop_duplicates()
            ensure = np.array(meta['ensure'])
            quinine = np.array(meta['quinine'])
            firstlick = np.array(meta['firstlick'])
            css = meta['condition']
            run_type = meta['run_type']

            ori_inds = np.array(toplot.index.get_level_values('orientation'))
            ori_inds = ori_inds == oris[count]

            # set labels
            if count == 0:
                ax.set_ylabel('Trials')
            ax.set_xlabel('Time (sec)')

            # plot cs color bar/line
            if cs_bar:
                css[run_type == 'naive'] = 'naive'
                for cs in np.unique(css):
                    cs_line_color = colors[cs_colors[cs]]
                    cs_y = np.where(css == cs)[0]
                    cs_y = [cs_y[0]-0.5, cs_y[-1]+0.5]
                    ax.plot((5, 5), cs_y, color=cs_line_color, ls='-',
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
                    day_y = [day_y[0]-0.5, day_y[-1]+0.5]
                    day_bar_color = day_colors[sorted(day_colors.keys())[count_d%2]]
                    ax.plot((9, 9), day_y, color=day_bar_color, ls='-',
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
                    day_y = [y, y]
                    ax.plot(x_lim, day_y, color='#8e8e8e', ls='-', lw=1, alpha=0.8)

            # plot lines between runs
            if run_line:
                runs = np.array(toplot.index.get_level_values('run'))
                runs = runs[ori_inds]
                runs = np.diff(runs)
                run_ind = np.where(runs > 0)[0]
                for y in run_ind:
                    run_y = [y, y]
                    ax.plot(x_lim, run_y, color='#bababa', ls='-', lw=1,  alpha=0.8)

            # plot onset/offest lines
            ax.plot((zero_sec, zero_sec), y_lim, color='#8e8e8e', ls='-', lw=2, alpha=0.8)
            ax.plot((three_sec, three_sec), y_lim, color='#8e8e8e', ls='-', lw=2, alpha=0.8)

            # plot quinine
            for l in range(len(quinine)):
                if np.isfinite(quinine[l]):
                    x = [quinine[l], quinine[l]]
                    y = [l-.25, l+.25]
                    ax.plot(x, y, color='#0fffc3', ls='-', lw=2.5)

            # plot ensure
            for l in range(len(ensure)):
                if np.isfinite(ensure[l]):
                    x = [ensure[l], ensure[l]]
                    y = [l-.25, l+.25]
                    ax.plot(x, y, color='#ffb30f', ls='-', lw=2.5)

            # plot licks
            for l in range(len(firstlick)):
                if np.isfinite(firstlick[l]):
                    x = [firstlick[l], firstlick[l]]
                    y = [l-.25, l+.25]
                    ax.plot(x, y, color='#7237f2', ls='-', lw=2.5)

            # reset ylabels
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

            date_rel = flow.metadata.DateSorter.frommeta(mice=[mouse])
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

            count = count + 1

        # match vmin and max across plots (using cmax to choose cmap and cmin)
        if match_clim and vmax is None:
            max_ind = np.nanargmax(cmax)
            cmin = cmin[max_ind]
            cmax = cmax[max_ind]
            ccmap = ccmap[max_ind]

            for ax in g.axes[0]:
                ax.collections[0].set_clim(vmax=cmax, vmin=cmin)
                ax.collections[0].set_cmap(ccmap)

        # save figures into folder
        save_dir = os.path.join(flow.paths.graphd, str(mouse))
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        path = os.path.join(save_dir, str(mouse) + '_cell_' + str(cell_idx) + '_' + trace_type + '.png')
        print('Cell: ' + str(cell_idx) + ': done.')
        g.savefig(path)
        plt.close()


def _myheatmap(data, **kwargs):
    """ Helper function for FacetGrid heatmap."""

    mydata = data.set_index('orientation', append=True)
    sns.heatmap(mydata, **kwargs)

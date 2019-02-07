"""Functions for plotting dataframes and tca decomp."""
import os
import flow
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import tensortools as tt
import seaborn as sns
import pandas as pd
from copy import deepcopy
from .. import df
from .. import tca


def xdayheatmap(mouse, cell_id=None, trace_type='dff', cs_bar=True, day_bar=True,
                day_line=True, run_line=False, match_clim=True,
                vmin=None, vmax=None, smooth=True, verbose=False):
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

        # plot main figure
        toplot = dff.pivot_table(index=['date', 'run','trial_idx','orientation'],
                                columns='timestamp', values='trace')
        g = sns.FacetGrid(toplot.reset_index('orientation'), col='orientation',
                          height=8, sharey=False, dropna=False)
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

            # plot quinine
            for l in range(len(quinine)):
                if np.isfinite(quinine[l]):
                    x = [quinine[l], quinine[l]]
                    y = [l+0.5, l+0.5]
                    ax.plot(x, y, color='#0fffc3', ls='', marker='.', markersize=2)

            # plot ensure
            for l in range(len(ensure)):
                if np.isfinite(ensure[l]):
                    x = [ensure[l], ensure[l]]
                    y = [l+0.5, l+0.5]
                    ax.plot(x, y, color='#ffb30f', ls='', marker='.', markersize=2)

            # plot licks
            for l in range(len(firstlick)):
                if np.isfinite(firstlick[l]):
                    x = [firstlick[l], firstlick[l]]
                    y = [l+0.5, l+0.5]
                    ax.plot(x, y, color='#7237f2', ls='', marker='.', markersize=2)

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

            # reset xticklabels
            xticklabels = np.array([-1, 0, 1, 2, 3, 4, 5, 6])
            xticklabels = xticklabels[(xticklabels > times[0]) & (xticklabels < times[-1])]
            xticks = [np.where(times <= s)[0][-1] for s in xticklabels]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation='horizontal')

            # update count through loops
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

        # save figures into folder structure
        save_dir = os.path.join(flow.paths.graphd, str(mouse))
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        save_dir = os.path.join(save_dir, 'heatmaps ' + str(trace_type))
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        path = os.path.join(save_dir, str(mouse) + '_cell_' + str(cell_idx) + '_' + trace_type + '.png')
        print('Cell: ' + str(cell_idx) + ': done.')
        g.savefig(path)

        if not verbose:
            plt.close()


def pairdaytcaqc(mouse, trace_type='zscore', verbose=False):
    """ Plot similarity and error plots for TCA decomposition ensembles."""

    # plotting options for the unconstrained and nonnegative models.
    plot_options = {
      'cp_als': {
        'line_kw': {
          'color': 'green',
          'label': 'cp_als',
        },
        'scatter_kw': {
          'color': 'green',
          'alpha': 0.5,
        },
      },
      'ncp_hals': {
        'line_kw': {
          'color': 'blue',
          'alpha': 0.5,
          'label': 'ncp_hals',
        },
        'scatter_kw': {
          'color': 'blue',
          'alpha': 0.5,
        },
      },
      'ncp_bcd': {
        'line_kw': {
          'color': 'red',
          'alpha': 0.5,
          'label': 'ncp_bcd',
        },
        'scatter_kw': {
          'color': 'red',
          'alpha': 0.5,
        },
      },
    }

    days = flow.metadata.DateSorter.frommeta(mice=[mouse], tags=None)

    for c, day1 in enumerate(days, 0):
        try:
            day2 = days[c+1]
        except IndexError:
            print('done.')
            break

        # create folder structure if needed
        # load
        out_dir = os.path.join(flow.paths.outd, str(day1.mouse))
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        load_dir = os.path.join(out_dir, 'tensors paired ' + str(trace_type))
        if not os.path.isdir(load_dir):
            os.mkdir(load_dir)
        tensor_path = os.path.join(load_dir, str(day1.mouse) + '_' + str(day1.date)
                         + '_' + str(day2.date) + '_pair_decomp_' + str(trace_type) + '.npy')
        # save
        ana_dir = os.path.join(flow.paths.graphd, str(day1.mouse))
        if not os.path.isdir(ana_dir):
            os.mkdir(ana_dir)
        save_dir = os.path.join(ana_dir, 'tensors paired ' + str(trace_type))
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        save_dir = os.path.join(save_dir, 'qc')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        error_path = os.path.join(save_dir, str(day1.mouse) + '_' + str(day1.date)
                         + '_' + str(day2.date) + '_objective.png')
        sim_path = os.path.join(save_dir, str(day1.mouse) + '_' + str(day1.date)
                         + '_' + str(day2.date) + '_similarity.png')

        # load your data
        ensemble = np.load(tensor_path)
        ensemble = ensemble.item()

        # plot error and similarity plots across rank number
        plt.figure()
        for m in ensemble:
            tt.plot_objective(ensemble[m], **plot_options[m])  # ax=ax[0])
        plt.legend()
        plt.title('Objective Function')
        plt.savefig(error_path)
        if verbose:
            plt.show()
        plt.clf()

        for m in ensemble:
            tt.plot_similarity(ensemble[m], **plot_options[m])  # ax=ax[1])
        plt.legend()
        plt.title('Iteration Similarity')
        plt.savefig(sim_path)
        if verbose:
            plt.show()
        plt.close()


def pairdaytcaqcsummary(mouse, trace_type='zscore', method='ncp_bcd', verbose=False):
    """ Plot similarity and error plots for TCA decomposition ensembles."""

    days = flow.metadata.DateSorter.frommeta(mice=[mouse], tags=None)

    cmap = sns.color_palette('hls', n_colors=len(days))

    # create figure and axes
    buffer = 5
    right_pad = 5

    fig0 = plt.figure(figsize=(10, 8))
    gs0 = GridSpec(100, 100, figure=fig0, left=0.05, right=.95, top=.95, bottom=0.05)
    ax0 = fig0.add_subplot(gs0[10:90-buffer, :90-right_pad])

    fig1 = plt.figure(figsize=(10, 8))
    gs1 = GridSpec(100, 100, figure=fig1, left=0.05, right=.95, top=.95, bottom=0.05)
    ax1 = fig1.add_subplot(gs1[10:90-buffer, :90-right_pad])

    # plt.figure()
    for c, day1 in enumerate(days, 0):
        try:
            day2 = days[c+1]
        except IndexError:
            print('done.')
            break

        # create folder structure if needed
        # load paths
        out_dir = os.path.join(flow.paths.outd, str(day1.mouse))
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        load_dir = os.path.join(out_dir, 'tensors paired ' + str(trace_type))
        if not os.path.isdir(load_dir):
            os.mkdir(load_dir)
        tensor_path = os.path.join(load_dir, str(day1.mouse) + '_' + str(day1.date)
                                   + '_' + str(day2.date) + '_pair_decomp_' + str(trace_type) + '.npy')
        # save paths
        ana_dir = os.path.join(flow.paths.graphd, str(day1.mouse))
        if not os.path.isdir(ana_dir):
            os.mkdir(ana_dir)
        save_dir = os.path.join(ana_dir, 'tensors paired ' + str(trace_type))
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        save_dir = os.path.join(save_dir, 'qc')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        error_path = os.path.join(save_dir, str(day1.mouse) + '_summary_objective.png')
        sim_path = os.path.join(save_dir, str(day1.mouse) + '_summary_similarity.png')

        # plotting options for the unconstrained and nonnegative models.
        plot_options = {
          'cp_als': {
            'line_kw': {
              'color': cmap[c],
              'label': 'pair ' + str(c),
            },
            'scatter_kw': {
              'color': cmap[c],
              'alpha': 0.5,
            },
          },
          'ncp_hals': {
            'line_kw': {
              'color': cmap[c],
              'alpha': 0.5,
              'label': 'pair ' + str(c),
            },
            'scatter_kw': {
              'color': cmap[c],
              'alpha': 0.5,
            },
          },
          'ncp_bcd': {
            'line_kw': {
              'color': cmap[c],
              'alpha': 0.5,
              'label': 'pair ' + str(c),
            },
            'scatter_kw': {
              'color': cmap[c],
              'alpha': 0.5,
            },
          },
        }

        # load your data
        ensemble = np.load(tensor_path)
        ensemble = ensemble.item()

        # plot error and similarity plots across rank number
        tt.plot_objective(ensemble[method], **plot_options[method], ax=ax0)
        tt.plot_similarity(ensemble[method], **plot_options[method], ax=ax1)

    # add legend, title
    ax0.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
    ax0.set_title('Objective Function: ' + str(method) + ', ' + mouse)
    ax1.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
    ax1.set_title('Iteration Similarity: ' + str(method) + ', ' + mouse)

    # save figs
    fig0.savefig(error_path, bbox_inches='tight')
    fig1.savefig(sim_path, bbox_inches='tight')

    if verbose:
        fig0.show()
        fig1.show()


def pairdaytcafactors(mouse, trace_type='zscore', method='ncp_bcd', verbose=False):
    """
    Plot factors for TCA decomposition ensembles.
    """

    # plotting options for the unconstrained and nonnegative models.
    plot_options = {
      'cp_als': {
        'line_kw': {
          'color': 'red',
          'label': 'cp_als',
        },
        'scatter_kw': {
          'color': 'green',
          'alpha': 0.5,
        },
        'bar_kw': {
          'color': 'blue',
          'alpha': 0.5,
        },
      },
      'ncp_hals': {
        'line_kw': {
          'color': 'red',
          'label': 'ncp_hals',
        },
        'scatter_kw': {
          'color': 'green',
          'alpha': 0.5,
        },
        'bar_kw': {
          'color': 'blue',
          'alpha': 0.5,
        },
      },
      'ncp_bcd': {
        'line_kw': {
          'color': 'red',
          'label': 'ncp_bcd',
        },
        'scatter_kw': {
          'color': 'green',
          'alpha': 0.5,
        },
        'bar_kw': {
          'color': 'blue',
          'alpha': 0.5,
        },
      },
    }

    days = flow.metadata.DateSorter.frommeta(mice=[mouse], tags=None)

    for c, day1 in enumerate(days, 0):
        try:
            day2 = days[c+1]
        except IndexError:
            print('done.')
            break

        # create folder structure if needed
        # load
        out_dir = os.path.join(flow.paths.outd, str(day1.mouse))
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        load_dir = os.path.join(out_dir, 'tensors paired ' + str(trace_type))
        if not os.path.isdir(load_dir):
            os.mkdir(load_dir)
        tensor_path = os.path.join(load_dir, str(day1.mouse) + '_' + str(day1.date)
                         + '_' + str(day2.date) + '_pair_decomp_' + str(trace_type) + '.npy')
        # save
        ana_dir = os.path.join(flow.paths.graphd, str(day1.mouse))
        if not os.path.isdir(ana_dir):
            os.mkdir(ana_dir)
        save_dir = os.path.join(ana_dir, 'tensors paired ' + str(trace_type))
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        save_dir = os.path.join(save_dir, 'factors')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        # load your data
        ensemble = np.load(tensor_path)
        ensemble = ensemble.item()


        # make necessary dirs
        date_dir = os.path.join(save_dir, str(day1.date) + '_' + str(day2.date) + ' ' + method)
        if not os.path.isdir(date_dir):
            os.mkdir(date_dir)

        # sort neuron factors by component they belong to most
        sort_ensemble, my_sorts = tca._sortfactors(ensemble[method])

        for r in sort_ensemble.results:

            fig = tt.plot_factors(sort_ensemble.results[r][0].factors, plots=['bar', 'line', 'scatter'],
                            axes=None,
                            scatter_kw=plot_options[method]['scatter_kw'],
                            line_kw=plot_options[method]['line_kw'],
                            bar_kw=plot_options[method]['bar_kw'])

            ax = fig[0].axes
            ax[0].set_title('Neuron factors')
            ax[1].set_title('Temporal factors')
            ax[2].set_title('Trial factors')

            count = 1
            for k in range(0, len(ax)):
                if np.mod(k+1, 3) == 1:
                    ax[k].set_ylabel('Component #' + str(count), rotation=0,
                                     labelpad=45, verticalalignment='center', fontstyle='oblique')
                    count = count + 1

            # Show plots.
            plt.savefig(os.path.join(date_dir, 'rank_' + str(int(r)) + '.png'), bbox_inches='tight')
            if verbose:
                plt.show()
            plt.close()


def _myheatmap(data, **kwargs):
    """ Helper function for FacetGrid heatmap."""

    mydata = data.set_index('orientation', append=True)
    sns.heatmap(mydata, **kwargs)

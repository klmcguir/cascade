"""Functions for plotting clustered factors from tca decomp."""
import os
import flow
import pool
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import tensortools as tt
import seaborn as sns
# from seaborn.matrix import ClusterGrid
import pandas as pd
from copy import deepcopy
from .. import df
from .. import tca
from .. import paths
from .. import utils
from .. import cluster
import warnings


# added functionality from https://github.com/mwaskom/seaborn/pull/1393/files
class MyClusterGrid(sns.matrix.ClusterGrid):
    def __init__(self, data, pivot_kws=None, z_score=None, standard_scale=None,
                 figsize=None, row_colors=None, col_colors=None, mask=None,
                 expected_size_dendrogram=1.0,
                 expected_size_colors=0.25):
        """Grid object for organizing clustered heatmap input on to axes"""

        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            self.data = pd.DataFrame(data)

        self.data2d = self.format_data(self.data, pivot_kws, z_score,
                                       standard_scale)

        self.mask = _matrix_mask(self.data2d, mask)

        if figsize is None:
            width, height = 10, 10
            figsize = (width, height)
        self.fig = plt.figure(figsize=figsize)

        self.expected_size_dendrogram = expected_size_dendrogram
        self.expected_size_side_colors = expected_size_colors

        self.row_colors, self.row_color_labels = \
            self._preprocess_colors(data, row_colors, axis=0)
        self.col_colors, self.col_color_labels = \
            self._preprocess_colors(data, col_colors, axis=1)

        width_ratios = self.dim_ratios(self.row_colors,
                                       figsize=figsize,
                                       axis=0)

        height_ratios = self.dim_ratios(self.col_colors,
                                        figsize=figsize,
                                        axis=1)
        nrows = 3 if self.col_colors is None else 4
        ncols = 3 if self.row_colors is None else 4

        self.gs = gridspec.GridSpec(nrows, ncols, wspace=0.01, hspace=0.01,
                                    width_ratios=width_ratios,
                                    height_ratios=height_ratios)

        self.ax_row_dendrogram = self.fig.add_subplot(self.gs[nrows - 1, 0:2])
        self.ax_col_dendrogram = self.fig.add_subplot(self.gs[0:2, ncols - 1])
        self.ax_row_dendrogram.set_axis_off()
        self.ax_col_dendrogram.set_axis_off()

        self.ax_row_colors = None
        self.ax_col_colors = None

        if self.row_colors is not None:
            self.ax_row_colors = self.fig.add_subplot(
                self.gs[nrows - 1, ncols - 2])
        if self.col_colors is not None:
            self.ax_col_colors = self.fig.add_subplot(
                self.gs[nrows - 2, ncols - 1])

        self.ax_heatmap = self.fig.add_subplot(self.gs[nrows - 1, ncols - 1])

        # colorbar for scale to left corner

        if self.col_colors is not None:
            cbar_max = 3
        else:
            cbar_max = 2

        self.cax = self.fig.add_subplot(self.gs[0:cbar_max, 0])

        self.dendrogram_row = None
        self.dendrogram_col = None

    def dim_ratios(self, side_colors, axis, figsize):
        """Get the proportions of the figure taken up by each axes
        """
        figdim = figsize[axis]

        expected_size_for_dendrogram = self.expected_size_dendrogram  # Inches
        expected_size_for_side_colors = self.expected_size_side_colors  # Inches

        # Get resizing proportion of this figure for the dendrogram and
        # colorbar, so only the heatmap gets bigger but the dendrogram stays
        # the same size.
        dendrogram = expected_size_for_dendrogram / figdim

        # add the colorbar
        colorbar_width = .8 * dendrogram
        colorbar_height = .2 * dendrogram
        if axis == 1:
            ratios = [colorbar_width, colorbar_height]
        else:
            ratios = [colorbar_height, colorbar_width]

        if side_colors is not None:
            colors_shape = np.asarray(side_colors).shape
            # This happens when a series or a list is passed
            if len(colors_shape) <= 2:
                n_colors = 1
            # And this happens when a dataframe is passed, the first dimension is number of colors
            else:
                n_colors = colors_shape[0]

            # Multiply side colors size by the number of colors
            expected_size_for_side_colors = n_colors * expected_size_for_side_colors

            side_colors_ratio = expected_size_for_side_colors / figdim

            # Add room for the colors
            ratios += [side_colors_ratio]

        # Add the ratio for the heatmap itself
        ratios.append(1 - sum(ratios))

        return ratios


def _matrix_mask(data, mask):
    """Ensure that data and mask are compatabile and add missing values.
    Values will be plotted for cells where ``mask`` is ``False``.
    ``data`` is expected to be a DataFrame; ``mask`` can be an array or
    a DataFrame.
    """
    if mask is None:
        mask = np.zeros(data.shape, np.bool)
    if isinstance(mask, np.ndarray):
        # For array masks, ensure that shape matches data then convert
        if mask.shape != data.shape:
            raise ValueError("Mask must have the same shape as data.")
        mask = pd.DataFrame(mask,
                            index=data.index,
                            columns=data.columns,
                            dtype=np.bool)
    elif isinstance(mask, pd.DataFrame):
        # For DataFrame masks, ensure that semantic labels match data
        if not mask.index.equals(data.index) \
           and mask.columns.equals(data.columns):
            err = "Mask must have the same index and columns as data."
            raise ValueError(err)
    # Add any cells with missing data to the mask
    # This works around an issue where `plt.pcolormesh` doesn't represent
    # missing data properly
    mask = mask | pd.isnull(data)
    return mask


def clustermap(data, pivot_kws=None, method='average', metric='euclidean',
               z_score=None, standard_scale=None, figsize=None, cbar_kws=None,
               row_cluster=True, col_cluster=True,
               row_linkage=None, col_linkage=None,
               row_colors=None, col_colors=None, mask=None,
               expected_size_dendrogram=1.0,
               expected_size_colors=0.25,
               **kwargs):
    """Plot a matrix dataset as a hierarchically-clustered heatmap.
    Parameters
    ----------
    data: 2D array-like
        Rectangular data for clustering. Cannot contain NAs.
    pivot_kws : dict, optional
        If `data` is a tidy dataframe, can provide keyword arguments for
        pivot to create a rectangular dataframe.
    method : str, optional
        Linkage method to use for calculating clusters.
        See scipy.cluster.hierarchy.linkage documentation for more information:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    metric : str, optional
        Distance metric to use for the data. See
        scipy.spatial.distance.pdist documentation for more options
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
        To use different metrics (or methods) for rows and columns, you may
        construct each linkage matrix yourself and provide them as
        {row,col}_linkage.
    z_score : int or None, optional
        Either 0 (rows) or 1 (columns). Whether or not to calculate z-scores
        for the rows or the columns. Z scores are: z = (x - mean)/std, so
        values in each row (column) will get the mean of the row (column)
        subtracted, then divided by the standard deviation of the row (column).
        This ensures that each row (column) has mean of 0 and variance of 1.
    standard_scale : int or None, optional
        Either 0 (rows) or 1 (columns). Whether or not to standardize that
        dimension, meaning for each row or column, subtract the minimum and
        divide each by its maximum.
    figsize: tuple of two ints, optional
        Size of the figure to create.
    cbar_kws : dict, optional
        Keyword arguments to pass to ``cbar_kws`` in ``heatmap``, e.g. to
        add a label to the colorbar.
    {row,col}_cluster : bool, optional
        If True, cluster the {rows, columns}.
    {row,col}_linkage : numpy.array, optional
        Precomputed linkage matrix for the rows or columns. See
        scipy.cluster.hierarchy.linkage for specific formats.
    {row,col}_colors : list-like or pandas DataFrame/Series, optional
        List of colors to label for either the rows or columns. Useful to
        evaluate whether samples within a group are clustered together. Can
        use nested lists or DataFrame for multiple color levels of labeling.
        If given as a DataFrame or Series, labels for the colors are extracted
        from the DataFrames column names or from the name of the Series.
        DataFrame/Series colors are also matched to the data by their
        index, ensuring colors are drawn in the correct order.
    mask : boolean array or DataFrame, optional
        If passed, data will not be shown in cells where ``mask`` is True.
        Cells with missing values are automatically masked. Only used for
        visualizing, not for calculating.
    expected_size_dendrogram: float, optional
        Size (in the same units as figsize) of the dendrogram size
    expected_size_colors: float, optional
        Size (in the same units as figsize) of size for row/col columns, if passed.
    kwargs : other keyword arguments
        All other keyword arguments are passed to ``sns.heatmap``
    Returns
    -------
    clustergrid : ClusterGrid
        A ClusterGrid instance.
    Notes
    -----
    The returned object has a ``savefig`` method that should be used if you
    want to save the figure object without clipping the dendrograms.
    To access the reordered row indices, use:
    ``clustergrid.dendrogram_row.reordered_ind``
    Column indices, use:
    ``clustergrid.dendrogram_col.reordered_ind``
    Examples
    --------
    Plot a clustered heatmap:
    .. plot::
        :context: close-figs
        >>> import seaborn as sns; sns.set(color_codes=True)
        >>> iris = sns.load_dataset("iris")
        >>> species = iris.pop("species")
        >>> g = sns.clustermap(iris)
    Use a different similarity metric:
    .. plot::
        :context: close-figs
        >>> g = sns.clustermap(iris, metric="correlation")
    Use a different clustering method:
    .. plot::
        :context: close-figs
        >>> g = sns.clustermap(iris, method="single")
    Use a different colormap and ignore outliers in colormap limits:
    .. plot::
        :context: close-figs
        >>> g = sns.clustermap(iris, cmap="mako", robust=True)
    Change the size of the figure:
    .. plot::
        :context: close-figs
        >>> g = sns.clustermap(iris, figsize=(6, 7))
    Plot one of the axes in its original organization:
    .. plot::
        :context: close-figs
        >>> g = sns.clustermap(iris, col_cluster=False)
    Add colored labels:
    .. plot::
        :context: close-figs
        >>> lut = dict(zip(species.unique(), "rbg"))
        >>> row_colors = species.map(lut)
        >>> g = sns.clustermap(iris, row_colors=row_colors)
    Standardize the data within the columns:
    .. plot::
        :context: close-figs
        >>> g = sns.clustermap(iris, standard_scale=1)
    Normalize the data within the rows:
    .. plot::
        :context: close-figs
        >>> g = sns.clustermap(iris, z_score=0)
    """
    plotter = MyClusterGrid(data, pivot_kws=pivot_kws, figsize=figsize,
                          row_colors=row_colors, col_colors=col_colors,
                          z_score=z_score, standard_scale=standard_scale,
                          mask=mask, expected_size_dendrogram=expected_size_dendrogram,
                          expected_size_colors=expected_size_colors)

    return plotter.plot(metric=metric, method=method,
                        colorbar_kws=cbar_kws,
                        row_cluster=row_cluster, col_cluster=col_cluster,
                        row_linkage=row_linkage, col_linkage=col_linkage,
                        **kwargs)


def groupday_longform_factors_annotated_clusfolders(
        mice=['OA27', 'OA26', 'OA67', 'VF226', 'CC175'],
        trace_type='zscore_day',
        method='ncp_bcd',
        cs='',
        warp=False,
        words=['rochester', 'convinced', 'convinced', 'convinced', 'convinced'],
        group_by='all',
        nan_thresh=0.85,
        rank_num=14,
        clus_num=10,
        use_dprime=False,
        extra_col=1,
        alpha=0.6,
        plot_running=True,
        filetype='pdf',
        verbose=False):

    """
    Plot TCA factors with trial metadata annotations for all days
    and ranks/components for TCA decomposition ensembles. Save factors
    with the same cluster into the same folder so that user can quickly
    check qualitative similarities between clusters.

    Parameters:
    -----------
    mouse : list of str
        Mouse names.
    trace_type : str
        dff, zscore, zscore_iti, zscore_day, deconvolved
    method : str
        TCA fit method from tensortools
    cs : str
        Cs stimuli to include, plus/minus/neutral, 0/135/270, etc. '' empty
        includes all stimuli
    warp : bool
        Use traces with time-warped outcome.
    words : list of str
        List of hash words that match mice.
    group_by : str
        Metric used for building tensor across days. Usually defines a stage
        of learning.
    nan_thresh : float
        Maximum proportion of nan trials per cell to include in the tensor.
        Cells with more than this ratio are removed from analysis.
    extra_col : int
        Number of columns to add to the original three factor columns
    alpha : float
        Value between 0 and 1 for transparency of markers
    plot_running : bool
        Include trace of scaled (to plot max) average running speed during trial
    verbose : bool
        Show plots as they are made.

    Returns:
    --------
    Saves figures to .../analysis folder  .../factors annotated
    """

    # use matplotlib plotting defaults
    mpl.rcParams.update(mpl.rcParamsDefault)

    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}
    group_pars = {'group_by': group_by}

    # if cells were removed with too many nan trials
    if nan_thresh:
        nt_tag = '_nantrial' + str(nan_thresh)
        nt_save_tag = ' nantrial ' + str(nan_thresh)
    else:
        nt_tag = ''
        nt_save_tag = ''

    # save dir
    group_word = paths.groupmouse_word({'mice': mice})
    mouse = 'Group-' + group_word
    save_dir = paths.tca_plots(
        mouse, 'group', pars=pars, word=words[0], group_pars=group_pars)
    save_dir = os.path.join(save_dir, 'factors annotated long-form' + nt_save_tag)
    if not os.path.isdir(save_dir): os.mkdir(save_dir)

    # get clusters
    if use_dprime:
        clus_df, temp_df = cluster.trial_factors_across_mice_dprime(
            mice=mice,
            trace_type=trace_type,
            method=method,
            cs=cs,
            warp=warp,
            words=words,
            group_by=group_by,
            nan_thresh=nan_thresh,
            verbose=verbose,
            rank_num=rank_num)
    else:
        clus_df, temp_df = cluster.trial_factors_across_mice(
            mice=mice,
            trace_type=trace_type,
            method=method,
            cs=cs,
            warp=warp,
            words=words,
            group_by=group_by,
            nan_thresh=nan_thresh,
            verbose=verbose,
            rank_num=rank_num)
    clus_df = clus_df.dropna(axis='rows')
    clus_df = cluster.get_component_clusters(clus_df, clus_num)

    for mnum, mouse in enumerate(mice):
        # load dir
        load_dir = paths.tca_path(
            mouse, 'group', pars=pars, word=words[mnum], group_pars=group_pars)
        tensor_path = os.path.join(
            load_dir, str(mouse) + '_' + str(group_by) + nt_tag
            + '_group_decomp_' + str(trace_type) + '.npy')
        ids_path = os.path.join(
            load_dir, str(mouse) + '_' + str(group_by) + nt_tag
            + '_group_ids_' + str(trace_type) + '.npy')
        input_tensor_path = os.path.join(
            load_dir, str(mouse) + '_' + str(group_by) + nt_tag
            + '_group_tensor_' + str(trace_type) + '.npy')
        meta_path = os.path.join(
            load_dir, str(mouse) + '_' + str(group_by) + nt_tag
            + '_df_group_meta.pkl')

        # load your data
        ensemble = np.load(tensor_path, allow_pickle=True)
        ensemble = ensemble.item()
        meta = pd.read_pickle(meta_path)
        meta = utils.update_naive_cs(meta)
        orientation = meta['orientation']
        trial_num = np.arange(0, len(orientation))
        condition = meta['condition']
        trialerror = meta['trialerror']
        hunger = deepcopy(meta['hunger'])
        speed = meta['speed']
        dates = meta.reset_index()['date']
        learning_state = meta['learning_state']

        # calculate change indices for days and reversal/learning
        udays = {d: c for c, d in enumerate(np.unique(dates))}
        ndays = np.diff([udays[i] for i in dates])
        day_x = np.where(ndays)[0] + 0.5
        ustate = {d: c for c, d in enumerate(np.unique(learning_state))}
        nstate = np.diff([ustate[i] for i in learning_state])
        lstate_x = np.where(nstate)[0] + 0.5

        # merge hunger and tag info for plotting hunger
        tags = meta['tag']
        hunger[tags == 'disengaged'] = 'disengaged'

        # sort neuron factors by component they belong to most
        # if 'mcp_als' has been run make sure the variable is in
        # the correct format
        if isinstance(ensemble[method], dict):
            ensemble2 = {}
            ensemble2[method] = lambda: None
            ensemble[method] = {k: [v] for k, v in ensemble[method].items()}
            ensemble2[method].results = ensemble[method]
            sort_ensemble, my_sorts = tca._sortfactors(ensemble2[method])
        else:
            sort_ensemble, my_sorts = tca._sortfactors(ensemble[method])

        # save into each cluster directory
        for clus in range(1, clus_num+1):

            # save dir for that cluster
            clus_dir = os.path.join(save_dir, 'cluster ' + str(clus))
            if not os.path.isdir(clus_dir): os.mkdir(clus_dir)

            rows = 5
            cols = 3
            U = sort_ensemble.results[rank_num][0].factors
            for comp in range(U.rank):

                # check that comp belongs in cluster
                lookup_df = clus_df.reset_index()
                comp_cluster = lookup_df.loc[
                    (lookup_df['mouse'] == mouse) &
                    (lookup_df['component'] == (comp + 1)),
                    'cluster'].values[0]
                if comp_cluster != clus:
                    continue

                fig, axes = plt.subplots(
                    rows, cols, figsize=(17, rows),
                    gridspec_kw={'width_ratios': [2, 2, 17]})

                # reshape for easier indexing
                ax = np.array(axes).reshape((rows, -1))
                ax[0, 0].set_title('Neuron factors')
                ax[0, 1].set_title('Temporal factors')
                ax[0, 2].set_title('Trial factors')

                # add title to whole figure
                ax[0, 0].text(
                    -1.2, 4,
                    '\n' + mouse + ': \n\nrank: ' + str(int(rank_num))
                    + '\nmethod: ' + method + ' \ngroup_by: '
                    + group_by + ' \ncluster: ' + str(clus),
                    fontsize=12,
                    transform=ax[0, 0].transAxes, color='#969696')

                # plot cell factors
                ax[0, 0].bar(
                    np.arange(0, len(U.factors[0][:, comp])),
                    U.factors[0][:, comp], color='b')
                ax[0, 0].autoscale(enable=True, axis='both', tight=True)

                # plot temporal factors
                ax[0, 1].plot(U.factors[1][:, comp], color='r', linewidth=1.5)
                ax[0, 1].autoscale(enable=True, axis='both', tight=True)

                # add a line for stim onset and offset
                # NOTE: assumes downsample, 1 sec before onset, 3 sec stim
                y_lim = ax[0, 1].get_ylim()
                ons = 15.5*1
                offs = ons+15.5*3
                ax[0, 1].plot([ons, ons], y_lim, ':k')
                ax[0, 1].plot([offs, offs], y_lim, ':k')

                col = cols - 1
                for i in range(rows):

                    # get axis values
                    y_lim = [0, np.nanmax(U.factors[2][:, comp])]

                    # running
                    if plot_running:
                        scale_by = np.nanmax(speed)/y_lim[1]
                        if not np.isnan(scale_by):
                            ax[i, col].plot(
                                np.array(speed.tolist())/scale_by,
                                color=[1, 0.1, 0.6, 0.2])
                            # , label='speed')

                    # Orientation - main variable to plot
                    if i == 0:
                        ori_vals = [0, 135, 270]
                        # color_vals = [[0.28, 0.68, 0.93, alpha],
                        #               [0.84, 0.12, 0.13, alpha],
                        #               [0.46, 0.85, 0.47, alpha]]
                        color_vals = sns.color_palette('husl', 3)
                        for k in range(0, 3):
                            ax[i, col].plot(
                                trial_num[orientation == ori_vals[k]],
                                U.factors[2][orientation == ori_vals[k], comp],
                                'o', label=str(ori_vals[k]), color=color_vals[k],
                                markersize=2, alpha=alpha)

                        ax[i, col].set_title(
                            'Component ' + str(comp + 1) + '\n\n\nTrial factors')
                        ax[i, col].legend(
                            bbox_to_anchor=(1.02, 1), loc='upper left',
                            borderaxespad=0, title='Orientation', markerscale=2,
                            prop={'size': 8})
                        ax[i, col].autoscale(enable=True, axis='both', tight=True)
                        ax[i, col].set_xticklabels([])

                    # Condition - main variable to plot
                    elif i == 1:
                        cs_vals = ['plus', 'minus', 'neutral']
                        cs_labels = ['plus', 'minus', 'neutral']
                        color_vals = [[0.46, 0.85, 0.47, alpha],
                                      [0.84, 0.12, 0.13, alpha],
                                      [0.28, 0.68, 0.93, alpha]]
                        for k in range(0, 3):
                            ax[i, col].plot(
                                trial_num[condition == cs_vals[k]],
                                U.factors[2][condition == cs_vals[k], comp], 'o',
                                label=str(cs_labels[k]), color=color_vals[k],
                                markersize=2)

                        ax[i, col].legend(
                            bbox_to_anchor=(1.02, 1), loc='upper left',
                            borderaxespad=0, title='Condition', markerscale=2,
                            prop={'size': 8})
                        ax[i, col].autoscale(enable=True, axis='both', tight=True)
                        ax[i, col].set_xticklabels([])

                    # Trial error - main variable to plot
                    elif i == 2:
                        color_counter = 0
                        error_colors = sns.color_palette(
                            palette='muted', n_colors=10)
                        trialerror_vals = [0, 1]  # 2, 3, 4, 5,] # 6, 7, 8, 9]
                        trialerror_labels = ['hit',
                                             'miss',
                                             'neutral correct reject',
                                             'neutral false alarm',
                                             'minus correct reject',
                                             'minus false alarm',
                                             'blank correct reject',
                                             'blank false alarm',
                                             'pav early licking',
                                             'pav late licking']
                        for k in range(len(trialerror_vals)):
                            ax[i, col].plot(
                                trial_num[trialerror == trialerror_vals[k]],
                                U.factors[2][trialerror == trialerror_vals[k], comp],
                                'o', label=str(trialerror_labels[k]), alpha=alpha,
                                markersize=2, color=error_colors[color_counter])
                            color_counter = color_counter + 1

                        ax[i, col].legend(
                            bbox_to_anchor=(1.02, 1), loc='upper left',
                            borderaxespad=0, title='Trialerror', markerscale=2,
                            prop={'size': 8})
                        ax[i, col].autoscale(enable=True, axis='both', tight=True)
                        ax[i, col].set_xticklabels([])

                    # Trial error 2.0 - main variable to plot
                    elif i == 3:
                        trialerror_vals = [2, 3]
                        trialerror_labels = ['neutral correct reject',
                                             'neutral false alarm']
                        for k in range(len(trialerror_vals)):
                            ax[i, col].plot(
                                trial_num[trialerror == trialerror_vals[k]],
                                U.factors[2][trialerror == trialerror_vals[k], comp],
                                'o', label=str(trialerror_labels[k]), alpha=alpha,
                                markersize=2, color=error_colors[color_counter])
                            color_counter = color_counter + 1

                        ax[i, col].legend(
                            bbox_to_anchor=(1.02, 1), loc='upper left',
                            borderaxespad=0, title='Trialerror', markerscale=2,
                            prop={'size': 8})
                        ax[i, col].autoscale(enable=True, axis='both', tight=True)
                        ax[i, col].set_xticklabels([])

                    # Trial error 3.0 - main variable to plot
                    elif i == 4:
                        trialerror_vals = [4, 5]
                        trialerror_labels = ['minus correct reject',
                                             'minus false alarm']
                        for k in range(len(trialerror_vals)):
                            ax[i, col].plot(
                                trial_num[trialerror == trialerror_vals[k]],
                                U.factors[2][trialerror == trialerror_vals[k], comp],
                                'o', label=str(trialerror_labels[k]), alpha=alpha,
                                markersize=2, color=error_colors[color_counter])
                            color_counter = color_counter + 1

                            ax[i, col].legend(
                                bbox_to_anchor=(1.02, 1), loc='upper left',
                                borderaxespad=0, title='Trialerror', markerscale=2,
                                prop={'size': 8})
                        ax[i, col].autoscale(enable=True, axis='both', tight=True)

                    # plot days, reversal, or learning lines if there are any
                    if col >= 1:
                        y_lim = ax[i, col].get_ylim()
                        if len(day_x) > 0:
                            for k in day_x:
                                ax[i, col].plot(
                                    [k, k], y_lim, color='#969696', linewidth=1)
                        if len(lstate_x) > 0:
                            ls_vals = ['naive', 'learning', 'reversal1']
                            ls_colors = ['#66bd63', '#d73027', '#a50026']
                            for k in lstate_x:
                                ls = learning_state[int(k-0.5)]
                                ax[i, col].plot(
                                    [k, k], y_lim,
                                    color=ls_colors[ls_vals.index(ls)],
                                    linewidth=1.5)

                    # hide subplots that won't be used
                    if i > 0:
                        ax[i, 0].axis('off')
                        ax[i, 1].axis('off')

                    # despine plots to look like sns defaults
                    sns.despine()

                # save
                if filetype.lower() == 'pdf':
                    suffix = '.pdf'
                elif filetype.lower() == 'eps':
                    suffix = '.eps'
                else:
                    suffix = '.png'
                plt.savefig(
                    os.path.join(
                        clus_dir, mouse + '_rank_' + str(int(rank_num)) +
                        '_component_' + str(comp + 1) + '_cluster_' +
                        str(clus) + suffix),
                    bbox_inches='tight')
                if verbose:
                    plt.show()
                plt.close('all')

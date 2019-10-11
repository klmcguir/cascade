"""Functions for plotting clustered factors from tca decomp."""
import flow
import pool

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import tensortools as tt
import seaborn as sns
import pandas as pd
from scipy.cluster import hierarchy
from scipy.stats import pearsonr

from copy import deepcopy
import warnings

from .. import df
from .. import tca
from .. import paths
from .. import utils
from .. import cluster
from .. import calc
from .. import load


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

        self.mask = sns.matrix._matrix_mask(self.data2d, mask)

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


def lineplot_grid_factor_tuning_byday(

        # plotting params
        var_list=['minus', 'mag_pref_response', 'plus', 'hit', 'neutral',
                  'dprime', 'miss'],

        # dataframe parameters
        mice=['OA27', 'OA26', 'OA67', 'VF226', 'CC175'],
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        words=['orlando', 'already', 'already', 'already', 'already'],
        group_by='all',
        nan_thresh=0.85,
        speed_thresh=5,
        rank_num=18,
        verbose=False):

    """
    Plotting function to make a mouse x component seaborn facetgrid. Plots the
    daily mean of plus, minus, neutral tuning for all components on all days.
    Plots a magnitude of response in gold to help parse tuning. Trialerror hit
    and miss tuning are plotted as doted lines. Dprime on a separate y-axis in
    purple. Information dense! Not for presenting.
    """

    # deal with saving dir
    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}
    group_pars = {'group_by': group_by}
    if nan_thresh:
        nt_tag = '_nantrial' + str(nan_thresh)
        nt_save_tag = ' nantrial ' + str(nan_thresh)
    else:
        nt_tag = ''
        nt_save_tag = ''
    group_word = paths.groupmouse_word({'mice': mice})
    mouse = 'Group-' + group_word
    save_dir = paths.tca_plots(
        mouse, 'group', pars=pars, word=words[0], group_pars=group_pars)
    save_dir = os.path.join(save_dir, 'lineplots' + nt_save_tag)
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    var_path = os.path.join(
        save_dir, str(mouse) + '_rank' + str(rank_num) + '_lineplot_trialfac_byday'
        + '_n' + str(len(mice)) + nt_tag + '.pdf')

    # create dataframes - ignore python and numpy divide by zero warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with np.errstate(invalid='ignore', divide='ignore'):
            var_df, b = df.groupmouse_trialfac_summary_days(
                mice=mice,
                trace_type=trace_type,
                method=method,
                cs=cs,
                warp=warp,
                words=words,
                group_by=group_by,
                nan_thresh=nan_thresh,
                speed_thresh=speed_thresh,
                rank_num=rank_num,
                verbose=verbose)

    # for i in ['fano_factor_pref']:
    # for i in ['t0', 't135', 't270']:
    # for i in ['sc0', 'sc135', 'sc270']:
    # for i in ['fano_factor_0', 'fano_factor_135', 'fano_factor_270']:
    new_list = []
    for i in var_list:

        tuning_vals = var_df[i].values
        condition = [i]*len(tuning_vals)
        data = {'tuning': tuning_vals, 'condition': condition}
        tuning_df = pd.DataFrame(data, index=var_df.index)
        new_list.append(tuning_df)
    new_df = pd.concat(new_list, axis=0)
    grid = sns.FacetGrid(
        new_df.reset_index(), row='mouse', col='component',
        hue='condition', aspect=2, height=3,
        palette=sns.color_palette('hls', 7), dropna=False,
        row_order=['OA27', 'OA26', 'OA67', 'VF226', 'CC175'])

    # plot
    grid.map(twiny_plot, 'tuning').add_legend()

    # save
    plt.savefig(var_path, bbox_inches='tight')


def twiny_plot(*args, **kwargs):
    """
    Helper plotting function for making twin y-axis plots in a facetgrid.
    """

    # get useful label and color info from inputs and current axis
    lbl = kwargs.pop('label')
    clr = kwargs.pop('color')
    ax = plt.gca()

    # plot dprime on a separate y-axis from other variables
    if lbl != 'dprime':
        if lbl == 'hit':
            ax.plot(range(len(*args)), *args, **kwargs,
                    linestyle='--', marker='.', color=clr, alpha=1)
        elif lbl == 'miss':
            ax.plot(range(len(*args)), *args, **kwargs,
                    linestyle='--', marker='.', color=clr, alpha=1)
        elif lbl == 'mag_pref_response':
            ax.plot(range(len(*args)), *args, **kwargs,
                    color=clr, alpha=0.5, linewidth=5)
        else:
            ax.plot(range(len(*args)), *args, **kwargs,
                    color=clr, marker='.', alpha=1)
    else:
        ax2 = ax.twinx()
        ax2.set_ylabel('dprime')
        ax2.plot(range(len(*args)), *args, **kwargs,
                color=clr, alpha=0.5, linewidth=5)


def corr_ramp_indices(

        # df params
        mice=['OA27', 'OA26', 'OA67', 'VF226', 'CC175'],
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        words=['orlando', 'already', 'already', 'already', 'already'],
        group_by='all',
        nan_thresh=0.85,
        speed_thresh=5,
        rank_num=18,
        auto_drop=True,
        annot=False):

    """
    Cluster weights from your trial factors and hierarchically cluster using
    seaborn.clustermap. Annotate plots with useful summary metrics.
    """
    # deal with saving dir
    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}
    group_pars = {'group_by': group_by}
    if nan_thresh:
        nt_tag = '_nantrial' + str(nan_thresh)
        nt_save_tag = ' nantrial ' + str(nan_thresh)
    else:
        nt_tag = ''
        nt_save_tag = ''
    if annot:
        a_tag = '_annot'
    else:
        a_tag = ''
    group_word = paths.groupmouse_word({'mice': mice})
    mouse = 'Group-' + group_word
    save_dir = paths.tca_plots(
        mouse, 'group', pars=pars, word=words[0], group_pars=group_pars)
    save_dir = os.path.join(save_dir, 'correlations' + nt_save_tag)
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    var_path_prefix = os.path.join(
        save_dir, str(mouse) + '_rank' + str(rank_num) + '_pearsonR_trialfac_bystage'
        + '_n' + str(len(mice)) + nt_tag + a_tag)

    # create dataframes - ignore python and numpy divide by zero warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with np.errstate(invalid='ignore', divide='ignore'):
            clustering_df, t_df = \
                df.groupmouse_trialfac_summary_stages(
                    mice=mice,
                    trace_type=trace_type,
                    method=method,
                    cs=cs,
                    warp=warp,
                    words=words,
                    group_by=group_by,
                    nan_thresh=nan_thresh,
                    speed_thresh=speed_thresh,
                    rank_num=rank_num,
                    verbose=False)

    # create dis/sated modulation index (ramp index) df
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with np.errstate(invalid='ignore', divide='ignore'):
            dfdis = calc.fits.groupmouse_fit_disengaged_sated_mean_per_comp(
                mice=mice,
                trace_type=trace_type,
                method=method,
                cs=cs,
                warp=warp,
                words=words,
                group_by=group_by,
                nan_thresh=nan_thresh,
                rank=rank_num,
                verbose=True)
    # just use index
    ri_dis = dfdis.set_index('component', append=True).loc[:, 'dis_index']

    # if running mod, center of mass, or ramp indices are included, remove
    # from columns (make these into a color df for annotating y-axis)
    learning_stages = ['pre_rev1']
    run_stage = ['running_modulation_' + stage for stage in learning_stages]
    ramp_stage = ['ramp_index_trials_' + stage for stage in learning_stages]
    mean_running_mod = clustering_df.loc[:, run_stage].mean(axis=1)
    ri_trials = clustering_df.loc[:, ramp_stage].mean(axis=1)
    ri_learning = clustering_df.loc[:, 'ramp_index_learning']
    ri_trace = clustering_df.loc[:, 'ramp_index_trace']
    ri_offset = clustering_df.loc[:, 'ramp_index_trace_offset']
    ri_speed = clustering_df.loc[:, 'ramp_index_speed_learning']
    center_of_mass = clustering_df.loc[:, 'center_of_mass']

    if auto_drop:

        # create df only early and high dp learning stages
        keep_cols = [
            'plus_high_dp_learning', 'neutral_high_dp_learning',
            'minus_high_dp_learning', 'plus_high_dp_rev1',
            'minus_high_dp_rev1', 'neutral_high_dp_rev1',
            'plus_naive', 'minus_naive', 'neutral_naive']
        drop_inds = ~clustering_df.columns.isin(keep_cols)
        drop_cols = clustering_df.columns[drop_inds]
        clustering_df = clustering_df.drop(columns=drop_cols)
        nan_indexer = clustering_df.isna().any(axis=1)  # this has to be here
        clustering_df = clustering_df.dropna(axis='rows')

        # remove nanned rows from other dfs
        mean_running_mod = mean_running_mod.loc[~nan_indexer, :]
        ri_trials = ri_trials.loc[~nan_indexer, :]
        ri_learning = ri_learning.loc[~nan_indexer, :]
        ri_trace = ri_trace.loc[~nan_indexer, :]
        ri_offset = ri_offset.loc[~nan_indexer, :]
        ri_speed = ri_speed.loc[~nan_indexer, :]
        center_of_mass = center_of_mass.loc[~nan_indexer, :]
        t_df = t_df.loc[~nan_indexer, :]
        ri_dis = ri_dis.loc[~nan_indexer, :]

    corr_df = pd.concat(
        [mean_running_mod, ri_speed, ri_learning, ri_trials, ri_trace,
         ri_offset, ri_dis], axis=1)
    with pd.option_context('mode.use_inf_as_null', True):
        corr_df[corr_df.isna()] = 0

    num_corr = 7
    corrmat = np.zeros((num_corr, num_corr))
    pmat = np.zeros((num_corr, num_corr))
    for i in range(num_corr):
        for k in range(num_corr):
            corA, corP = pearsonr(corr_df.values[:, i], corr_df.values[:, k])
            corrmat[i, k] = corA
            pmat[i, k] = corP

    labels = ['running modulation', 'speed RI', 'learning RI',
              'daily trial RI', 'trace RI', 'trace offset RI', 'disengaged I']
    plt.figure()
    sns.heatmap(corrmat, annot=annot, xticklabels=labels, yticklabels=labels,
                square=True, cbar_kws={'label': 'correlation (R)'})
    plt.xticks(rotation=45, ha='right')
    plt.title('Pearson-R corrcoef')
    plt.savefig(
        var_path_prefix + '_corr.pdf', bbox_inches='tight')

    plt.figure()
    sns.heatmap(pmat, annot=annot, xticklabels=labels, yticklabels=labels,
                square=True, cbar_kws={'label': 'p-value'})
    plt.xticks(rotation=45, ha='right')
    plt.title('Pearson-R p-values')
    plt.savefig(
        var_path_prefix + '_pvals.pdf', bbox_inches='tight')

    plt.figure()
    logger = np.log10(pmat).flatten()
    vmin = np.nanmin(logger[np.isfinite(logger)])
    vmax = 0
    sns.heatmap(np.log10(pmat), annot=annot, xticklabels=labels,
                yticklabels=labels, vmin=vmin, vmax=vmax,
                square=True, cbar_kws={'label': 'log$_{10}$(p-value)'})
    plt.xticks(rotation=45, ha='right')
    plt.title('Pearson-R log$_{10}$(p-values)')
    plt.savefig(
        var_path_prefix + '_log10pvals.pdf', bbox_inches='tight')


def corr_ramp_indices_bymouse(

        # df params
        mice=['OA27', 'OA26', 'OA67', 'VF226', 'CC175'],
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        words=['orlando', 'already', 'already', 'already', 'already'],
        group_by='all',
        nan_thresh=0.85,
        speed_thresh=5,
        rank_num=18,
        auto_drop=True,
        annot=False):

    """
    Cluster weights from your trial factors and hierarchically cluster using
    seaborn.clustermap. Annotate plots with useful summary metrics.
    """
    # deal with saving dir
    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}
    group_pars = {'group_by': group_by}
    if nan_thresh:
        nt_tag = '_nantrial' + str(nan_thresh)
        nt_save_tag = ' nantrial ' + str(nan_thresh)
    else:
        nt_tag = ''
        nt_save_tag = ''
    if annot:
        a_tag = '_annot'
    else:
        a_tag = ''
    group_word = paths.groupmouse_word({'mice': mice})
    mouse = 'Group-' + group_word
    save_dir = paths.tca_plots(
        mouse, 'group', pars=pars, word=words[0], group_pars=group_pars)
    save_dir = os.path.join(save_dir, 'correlations' + nt_save_tag)
    if not os.path.isdir(save_dir): os.mkdir(save_dir)

    # create dataframes - ignore python and numpy divide by zero warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with np.errstate(invalid='ignore', divide='ignore'):
            clustering_df, t_df = \
                df.groupmouse_trialfac_summary_stages(
                    mice=mice,
                    trace_type=trace_type,
                    method=method,
                    cs=cs,
                    warp=warp,
                    words=words,
                    group_by=group_by,
                    nan_thresh=nan_thresh,
                    speed_thresh=speed_thresh,
                    rank_num=rank_num,
                    verbose=False)

    # if running mod, center of mass, or ramp indices are included, remove
    # from columns (make these into a color df for annotating y-axis)
    learning_stages = ['pre_rev1']
    run_stage = ['running_modulation_' + stage for stage in learning_stages]
    ramp_stage = ['ramp_index_trials_' + stage for stage in learning_stages]
    mean_running_mod = clustering_df.loc[:, run_stage].mean(axis=1)
    ri_trials = clustering_df.loc[:, ramp_stage].mean(axis=1)
    ri_learning = clustering_df.loc[:, 'ramp_index_learning']
    ri_trace = clustering_df.loc[:, 'ramp_index_trace']
    ri_offset = clustering_df.loc[:, 'ramp_index_trace_offset']
    ri_speed = clustering_df.loc[:, 'ramp_index_speed_learning']
    center_of_mass = clustering_df.loc[:, 'center_of_mass']

    if auto_drop:

        # create df only early and high dp learning stages
        keep_cols = [
            'plus_high_dp_learning', 'neutral_high_dp_learning',
            'minus_high_dp_learning', 'plus_high_dp_rev1',
            'minus_high_dp_rev1', 'neutral_high_dp_rev1',
            'plus_naive', 'minus_naive', 'neutral_naive']
        drop_inds = ~clustering_df.columns.isin(keep_cols)
        drop_cols = clustering_df.columns[drop_inds]
        clustering_df = clustering_df.drop(columns=drop_cols)
        nan_indexer = clustering_df.isna().any(axis=1)  # this has to be here
        clustering_df = clustering_df.dropna(axis='rows')

        # remove nanned rows from other dfs
        mean_running_mod = mean_running_mod.loc[~nan_indexer, :]
        ri_trials = ri_trials.loc[~nan_indexer, :]
        ri_learning = ri_learning.loc[~nan_indexer, :]
        ri_trace = ri_trace.loc[~nan_indexer, :]
        ri_offset = ri_offset.loc[~nan_indexer, :]
        ri_speed = ri_speed.loc[~nan_indexer, :]
        center_of_mass = center_of_mass.loc[~nan_indexer, :]
        t_df = t_df.loc[~nan_indexer, :]


    corr_df = pd.concat(
        [mean_running_mod, ri_speed, ri_learning, ri_trials, ri_trace,
         ri_offset], axis=1)
    with pd.option_context('mode.use_inf_as_null', True):
        corr_df[corr_df.isna()] = 0

    for ms in np.unique(corr_df.reset_index()['mouse']):
        var_path_prefix = os.path.join(
            save_dir, str(ms) + '_rank' + str(rank_num) +
            '_pearsonR_trialfac_bystage' + nt_tag + a_tag)

        ms_indexer = corr_df.reset_index()['mouse'] == ms
        corrmat = np.zeros((6, 6))
        pmat = np.zeros((6, 6))
        for i in range(6):
            for k in range(6):
                corA, corP = pearsonr(
                    corr_df.values[ms_indexer, i],
                    corr_df.values[ms_indexer, k])
                corrmat[i, k] = corA
                pmat[i, k] = corP

        labels = ['running modulation', 'speed RI', 'learning RI',
                  'daily trial RI', 'trace RI', 'trace offset RI']
        plt.figure()
        sns.heatmap(corrmat, annot=annot, xticklabels=labels,
                    yticklabels=labels,
                    square=True, cbar_kws={'label': 'correlation (R)'})
        plt.xticks(rotation=45, ha='right')
        plt.title(ms + ' Pearson-R corrcoef')
        plt.savefig(
            var_path_prefix + '_corr.png', bbox_inches='tight')

        plt.figure()
        sns.heatmap(pmat, annot=annot, xticklabels=labels, yticklabels=labels,
                    square=True, cbar_kws={'label': 'p-value'})
        plt.xticks(rotation=45, ha='right')
        plt.title(ms + ' Pearson-R p-values')
        plt.savefig(
            var_path_prefix + '_pvals.png', bbox_inches='tight')

        plt.figure()
        logger = np.log10(pmat).flatten()
        vmin = np.nanmin(logger[np.isfinite(logger)])
        vmax = 0
        sns.heatmap(np.log10(pmat), annot=annot, xticklabels=labels,
                    yticklabels=labels, vmin=vmin, vmax=vmax,
                    square=True, cbar_kws={'label': 'log$_{10}$(p-value)'})
        plt.xticks(rotation=45, ha='right')
        plt.title(ms + ' Pearson-R log$_{10}$(p-values)')
        plt.savefig(
            var_path_prefix + '_log10pvals.png', bbox_inches='tight')


def hierclus_on_trials_learning_stages(

        # df params
        mice=['OA27', 'OA26', 'OA67', 'VF226', 'CC175'],
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        words=['restaurant', 'whale', 'whale', 'whale', 'whale'],
        group_by='all',
        nan_thresh=0.85,
        score_threshold=0.8,
        speed_thresh=5,

        # clustering/plotting params
        rank_num=15,
        cluster_number=8,
        cluster_method='ward',
        expected_size_colors=0.5,
        auto_drop=True,

        # save params
        filetype='png'):

    """
    Cluster weights from your trial factors and hierarchically cluster using
    seaborn.clustermap. Annotate plots with useful summary metrics.
    """
    # deal with saving dir
    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}
    group_pars = {'group_by': group_by}

    # if cells were removed with too many nan trials
    if nan_thresh:
        load_tag = '_nantrial' + str(nan_thresh)
        save_tag = ' nantrial ' + str(nan_thresh)
    else:
        load_tag = ''
        save_tag = ''

    # update saving tag if you used a cell score threshold
    if score_threshold:
        load_tag = '_score0pt' + str(int(score_threshold*10)) + load_tag
        save_tag = ' score0pt' + str(int(score_threshold*10)) + save_tag

    met_tag = '_' + cluster_method
    group_word = paths.groupmouse_word({'mice': mice})
    mouse = 'Group-' + group_word
    save_dir = paths.tca_plots(
        mouse, 'group', pars=pars, word=words[0], group_pars=group_pars)
    save_dir = os.path.join(save_dir, 'clustering' + save_tag)
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    var_path_prefix = os.path.join(
        save_dir, str(mouse) + met_tag + '_rank' + str(rank_num) + '_clus' +
        str(cluster_number) + '_heirclus_trialfac_bystage'
        + '_n' + str(len(mice)) + load_tag)

    # create dataframes - ignore python and numpy divide by zero warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with np.errstate(invalid='ignore', divide='ignore'):
            clustering_df, t_df = \
                df.groupmouse_trialfac_summary_stages(
                    mice=mice,
                    trace_type=trace_type,
                    method=method,
                    cs=cs,
                    warp=warp,
                    words=words,
                    group_by=group_by,
                    nan_thresh=nan_thresh,
                    score_threshold=score_threshold,
                    speed_thresh=speed_thresh,
                    rank_num=rank_num,
                    verbose=False)
    clustering_df2 = deepcopy(clustering_df)
    clustering_df3 = deepcopy(clustering_df)

    # if running mod, center of mass, or ramp indices are included, remove
    # from columns (make these into a color df for annotating y-axis)
    learning_stages = ['pre_rev1']
    run_stage = ['running_modulation_' + stage for stage in learning_stages]
    ramp_stage = ['ramp_index_trials_' + stage for stage in learning_stages]
    mean_running_mod = clustering_df.loc[:, run_stage].mean(axis=1)
    ri_trials = clustering_df.loc[:, ramp_stage].mean(axis=1)
    ri_learning = clustering_df.loc[:, 'ramp_index_learning']
    ri_trace = clustering_df.loc[:, 'ramp_index_trace']
    ri_offset = clustering_df.loc[:, 'ramp_index_trace_offset']
    ri_speed = clustering_df.loc[:, 'ramp_index_speed_learning']
    center_of_mass = clustering_df.loc[:, 'center_of_mass']

    if auto_drop:

        # create df only early and high dp learning stages
        keep_cols = [
            'plus_high_dp_learning', 'neutral_high_dp_learning',
            'minus_high_dp_learning', 'plus_high_dp_rev1',
            'minus_high_dp_rev1', 'neutral_high_dp_rev1',
            'plus_naive', 'minus_naive', 'neutral_naive']
        drop_inds = ~clustering_df.columns.isin(keep_cols)
        drop_cols = clustering_df.columns[drop_inds]
        clustering_df = clustering_df.drop(columns=drop_cols)
        nan_indexer = clustering_df.isna().any(axis=1)  # this has to be here
        clustering_df = clustering_df.dropna(axis='rows')

        # create df containing all learning stages through rev1
        keep_cols2 = [
            'plus_naive', 'plus_low_dp_learning', 'plus_high_dp_learning',
            'plus_low_dp_rev1', 'plus_high_dp_rev1', 'neutral_naive',
            'neutral_low_dp_learning', 'neutral_high_dp_learning',
            'neutral_low_dp_rev1', 'neutral_high_dp_rev1', 'minus_naive',
            'minus_low_dp_learning', 'minus_high_dp_learning',
            'minus_low_dp_rev1', 'minus_high_dp_rev1']
        drop_inds2 = ~clustering_df2.columns.isin(keep_cols2)
        drop_cols2 = clustering_df2.columns[drop_inds2]
        clustering_df2 = clustering_df2.drop(columns=drop_cols2)
        clustering_df2 = clustering_df2.dropna(axis='rows')

        # create df containing AMPLITUDE all learning stages through rev1
        keep_cols3 = [
            'plus_amp_naive', 'plus_amp_low_dp_learning',
            'plus_amp_high_dp_learning',
            'plus_amp_low_dp_rev1', 'plus_amp_high_dp_rev1',
            'neutral_amp_naive',
            'neutral_amp_low_dp_learning',
            'neutral_amp_high_dp_learning',
            'neutral_amp_low_dp_rev1', 'neutral_amp_high_dp_rev1',
            'minus_amp_naive',
            'minus_amp_low_dp_learning', 'minus_amp_high_dp_learning',
            'minus_amp_low_dp_rev1', 'minus_amp_high_dp_rev1']
        drop_inds3 = ~clustering_df3.columns.isin(keep_cols3)
        drop_cols3 = clustering_df3.columns[drop_inds3]
        clustering_df3 = clustering_df3.drop(columns=drop_cols3)
        clustering_df3 = clustering_df3.dropna(axis='rows')

        # remove nanned rows from other dfs
        mean_running_mod = mean_running_mod.loc[~nan_indexer, :]
        ri_trials = ri_trials.loc[~nan_indexer, :]
        ri_learning = ri_learning.loc[~nan_indexer, :]
        ri_trace = ri_trace.loc[~nan_indexer, :]
        ri_offset = ri_offset.loc[~nan_indexer, :]
        ri_speed = ri_speed.loc[~nan_indexer, :]
        center_of_mass = center_of_mass.loc[~nan_indexer, :]
        t_df = t_df.loc[~nan_indexer, :]

        # reorder df columns for plots with no clustering on columns
        clustering_df2 = clustering_df2[
            ['plus_naive', 'plus_low_dp_learning', 'plus_high_dp_learning',
             'plus_low_dp_rev1', 'plus_high_dp_rev1', 'neutral_naive',
             'neutral_low_dp_learning', 'neutral_high_dp_learning',
             'neutral_low_dp_rev1', 'neutral_high_dp_rev1',
             'minus_naive', 'minus_low_dp_learning', 'minus_high_dp_learning',
             'minus_low_dp_rev1', 'minus_high_dp_rev1']]

        # reorder df columns for plots with no clustering on columns
        clustering_df3 = clustering_df3[
            ['plus_amp_naive', 'plus_amp_low_dp_learning',
             'plus_amp_high_dp_learning',
             'plus_amp_low_dp_rev1', 'plus_amp_high_dp_rev1',
             'neutral_amp_naive',
             'neutral_amp_low_dp_learning', 'neutral_amp_high_dp_learning',
             'neutral_amp_low_dp_rev1', 'neutral_amp_high_dp_rev1',
             'minus_amp_naive', 'minus_amp_low_dp_learning',
             'minus_amp_high_dp_learning',
             'minus_amp_low_dp_rev1', 'minus_amp_high_dp_rev1']]

    # cluster to get cluster color labels for each component
    g = sns.clustermap(clustering_df, method=cluster_method)
    row_sorter = g.dendrogram_row.reordered_ind
    clusters = hierarchy.fcluster(
        g.dendrogram_row.linkage, cluster_number, criterion='maxclust')
    cluster_color_options = sns.color_palette('hls', cluster_number)
    cluster_colors = [cluster_color_options[i-1] for i in clusters]

    # create mouse color labels
    mouse_list = clustering_df.reset_index().loc[:, 'mouse']
    mouse_color_options = sns.light_palette('navy', len(mouse_list.unique()))
    mouse_color_dict = {k: v for k, v in zip(mouse_list.unique(),
                                             mouse_color_options)}
    mouse_colors = [mouse_color_dict[m] for m in mouse_list]

    # create center of mass color labels
    binned_cm = pd.cut(center_of_mass, 10, labels=range(0, 10))
    cm_color_options = sns.light_palette('red', 10)
    cm_color_dict = {k: v for k, v in zip(np.unique(binned_cm),
                     cm_color_options)}
    cm_colors = [cm_color_dict[m] for m in binned_cm]

    # bins for creating custom heatmaps
    bins = [
        -np.inf, -1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, np.inf]

    # create running mod color labels
    binned_run = pd.cut(mean_running_mod, bins, labels=range(0, len(bins)-1))
    run_color_options = sns.diverging_palette(220, 10, n=(len(bins)-1))
    run_color_dict = {k: v for k, v in zip(np.unique(binned_run),
                      run_color_options)}
    run_colors = [run_color_dict[m] for m in binned_run]

    # create trial ramp index color labels
    binned_ramp = pd.cut(ri_trials, bins, labels=range(0, len(bins)-1))
    ramp_color_options = sns.diverging_palette(220, 10, n=(len(bins)-1))
    ramp_color_dict = {k: v for k, v in zip(np.unique(binned_ramp),
                       ramp_color_options)}
    trial_ramp_colors = [ramp_color_dict[m] for m in binned_ramp]

    # create learning ramp index color labels
    binned_ramp = pd.cut(ri_learning, bins, labels=range(0, len(bins)-1))
    ramp_color_options = sns.diverging_palette(220, 10, n=(len(bins)-1))
    ramp_color_dict = {k: v for k, v in zip(np.unique(binned_ramp),
                       ramp_color_options)}
    learning_ramp_colors = [ramp_color_dict[m] for m in binned_ramp]

    # create trace ramp index color labels
    binned_ramp = pd.cut(ri_trace, bins, labels=range(0, len(bins)-1))
    ramp_color_options = sns.diverging_palette(220, 10, n=(len(bins)-1))
    ramp_color_dict = {k: v for k, v in zip(np.unique(binned_ramp),
                       ramp_color_options)}
    trace_ramp_colors = [ramp_color_dict[m] if ~np.isnan(m) else
                         [.5, .5, .5, 1.] for m in binned_ramp]

    # create trace ramp index color labels
    binned_ramp = pd.cut(ri_offset, bins, labels=range(0, len(bins)-1))
    ramp_color_options = sns.diverging_palette(220, 10, n=(len(bins)-1))
    ramp_color_dict = {k: v for k, v in zip(np.unique(binned_ramp),
                                            ramp_color_options)}
    offset_ramp_colors = [ramp_color_dict[m] if ~np.isnan(m) else
                          [.5, .5, .5, 1.] for m in binned_ramp]

    # create color vector for dis/sated modulation index (ramp index)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with np.errstate(invalid='ignore', divide='ignore'):
            dfdis = calc.fits.groupmouse_fit_disengaged_sated_mean_per_comp(
                mice=mice,
                trace_type=trace_type,
                method=method,
                cs=cs,
                warp=warp,
                words=words,
                group_by=group_by,
                nan_thresh=nan_thresh,
                score_threshold=score_threshold,
                rank=rank_num,
                verbose=True)
    # just use index
    dis = dfdis.loc[:, 'dis_index']
    # removes rows that were already dropped
    dis = dis.loc[~nan_indexer, :]
    binned_dis = pd.cut(dis, bins, labels=range(0, len(bins)-1))
    dis_color_options = sns.diverging_palette(220, 10, n=(len(bins)-1))
    dis_color_dict = {k: v for k, v in zip(np.unique(binned_dis),
                                           dis_color_options)}
    dis_colors = [dis_color_dict[m] if ~np.isnan(m) else
                  [.5, .5, .5, 1.] for m in binned_dis]

    # colors for columns learning stages
    col_colors = []
    for col_name in clustering_df2.columns:
        if 'plus' in col_name.lower():
            col_colors.append('lightgreen')
        elif 'neutral' in col_name.lower():
            col_colors.append('lightskyblue')
        elif 'minus' in col_name.lower():
            col_colors.append('lightcoral')

    # create df of running colors for row colors
    data = {'mouse': mouse_colors,
            'running-modulation': run_colors,
            'ramp-index-learning': learning_ramp_colors,
            'dis-index': dis_colors,
            'ramp-index-daily-trials': trial_ramp_colors,
            'ramp-index-trace': trace_ramp_colors,
            'ramp-index-trace-offset': offset_ramp_colors,
            # 'center-of-mass-trace': cm_colors,
            'cluster': cluster_colors}
    color_df = pd.DataFrame(data=data, index=clustering_df.index)

    # plot
    plt.close('all')

    # parse save params
    if filetype.lower() == 'png' or filetype.lower() == '.png':
        suf = '.png'
    elif filetype.lower() == 'pdf' or filetype.lower() == '.pdf':
        suf = '.pdf'
    elif filetype.lower() == 'eps' or filetype.lower() == '.eps':
        suf = '.eps'
    else:
        print('File-type not recognized: {}'.format(filetype))

    fig1 = clustermap(
        clustering_df, row_colors=color_df, figsize=(13, 13),
        xticklabels=True, yticklabels=True, col_cluster=True,
        row_cluster=True, expected_size_colors=0.5, method=cluster_method)
    fig1.savefig(
        var_path_prefix + '_trialfac{}'.format(suf), bbox_inches='tight')

    fig2 = clustermap(
        t_df.iloc[row_sorter, :], row_colors=color_df.iloc[row_sorter, :],
        figsize=(13, 13), xticklabels=False, yticklabels=True,
        col_cluster=False, row_cluster=False, expected_size_colors=0.5,
        method='ward')
    fig2.savefig(
        var_path_prefix + '_tempofac{}'.format(suf), bbox_inches='tight')

    fig3 = clustermap(
        t_df, row_colors=color_df, figsize=(13, 13),
        xticklabels=False, yticklabels=True, col_cluster=False,
        row_cluster=True, expected_size_colors=0.5, method=cluster_method)
    fig3.savefig(
        var_path_prefix + '_tempofac_sort{}'.format(suf), bbox_inches='tight')

    fig4 = clustermap(
        clustering_df2.iloc[row_sorter, :], figsize=(13, 13),
        row_colors=color_df.iloc[row_sorter, :], col_colors=col_colors,
        xticklabels=True, yticklabels=True, col_cluster=False,
        row_cluster=False, expected_size_colors=0.5, method=cluster_method)
    fig4.savefig(
        var_path_prefix + '_5ptstages{}'.format(suf), bbox_inches='tight')

    col_sorter = [0, 1, 2, 8, 9, 5, 6, 7, 13, 14, 10, 11, 12, 3, 4]
    ori_col_df = clustering_df2.iloc[row_sorter, :]
    ori_col_df = ori_col_df.iloc[:, col_sorter]
    fig5 = clustermap(ori_col_df, figsize=(13, 13),
        row_colors=color_df.iloc[row_sorter, :],
        col_colors=[col_colors[s] for s in col_sorter],
        xticklabels=True, yticklabels=True, col_cluster=False,
        row_cluster=False, expected_size_colors=0.5, method=cluster_method)
    fig5.savefig(
        var_path_prefix + '_5ptstages_oricolsort{}'.format(suf), bbox_inches='tight')

    fig6 = clustermap(
        clustering_df3.iloc[row_sorter, :], figsize=(13, 13),
        row_colors=color_df.iloc[row_sorter, :], col_colors=col_colors,
        xticklabels=True, yticklabels=True, col_cluster=False,
        row_cluster=False, expected_size_colors=0.5, method=cluster_method,
        standard_scale=0)
    fig6.savefig(
        var_path_prefix + '_5ptstages_amp{}'.format(suf), bbox_inches='tight')

    col_sorter = [0, 1, 2, 8, 9, 5, 6, 7, 13, 14, 10, 11, 12, 3, 4]
    ori_col_df = clustering_df3.iloc[row_sorter, :]
    ori_col_df = ori_col_df.iloc[:, col_sorter]
    fig7 = clustermap(ori_col_df, figsize=(13, 13),
        row_colors=color_df.iloc[row_sorter, :],
        col_colors=[col_colors[s] for s in col_sorter],
        xticklabels=True, yticklabels=True, col_cluster=False,
        row_cluster=False, expected_size_colors=0.5, method=cluster_method,
        standard_scale=0)
    fig7.savefig(
        var_path_prefix + '_5ptstages_amp_oricolsort{}'.format(suf), bbox_inches='tight')

    fig8 = clustermap(
        clustering_df3.iloc[row_sorter, :], figsize=(13, 13),
        row_colors=color_df.iloc[row_sorter, :], col_colors=col_colors,
        xticklabels=True, yticklabels=True, col_cluster=False,
        row_cluster=True, expected_size_colors=0.5, method=cluster_method,
        standard_scale=0)
    fig8.savefig(
        var_path_prefix + '_5ptstages_ampclus{}'.format(suf), bbox_inches='tight')


def hierclus_on_amp_trials_learning_stages(

        # df params
        mice=['OA27', 'OA26', 'OA67', 'VF226', 'CC175'],
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        words=['restaurant', 'whale', 'whale', 'whale', 'whale'],
        group_by='all',
        nan_thresh=0.85,
        score_threshold=0.8,
        speed_thresh=5,

        # clustering/plotting params
        rank_num=15,
        cluster_number=8,
        cluster_method='ward',
        expected_size_colors=0.5,
        auto_drop=True,

        # save params
        filetype='png'):

    """
    Cluster weights from your trial factors and hierarchically cluster using
    seaborn.clustermap. Annotate plots with useful summary metrics.
    """
    # deal with saving dir
    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}
    group_pars = {'group_by': group_by}

    # if cells were removed with too many nan trials
    if nan_thresh:
        load_tag = '_nantrial' + str(nan_thresh)
        save_tag = ' nantrial ' + str(nan_thresh)
    else:
        load_tag = ''
        save_tag = ''

    # update saving tag if you used a cell score threshold
    if score_threshold:
        load_tag = '_score0pt' + str(int(score_threshold*10)) + load_tag
        save_tag = ' score0pt' + str(int(score_threshold*10)) + save_tag

    met_tag = '_' + cluster_method
    group_word = paths.groupmouse_word({'mice': mice})
    mouse = 'Group-' + group_word
    save_dir = paths.tca_plots(
        mouse, 'group', pars=pars, word=words[0], group_pars=group_pars)
    save_dir = os.path.join(save_dir, 'clustering' + save_tag)
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    var_path_prefix = os.path.join(
        save_dir, str(mouse) + '_AMP' + met_tag + '_rank' + str(rank_num) +
        '_clus' + str(cluster_number) + '_heirclus_trialfac_bystage'
        + '_n' + str(len(mice)) + load_tag)

    # create dataframes - ignore python and numpy divide by zero warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with np.errstate(invalid='ignore', divide='ignore'):
            clustering_df, t_df = \
                df.groupmouse_trialfac_summary_stages(
                    mice=mice,
                    trace_type=trace_type,
                    method=method,
                    cs=cs,
                    warp=warp,
                    words=words,
                    group_by=group_by,
                    nan_thresh=nan_thresh,
                    score_threshold=score_threshold,
                    speed_thresh=speed_thresh,
                    rank_num=rank_num,
                    verbose=False)
    clustering_df2 = deepcopy(clustering_df)
    clustering_df3 = deepcopy(clustering_df)

    # if running mod, center of mass, or ramp indices are included, remove
    # from columns (make these into a color df for annotating y-axis)
    learning_stages = ['pre_rev1']
    run_stage = ['running_modulation_' + stage for stage in learning_stages]
    ramp_stage = ['ramp_index_trials_' + stage for stage in learning_stages]
    mean_running_mod = clustering_df.loc[:, run_stage].mean(axis=1)
    ri_trials = clustering_df.loc[:, ramp_stage].mean(axis=1)
    ri_learning = clustering_df.loc[:, 'ramp_index_learning']
    ri_trace = clustering_df.loc[:, 'ramp_index_trace']
    ri_offset = clustering_df.loc[:, 'ramp_index_trace_offset']
    ri_speed = clustering_df.loc[:, 'ramp_index_speed_learning']
    center_of_mass = clustering_df.loc[:, 'center_of_mass']

    if auto_drop:

        # create df only early and high dp learning stages
        keep_cols = [
            'plus_high_dp_learning', 'neutral_high_dp_learning',
            'minus_high_dp_learning', 'plus_high_dp_rev1',
            'minus_high_dp_rev1', 'neutral_high_dp_rev1',
            'plus_naive', 'minus_naive', 'neutral_naive']
        drop_inds = ~clustering_df.columns.isin(keep_cols)
        drop_cols = clustering_df.columns[drop_inds]
        clustering_df = clustering_df.drop(columns=drop_cols)
        nan_indexer = clustering_df.isna().any(axis=1)  # this has to be here
        clustering_df = clustering_df.dropna(axis='rows')

        # create df containing all learning stages through rev1
        keep_cols2 = [
            'plus_naive', 'plus_low_dp_learning', 'plus_high_dp_learning',
            'plus_low_dp_rev1', 'plus_high_dp_rev1', 'neutral_naive',
            'neutral_low_dp_learning', 'neutral_high_dp_learning',
            'neutral_low_dp_rev1', 'neutral_high_dp_rev1', 'minus_naive',
            'minus_low_dp_learning', 'minus_high_dp_learning',
            'minus_low_dp_rev1', 'minus_high_dp_rev1']
        drop_inds2 = ~clustering_df2.columns.isin(keep_cols2)
        drop_cols2 = clustering_df2.columns[drop_inds2]
        clustering_df2 = clustering_df2.drop(columns=drop_cols2)
        clustering_df2 = clustering_df2.dropna(axis='rows')

        # create df containing AMPLITUDE all learning stages through rev1
        keep_cols3 = [
            'plus_amp_naive', 'plus_amp_low_dp_learning',
            'plus_amp_high_dp_learning',
            'plus_amp_low_dp_rev1', 'plus_amp_high_dp_rev1',
            'neutral_amp_naive',
            'neutral_amp_low_dp_learning',
            'neutral_amp_high_dp_learning',
            'neutral_amp_low_dp_rev1', 'neutral_amp_high_dp_rev1',
            'minus_amp_naive',
            'minus_amp_low_dp_learning', 'minus_amp_high_dp_learning',
            'minus_amp_low_dp_rev1', 'minus_amp_high_dp_rev1']
        drop_inds3 = ~clustering_df3.columns.isin(keep_cols3)
        drop_cols3 = clustering_df3.columns[drop_inds3]
        clustering_df3 = clustering_df3.drop(columns=drop_cols3)
        clustering_df3 = clustering_df3.dropna(axis='rows')

        # remove nanned rows from other dfs
        mean_running_mod = mean_running_mod.loc[~nan_indexer, :]
        ri_trials = ri_trials.loc[~nan_indexer, :]
        ri_learning = ri_learning.loc[~nan_indexer, :]
        ri_trace = ri_trace.loc[~nan_indexer, :]
        ri_offset = ri_offset.loc[~nan_indexer, :]
        ri_speed = ri_speed.loc[~nan_indexer, :]
        center_of_mass = center_of_mass.loc[~nan_indexer, :]
        t_df = t_df.loc[~nan_indexer, :]

        # reorder df columns for plots with no clustering on columns
        clustering_df2 = clustering_df2[
            ['plus_naive', 'plus_low_dp_learning', 'plus_high_dp_learning',
             'plus_low_dp_rev1', 'plus_high_dp_rev1', 'neutral_naive',
             'neutral_low_dp_learning', 'neutral_high_dp_learning',
             'neutral_low_dp_rev1', 'neutral_high_dp_rev1',
             'minus_naive', 'minus_low_dp_learning', 'minus_high_dp_learning',
             'minus_low_dp_rev1', 'minus_high_dp_rev1']]

        # reorder df columns for plots with no clustering on columns
        clustering_df3 = clustering_df3[
            ['plus_amp_naive', 'plus_amp_low_dp_learning',
             'plus_amp_high_dp_learning',
             'plus_amp_low_dp_rev1', 'plus_amp_high_dp_rev1',
             'neutral_amp_naive',
             'neutral_amp_low_dp_learning', 'neutral_amp_high_dp_learning',
             'neutral_amp_low_dp_rev1', 'neutral_amp_high_dp_rev1',
             'minus_amp_naive', 'minus_amp_low_dp_learning',
             'minus_amp_high_dp_learning',
             'minus_amp_low_dp_rev1', 'minus_amp_high_dp_rev1']]

        # create an additional amplitude df
        clustering_df4 = clustering_df3[
            ['plus_amp_naive',
             'plus_amp_high_dp_learning',
             'plus_amp_high_dp_rev1',
             'neutral_amp_naive',
             'neutral_amp_high_dp_learning',
             'neutral_amp_high_dp_rev1',
             'minus_amp_naive',
             'minus_amp_high_dp_learning',
             'minus_amp_high_dp_rev1']]

    # cluster to get cluster color labels for each component
    g = sns.clustermap(clustering_df4, method=cluster_method, standard_scale=0)
    row_sorter = g.dendrogram_row.reordered_ind
    clusters = hierarchy.fcluster(
        g.dendrogram_row.linkage, cluster_number, criterion='maxclust')
    cluster_color_options = sns.color_palette('hls', cluster_number)
    cluster_colors = [cluster_color_options[i-1] for i in clusters]

    # create mouse color labels
    mouse_list = clustering_df.reset_index().loc[:, 'mouse']
    mouse_color_options = sns.light_palette('navy', len(mouse_list.unique()))
    mouse_color_dict = {k: v for k, v in zip(mouse_list.unique(),
                                             mouse_color_options)}
    mouse_colors = [mouse_color_dict[m] for m in mouse_list]

    # create center of mass color labels
    binned_cm = pd.cut(center_of_mass, 10, labels=range(0, 10))
    cm_color_options = sns.light_palette('red', 10)
    cm_color_dict = {k: v for k, v in zip(np.unique(binned_cm),
                     cm_color_options)}
    cm_colors = [cm_color_dict[m] for m in binned_cm]

    # bins for creating custom heatmaps
    bins = [
        -np.inf, -1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, np.inf]

    # create running mod color labels
    binned_run = pd.cut(mean_running_mod, bins, labels=range(0, len(bins)-1))
    run_color_options = sns.diverging_palette(220, 10, n=(len(bins)-1))
    run_color_dict = {k: v for k, v in zip(np.unique(binned_run),
                      run_color_options)}
    run_colors = [run_color_dict[m] for m in binned_run]

    # create trial ramp index color labels
    binned_ramp = pd.cut(ri_trials, bins, labels=range(0, len(bins)-1))
    ramp_color_options = sns.diverging_palette(220, 10, n=(len(bins)-1))
    ramp_color_dict = {k: v for k, v in zip(np.unique(binned_ramp),
                       ramp_color_options)}
    trial_ramp_colors = [ramp_color_dict[m] for m in binned_ramp]

    # create learning ramp index color labels
    binned_ramp = pd.cut(ri_learning, bins, labels=range(0, len(bins)-1))
    ramp_color_options = sns.diverging_palette(220, 10, n=(len(bins)-1))
    ramp_color_dict = {k: v for k, v in zip(np.unique(binned_ramp),
                       ramp_color_options)}
    learning_ramp_colors = [ramp_color_dict[m] for m in binned_ramp]

    # create trace ramp index color labels
    binned_ramp = pd.cut(ri_trace, bins, labels=range(0, len(bins)-1))
    ramp_color_options = sns.diverging_palette(220, 10, n=(len(bins)-1))
    ramp_color_dict = {k: v for k, v in zip(np.unique(binned_ramp),
                       ramp_color_options)}
    trace_ramp_colors = [ramp_color_dict[m] if ~np.isnan(m) else
                         [.5, .5, .5, 1.] for m in binned_ramp]

    # create trace ramp index color labels
    binned_ramp = pd.cut(ri_offset, bins, labels=range(0, len(bins)-1))
    ramp_color_options = sns.diverging_palette(220, 10, n=(len(bins)-1))
    ramp_color_dict = {k: v for k, v in zip(np.unique(binned_ramp),
                                            ramp_color_options)}
    offset_ramp_colors = [ramp_color_dict[m] if ~np.isnan(m) else
                          [.5, .5, .5, 1.] for m in binned_ramp]

    # create color vector for dis/sated modulation index (ramp index)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with np.errstate(invalid='ignore', divide='ignore'):
            dfdis = calc.fits.groupmouse_fit_disengaged_sated_mean_per_comp(
                mice=mice,
                trace_type=trace_type,
                method=method,
                cs=cs,
                warp=warp,
                words=words,
                group_by=group_by,
                nan_thresh=nan_thresh,
                score_threshold=score_threshold,
                rank=rank_num,
                verbose=True)
    # just use index
    dis = dfdis.loc[:, 'dis_index']
    # removes rows that were already dropped
    dis = dis.loc[~nan_indexer, :]
    binned_dis = pd.cut(dis, bins, labels=range(0, len(bins)-1))
    dis_color_options = sns.diverging_palette(220, 10, n=(len(bins)-1))
    dis_color_dict = {k: v for k, v in zip(np.unique(binned_dis),
                                           dis_color_options)}
    dis_colors = [dis_color_dict[m] if ~np.isnan(m) else
                  [.5, .5, .5, 1.] for m in binned_dis]

    # colors for columns learning stages
    col_colors = []
    for col_name in clustering_df2.columns:
        if 'plus' in col_name.lower():
            col_colors.append('lightgreen')
        elif 'neutral' in col_name.lower():
            col_colors.append('lightskyblue')
        elif 'minus' in col_name.lower():
            col_colors.append('lightcoral')

    # create df of running colors for row colors
    data = {'mouse': mouse_colors,
            'running-modulation': run_colors,
            'ramp-index-learning': learning_ramp_colors,
            'dis-index': dis_colors,
            'ramp-index-daily-trials': trial_ramp_colors,
            'ramp-index-trace': trace_ramp_colors,
            'ramp-index-trace-offset': offset_ramp_colors,
            # 'center-of-mass-trace': cm_colors,
            'cluster': cluster_colors}
    color_df = pd.DataFrame(data=data, index=clustering_df.index)

    # plot
    plt.close('all')

    # parse save params
    if filetype.lower() == 'png' or filetype.lower() == '.png':
        suf = '.png'
    elif filetype.lower() == 'pdf' or filetype.lower() == '.pdf':
        suf = '.pdf'
    elif filetype.lower() == 'eps' or filetype.lower() == '.eps':
        suf = '.eps'
    else:
        print('File-type not recognized: {}'.format(filetype))

    fig1 = clustermap(
        clustering_df4, row_colors=color_df, figsize=(13, 13),
        xticklabels=True, yticklabels=True, col_cluster=False,
        row_cluster=True, expected_size_colors=0.5, method=cluster_method,
        standard_scale=0)
    fig1.savefig(
        var_path_prefix + '_trialfac{}'.format(suf), bbox_inches='tight')

    fig2 = clustermap(
        t_df.iloc[row_sorter, :], row_colors=color_df.iloc[row_sorter, :],
        figsize=(13, 13), xticklabels=False, yticklabels=True,
        col_cluster=False, row_cluster=False, expected_size_colors=0.5,
        method='ward')
    fig2.savefig(
        var_path_prefix + '_tempofac{}'.format(suf), bbox_inches='tight')

    fig3 = clustermap(
        t_df, row_colors=color_df, figsize=(13, 13),
        xticklabels=False, yticklabels=True, col_cluster=False,
        row_cluster=True, expected_size_colors=0.5, method=cluster_method)
    fig3.savefig(
        var_path_prefix + '_tempofac_sort{}'.format(suf), bbox_inches='tight')

    fig4 = clustermap(
        clustering_df2.iloc[row_sorter, :], figsize=(13, 13),
        row_colors=color_df.iloc[row_sorter, :], col_colors=col_colors,
        xticklabels=True, yticklabels=True, col_cluster=False,
        row_cluster=False, expected_size_colors=0.5, method=cluster_method)
    fig4.savefig(
        var_path_prefix + '_5ptstages{}'.format(suf), bbox_inches='tight')

    col_sorter = [0, 1, 2, 8, 9, 5, 6, 7, 13, 14, 10, 11, 12, 3, 4]
    ori_col_df = clustering_df2.iloc[row_sorter, :]
    ori_col_df = ori_col_df.iloc[:, col_sorter]
    fig5 = clustermap(ori_col_df, figsize=(13, 13),
        row_colors=color_df.iloc[row_sorter, :],
        col_colors=[col_colors[s] for s in col_sorter],
        xticklabels=True, yticklabels=True, col_cluster=False,
        row_cluster=False, expected_size_colors=0.5, method=cluster_method)
    fig5.savefig(
        var_path_prefix + '_5ptstages_oricolsort{}'.format(suf), bbox_inches='tight')

    fig6 = clustermap(
        clustering_df3.iloc[row_sorter, :], figsize=(13, 13),
        row_colors=color_df.iloc[row_sorter, :], col_colors=col_colors,
        xticklabels=True, yticklabels=True, col_cluster=False,
        row_cluster=False, expected_size_colors=0.5, method=cluster_method,
        standard_scale=0)
    fig6.savefig(
        var_path_prefix + '_5ptstages_amp{}'.format(suf), bbox_inches='tight')

    col_sorter = [0, 1, 2, 8, 9, 5, 6, 7, 13, 14, 10, 11, 12, 3, 4]
    ori_col_df = clustering_df3.iloc[row_sorter, :]
    ori_col_df = ori_col_df.iloc[:, col_sorter]
    fig7 = clustermap(ori_col_df, figsize=(13, 13),
        row_colors=color_df.iloc[row_sorter, :],
        col_colors=[col_colors[s] for s in col_sorter],
        xticklabels=True, yticklabels=True, col_cluster=False,
        row_cluster=False, expected_size_colors=0.5, method=cluster_method,
        standard_scale=0)
    fig7.savefig(
        var_path_prefix + '_5ptstages_amp_oricolsort{}'.format(suf), bbox_inches='tight')

    fig8 = clustermap(
        clustering_df3.iloc[row_sorter, :], figsize=(13, 13),
        row_colors=color_df.iloc[row_sorter, :], col_colors=col_colors,
        xticklabels=True, yticklabels=True, col_cluster=False,
        row_cluster=True, expected_size_colors=0.5, method=cluster_method,
        standard_scale=0)
    fig8.savefig(
        var_path_prefix + '_5ptstages_ampclus{}'.format(suf), bbox_inches='tight')


def hierclus_simple_on_trials_learning_stages(

        # df params
        mice=['OA27', 'OA26', 'OA67', 'VF226', 'CC175'],
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        words=['orlando', 'already', 'already', 'already', 'already'],
        group_by='all',
        nan_thresh=0.85,
        score_threshold=None,
        speed_thresh=5,

        # clustering/plotting params
        rank_num=18,
        cluster_number=8,
        cluster_method='ward',
        expected_size_colors=0.5,
        filetype='png',
        auto_drop=True):

    """
    Cluster weights from your trial factors and hierarchically cluster using
    seaborn.clustermap. Annotate plots with useful summary metrics.
    """
    # deal with saving dir
    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}
    group_pars = {'group_by': group_by}

    # if cells were removed with too many nan trials
    if nan_thresh:
        load_tag = '_nantrial' + str(nan_thresh)
        save_tag = ' nantrial ' + str(nan_thresh)
    else:
        load_tag = ''
        save_tag = ''

    # update saving tag if you used a cell score threshold
    if score_threshold:
        load_tag = '_score0pt' + str(int(score_threshold*10)) + load_tag
        save_tag = ' score0pt' + str(int(score_threshold*10)) + save_tag

    met_tag = '_' + cluster_method
    group_word = paths.groupmouse_word({'mice': mice})
    mouse = 'Group-' + group_word
    save_dir = paths.tca_plots(
        mouse, 'group', pars=pars, word=words[0], group_pars=group_pars)
    save_dir = os.path.join(save_dir, 'clustering simple' + save_tag)
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    var_path_prefix = os.path.join(
        save_dir, str(mouse) + met_tag + '_rank' + str(rank_num) + '_sclus' +
        str(cluster_number) + '_heirclus_trialfac_bystage'
        + '_n' + str(len(mice)) + load_tag)

    # create dataframes - ignore python and numpy divide by zero warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with np.errstate(invalid='ignore', divide='ignore'):
            clustering_df, t_df = \
                df.groupmouse_trialfac_summary_stages(
                    mice=mice,
                    trace_type=trace_type,
                    method=method,
                    cs=cs,
                    warp=warp,
                    words=words,
                    group_by=group_by,
                    nan_thresh=nan_thresh,
                    score_threshold=score_threshold,
                    speed_thresh=speed_thresh,
                    rank_num=rank_num,
                    verbose=False)
    clustering_df2 = deepcopy(clustering_df)
    clustering_df3 = deepcopy(clustering_df)

    # if running mod, center of mass, or ramp indices are included, remove
    # from columns (make these into a color df for annotating y-axis)
    learning_stages = ['pre_rev1']
    run_stage = ['running_modulation_' + stage for stage in learning_stages]
    ramp_stage = ['ramp_index_trials_' + stage for stage in learning_stages]
    mean_running_mod = clustering_df.loc[:, run_stage].mean(axis=1)
    ri_trials = clustering_df.loc[:, ramp_stage].mean(axis=1)
    ri_learning = clustering_df.loc[:, 'ramp_index_learning']
    ri_trace = clustering_df.loc[:, 'ramp_index_trace']
    ri_offset = clustering_df.loc[:, 'ramp_index_trace_offset']
    ri_speed = clustering_df.loc[:, 'ramp_index_speed_learning']
    center_of_mass = clustering_df.loc[:, 'center_of_mass']

    if auto_drop:

        # create df only early and high dp learning stages
        keep_cols = [
            'plus_high_dp_learning', 'neutral_high_dp_learning',
            'minus_high_dp_learning', 'plus_high_dp_rev1',
            'minus_high_dp_rev1', 'neutral_high_dp_rev1',
            'plus_naive', 'minus_naive', 'neutral_naive']
        drop_inds = ~clustering_df.columns.isin(keep_cols)
        drop_cols = clustering_df.columns[drop_inds]
        clustering_df = clustering_df.drop(columns=drop_cols)
        nan_indexer = clustering_df.isna().any(axis=1)  # this has to be here
        clustering_df = clustering_df.dropna(axis='rows')

        # create df containing all learning stages through rev1
        keep_cols2 = [
            'plus_naive', 'plus_low_dp_learning', 'plus_high_dp_learning',
            'plus_low_dp_rev1', 'plus_high_dp_rev1', 'neutral_naive',
            'neutral_low_dp_learning', 'neutral_high_dp_learning',
            'neutral_low_dp_rev1', 'neutral_high_dp_rev1', 'minus_naive',
            'minus_low_dp_learning', 'minus_high_dp_learning',
            'minus_low_dp_rev1', 'minus_high_dp_rev1']
        drop_inds2 = ~clustering_df2.columns.isin(keep_cols2)
        drop_cols2 = clustering_df2.columns[drop_inds2]
        clustering_df2 = clustering_df2.drop(columns=drop_cols2)
        clustering_df2 = clustering_df2.dropna(axis='rows')

        # create df containing AMPLITUDE all learning stages through rev1
        keep_cols3 = [
            'plus_amp_naive', 'plus_amp_low_dp_learning',
            'plus_amp_high_dp_learning',
            'plus_amp_low_dp_rev1', 'plus_amp_high_dp_rev1',
            'neutral_amp_naive',
            'neutral_amp_low_dp_learning',
            'neutral_amp_high_dp_learning',
            'neutral_amp_low_dp_rev1', 'neutral_amp_high_dp_rev1',
            'minus_amp_naive',
            'minus_amp_low_dp_learning', 'minus_amp_high_dp_learning',
            'minus_amp_low_dp_rev1', 'minus_amp_high_dp_rev1']
        drop_inds3 = ~clustering_df3.columns.isin(keep_cols3)
        drop_cols3 = clustering_df3.columns[drop_inds3]
        clustering_df3 = clustering_df3.drop(columns=drop_cols3)
        clustering_df3 = clustering_df3.dropna(axis='rows')

        # remove nanned rows from other dfs
        mean_running_mod = mean_running_mod.loc[~nan_indexer, :]
        ri_trials = ri_trials.loc[~nan_indexer, :]
        ri_learning = ri_learning.loc[~nan_indexer, :]
        ri_trace = ri_trace.loc[~nan_indexer, :]
        ri_offset = ri_offset.loc[~nan_indexer, :]
        ri_speed = ri_speed.loc[~nan_indexer, :]
        center_of_mass = center_of_mass.loc[~nan_indexer, :]
        t_df = t_df.loc[~nan_indexer, :]

        # reorder df columns for plots with no clustering on columns
        clustering_df2 = clustering_df2[
            ['plus_naive', 'plus_low_dp_learning', 'plus_high_dp_learning',
             'plus_low_dp_rev1', 'plus_high_dp_rev1', 'neutral_naive',
             'neutral_low_dp_learning', 'neutral_high_dp_learning',
             'neutral_low_dp_rev1', 'neutral_high_dp_rev1',
             'minus_naive', 'minus_low_dp_learning', 'minus_high_dp_learning',
             'minus_low_dp_rev1', 'minus_high_dp_rev1']]

        # reorder df columns for plots with no clustering on columns
        clustering_df3 = clustering_df3[
            ['plus_amp_naive', 'plus_amp_low_dp_learning',
             'plus_amp_high_dp_learning',
             'plus_amp_low_dp_rev1', 'plus_amp_high_dp_rev1',
             'neutral_amp_naive',
             'neutral_amp_low_dp_learning', 'neutral_amp_high_dp_learning',
             'neutral_amp_low_dp_rev1', 'neutral_amp_high_dp_rev1',
             'minus_amp_naive', 'minus_amp_low_dp_learning',
             'minus_amp_high_dp_learning',
             'minus_amp_low_dp_rev1', 'minus_amp_high_dp_rev1']]

    # cluster to get cluster color labels for each component
    g = sns.clustermap(clustering_df, method=cluster_method)
    row_sorter = g.dendrogram_row.reordered_ind
    clusters = hierarchy.fcluster(
        g.dendrogram_row.linkage, cluster_number, criterion='maxclust')
    cluster_color_options = sns.color_palette('deep', cluster_number)
    cluster_colors = [cluster_color_options[i-1] for i in clusters]

    # create mouse color labels
    mouse_list = clustering_df.reset_index().loc[:, 'mouse']
    mouse_color_options = sns.light_palette('navy', len(mouse_list.unique()))
    mouse_color_dict = {k: v for k, v in zip(mouse_list.unique(),
                                             mouse_color_options)}
    mouse_colors = [mouse_color_dict[m] for m in mouse_list]

    # create center of mass color labels
    binned_cm = pd.cut(center_of_mass, 10, labels=range(0, 10))
    cm_color_options = sns.light_palette('red', 10)
    cm_color_dict = {k: v for k, v in zip(np.unique(binned_cm),
                     cm_color_options)}
    cm_colors = [cm_color_dict[m] for m in binned_cm]

    # bins for creating custom heatmaps
    bins = [
        -np.inf, -1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, np.inf]

    # create running mod color labels
    binned_run = pd.cut(mean_running_mod, bins, labels=range(0, len(bins)-1))
    run_color_options = sns.diverging_palette(220, 10, n=(len(bins)-1))
    run_color_dict = {k: v for k, v in zip(np.unique(binned_run),
                      run_color_options)}
    run_colors = [run_color_dict[m] for m in binned_run]

    # create trial ramp index color labels
    binned_ramp = pd.cut(ri_trials, bins, labels=range(0, len(bins)-1))
    ramp_color_options = sns.diverging_palette(220, 10, n=(len(bins)-1))
    ramp_color_dict = {k: v for k, v in zip(np.unique(binned_ramp),
                       ramp_color_options)}
    trial_ramp_colors = [ramp_color_dict[m] for m in binned_ramp]

    # create learning ramp index color labels
    binned_ramp = pd.cut(ri_learning, bins, labels=range(0, len(bins)-1))
    ramp_color_options = sns.diverging_palette(220, 10, n=(len(bins)-1))
    ramp_color_dict = {k: v for k, v in zip(np.unique(binned_ramp),
                       ramp_color_options)}
    learning_ramp_colors = [ramp_color_dict[m] for m in binned_ramp]

    # create trace ramp index color labels
    binned_ramp = pd.cut(ri_trace, bins, labels=range(0, len(bins)-1))
    ramp_color_options = sns.diverging_palette(220, 10, n=(len(bins)-1))
    ramp_color_dict = {k: v for k, v in zip(np.unique(binned_ramp),
                       ramp_color_options)}
    trace_ramp_colors = [ramp_color_dict[m] if ~np.isnan(m) else
                         [.5, .5, .5, 1.] for m in binned_ramp]

    # create trace ramp index color labels
    binned_ramp = pd.cut(ri_offset, bins, labels=range(0, len(bins)-1))
    ramp_color_options = sns.diverging_palette(220, 10, n=(len(bins)-1))
    ramp_color_dict = {k: v for k, v in zip(np.unique(binned_ramp),
                                            ramp_color_options)}
    offset_ramp_colors = [ramp_color_dict[m] if ~np.isnan(m) else
                          [.5, .5, .5, 1.] for m in binned_ramp]

    # create colormap for variance explained by each component (as a fraction
    # of the total variance explained
    mouse_list = clustering_df.reset_index().loc[:, 'mouse'].unique()
    var_list = []
    for mouse in mouse_list:
        word = words[np.where(np.isin(mice, mouse))[0][0]]
        var_df = calc.var.groupday_varex_bycomp(
            flow.Mouse(mouse=mouse), word=word)
        var_df = var_df.loc[(var_df['rank'] == rank_num), :]
        scalar = var_df.sum()['variance_explained_tcamodel']
        var_df = var_df['variance_explained_tcamodel'] / scalar
        var_list.append(var_df)
    print(var_list)
    varex = pd.concat(var_list, axis=0)
    bins = list(np.arange(0, 0.2, 0.01))
    bins.append(np.inf)
    varex_color_options = sns.cubehelix_palette(
        len(bins)-1, start=.5, rot=-.75, reverse=True)
    binned_varex = pd.cut(varex, bins, labels=range(0, len(bins)-1))
    varex_color_dict = {k: v for k, v in zip(np.unique(binned_varex),
                                             varex_color_options)}
    varex_colors = [varex_color_dict[m] if ~np.isnan(m) else
                    [.5, .5, .5, 1.] for m in binned_varex]

    # colors for columns learning stages
    col_colors = []
    for col_name in clustering_df2.columns:
        if 'plus' in col_name.lower():
            col_colors.append('lightgreen')
        elif 'neutral' in col_name.lower():
            col_colors.append('lightskyblue')
        elif 'minus' in col_name.lower():
            col_colors.append('lightcoral')

    # create df of running colors for row colors
    data = {'mouse': mouse_colors,
            # 'running-modulation': run_colors,
            # 'ramp-index-learning': learning_ramp_colors,
            # 'ramp-index-daily-trials': trial_ramp_colors,
            # 'ramp-index-trace': trace_ramp_colors,
            # 'ramp-index-trace-offset': offset_ramp_colors,
            # 'center-of-mass-trace': cm_colors,
            # 'variance explained': varex_colors,
            'cluster': cluster_colors}
    color_df = pd.DataFrame(data=data, index=clustering_df.index)
    data = {'mouse': mouse_colors}
    mcolor_df = pd.DataFrame(data=data, index=clustering_df.index)

    # plot
    plt.close('all')
    sns.set_context("talk")
    figy = 15
    figx = 12
    yfontsize = 12
    xlabl = ['naive', 'early learning', 'late learning',
             'early reversal', 'late reversal',
             'naive', 'early learning', 'late learning',
             'early reversal', 'late reversal',
             'naive', 'early learning', 'late learning',
             'early reversal', 'late reversal']

    fig1 = clustermap(
        clustering_df, row_colors=color_df, figsize=(figx, figy),
        xticklabels=True, yticklabels=True, col_cluster=True,
        row_cluster=True, expected_size_colors=0.5, method=cluster_method)
    fig1.ax_heatmap.set_yticklabels(
        fig1.ax_heatmap.get_yticklabels(), fontsize=yfontsize)
    fig1.savefig(
        var_path_prefix + '_trialfac.' + filetype, bbox_inches='tight')

    fig2 = clustermap(
        t_df.iloc[row_sorter, :], row_colors=color_df.iloc[row_sorter, :],
        figsize=(figx, figy), xticklabels=False, yticklabels=True,
        col_cluster=False, row_cluster=False, expected_size_colors=0.5,
        method='ward')
    fig2.ax_heatmap.set_yticklabels(
        fig2.ax_heatmap.get_yticklabels(), fontsize=yfontsize)
    fig2.savefig(
        var_path_prefix + '_tempofac.' + filetype, bbox_inches='tight')

    fig3 = clustermap(
        t_df, row_colors=color_df, figsize=(figx, figy),
        xticklabels=False, yticklabels=True, col_cluster=False,
        row_cluster=True, expected_size_colors=0.5, method=cluster_method)
    fig3.ax_heatmap.set_yticklabels(
        fig3.ax_heatmap.get_yticklabels(), fontsize=yfontsize)
    fig3.savefig(
        var_path_prefix + '_tempofac_sort.' + filetype, bbox_inches='tight')

    fig4 = clustermap(
        clustering_df2.iloc[row_sorter, :], figsize=(figx, figy),
        row_colors=color_df.iloc[row_sorter, :], col_colors=col_colors,
        xticklabels=xlabl, yticklabels=True, col_cluster=False,
        row_cluster=False, expected_size_colors=0.5, method=cluster_method)
    fig4.ax_heatmap.set_yticklabels(
        fig4.ax_heatmap.get_yticklabels(), fontsize=yfontsize)
    fig4.savefig(
        var_path_prefix + '_5ptstages.' + filetype, bbox_inches='tight')

    col_sorter = [0, 1, 2, 8, 9, 5, 6, 7, 13, 14, 10, 11, 12, 3, 4]
    ori_col_df = clustering_df2.iloc[row_sorter, :]
    ori_col_df = ori_col_df.iloc[:, col_sorter]
    fig5 = clustermap(ori_col_df, figsize=(figx, figy),
        row_colors=color_df.iloc[row_sorter, :],
        col_colors=[col_colors[s] for s in col_sorter],
        xticklabels=xlabl, yticklabels=True, col_cluster=False,
        row_cluster=False, expected_size_colors=0.5, method=cluster_method)
    fig5.ax_heatmap.set_yticklabels(
        fig5.ax_heatmap.get_yticklabels(), fontsize=yfontsize)
    fig5.savefig(
        var_path_prefix + '_5ptstages_oricolsort.' + filetype, bbox_inches='tight')

    fig6 = clustermap(
        clustering_df3.iloc[row_sorter, :], figsize=(figx, figy),
        row_colors=color_df.iloc[row_sorter, :], col_colors=col_colors,
        xticklabels=xlabl, yticklabels=True, col_cluster=False,
        row_cluster=False, expected_size_colors=0.5, method=cluster_method,
        standard_scale=0)
    fig6.ax_heatmap.set_yticklabels(
        fig6.ax_heatmap.get_yticklabels(), fontsize=yfontsize)
    fig6.savefig(
        var_path_prefix + '_5ptstages_amp.' + filetype, bbox_inches='tight')

    col_sorter = [0, 1, 2, 8, 9, 5, 6, 7, 13, 14, 10, 11, 12, 3, 4]
    ori_col_df = clustering_df3.iloc[row_sorter, :]
    ori_col_df = ori_col_df.iloc[:, col_sorter]
    fig7 = clustermap(ori_col_df, figsize=(figx, figy),
        row_colors=color_df.iloc[row_sorter, :],
        col_colors=[col_colors[s] for s in col_sorter],
        xticklabels=xlabl, yticklabels=True, col_cluster=False,
        row_cluster=False, expected_size_colors=0.5, method=cluster_method,
        standard_scale=0)
    fig7.ax_heatmap.set_yticklabels(
        fig7.ax_heatmap.get_yticklabels(), fontsize=yfontsize)
    fig7.savefig(
        var_path_prefix + '_5ptstages_amp_oricolsort.' + filetype, bbox_inches='tight')

    col_sorter = [0, 1, 2, 8, 9, 5, 6, 7, 13, 14, 10, 11, 12, 3, 4]
    fig8 = clustermap(
        clustering_df3.iloc[row_sorter, col_sorter],
        figsize=(figx, figy),
        row_colors=color_df.iloc[row_sorter, :],
        col_colors=[col_colors[s] for s in col_sorter],
        xticklabels=xlabl, yticklabels=True, col_cluster=False,
        row_cluster=True, expected_size_colors=0.5,
        method=cluster_method,
        standard_scale=0)
    fig8.ax_heatmap.set_yticklabels(
        fig8.ax_heatmap.get_yticklabels(), fontsize=yfontsize)
    fig8.savefig(
        var_path_prefix + '_5ptstages_ampclus.' + filetype, bbox_inches='tight')

    col_sorter = [0, 1, 2, 8, 9, 5, 6, 7, 13, 14, 10, 11, 12, 3, 4]
    ori_col_df = clustering_df2
    ori_col_df = ori_col_df.iloc[:, col_sorter]
    fig9 = clustermap(ori_col_df, figsize=(figx, figy),
        row_colors=mcolor_df,
        col_colors=[col_colors[s] for s in col_sorter],
        xticklabels=xlabl, yticklabels=True, col_cluster=False,
        row_cluster=False, expected_size_colors=0.5, method=cluster_method)
    fig9.ax_heatmap.set_yticklabels(
        fig9.ax_heatmap.get_yticklabels(), fontsize=yfontsize)
    fig9.savefig(
        var_path_prefix + '_5ptstages_oricolsort_nosort.' + filetype,
        bbox_inches='tight')

    col_sorter = [0, 1, 2, 8, 9, 5, 6, 7, 13, 14, 10, 11, 12, 3, 4]
    row_indexer = (clustering_df2.reset_index()['mouse'] == 'OA27').values
    ori_col_df = clustering_df2.iloc[row_indexer, :]
    ori_col_df = ori_col_df.iloc[:, col_sorter]
    g = clustermap(ori_col_df,  # figsize=(figx, figy),
        row_colors=mcolor_df.iloc[row_indexer, :],
        col_colors=[col_colors[s] for s in col_sorter],
        xticklabels=xlabl, yticklabels=True, col_cluster=False,
        row_cluster=False, expected_size_colors=0.5, method=cluster_method)
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(), fontsize=yfontsize)
    g.savefig(
        var_path_prefix + '_5ptstages_oricolsort_ex.' + filetype,
        bbox_inches='tight')


def groupday_longform_factors_annotated_clusfolders(
        mice=['OA27', 'OA26', 'OA67', 'VF226', 'CC175'],
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        words=['orlando', 'already', 'already', 'already', 'already'],
        group_by='all',
        nan_thresh=0.85,
        rank_num=18,
        clus_num=10,
        use_dprime=False,
        extra_col=1,
        alpha=0.6,
        plot_running=True,
        log_scale=True,
        cluster_method='ward',
        speed_thresh=5,
        filetype='png',
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

    # plot cell factors using a log scale
    if log_scale:
        log_tag = 'logsc '
    else:
        log_tag = ''

    # save dir
    group_word = paths.groupmouse_word({'mice': mice})
    mouse = 'Group-' + group_word
    save_dir = paths.tca_plots(
        mouse, 'group', pars=pars, word=words[0], group_pars=group_pars)
    save_dir = os.path.join(save_dir, 'factors annotated long-form'
                            + nt_save_tag + ' nclus ' + str(clus_num))
    if not os.path.isdir(save_dir): os.mkdir(save_dir)

    # get clusters
    clus_df, temp_df = \
        df.groupmouse_trialfac_summary_stages(
            mice=mice,
            trace_type=trace_type,
            method=method,
            cs=cs,
            warp=warp,
            words=words,
            group_by=group_by,
            nan_thresh=nan_thresh,
            speed_thresh=speed_thresh,
            rank_num=rank_num,
            verbose=False)
    # if use_dprime:
    #     clus_df, temp_df = cluster.trial_factors_across_mice_dprime(
    #         mice=mice,
    #         trace_type=trace_type,
    #         method=method,
    #         cs=cs,
    #         warp=warp,
    #         words=words,
    #         group_by=group_by,
    #         nan_thresh=nan_thresh,
    #         verbose=verbose,
    #         rank_num=rank_num)
    # else:
    #     clus_df, temp_df = cluster.trial_factors_across_mice(
    #         mice=mice,
    #         trace_type=trace_type,
    #         method=method,
    #         cs=cs,
    #         warp=warp,
    #         words=words,
    #         group_by=group_by,
    #         nan_thresh=nan_thresh,
    #         verbose=verbose,
    #         rank_num=rank_num)
    # clus_df = clus_df.dropna(axis='rows')
    # clus_df = cluster.get_component_clusters(clus_df, clus_num)
    clus_df = cluster.get_groupday_stage_clusters(
        clus_df, clus_num, method=cluster_method)

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
            clus_dir = os.path.join(save_dir, 'cluster ' + log_tag + str(clus))
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
                if not log_scale:
                    ax[0, 0].plot(
                        np.arange(0, len(U.factors[0][:, comp])),
                        U.factors[0][:, comp], '.', color='b', alpha=0.7)
                    ax[0, 0].autoscale(enable=True, axis='both', tight=True)
                else:
                    ax[0, 0].plot(
                        np.arange(0, len(U.factors[0][:, comp])),
                        np.log2(U.factors[0][:, comp]), '.', color='b',
                        alpha=0.7)
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
                                    linewidth=3, alpha=0.7)

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

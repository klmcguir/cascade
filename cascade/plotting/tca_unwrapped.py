"""Functions for plotting tca decomp run a new way."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensortools.tensors import KTensor
import seaborn as sns
from copy import deepcopy
from .. import paths
from .. import lookups
from .. import utils


def longform_factors_annotated(
        meta,
        ktensor,
        mod,
        unwrapped_ktensor=None,
        alpha=0.6,
        plot_running=True,
        filetype='png',
        scale_y=False,
        hmm_engaged=True,
        add_prev_cols=True,
        folder_tag='',
        verbose=False):
    """Plot trial factors as scatter and pseduocolor with various labels.

    Parameters
    ----------
    meta : pandas.DataFrame
        Trial metadata.
    ktensor : tensortools.KTensor
        Your factors containing your trial factors[2] for a given rank of "mod".
    mod : str
        Model name.
    unwrapped_ktensor : tensortools.KTensor, optional
        The factors from a given model (should match "mod"), contains tuning factors, by default None
    alpha : float, optional
        alpha for scatter, by default 0.6
    plot_running : bool, optional
        Plot running as a pink trace (normalized) behind scatter, by default True
    filetype : str, optional
        File extensions for saving, by default 'png'
    scale_y : bool, optional
        Rescale your y_axis (i.e., if outliers make it hard to see your scatter), by default False
    hmm_engaged : bool, optional
        Add a row for HMM engagement, by default True
    add_prev_cols : bool, optional
        Add columns that deal with previous cues or rewards, etc., by default True
    verbose : bool, optional
        Print terminal output, by default False
    """

    # get your mouse from metadata
    mouse = utils.meta_mouse(meta)

    # use matplotlib plotting defaults
    mpl.rcParams.update(mpl.rcParamsDefault)

    # sort your cell factors by max
    sort_ktensor, _ = _sortcellfactor(ktensor)

    # get rank of model
    r = ktensor.rank

    # if unwrapped ktensor is provided, get "best" tuning
    if unwrapped_ktensor is not None:
        assert ktensor.rank == unwrapped_ktensor.rank
        tune_order = ['becomes_unrewarded', 'remains_unrewarded', 'becomes_rewarded']
        inverted_lookup = {v:k for k, v in lookups.lookup_mm[mouse].items()}
        tuning_factors = utils.rescale_factors(unwrapped_ktensor)[2]
        max_tune = np.argmax(tuning_factors, axis=0)
        sum_tune = np.nansum(tuning_factors, axis=0)
        tuning_oris = [[lookups.lookup[mouse][inverted_lookup[tune_order[s]]]]
                       if su <= 1.5 else [0, 135, 270]
                       for s, su in zip(max_tune, sum_tune)]
        tuning_type = [
            lookups.lookup_mm[mouse][lookups.lookup_ori[mouse][s[0]]]
            if len(s) == 1 else 'broad' for s in tuning_oris
        ]

    orientation = meta['orientation']
    trial_num = np.arange(0, len(orientation))
    condition = meta['condition']
    trialerror = meta['trialerror']
    hunger = deepcopy(meta['hunger'])
    speed = meta['speed']
    dates = meta.reset_index()['date']
    learning_state = meta['learning_state']
    if hmm_engaged and 'hmm_engaged' in meta.columns:
        hmm = meta['hmm_engaged']
    else:
        if verbose:
            print('hmm_engaged not in columns: Final row removed.')

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

    # plot
    if hmm_engaged:
        rows = 7
    else:
        rows = 6
    if add_prev_cols:
        rows += 6

    cols = 3
    U = sort_ktensor
    for comp in range(U.rank):
        _, axes = plt.subplots(
            rows, cols, figsize=(17, rows),
            gridspec_kw={'width_ratios': [2, 2, 17]})

        # reset previous col (trial history variables) counter
        prev_col_counter = 0

        # reshape for easier indexing
        ax = np.array(axes).reshape((rows, -1))
        ax[0, 0].set_title('Neuron factors')
        ax[0, 1].set_title('Temporal factors')
        ax[0, 2].set_title('Trial factors')

        # add title to whole figure
        ax[0, 0].text(-1.2, 4,
            f'\n{mouse}:\n\n rank: {str(int(r))}' +
            f'\n model: {mod}' +
            (f'\n component pref. tuning: {tuning_oris[comp]}' if  unwrapped_ktensor is not None else '') +
            (f'\n component pref. tuning: {tuning_type[comp]}' if  unwrapped_ktensor is not None else ''),
            fontsize=12,
            transform=ax[0, 0].transAxes, color='#969696')

        # plot cell factors
        ax[0, 0].plot(
            np.arange(0, len(U.factors[0][:, comp])),
            U.factors[0][:, comp], '.', color='b', alpha=0.7)
        ax[0, 0].autoscale(enable=True, axis='both', tight=True)

        # plot temporal factors
        ax[0, 1].plot(U.factors[1][:, comp], color='r', linewidth=1.5)
        ax[0, 1].autoscale(enable=True, axis='both', tight=True)

        # add a line for stim onset and offset
        # NOTE: assumes downsample, 1 sec before onset, default is 15.5 Hz
        # if '_bin' in trace_type.lower():
        #     one_sec = 3.9  # 27 frames for 7 sec, 1 pre, 6, post
        # else:
        #     one_sec = 15.5
        # off_time = lookups.stim_length[mouse]
        # y_lim = ax[0, 1].get_ylim()
        # ons = one_sec * 1
        # offs = ons + one_sec * off_time
        # ax[0, 1].plot([ons, ons], y_lim, ':k')
        # if '_onset' not in trace_type.lower():
        #     ax[0, 1].plot([offs, offs], y_lim, ':k')

        col = cols - 1
        for i in range(rows):

            # get axis values
            if i == 0:
                y_sc_factor = 4
                if scale_y:
                    ystd3 = np.nanstd(U.factors[2][:, comp]) * y_sc_factor
                    ymax = np.nanmax(U.factors[2][:, comp])
                    if ystd3 < ymax:
                        y_lim = [0, ystd3]
                    else:
                        y_lim = [0, ymax]
                else:
                    y_lim = [0, np.nanmax(U.factors[2][:, comp])]

            # running
            if plot_running:
                scale_by = np.nanmax(speed) / y_lim[1]
                if not np.isnan(scale_by):
                    ax[i, col].plot(
                        np.array(speed.tolist()) / scale_by,
                        color=[1, 0.1, 0.6, 0.2])
                    # , label='speed')

            # Orientation - main variable to plot
            if i == 0:
                ori_vals = [0, 135, 270]
                # color_vals = [[0.28, 0.68, 0.93, alpha],
                #               [0.84, 0.12, 0.13, alpha],
                #               [0.46, 0.85, 0.47, alpha]]
                color_vals = sns.color_palette('BuPu', 3)
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
                ax[i, col].set_xticklabels([])

            # High/low speed 5 cm/s threshold - main variable to plot
            elif i == 5:
                speed_bool = speed.values > 4
                color_vals = sns.color_palette("hls", 2)

                ax[i, col].plot(trial_num[~speed_bool],
                                U.factors[2][~speed_bool, comp], 'o',
                                label='stationary',
                                color=color_vals[1],
                                alpha=0.3,
                                markersize=2)
                ax[i, col].plot(trial_num[speed_bool],
                                U.factors[2][speed_bool, comp], 'o',
                                label='running',
                                color=color_vals[0],
                                alpha=0.3,
                                markersize=2)

                ax[i, col].legend(
                    bbox_to_anchor=(1.02, 1), loc='upper left',
                    borderaxespad=0, title='Running', markerscale=2,
                    prop={'size': 8})
                ax[i, col].autoscale(enable=True, axis='both', tight=True)

                if hmm_engaged or add_prev_cols:
                    ax[i, col].set_xticklabels([])

            # HMM engagement - main variable to plot
            elif i == 6:
                h_vals = ['engaged', 'disengaged']
                h_labels = ['engaged', 'disengaged']
                color_vals = [[1, 0.6, 0.3, alpha],
                                [0.7, 0.9, 0.4, alpha]]

                ax[i, col].plot(trial_num[hmm],
                                U.factors[2][hmm, comp], 'o',
                                label=str(h_labels[0]),
                                color=color_vals[0],
                                markersize=2)
                ax[i, col].plot(trial_num[~hmm],
                                U.factors[2][~hmm, comp], 'o',
                                label=str(h_labels[1]),
                                color=color_vals[1],
                                markersize=2)
                ax[i, col].legend(
                    bbox_to_anchor=(1.02, 1), loc='upper left',
                    borderaxespad=0, title='HMM engaged', markerscale=2,
                    prop={'size': 8})
                ax[i, col].autoscale(enable=True, axis='both', tight=True)
                if add_prev_cols:
                    ax[i, col].set_xticklabels([])

            if add_prev_cols:
                # Trial history in some form - main variable to plot
                if i >= 7:
                    on_color = ['#9fff73', '#ff663c', '#a5ff89', '#63e5ff', '#ff5249', '#6b54ff']
                    off_color = ['#ff739f', '#3cffec', '#ff89a5', '#ff8f63', '#49ff6a', '#ffb554']
                    # here CS is for the initial learning period
                    prev_col_list = [
                        'prev_reward',
                        'prev_punish',
                        'prev_same_plus',
                        'prev_same_neutral',
                        'prev_same_minus',
                        'prev_blank']
                    prev_col_titles = [
                        'Prev Reward',
                        'Prev Punishment',
                        'Prev Same Cue: initial plus',
                        'Prev Same Cue: initial neutral',
                        'Prev Same Cue: initial minus',
                        'Prev Blank']
                    prev_col_labels = [
                        'rewarded [-1]',
                        'punishment [-1]',
                        'initial plus [-1]',
                        'initial neutral [-1]',
                        'initial minus [-1]',
                        'blank [-1]']
                    current_col = prev_col_list[prev_col_counter]

                    # skip column if it is not in metadata (will result
                    # in blank axes at end)
                    if current_col not in meta.columns:
                        continue

                    # boolean of trial history
                    prev_same_bool = meta[current_col].values
                    if 'plus' in current_col:
                        matched_ori = [lookups.lookup[mouse]['plus']]
                    elif 'minus' in current_col:
                        matched_ori = [lookups.lookup[mouse]['minus']]
                    elif 'neutral' in current_col:
                        matched_ori = [lookups.lookup[mouse]['neutral']]
                    elif unwrapped_ktensor is not None:
                        matched_ori = tuning_oris[comp]
                    else:
                        matched_ori = [0, 135, 270]
                    same_ori_bool = meta['orientation'].isin(matched_ori).values

                    ax[i, col].plot(
                        trial_num[~prev_same_bool & same_ori_bool],
                        U.factors[2][~prev_same_bool & same_ori_bool, comp],
                        'o',
                        label='not {}'.format(prev_col_labels[prev_col_counter]),
                        color=off_color[prev_col_counter],
                        alpha=alpha,
                        markersize=2)

                    ax[i, col].plot(
                        trial_num[prev_same_bool & same_ori_bool],
                        U.factors[2][prev_same_bool & same_ori_bool, comp],
                        'o',
                        label=prev_col_labels[prev_col_counter],
                        color=on_color[prev_col_counter],
                        alpha=alpha,
                        markersize=2)

                    ax[i, col].legend(
                        bbox_to_anchor=(1.02, 1), loc='upper left',
                        borderaxespad=0,
                        title=prev_col_titles[prev_col_counter],
                        markerscale=2,
                        prop={'size': 8})
                    ax[i, col].autoscale(enable=True, axis='both', tight=True)

                    # if i is less than the last row
                    if i < rows - 1:
                        ax[i, col].set_xticklabels([])

                    # increment counter
                    prev_col_counter += 1

            # plot days, reversal, or learning lines if there are any
            if col >= 1:
                # y_lim = ax[i, col].get_ylim()
                if len(day_x) > 0:
                    for k in day_x:
                        ax[i, col].plot(
                            [k, k], y_lim, color='#969696', linewidth=1)
                if len(lstate_x) > 0:
                    ls_vals = ['naive', 'learning', 'reversal1']
                    ls_colors = ['#66bd63', '#d73027', '#a50026']
                    for k in lstate_x:
                        ls = learning_state[int(k - 0.5)]
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

            # rescale the y-axis for trial factors if you
            if i == 0 and scale_y:
                ystd3 = np.nanstd(U.factors[2][:, comp]) * y_sc_factor
                ymax = np.nanmax(U.factors[2][:, comp])
            if scale_y:
                if ystd3 < ymax:
                    y_ticks = np.round([0, ystd3 / 2, ystd3], 2)
                    ax[i, 2].set_ylim([0, ystd3])
                    ax[i, 2].set_yticks(y_ticks)
                    ax[i, 2].set_yticklabels(y_ticks)
                else:
                    y_ticks = np.round([0, ymax / 2, ymax], 2)
                    ax[i, 2].set_ylim([0, ymax])
                    ax[i, 2].set_yticks(y_ticks)
                    ax[i, 2].set_yticklabels(y_ticks)

        # save
        if filetype.lower() == 'pdf':
            suffix = '.pdf'
        elif filetype.lower() == 'eps':
            suffix = '.eps'
        else:
            suffix = '.png'
        save_path = paths.analysis_file(
            f'component_{comp + 1}_rank_{r}_mouse_{mouse}_{mod}{suffix}',
            f'tca_dfs/TCA_factor_fitting{folder_tag}/{mod}/factors_longform/{mouse}')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close('all')


def _sortcellfactor(ktensor):
    """
    Sort your cell factors then return a resorted ktensor. 
    """

    # use the lowest index (and lowest error objective) to create sort order
    factors = deepcopy(ktensor[0])

    # sort neuron factors according to which component had highest weight
    max_fac = np.argmax(factors, axis=1)
    sort_fac = np.argsort(max_fac)
    sort_max_fac = max_fac[sort_fac]
    first_sort = factors[sort_fac, :]

    # descending sort within each group of sorted neurons
    second_sort = []
    for i in np.unique(max_fac):
        second_inds = (np.where(sort_max_fac == i)[0])
        second_sub_sort = np.argsort(first_sort[sort_max_fac == i, i])
        second_sort.extend(second_inds[second_sub_sort][::-1])

    # apply the second sort
    full_sort = sort_fac[second_sort]
    sorted_factors = factors[full_sort, :]

    # check for zero-weight factors
    no_weight_binary = np.max(sorted_factors, axis=1) == 0
    no_weight_binary = np.max(sorted_factors, axis=1) == 0
    inds_to_end = full_sort[no_weight_binary]
    full_sort = np.concatenate((full_sort[np.invert(no_weight_binary)], inds_to_end), axis=0)

    sorted_kt = KTensor([factors[full_sort, :], ktensor[1], ktensor[2]])

    return sorted_kt, full_sort
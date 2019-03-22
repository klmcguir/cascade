"""Functions for plotting stitched tca decomp."""
import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from .. import df
from .. import stitch as stch
from .. import paths


def plot_matched_factors(
        mouse,
        rank_num=10,
        match_by='tri_sim',
        trace_type='zscore_day',
        method='ncp_bcd',
        cs='',
        warp=False,
        word=None,
        plot_sim_mats=True):
    """
    Wrapper function for stitch.neuron_similarity or
    stitch.tri_factor_similarity. Plot your best matched neuron
    factor weight correlation transitions across time as well as
    your best matched temporal factors.

    Parameters
    ----------

    """

    # pars for loading tca data
    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}

    # run neuron factor stitching
    # {'trans': transition_weights,
    #  'tempo_fac': temporal_factors_list,
    #  'neuro_sim': sim_mat_by_day,
    #  'tempo_sim': sim_mat_tempo_by_day}
    if match_by == 'tri_sim' or match_by == 'tri_sim_prob':
        out = stch.tri_factor_similarity(
            mouse, rank_num=rank_num, match_by=match_by,
            trace_type=trace_type, method=method,
            cs=cs, warp=warp, word=word)
    else:
        out = stch.neuron_similarity(
            mouse, rank_num=rank_num, match_by=match_by,
            trace_type=trace_type, method=method,
            cs=cs, warp=warp, word=word)
    npdays = np.shape(out['trans'])[1]
    ndays = np.shape(out['tempo_fac'][0])[0]

    # set up saving dir
    save_dir = paths.tca_plots(mouse, 'single', pars=pars, word=word)
    save_dir = os.path.join(save_dir, 'factor stitching')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    save_dir2 = os.path.join(save_dir, 'matched temporal factors')
    if not os.path.isdir(save_dir2): os.mkdir(save_dir2)

    # plot your correlation corr matrix of best matched factors
    plt.figure(figsize=(9, 4))
    plt.imshow(out['trans'], aspect='auto', vmin=0, vmax=1)
    plt.ylabel('component #')
    plt.xlabel('day pair #')
    plt.xticks(range(0, npdays), range(1, npdays+1))
    plt.yticks(range(0, rank_num), range(1, rank_num+1))
    plt.title('Best Matched Ensemble Transitions')
    plt.colorbar(label='correlation coefficient (R)')

    # save
    file_name = mouse + ' best ensemble transitions' + ' -' + match_by + '.pdf'
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path)
    plt.close()

    for z in range(rank_num):
        plt.figure()
        plt.imshow(out['tempo_fac'][z], aspect='auto')
        plt.title('Temporal Factors for Matched Ensembles, component ' + str(z+1))
        plt.ylabel('day #')
        plt.xlabel('time from stimulus onset (sec)')
        plt.xticks(list(np.arange(15.5, 15.5*7, 15.5)), range(0, 6))
        plt.yticks(range(0, ndays), range(1, ndays+1))
        y_lim = plt.ylim()
        ons = 15.5*1
        offs = ons+15.5*3
        plt.plot([ons, ons], y_lim, ':r')
        plt.plot([offs, offs], y_lim, ':r')
        file_name = mouse + ' stitched tempo component ' + str(z+1) + ' -' + match_by + '.pdf'
        save_path = os.path.join(save_dir2, file_name)
        plt.savefig(save_path)
        plt.close()

    # plot sim matrices without rerunning matching algo
    if plot_sim_mats:
        _plot_neuro_similarity_matrices(out, mouse, rank_num,
                                        match_by, pars, word)
        _plot_tempo_similarity_matrices(out, mouse, rank_num,
                                        match_by, pars, word)
        _plot_tuning_similarity_matrices(out, mouse, rank_num,
                                         match_by, pars, word)
        if match_by == 'tri_sim' or match_by == 'tri_sim_prob':
            _plot_tri_similarity_matrices(out, mouse, rank_num,
                                          match_by, pars, word)


def plot_neuro_similarity_matrices(
        mouse,
        rank_num=10,
        match_by='similarity',
        trace_type='zscore_day',
        method='ncp_bcd',
        cs='',
        warp=False,
        word=None):

    """
    Wrapper function for stitch.neuron_factors. Plot your
    similarity matrices (neuron factor weight correlations)
    for pairs of days.

    Parameters
    ----------

    """

    # pars for loading tca data
    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}

    # run neuron factor stitching
    # {'trans': transition_weights,
    # 'tempo_fac': temporal_factors_list,
    # 'neuro_sim': sim_mat_neuro_by_day,
    # 'tempo_sim': sim_mat_tempo_by_day,
    # 'ttuning_sim': sim_mat_trial_tuning_by_day,
    # 'tri_sim': match_mat}
    if match_by == 'tri_sim' or match_by == 'tri_sim_prob':
        out = stch.tri_factor_similarity(
            mouse, rank_num=rank_num, match_by=match_by,
            trace_type=trace_type, method=method,
            cs=cs, warp=warp, word=word)
    else:
        out = stch.neuron_similarity(
            mouse, rank_num=rank_num, match_by=match_by,
            trace_type=trace_type, method=method,
            cs=cs, warp=warp, word=word)

    # set up saving dir
    save_dir = paths.tca_plots(mouse, 'single', pars=pars, word=word)
    save_dir = os.path.join(save_dir, 'factor stitching')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, 'similarity matrices')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)

    # plot
    for i in range(len(out['neuro_sim'])):
        plt.figure(figsize=(5, 4))
        plt.imshow(out['neuro_sim'][i], aspect='auto',
                   vmin=0, vmax=1)
        plt.ylabel('day ' + str(i+2) + ' component #')
        plt.xlabel('day ' + str(i+1) + ' component #')
        plt.xticks(range(0, rank_num), range(1, rank_num+1))
        plt.yticks(range(0, rank_num), range(1, rank_num+1))
        plt.title('Neuron Factor Similarity Matrix')
        plt.colorbar(label='correlation coefficient (R)')

        file_name = mouse +' neuro similarity day ' + str(i+1) + ' day ' + str(i+2) + ' -' + match_by + '.pdf'
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path)
        plt.close()


def plot_tempo_similarity_matrices(
        mouse,
        rank_num=10,
        match_by='similarity',
        trace_type='zscore_day',
        method='ncp_bcd',
        cs='',
        warp=False,
        word=None):

    """
    Wrapper function for stitch.neuron_factors. Plot your
    similarity matrices (neuron factor weight correlations)
    for pairs of days.

    Parameters
    ----------

    """

    # pars for loading tca data
    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}

    # run neuron factor stitching
    # {'trans': transition_weights,
    # 'tempo_fac': temporal_factors_list,
    # 'neuro_sim': sim_mat_neuro_by_day,
    # 'tempo_sim': sim_mat_tempo_by_day,
    # 'ttuning_sim': sim_mat_trial_tuning_by_day,
    # 'tri_sim': match_mat}
    if match_by == 'tri_sim' or match_by == 'tri_sim_prob':
        out = stch.tri_factor_similarity(
            mouse, rank_num=rank_num, match_by=match_by,
            trace_type=trace_type, method=method,
            cs=cs, warp=warp, word=word)
    else:
        out = stch.neuron_similarity(
            mouse, rank_num=rank_num, match_by=match_by,
            trace_type=trace_type, method=method,
            cs=cs, warp=warp, word=word)

    # set up saving dir
    save_dir = paths.tca_plots(mouse, 'single', pars=pars, word=word)
    save_dir = os.path.join(save_dir, 'factor stitching')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, 'similarity matrices')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)

    # plot
    for i in range(len(out['tempo_sim'])):
        plt.figure(figsize=(5, 4))
        plt.imshow(out['tempo_sim'][i], aspect='auto',
                   vmin=0, vmax=1)
        plt.ylabel('day ' + str(i+2) + ' component #')
        plt.xlabel('day ' + str(i+1) + ' component #')
        plt.xticks(range(0, rank_num), range(1, rank_num+1))
        plt.yticks(range(0, rank_num), range(1, rank_num+1))
        plt.title('Temporal Factor Similarity Matrix')
        plt.colorbar(label='correlation coefficient (R)')

        file_name = mouse + ' tempo similarity day ' + str(i+1) + ' day ' + str(i+2) + ' -' + match_by + '.pdf'
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path)
        plt.close()


def plot_tuning_similarity_matrices(
        mouse,
        rank_num=10,
        match_by='similarity',
        trace_type='zscore_day',
        method='ncp_bcd',
        cs='',
        warp=False,
        word=None):

    """
    Wrapper function for stitch.neuron_similarity. Plot your
    similarity matrices (neuron factor weight correlations)
    for pairs of days.

    Parameters
    ----------

    """

    # pars for loading tca data
    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}

    # run neuron factor stitching
    # {'trans': transition_weights,
    # 'tempo_fac': temporal_factors_list,
    # 'neuro_sim': sim_mat_neuro_by_day,
    # 'tempo_sim': sim_mat_tempo_by_day,
    # 'ttuning_sim': sim_mat_trial_tuning_by_day,
    # 'tri_sim': match_mat}
    if match_by == 'tri_sim' or match_by == 'tri_sim_prob':
        out = stch.tri_factor_similarity(
            mouse, rank_num=rank_num, match_by=match_by,
            trace_type=trace_type, method=method,
            cs=cs, warp=warp, word=word)
    else:
        out = stch.neuron_similarity(
            mouse, rank_num=rank_num, match_by=match_by,
            trace_type=trace_type, method=method,
            cs=cs, warp=warp, word=word)

    # set up saving dir
    save_dir = paths.tca_plots(mouse, 'single', pars=pars, word=word)
    save_dir = os.path.join(save_dir, 'factor stitching')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, 'similarity matrices')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)

    # plot
    for i in range(len(out['ttuning_sim'])):
        plt.figure(figsize=(5, 4))
        plt.imshow(out['ttuning_sim'][i], aspect='auto',
                   vmin=0, vmax=1)
        plt.ylabel('day ' + str(i+2) + ' component #')
        plt.xlabel('day ' + str(i+1) + ' component #')
        plt.xticks(range(0, rank_num), range(1, rank_num+1))
        plt.yticks(range(0, rank_num), range(1, rank_num+1))
        plt.title('Trial Factor Tuning Similarity Matrix')
        plt.colorbar(label='correlation coefficient (R)')

        file_name = mouse + ' tuning similarity day ' + str(i+1) + ' day ' + str(i+2) + ' -' + match_by + '.pdf'
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path)
        plt.close()


def plot_tri_similarity_matrices(
        mouse,
        rank_num=10,
        match_by='tri_sim',
        trace_type='zscore_day',
        method='ncp_bcd',
        cs='',
        warp=False,
        word=None):

    """
    Wrapper function for stitch.neuron_similarity. Plot your
    similarity matrices (neuron factor weight correlations)
    for pairs of days.

    Parameters
    ----------

    """

    # pars for loading tca data
    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}

    # run neuron factor stitching
    # {'trans': transition_weights,
    # 'tempo_fac': temporal_factors_list,
    # 'neuro_sim': sim_mat_neuro_by_day,
    # 'tempo_sim': sim_mat_tempo_by_day,
    # 'ttuning_sim': sim_mat_trial_tuning_by_day,
    # 'tri_sim': match_mat}
    out = stch.tri_factor_similarity(
        mouse, rank_num=rank_num, match_by=match_by,
        trace_type=trace_type, method=method,
        cs=cs, warp=warp, word=word)

    # set up saving dir
    save_dir = paths.tca_plots(mouse, 'single', pars=pars, word=word)
    save_dir = os.path.join(save_dir, 'factor stitching')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, 'similarity matrices')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)

    # set tag for distinguishing match_by with prob
    if match_by == 'tri_sim_prob':
        title_tag = ' + prob'
    else:
        title_tag = ''

    # plot
    for i in range(len(out['tri_sim'])):
        plt.figure(figsize=(5, 4))
        plt.imshow(out['tri_sim'][i], aspect='auto',
                   vmin=0, vmax=1)
        plt.ylabel('day ' + str(i+2) + ' component #')
        plt.xlabel('day ' + str(i+1) + ' component #')
        plt.xticks(range(0, rank_num), range(1, rank_num+1))
        plt.yticks(range(0, rank_num), range(1, rank_num+1))
        plt.title('Combined Similarity Matrix' + title_tag)
        plt.colorbar(label='correlation coefficient (R)')

        file_name = mouse + ' tuning similarity day ' + str(i+1) + ' day ' + str(i+2) + ' -' + match_by + '.pdf'
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path)
        plt.close()


# --------------  INTERNAL PLOTTING FUNCTIONS  --------------


def _plot_neuro_similarity_matrices(
        out,
        mouse,
        rank_num,
        match_by,
        pars,
        word):

    """
    Takes input from stitch.neuron_factors. Plot your
    similarity matrices (neuron factor weight correlations)
    for pairs of days.

    Parameters
    ----------

    """

    # set up saving dir
    save_dir = paths.tca_plots(mouse, 'single', pars=pars, word=word)
    save_dir = os.path.join(save_dir, 'factor stitching')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, 'similarity matrices')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)

    # plot
    for i in range(len(out['neuro_sim'])):
        plt.figure(figsize=(5, 4))
        plt.imshow(out['neuro_sim'][i], aspect='auto',
                   vmin=0, vmax=1)
        plt.ylabel('day ' + str(i+2) + ' component #')
        plt.xlabel('day ' + str(i+1) + ' component #')
        plt.xticks(range(0, rank_num), range(1, rank_num+1))
        plt.yticks(range(0, rank_num), range(1, rank_num+1))
        plt.title('Neuron Factor Similarity Matrix')
        plt.colorbar(label='correlation coefficient (R)')

        file_name = mouse +' neuro similarity day ' + str(i+1) + ' day ' + str(i+2) + ' -' + match_by + '.pdf'
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path)
        plt.close()


def _plot_tempo_similarity_matrices(
        out,
        mouse,
        rank_num,
        match_by,
        pars,
        word):

    """
    Takes input from stitch.neuron_similarity. Plot your
    similarity matrices (neuron factor weight correlations)
    for pairs of days.

    Parameters
    ----------

    """

    # set up saving dir
    save_dir = paths.tca_plots(mouse, 'single', pars=pars, word=word)
    save_dir = os.path.join(save_dir, 'factor stitching')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, 'similarity matrices')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)

    # plot
    for i in range(len(out['tempo_sim'])):
        plt.figure(figsize=(5, 4))
        plt.imshow(out['tempo_sim'][i], aspect='auto',
                   vmin=0, vmax=1)
        plt.ylabel('day ' + str(i+2) + ' component #')
        plt.xlabel('day ' + str(i+1) + ' component #')
        plt.xticks(range(0, rank_num), range(1, rank_num+1))
        plt.yticks(range(0, rank_num), range(1, rank_num+1))
        plt.title('Temporal Factor Similarity Matrix')
        plt.colorbar(label='correlation coefficient (R)')

        file_name = mouse + ' tempo similarity day ' + str(i+1) + ' day ' + str(i+2) + ' -' + match_by + '.pdf'
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path)
        plt.close()


def _plot_tuning_similarity_matrices(
        out,
        mouse,
        rank_num,
        match_by,
        pars,
        word):

    """
    Wrapper function for stitch.neuron_similarity. Plot your
    similarity matrices (neuron factor weight correlations)
    for pairs of days.

    Parameters
    ----------

    """

    # set up saving dir
    save_dir = paths.tca_plots(mouse, 'single', pars=pars, word=word)
    save_dir = os.path.join(save_dir, 'factor stitching')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, 'similarity matrices')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)

    # plot
    for i in range(len(out['ttuning_sim'])):
        plt.figure(figsize=(5, 4))
        plt.imshow(out['ttuning_sim'][i], aspect='auto',
                   vmin=0, vmax=1)
        plt.ylabel('day ' + str(i+2) + ' component #')
        plt.xlabel('day ' + str(i+1) + ' component #')
        plt.xticks(range(0, rank_num), range(1, rank_num+1))
        plt.yticks(range(0, rank_num), range(1, rank_num+1))
        plt.title('Trial Factor Tuning Similarity Matrix')
        plt.colorbar(label='correlation coefficient (R)')

        file_name = mouse + ' tuning similarity day ' + str(i+1) + ' day ' + str(i+2) + ' -' + match_by + '.pdf'
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path)
        plt.close()


def _plot_tri_similarity_matrices(
        out,
        mouse,
        rank_num,
        match_by,
        pars,
        word):

    """
    Wrapper function for stitch.neuron_similarity. Plot your
    similarity matrices (neuron factor weight correlations)
    for pairs of days.

    Parameters
    ----------

    """

    # set up saving dir
    save_dir = paths.tca_plots(mouse, 'single', pars=pars, word=word)
    save_dir = os.path.join(save_dir, 'factor stitching')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, 'similarity matrices')
    if not os.path.isdir(save_dir): os.mkdir(save_dir)

    # set tag for distinguishing match_by with prob
    if match_by == 'tri_sim_prob':
        title_tag = ' + prob'
    else:
        title_tag = ''

    # plot
    for i in range(len(out['tri_sim'])):
        plt.figure(figsize=(5, 4))
        plt.imshow(out['tri_sim'][i], aspect='auto',
                   vmin=0, vmax=1)
        plt.ylabel('day ' + str(i+2) + ' component #')
        plt.xlabel('day ' + str(i+1) + ' component #')
        plt.xticks(range(0, rank_num), range(1, rank_num+1))
        plt.yticks(range(0, rank_num), range(1, rank_num+1))
        plt.title('Combined Similarity Matrix' + title_tag)
        plt.colorbar(label='correlation coefficient (R)')

        file_name = mouse + ' combined similarity day ' + str(i+1) + ' day ' + str(i+2) + ' -' + match_by + '.pdf'
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path)
        plt.close()

"""Calculations to be saved to mongoDB database"""
from pool.database import memoize
import numpy as np
import flow
import pandas as pd
from .. import load


# @memoize(across='mouse', updated=191002, returns='other', large_output=False)
def center_of_mass_tempofac(
        mouse,
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        word='restaurant',
        group_by='all',
        nan_thresh=0.85,
        score_threshold=0.8,
        rank_num=15):
    """
    Load in a model of your temporal factors and calculate their center
    of mass.
    """
    # load your data
    load_kwargs = {'mouse': mouse.mouse,
                   'method': method,
                   'cs': cs,
                   'warp': warp,
                   'word': word,
                   'group_by': group_by,
                   'nan_thresh': nan_thresh,
                   'score_threshold': score_threshold}
    V, _, _ = load.groupday_tca_model(**load_kwargs, full_output=True)

    # get array of temporal factors
    tr = V.results[rank_num][0].factors[1][:, :].T

    # create array of the total number of time points
    pos = np.arange(1, tr.shape[1]+1)

    # calculate center of mass
    center_of_mass = []
    factors = []
    data = {}
    for i in range(tr.shape[0]):
        center_of_mass.append(np.sum(tr[i, :] * pos)/np.sum(tr[i, :]))
        factors.append(int(i+1))

    # put center of mass into dataframe
    data = {'center_of_mass': center_of_mass, 'factor': factors}
    cm_df = pd.DataFrame(data=data)

    return cm_df


def is_center_of_mass_visual(
        mouse,
        trace_type='zscore_day',
        method='mncp_hals',
        cs='',
        warp=False,
        word='restaurant',
        group_by='all',
        nan_thresh=0.85,
        score_threshold=0.8,
        rank_num=15):
    """
    Check if the center of mass of your temporal factors is beyond the
    offset of the visual stimulus.
    """

    # load your data
    cm_kwargs = {'mouse': flow.Mouse(mouse=mouse),
                 'method': method,
                 'cs': cs,
                 'warp': warp,
                 'word': word,
                 'group_by': group_by,
                 'nan_thresh': nan_thresh,
                 'score_threshold': score_threshold,
                 'rank_num': rank_num}
    cm_df = center_of_mass_tempofac(**cm_kwargs)

    # set the stimulus offset time (stimulus length + baseline length)
    if mouse in ['OA32', 'OA34', 'OA37', 'OA36', 'CB173', 'AS20', 'AS41']:
        off_time = 2 + 1  # add 1 for the second before stimulus onset
    else:
        off_time = 3 + 1

    return cm_df['center_of_mass'].values <= 15*off_time

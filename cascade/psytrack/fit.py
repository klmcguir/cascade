"""Wrapper functions for running psytrack behavioral model."""
import flow
from .. import lookups

"""
Common hyperparameters for fitting psytrack model from simple go/nogo task
I use. 
"""
pars_orig = {'weights': {
        'bias': 1,
        'ori_0': 1,
        'ori_135': 1,
        'ori_270': 1,
        'prev_choice': 1,
        'prev_reward': 1,
        'prev_punish': 1}
        }

pars_prev = {'weights': {
        'bias': 1,
        'ori_0': 1,
        'ori_135': 1,
        'ori_270': 1,
        'prev_0': 1,
        'prev_135': 1,
        'prev_270': 1,
        'prev_choice': 1,
        'prev_reward': 1,
        'prev_punish': 1}
        }

pars_simp_th = {'weights': {
        'bias': 1,
        'ori_same_0': 1,
        'ori_same_135': 1,
        'ori_same_270': 1,
        'ori_other_0': 1,
        'ori_other_135': 1,
        'ori_other_270': 1,
        'prev_choice': 1,
        'prev_reward': 1,
        'prev_punish': 1}
        }

th_pars = {'weights': {
        'bias': 1,
        'ori_0_0': 1,
        'ori_135_0': 1,
        'ori_270_0': 1,
        'ori_0_135': 1,
        'ori_135_135': 1,
        'ori_270_135': 1,
        'ori_0_270': 1,
        'ori_135_270': 1,
        'ori_270_270': 1,
        'prev_choice': 1,
        'prev_reward': 1,
        'prev_punish': 1}
        }

"""
Functions 
"""
def run_psytrack(mouse, pars=pars_simp_th, include_pavlovian=False):
    """
    Simple function for running psytrack behavioral model on a single mouse
    """

    # upadate parameters to reflect pavlovian kwarg
    pars['include_pavlovian'] = include_pavlovian

    psy = flow.Mouse(mouse=mi).psytracker(verbose=True, force=force, pars=pars)

    return psy


def batch_run_psytrack(
        mice=lookups.mice['all15'],
        pars=pars_simp_th,
        include_pavlovian=False
        ):
    """
    Function for running psytrack behavioral model on a group of mouse
    """

    # loop over mice fitting behavioral model
    psy_list = []
    for mi in mice:
        psy = run_psytrack(mi, pars=pars, include_pavlovian=include_pavlovian)
        psy_list.append(psy)

    return psy_list

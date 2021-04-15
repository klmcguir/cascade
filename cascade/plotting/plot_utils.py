import numpy as np
from .. import lookups


def heatmap_xticks(staging='parsed_11stage_label',
                   additional_pt=1,
                   additional_text=True,
                   drop_stage_text=False,
                   drop_naive=True):

    # get the labels for your heatmap stages
    stages = lookups.staging[staging]
    if drop_naive:
        stages = stages[1:]

    # first create 0s pt with a stage label
    stim_starts = [15.5 + 47 * s for s in np.arange(len(stages))]
    if drop_stage_text:
        stim_labels = ['0' for _, _ in enumerate(stages)]
    else:
        stim_labels = [f'0\n\n{s}' if c % 2 == 0 else f'0\n{s}' for c, s in enumerate(stages)]

    # create a 1s pt alone
    if additional_pt is not None:
        if additional_pt == 1:
            stim_1s = [31 + 47 * s for s in np.arange(len(stages))]
            if additional_text:
                stim_1_labels = ['1' for _ in np.arange(len(stages))]
            else:
                stim_1_labels = ['' for _ in np.arange(len(stages))]
        elif additional_pt == 2:
            stim_1s = [46.5 + 47 * s for s in np.arange(len(stages))]
            if additional_text:
                stim_1_labels = ['2' for _ in np.arange(len(stages))]
            else:
                stim_1_labels = ['' for _ in np.arange(len(stages))]

        # zip and flatten into a single set of xticks and xtick labels
        xticks = list(sum(zip(stim_starts, stim_1s), ()))
        xticklabels = list(sum(zip(stim_labels, stim_1_labels), ()))
    else:
        xticks = stim_starts
        xticklabels = stim_labels

    return xticks, xticklabels


def choose_negative_modes(sort_ensemble, negative_modes=[]):
    """
    Helper function to make sure that any negativity in factors is expressed in the desired mode.

    :param sort_ensemble: method object from tensortools
    :param negative_modes: list of int
        factor modes that are permitted to be negative
    :return: copy of object with any negativity present expressed only in correct modes
    """

    # 3 modes is all that you should practically run, so doesn't make sense for this function
    assert len(negative_modes) < 3
    if len(negative_modes) == 0:
        return sort_ensemble

    for r in sort_ensemble.results:

        # check which modes have negatives
        current_negatives = []
        for fac_mode in range(3):
            neg_modes = np.sum(sort_ensemble.results[r][0].factors[fac_mode] < 0, axis=0) > 0
            current_negatives.append(neg_modes)
        neg_map = np.vstack(current_negatives)
        assert neg_map.shape[0] == 3 and neg_map.shape[1] == r

        # loop over components, identify improper negatives, flip
        not_allowed = np.array([int(s) for s in range(3) if s not in negative_modes])
        for c in range(r):

            # check for negatives where there should not be
            if np.any(neg_map[not_allowed, c]):

                # check an see if the modes allowed to negative are already negative
                if np.any(neg_map[np.array(negative_modes), c]) and len(negative_modes) > 1:
                    print('Right now this does not handle multiple negative modes')
                    raise NotImplementedError
                else:
                    # flip the approved negative mode with the unapproved negative mode
                    bad_mode = np.where(neg_map[:, c])[0]
                    for b in bad_mode:
                        sort_ensemble.results[r][0].factors[b][:, c] = \
                            sort_ensemble.results[r][0].factors[b][:, c] * -1
                        sort_ensemble.results[r][0].factors[negative_modes[0]][:, c] = \
                            sort_ensemble.results[r][0].factors[negative_modes[0]][:, c] * -1

    return sort_ensemble

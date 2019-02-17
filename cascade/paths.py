"""Functions for tca paths."""
import os
import json
import flow


def tca_path(mouse, grouping, pars=None, word=None):
    """
    Create directory for TCA. Hash TCA parameters and save
    into TCA directory as a .txt file.

    Parameters
    ----------
    mouse : str
        Mouse.
    grouping : str
        Grouping of TCA being run. i.e., 'pair' or 'single'
        day TCA.
    pars : dict
        Dict of all parameters for TCA.
    word : str
        Word created by hashing pars. For loading pars can be
        truncated to include only cs, warp, trace_type, AND
        the pars word entered here.

    Returns
    -------
    save_dir : str
        Path of TCA with specific parameters.
    """
    # check inputs
    if 'single' != grouping and 'pair' != grouping:
        print('Unacceptable grouping: try: single or pair.')
        return

    # get word
    if pars and not word:
        pars_word = flow.misc.wordhash.word(pars)
        print('TCA parameters hashed: ' + pars_word)
    elif pars and word:
        pars_word = word
    else:
        print('ERROR: Neither pars or word were passed to tca_path.')
        return

    # create folder structure and save dir
    cs_tag = '' if len(pars['cs']) == 0 else '-' + str(pars['cs'])
    warp_tag = '' if pars['warp'] is False else '-warp'
    trace_tag = '-' + pars['trace_type']
    pars_tag = '-' + pars_word
    tca_tag = 'tensors-' + grouping
    folder_name = tca_tag + trace_tag + cs_tag + warp_tag + pars_tag
    save_dir = os.path.join(flow.paths.outd, mouse)
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, folder_name)
    if not os.path.isdir(save_dir): os.mkdir(save_dir)

    # save your parameters into a text file
    pars_path = os.path.join(save_dir, 'tca_params.txt')
    if pars and not word:
        with open(pars_path, 'w') as file:
            file.write(json.dumps(pars))

    return save_dir


def tca_plots(mouse, grouping, pars=None, word=None):
    """
    Create directory for TCA plots. Hash TCA parameters and save
    into TCA directory as a .txt file.

    Parameters
    ----------
    mouse : str
        Mouse.
    grouping : str
        Grouping of TCA being run. i.e., 'pair' or 'single'
        day TCA.
    pars : dict
        Dict of truncated parameters for TCA.
    word : str
        Word created by hashing pars. For loading pars can be
        truncated to include only cs, warp, trace_type, AND
        the pars word entered here.

    Returns
    -------
    save_dir : str
        Path of TCA with specific parameters.
    """
    # check inputs
    if 'single' != grouping and 'pair' != grouping:
        print('Unacceptable grouping: try: single or pair.')
        return

    # get word
    if pars and word:
        pars_word = word
    else:
        print('ERROR: Word was not passed to tca_plots.')
        return

    # create folder structure and save dir
    cs_tag = '' if len(pars['cs']) == 0 else '-' + str(pars['cs'])
    warp_tag = '' if pars['warp'] is False else '-warp'
    trace_tag = '-' + pars['trace_type']
    pars_tag = '-' + pars_word
    tca_tag = 'tensors-' + grouping
    folder_name = tca_tag + trace_tag + cs_tag + warp_tag + pars_tag
    save_dir = os.path.join(flow.paths.graphd, mouse)
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, folder_name)
    if not os.path.isdir(save_dir): os.mkdir(save_dir)

    return save_dir

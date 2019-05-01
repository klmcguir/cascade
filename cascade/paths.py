"""Functions for tca paths."""
import os
import json
import flow


def groupmouse_word(mouse_dict):
    """
    Hash a dictionary of mouse names into a single identifying word.

    Parameters
    ----------
    mouse_dict : dict
        dictionary where key is mice and values are list of mouse names

    Returns
    -------
    word : str
        hash word
    """
    word = flow.misc.wordhash.word(mouse_dict)
    print('Mice hashed: ' + word)

    return word


def tca_path(mouse, grouping, pars=None, word=None, group_pars=None):
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
    if ('single' != grouping and
        'pair' != grouping and
        'group' != grouping):
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

    # check for grouping params
    if group_pars:
        group_tag = '' if group_pars['group_by'] is None else '-' + group_pars['group_by']
    else:
        group_tag = ''

    # create folder structure and save dir
    cs_tag = '' if len(pars['cs']) == 0 else '-' + str(pars['cs'])
    warp_tag = '' if pars['warp'] is False else '-warp'
    trace_tag = '-' + pars['trace_type']
    pars_tag = '-' + pars_word
    tca_tag = 'tensors-' + grouping
    folder_name = tca_tag + trace_tag + cs_tag + warp_tag + group_tag + pars_tag
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


def tca_plots(mouse, grouping, pars=None, word=None, group_pars=None):
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
    if ('single' != grouping and
        'pair' != grouping and
        'group' != grouping):
        print('Unacceptable grouping: try: single or pair.')
        return

    # get word
    if pars and word:
        pars_word = word
    else:
        print('ERROR: Word was not passed to tca_plots.')
        return

    # check for grouping params
    if group_pars:
        group_tag = '' if group_pars['group_by'] is None else '-' + group_pars['group_by']
    else:
        group_tag = ''

    # create folder structure and save dir
    cs_tag = '' if len(pars['cs']) == 0 else '-' + str(pars['cs'])
    warp_tag = '' if pars['warp'] is False else '-warp'
    trace_tag = '-' + pars['trace_type']
    pars_tag = '-' + pars_word
    tca_tag = 'tensors-' + grouping
    folder_name = tca_tag + trace_tag + cs_tag + warp_tag + group_tag + pars_tag
    save_dir = os.path.join(flow.paths.graphd, mouse)
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, folder_name)
    if not os.path.isdir(save_dir): os.mkdir(save_dir)

    return save_dir


def df_path(mouse, pars=None, word=None):
    """
    Create directory for dataframes. Hash trace parameters and save
    into directory as a .txt file.

    Parameters
    ----------
    mouse : str
        Mouse.
    pars : dict
        Dict of all parameters for TCA.
    word : str
        Word created by hashing pars. For loading pars can be
        truncated to include only cs, warp, trace_type, AND
        the pars word entered here.

    Returns
    -------
    save_dir : str
        Path of df with specific parameters.
    """

    # get word
    if pars and not word:
        pars_word = flow.misc.wordhash.word(pars)
        print('Trace parameters hashed for df: ' + pars_word)
    elif pars and word:
        pars_word = word
    else:
        print('ERROR: Neither pars or word were passed to df_path.')
        return

    # create folder structure and save dir
    trace_tag = '-' + pars['trace_type']
    pars_tag = '-' + pars_word
    folder_name = 'dfs' + trace_tag + pars_tag
    save_dir = os.path.join(flow.paths.outd, mouse)
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, folder_name)
    if not os.path.isdir(save_dir): os.mkdir(save_dir)

    # save your parameters into a text file
    pars_path = os.path.join(save_dir, 'df_trace_params.txt')
    if pars and not word:
        with open(pars_path, 'w') as file:
            file.write(json.dumps(pars))

    return save_dir


def df_plots(mouse, pars=None, word=None, plot_type='heatmap'):
    """
    Create directory for plotting xday dfs. Hash trace parameters and save
    into directory as a .txt file.

    Parameters
    ----------
    mouse : str
        Mouse.
    pars : dict
        Dict of all parameters for TCA.
    word : str
        Word created by hashing pars. For loading pars can be
        truncated to include only cs, warp, trace_type, AND
        the pars word entered here.

    Returns
    -------
    save_dir : str
        Path of df with specific parameters.
    """

    # get word
    if pars and not word:
        pars_word = flow.misc.wordhash.word(pars)
        print('Trace parameters hashed for df: ' + pars_word)
    elif pars and word:
        pars_word = word
    else:
        print('ERROR: Neither pars or word were passed to df_path.')
        return

    # create folder structure and save dir
    trace_tag = '-' + pars['trace_type']
    pars_tag = '-' + pars_word
    if plot_type.lower() == 'heatmap':
        folder_name = 'heatmaps' + trace_tag + pars_tag
    elif plot_type.lower() == 'traces':
        folder_name = 'traces' + trace_tag + pars_tag
    save_dir = os.path.join(flow.paths.graphd, mouse)
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, folder_name)
    if not os.path.isdir(save_dir): os.mkdir(save_dir)

    return save_dir

"""Useful lookup dicts for general calculations and plotting ease."""

# table of colors I like
color_dict = {
    'plus': '#75d978',
    'minus': '#d61f21',
    'neutral': '#47aded',
    'initial_plus': '#75d978',
    'initial_minus': '#d61f21',
    'initial_neutral': '#47aded',
    'learning': '#226e36',
    'reversal': '#ad261a',
    'none': '#e8d4d2',  # '#f9f9f9',
    'gray': '#a3a3a3',
    'plus1': '#cae85e',
    'plus2': '#61e6a4',
    'plus3': '#85eb60',
    'minus1': '#ff5e79',
    'minus2': '#ffa561',
    'minus3': '#ff7b60',
    'neutral1': '#cfc3ff',
    'neutral2': '#db77ff',
    'neutral3': '#9476ff',
    'dprime': '#7c7ce4',
    'dprime1': '#e47ce4',
    'plus-neutral': '#6ACE95',
    'plus-minus': '#8DAB62',
    'neutral-plus': '#53B8D0',
    'neutral-minus': '#6B8ABA',
    'minus-plus': '#BE4E37',
    'minus-neutral': '#B24354',
    'broad': '#ccc2ba',
    'broad3': 'black',
    'joint': '#dcd0ff',
    '0': '#56fff9',
    '0-135': '#8EF0C3',
    '135-0': '#C7E18C',
    '135': '#ffd256',
    '135-270': '#FFA973',
    '270-135': '#FF7F90',
    '270': '#ff56ad',
    '270-0': '#C78EC6',
    '0-270': '#8EC7E0',
    'becomes_unrewarded': '#27ffa9',
    'becomes_rewarded': '#ff27be',
    'remains_unrewarded': '#ffb327'
}

# common hardcoded save locations I don't want to type
saveroot = '/twophoton_analysis/Data/analysis/Group-attractive/'
coreroot = '/twophoton_analysis/Data/analysis/core_dfs/'

# get ordering used for categorical variables in plots
ax_order = {
    'cues': ['broad', 'plus-minus', 'plus', 'plus-neutral', 'neutral-plus',
             'neutral', 'neutral-minus', 'minus-neutral', 'minus', 'minus-plus'],
    'oris': ['0', '0-135', '135-0', '135', '135-270', '270-135', '270', '270-0', '0-270'],
}

# lookup table of conditions during initial learning matched to orientations
lookup = {'OA27': {'plus': 270, 'minus': 135, 'neutral': 0, 'blank': -1},
          'PD27': {'plus': 270, 'minus': 135, 'neutral': 0, 'blank': -1},
          'VF226': {'plus': 0, 'minus': 270, 'neutral': 135, 'blank': -1},
          'LM226': {'plus': 0, 'minus': 270, 'neutral': 135, 'blank': -1},
          'PD226': {'plus': 0, 'minus': 270, 'neutral': 135, 'blank': -1},
          'OA67': {'plus': 0, 'minus': 270, 'neutral': 135, 'blank': -1},
          'LM67': {'plus': 0, 'minus': 270, 'neutral': 135, 'blank': -1},
          'PD67': {'plus': 0, 'minus': 270, 'neutral': 135, 'blank': -1},
          'OA32': {'plus': 135, 'minus': 0, 'neutral': 270, 'blank': -1},
          'OA34': {'plus': 270, 'minus': 135, 'neutral': 0, 'blank': -1},
          'OA36': {'plus': 0, 'minus': 270, 'neutral': 135, 'blank': -1},
          'OA26': {'plus': 270, 'minus': 135, 'neutral': 0, 'blank': -1},
          'LM26': {'plus': 270, 'minus': 135, 'neutral': 0, 'blank': -1},
          'PD26': {'plus': 270, 'minus': 135, 'neutral': 0, 'blank': -1},
          'CC175': {'plus': 0, 'minus': 270, 'neutral': 135, 'blank': -1},
          'CB173': {'plus': 0, 'minus': 270, 'neutral': 135, 'blank': -1},
          'AS23': {'plus': 0, 'minus': 270, 'neutral': 135, 'blank': -1},
          'AS20': {'plus': 135, 'minus': 270, 'neutral': 0, 'blank': -1},  # TYPO WAS HERE
          'AS47': {'plus': 135, 'minus': 0, 'neutral': 270, 'blank': -1},
          'AS41': {'plus': 270, 'minus': 0, 'neutral': 135, 'blank': -1},
          'OA38': {'plus': 135, 'minus': 0, 'neutral': 270, 'blank': -1},
          'AS57': {'plus': 0, 'minus': 270, 'neutral': 135, 'blank': -1}}

# lookup table of orientations during initial learning matched to conditions
lookup_ori = {'OA27': {270: 'plus', 135: 'minus', 0: 'neutral', -1: 'blank'},
              'PD27': {270: 'plus', 135: 'minus', 0: 'neutral', -1: 'blank'},
              'VF226': {0: 'plus', 270: 'minus', 135: 'neutral', -1: 'blank'},
              'LM226': {0: 'plus', 270: 'minus', 135: 'neutral', -1: 'blank'},
              'PD226': {0: 'plus', 270: 'minus', 135: 'neutral', -1: 'blank'},
              'OA67': {0: 'plus', 270: 'minus', 135: 'neutral', -1: 'blank'},
              'LM67': {0: 'plus', 270: 'minus', 135: 'neutral', -1: 'blank'},
              'PD67': {0: 'plus', 270: 'minus', 135: 'neutral', -1: 'blank'},
              'OA32': {135: 'plus', 0: 'minus', 270: 'neutral', -1: 'blank'},
              'OA34': {270: 'plus', 135: 'minus', 0: 'neutral', -1: 'blank', 67.5: 'disengaged_ambiguous'},
              'OA36': {0: 'plus', 270: 'minus', 135: 'neutral', -1: 'blank'},
              'OA26': {270: 'plus', 135: 'minus', 0: 'neutral', -1: 'blank'},
              'LM26': {270: 'plus', 135: 'minus', 0: 'neutral', -1: 'blank'},
              'PD26': {270: 'plus', 135: 'minus', 0: 'neutral', -1: 'blank'},
              'CC175': {0: 'plus', 270: 'minus', 135: 'neutral', -1: 'blank'},
              'CB173': {0: 'plus', 270: 'minus', 135: 'neutral', -1: 'blank'},
              'AS23': {0: 'plus', 270: 'minus', 135: 'neutral', -1: 'blank'},
              'AS20': {135: 'plus', 270: 'minus', 0: 'neutral', -1: 'blank'},
              'AS47': {135: 'plus', 0: 'minus', 270: 'neutral', -1: 'blank'},
              'AS41': {270: 'plus', 0: 'minus', 135: 'neutral', -1: 'blank'},
              'OA38': {135: 'plus', 0: 'minus', 270: 'neutral', -1: 'blank'},
              'AS57': {0: 'plus', 270: 'minus', 135: 'neutral', -1: 'blank'}}

# mapping of what cues, defined by their condition during learning, do across reversal
lookup_mm = {'AS20': {'plus': 'becomes_unrewarded',
                      'neutral': 'becomes_rewarded',
                      'minus': 'remains_unrewarded'},
             'AS23': {'neutral': 'becomes_rewarded',
                      'minus': 'remains_unrewarded',
                      'plus': 'becomes_unrewarded'},
             'AS41': {'minus': 'none', 'plus': 'none', 'neutral': 'none'},
             'AS47': {'neutral': 'none', 'plus': 'none', 'minus': 'none'},
             'OA26': {'plus': 'becomes_unrewarded',
                      'neutral': 'remains_unrewarded',
                      'minus': 'becomes_rewarded'},
             'LM26': {'plus': 'becomes_unrewarded',
                      'neutral': 'remains_unrewarded',
                      'minus': 'becomes_rewarded'},
             'PD26': {'plus': 'becomes_unrewarded',
                      'neutral': 'remains_unrewarded',
                      'minus': 'becomes_rewarded'},
             'OA27': {'plus': 'becomes_unrewarded',
                      'minus': 'becomes_rewarded',
                      'neutral': 'remains_unrewarded'},
             'PD27': {'plus': 'becomes_unrewarded',
                      'minus': 'becomes_rewarded',
                      'neutral': 'remains_unrewarded'},
             'OA32': {'plus': 'becomes_unrewarded',
                      'minus': 'becomes_rewarded',
                      'neutral': 'remains_unrewarded'},
             'OA34': {'minus': 'becomes_rewarded',
                      'plus': 'becomes_unrewarded',
                      'neutral': 'remains_unrewarded'},
             'OA36': {'minus': 'becomes_rewarded',
                      'plus': 'becomes_unrewarded',
                      'neutral': 'remains_unrewarded'},
             'OA38': {'neutral': 'none', 'minus': 'none', 'plus': 'none'},
             'OA67': {'plus': 'becomes_unrewarded',
                      'neutral': 'remains_unrewarded',
                      'minus': 'becomes_rewarded'},
             'LM67': {'plus': 'becomes_unrewarded',
                      'neutral': 'remains_unrewarded',
                      'minus': 'becomes_rewarded'},
             'PD67': {'plus': 'becomes_unrewarded',
                      'neutral': 'remains_unrewarded',
                      'minus': 'becomes_rewarded'},
             'VF226': {'plus': 'becomes_unrewarded',
                       'neutral': 'remains_unrewarded',
                       'minus': 'becomes_rewarded'},
             'LM226': {'plus': 'becomes_unrewarded',
                       'neutral': 'remains_unrewarded',
                       'minus': 'becomes_rewarded'},
             'PD226': {'plus': 'becomes_unrewarded',
                       'neutral': 'remains_unrewarded',
                       'minus': 'becomes_rewarded'}}

# category names
category_names = {
    0: 'Stimulus, transient',
    2: 'Stimulus, ramping',
    4: 'Stimulus, flat',
    1: 'Offset, transient',
    3: 'Offset, delayed',
    5: 'Offset, other'
}

# lookup table of stimulus length
stim_length = {
    'OA27': 3,
    'PD27': 3,
    'VF226': 3,
    'LM226': 3,
    'PD226': 3,
    'OA67': 3,
    'LM67': 3,
    'PD67': 3,
    'OA32': 2,
    'OA34': 2,
    'OA36': 2,
    'OA26': 3,
    'LM26': 3,
    'PD26': 3,
    'CC175': 3,
    'CB173': 2,
    'AS23': 2,
    'AS20': 2,
    'AS47': 2,
    'AS41': 2,
    'OA38': 2,
    'AS57': 2}

# lookup table for staging types
staging = {
    'parsed_stage': ['naive', 'low_dp learning', 'high_dp learning', 'low_dp reversal1', 'high_dp reversal1'],
    'parsed_10stage': ['early naive', 'late naive', 'early low_dp learning', 'late low_dp learning',
                       'early high_dp learning', 'late high_dp learning', 'early low_dp reversal1',
                       'late low_dp reversal1', 'early high_dp reversal1' 'late high_dp reversal1'],
    'parsed_11stage': ['L0 naive', 'L1 learning', 'L2 learning', 'L3 learning', 'L4 learning', 'L5 learning',
                       'L1 reversal1', 'L2 reversal1', 'L3 reversal1', 'L4 reversal1', 'L5 reversal1'],
    'parsed_11stage_T': ['T0 naive', 'T1 learning', 'T2 learning', 'T3 learning', 'T4 learning', 'T5 learning',
                       'T1 reversal1', 'T2 reversal1', 'T3 reversal1', 'T4 reversal1', 'T5 reversal1'],
}

"""Useful lists so I don't have to type out names of mice I commonly use"""
mice = {
    # all mice with crossday alignment completed
    'all15': sorted(['AS23', 'AS20', 'CB173', 'AS47', 'AS41', 'AS57', 'OA38',
                     'OA27', 'OA67', 'VF226', 'OA32', 'OA34', 'OA36', 'OA26', 'CC175']),
    'all14': sorted(['AS23', 'AS20', 'CB173', 'AS47', 'AS41', 'AS57', 'OA38',
                     'OA67', 'VF226', 'OA32', 'OA34', 'OA36', 'OA26', 'CC175']),
    # removed AS57 for poor training, CB173 actually has behavioral understanding, included.
    # removed CC175 because it was mostly naive data
    'all13': sorted(['AS23', 'AS20', 'CB173', 'AS47', 'AS41', 'OA38',
                     'OA27', 'OA67', 'VF226', 'OA32', 'OA34', 'OA36', 'OA26']),
    # removed AS57 and CB173 for poor training, removed CC175 because it was mostly naive data
    'all12': sorted(['AS23', 'AS20', 'AS47', 'AS41', 'OA38',
                     'OA27', 'OA67', 'VF226', 'OA32', 'OA34', 'OA36', 'OA26']),

    # reversal mice. CB173 also has reversal but never looked well trained.
    # AS57 is only had one day that looked well trained before
    # reversal. AS23 had seizures. AS20 was well trained when AUS got him.
    # all good mice with reversals: CB173 and AS57 were never well trained, removed
    'rev9': sorted(['AS23', 'AS20',
                    'OA27', 'OA67', 'VF226', 'OA32', 'OA34', 'OA36', 'OA26']),
    # all possible mice with behavioral reversal: CB173 actually shows licking and running deltas to FC
    'rev10': sorted(['AS23', 'AS20', 'CB173',
                     'OA27', 'OA67', 'VF226', 'OA32', 'OA34', 'OA36', 'OA26']),
    # for posterity: AS57 shows poor performance and behavioral understanding, don't use
    'rev10v2': sorted(['AS23', 'AS20', 'AS57',
                       'OA27', 'OA67', 'VF226', 'OA32', 'OA34', 'OA36', 'OA26']),
    # all mice with reversals
    'rev11': sorted(['AS23', 'AS20', 'CB173', 'AS57',
                     'OA27', 'OA67', 'VF226', 'OA32', 'OA34', 'OA36', 'OA26']),
    # Original big learning animals
    'core4': sorted(['OA27', 'OA67', 'VF226', 'OA26']),
    # Big learning animals. OA38 has 4 days of naive immediately before
    # learning begins here that had too large a shift to align FOVs
    'core9': sorted(['OA27', 'OA67', 'VF226', 'OA26',
                     'OA32', 'OA34', 'OA36', 'AS47', 'OA38', 'AS23']),
    # Animals with naive. OA38 has 4 days of naive immediately before
    # learning begins here that we too large a shift to align FOVs
    'naive6': sorted(['OA27', 'OA67', 'VF226', 'OA26', 'AS47', 'CC175']),
    # all mice with 2 second stimulus
    'stim2': sorted(['AS23', 'AS20', 'CB173', 'AS47', 'AS41', 'AS57', 'OA38',
                     'OA32', 'OA34', 'OA36']),
    'stim3': sorted(['OA27', 'OA67', 'VF226', 'OA26', 'CC175']),

    # Other fields of view
    # LM mice
    'lm': sorted(['LM67', 'LM226', 'LM26']),
    # L5 (deep layer) POR mice
    # PD175 has some naive imaging but had major registration issues so was dropped
    'l5': sorted(['PD27', 'PD67', 'PD226', 'PD26']),
    # Other other FOVs
    'lml5': sorted(['LM67', 'LM226', 'LM26', 'PD27', 'PD67', 'PD226', 'PD26'])
}

""" Factors of different classes, ONE INDEXED, {category: {word: {mouse: {rank: [comp]}}}}"""
factor_categories = {
    'outcome': {
            'respondent': {
                'OA27': {15: [3], 10: [3]}
            },
            'computation': {
                'VF226': {15: [15], 10: [1]},  # weak
                'OA67': {15: [10], 10: [5]},
                'OA32': {15: [7], 10: [5]},
                'OA34': {15: [8], 10: [5]},  # also 15: 14 a little
                'OA36': {15: [4], 10: [3]},  # also 15: 15, 10: 10
                'OA26': {15: [4], 10: [3]},  # 10: 1, 3, 10
                'CC175': {15: [], 10: []},
                'CB173': {15: [3], 10: [2]},  # 15: 3 or 7
                'AS23': {15: [15], 10: [10]},  # weak
                'AS20': {15: [12], 10: [9]},
                'AS47': {15: [4], 10: [4]},  # 15: also 7
                'AS41': {15: [4], 10: [4]},  # check, few days
                'OA38': {15: [9], 10: []},  # check, few days
                'AS57': {15: [], 10: []},
            }
    },
    'offset': {
            'respondent': {
                'OA27': {15: [14], 10: []}  # or 15: 6
            },
            'computation': {
                'VF226': {15: [2], 10: [7]},  # has a few
                'OA67': {15: [3], 10: [1]},
                'OA32': {15: [12], 10: []},  # 15: 11,12 ; 10: 8, 9
                'OA34': {15: [11], 10: []},  # 15: 6,11
                'OA36': {15: [8], 10: []},  # maybe, think about ramp to very sharp cells
                'OA26': {15: [], 10: []},  #
                'CC175': {15: [1], 10: []},
                'CB173': {15: [14], 10: []},  #
                'AS23': {15: [13], 10: [8]},  #
                'AS20': {15: [8], 10: []},  # has many
                'AS47': {15: [14], 10: []},  # has many
                'AS41': {15: [8], 10: []},  # 15: 8, 15 tossup
                'OA38': {15: [6], 10: []},  # many
                'AS57': {15: [13], 10: []},
            }
    }
}

""" Suppressed factors trunc_zscore_day, lion/citation {mouse: {rank: comp}}"""
supress = {
    'OA27': {15: [14], 12: [11], 10: [9], 9: [8]},
    'VF226': {15: [1], 10: [6]},  # 10: 1 seems like an important comparison.
    'OA67': {15: [13], 10: [8, 5]},  # 15: 13 isn't great, 10: 8 has offset but is low for plus only.
    'OA32': {15: [15], 10: [10]},
    'OA34': {15: [], 10: [10]},
    'OA36': {15: [], 10: []},
    'OA26': {15: [], 10: []},
    'CC175': {15: [], 10: []},
    'CB173': {15: [], 10: []},
    'AS23': {15: [], 10: []},
    'AS20': {15: [12], 10: [9]},  # 15: sucks, 10: is great
    'AS47': {15: [], 10: []},
    'AS41': {15: [], 10: []},
    'OA38': {15: [], 10: []},
    'AS57': {15: [8, 14], 10: [9]}
}

offset = {
    'OA27': {15: [7], 12: [6], 10: [5], 9: []},
    'VF226': {15: [6, 11], 10: [5]},
    'OA67': {15: [2], 10: [2]},
    'OA32': {15: [14], 10: [8]},
    'OA34': {15: [], 10: [8]},  # 10: 3 looks like a plus offset
    'OA36': {15: [], 10: []},
    'OA26': {15: [], 10: []},
    'CC175': {15: [], 10: []},
    'CB173': {15: [], 10: []},
    'AS23': {15: [], 10: []},
    'AS20': {15: [9, 11, 15], 10: [6, 8]},  # 15:1 is longer like ensure, but tuned neut-minus
    'AS47': {15: [], 10: []},
    'AS41': {15: [], 10: []},
    'OA38': {15: [], 10: []},
    'AS57': {15: [4], 10: [3]}
}

ensure = {
    'OA27': {15: [4], 12: [3], 10: [3], 9: [3]},
    'VF226': {15: [8], 10: [1]},
    'OA67': {15: [6, 7], 10: [4]},
    'OA32': {15: [12], 10: [3]},
    'OA34': {15: [], 10: [1]},
    'OA36': {15: [], 10: []},
    'OA26': {15: [], 10: []},
    'CC175': {15: [], 10: []},
    'CB173': {15: [], 10: []},
    'AS23': {15: [], 10: []},
    'AS20': {15: [4], 10: [3]},
    'AS47': {15: [], 10: []},
    'AS41': {15: [], 10: []},
    'OA38': {15: [], 10: []},
    'AS57': {15: [10, 15], 10: [10]},  # unclear which it is
}

tt_plot_options = {
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
    'mcp_als': {
        'line_kw': {
            'color': 'red',
            'label': 'mcp_als',
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
    'mncp_hals': {
        'line_kw': {
            'color': 'red',
            'label': 'mcp_als',
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

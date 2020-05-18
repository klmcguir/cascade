"""Useful lookup dicts for general calculations and plotting ease."""

# table of colors I like
color_dict = {
    'plus': [0.46, 0.85, 0.47, 1],
    'minus': [0.84, 0.12, 0.13, 1],
    'neutral': [0.28, 0.68, 0.93, 1],
    'learning': [34/255, 110/255, 54/255, 1],
    'reversal': [173/255, 38/255, 26/255, 1],
    'gray': [163/255, 163/255, 163/255, 1],
    'plus1': '#cae85e',
    'plus2': '#61e6a4',
    'plus3': '#85eb60',
    'minus1': '#ff5e79',
    'minus2': '#ffa561',
    'minus3': '#ff7b60',
    'neutral1': '#74c3ff',
    'neutral2': '#db77ff',
    'neutral3': '#9476ff',
    }

# lookup table of conditions during initial learning matched to orientations
lookup = {'OA27': {'plus': 270, 'minus': 135, 'neutral': 0, 'blank': -1},
     'VF226': {'plus': 0, 'minus': 270, 'neutral': 135, 'blank': -1},
     'OA67': {'plus': 0, 'minus': 270, 'neutral': 135, 'blank': -1},
     'OA32': {'plus': 135, 'minus': 0, 'neutral': 270, 'blank': -1},
     'OA34': {'plus': 270, 'minus': 135, 'neutral': 0, 'blank': -1},
     'OA36': {'plus': 0, 'minus': 270, 'neutral': 135, 'blank': -1},
     'OA26': {'plus': 270, 'minus': 135, 'neutral': 0, 'blank': -1},
     'CC175': {'plus': 270, 'minus': 135, 'neutral': 0, 'blank': -1},
     'CB173': {'plus': 0, 'minus': 270, 'neutral': 135, 'blank': -1},
     'AS23': {'plus': 0, 'minus': 270, 'neutral': 135, 'blank': -1},
     'AS20': {'plus': 135, 'minus': 270, 'neutral': 135, 'blank': -1},
     'AS47': {'plus': 135, 'minus': 0, 'neutral': 270, 'blank': -1},
     'AS41': {'plus': 270, 'minus': 0, 'neutral': 135, 'blank': -1},
     'OA38': {'plus': 135, 'minus': 0, 'neutral': 270, 'blank': -1},
     'AS57': {'plus': 0, 'minus': 270, 'neutral': 135, 'blank': -1}}

# lookup table of orientations during initial learning matched to conditions
lookup_ori = {'OA27': {270: 'plus', 135: 'minus', 0: 'neutral', -1: 'blank'},
     'VF226': {0: 'plus', 270: 'minus', 135: 'neutral', -1: 'blank'},
     'OA67': {0: 'plus', 270: 'minus', 135: 'neutral', -1: 'blank'},
     'OA32': {135: 'plus', 0: 'minus', 270: 'neutral', -1: 'blank'},
     'OA34': {270: 'plus', 135: 'minus', 0: 'neutral', -1: 'blank'},
     'OA36': {0: 'plus', 270: 'minus', 135: 'neutral', -1: 'blank'},
     'OA26': {270: 'plus', 135: 'minus', 0: 'neutral', -1: 'blank'},
     'CC175': {270: 'plus', 135: 'minus', 0: 'neutral', -1: 'blank'},
     'CB173': {0: 'plus', 270: 'minus', 135: 'neutral', -1: 'blank'},
     'AS23': {0: 'plus', 270: 'minus', 135: 'neutral', -1: 'blank'},
     'AS20': {135: 'plus', 270: 'minus', 0: 'neutral', -1: 'blank'},
     'AS47': {135: 'plus', 0: 'minus', 270: 'neutral', -1: 'blank'},
     'AS41': {270: 'plus', 0: 'minus', 135: 'neutral', -1: 'blank'},
     'OA38': {135: 'plus', 0: 'minus', 270: 'neutral', -1: 'blank'},
     'AS57': {0: 'plus', 270: 'minus', 135: 'neutral', -1: 'blank'}}

# lookup table of stimulus length
stim_length = {
     'OA27': 3,
     'VF226': 3,
     'OA67': 3,
     'OA32': 2,
     'OA34': 2,
     'OA36': 2,
     'OA26': 3,
     'CC175': 3,
     'CB173': 2,
     'AS23': 2,
     'AS20': 2,
     'AS47': 2,
     'AS41': 2,
     'OA38': 2,
     'AS57': 2}

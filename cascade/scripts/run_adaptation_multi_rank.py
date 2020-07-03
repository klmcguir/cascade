import cascade as cas
import matplotlib.pyplot as plt


mice = cas.lookups.mice['all15']
nan_thresh = 0.95
score_threshold = 0.8


for rank in [5, 10, 15, 20]:

    temp = cas.adaptation.calc_daily_ramp(
            mice,
            words=['bookmarks' if m == 'OA27' else 'horrible' for m in mice],
            method='ncp_hals',
            cs='',
            warp=False,
            trace_type='zscore_day',
            group_by='all3',
            nan_thresh=nan_thresh,
            score_threshold=score_threshold,
            rank=rank,
            annotate=True,
            over_components=True)
    plt.close('all')

    test = cas.adaptation.calc_daily_ramp(
            mice,
            words=['facilitate']*len(mice),
            method='ncp_hals',
            cs='',
            warp=False,
            trace_type='zscore_day',
            group_by='learning',
            nan_thresh=nan_thresh,
            score_threshold=score_threshold,
            rank=rank,
            annotate=True,
            over_components=True)
    plt.close('all')

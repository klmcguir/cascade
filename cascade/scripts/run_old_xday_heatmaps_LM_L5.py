from cascade.df import trialmeta, trigger
from cascade import lookups
from cascade.plotting.xday import heatmap

# for new LM/L5 FOVs
mice = lookups.mice['lml5']

# loop over and use the "manual" triggering and metadata building to make triple heatmaps per cell
for mouse in mice:

    try:
        trialmeta(mouse, downsample=True, verbose=True)
        trigger(mouse, trace_type='zscore_day', cs='', downsample=True,
                start_time=-1, end_time=6, clean_artifacts=None,
                thresh=20, warp=False, smooth=True, smooth_win=6,
                verbose=True)
        trigger(mouse, trace_type='dff', cs='', downsample=True,
                start_time=-1, end_time=6, clean_artifacts=None,
                thresh=20, warp=False, smooth=True, smooth_win=6,
                verbose=True)
    except:
        print(f'1. skipped {mouse}')

# plot and save
for mouse in mice:

    try:
        heatmap(mouse, cell_id=None, trace_type='zscore_day', cs_bar=True, day_bar=True, day_line=True, run_line=False,
                match_clim=True, quinine_ticks=False, ensure_ticks=False, lick_ticks=False, label_cbar=True, vmin=None,
                vmax=None, smooth=False, save_tag='', word='infrastructure', verbose=False)
        heatmap(mouse, cell_id=None, trace_type='dff', cs_bar=True, day_bar=True, day_line=True, run_line=False,
                match_clim=True, quinine_ticks=False, ensure_ticks=False, lick_ticks=False, label_cbar=True, vmin=None,
                vmax=None, smooth=False, save_tag='', word='existing', verbose=False)
    except:
        print(f'2. skipped {mouse}')

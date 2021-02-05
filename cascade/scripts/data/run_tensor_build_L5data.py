""" Run 'TCA' function without the TCA, building tensor and metadata outputs"""
import cascade as cas

# matches COMPUTATION parameter hash word

# parameters
drive_threshold = 1.31  # usually 1.31 for significance
sc = 0.8
nant = 0.95  # 0.95
gb = 'all3'  # TESTING THIS TO GET PAVS FOR MISMATCH
tt = 'zscore_day'
method = 'ncp_hals'
rank = 20  # [10, 15, 20]  #usually 20
ww = False
cv = False
negative_modes = []  # [1]
smooth = True
# exclude_conds=('blank_reward', 'blank', 'pavlovian') # This is the wrong order being used to generate a unique hash
exclude_conds = ('blank', 'blank_reward', 'pavlovian')
clean_artifacts = None

update_meta = True

# for mouse in ['OA27']:
# for mouse in cas.lookups.mice['l5']:
for mouse in ['PD67', 'PD226']:

    cas.tca.groupday_tca(
        mouse,
        tags=None,

        # TCA params
        rank=rank,
        method=(method,),
        replicates=3,
        fit_options=None,
        negative_modes=negative_modes,

        # grouping params
        group_by=gb,
        up_or_down='up',
        use_dprime=False,
        dprime_threshold=2,

        # tensor params
        trace_type=tt,
        cs='',
        downsample=True,
        start_time=-1,
        end_time=6,
        clean_artifacts=clean_artifacts,
        thresh=20,
        warp=ww,
        smooth=smooth,  # True
        smooth_win=6,
        verbose=True,

        # filtering params
        exclude_tags=('disengaged', 'orientation_mapping', 'contrast', 'retinotopy', 'sated'),
        exclude_conds=exclude_conds,
        driven=True,
        drive_css=('0', '135', '270'),
        drive_threshold=drive_threshold,  # 3 for logn, #1.31
        drive_type='trial',
        nan_trial_threshold=nant,  # 0.95
        score_threshold=sc,

        # other params
        update_meta=update_meta,
        three_pt_tf=False,
        remove_stim_corr=False,
        cv=cv,
    )

"""Calculations to be saved to mongoDB database"""
from pool.database import memoize

@memoize(across='mouse', updated=190429, returns='other')
def groupday_varex_byday_bycell(
        mouse,
        trace_type='zscore_day',
        method='ncp_bcd',
        cs='',
        warp=False,
        word=None,
        group_by=None,
        nan_thresh=None,
        verbose=False):
    """
    Plot reconstruction error as variance explained across all whole groupday
    TCA decomposition ensemble.

    Parameters:
    -----------
    mouse : str; mouse object 
    trace_type : str; dff, zscore, deconvolved
    method : str; TCA fit method from tensortools

    Returns:
    --------
    Saves figures to .../analysis folder  .../qc
    """

    pars = {'trace_type': trace_type, 'cs': cs, 'warp': warp}
    group_pars = {'group_by': group_by}

    # if cells were removed with too many nan trials
    if nan_thresh:
        nt_tag = '_nantrial' + str(nan_thresh)
        nt_save_tag = ' nantrial ' + str(nan_thresh)
    else:
        nt_tag = ''
        nt_save_tag = ''

    # load dir
    load_dir = paths.tca_path(
        mouse, 'group', pars=pars, word=word, group_pars=group_pars)
    tensor_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_group_decomp_' + str(trace_type) + '.npy')
    input_tensor_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_group_tensor_' + str(trace_type) + '.npy')
    meta_path = os.path.join(
        load_dir, str(mouse) + '_' + str(group_by) + nt_tag
        + '_df_group_meta.pkl')

    # save dir
    save_dir = paths.tca_plots(
        mouse, 'group', pars=pars, word=word, group_pars=group_pars)
    save_dir = os.path.join(save_dir, 'qc' + nt_save_tag)
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    var_path = os.path.join(
        save_dir, str(mouse) + '_summary_variance_cubehelix.pdf')

    # load your data
    ensemble = np.load(tensor_path)
    ensemble = ensemble.item()
    V = ensemble[method]
    X = np.load(input_tensor_path)
    meta = pd.read_pickle(meta_path)
    orientation = meta['orientation']
    trial_num = np.arange(0, len(orientation))
    condition = meta['condition']
    trialerror = meta['trialerror']
    hunger = deepcopy(meta['hunger'])
    speed = meta['speed']
    dates = meta.reset_index()['date']
    learning_state = meta['learning_state']

    # get reconstruction error as variance explained per day
    var_by_rank = {}
    for r in V.results:
        var_by_day = {}
        bU = V.results[r][0].factors.full()
        for day in np.unique(dates):
            day_bool = dates.isin(day)
            bUd = bU[:, :, day_bool]
            bX = X[:, :, day_bool]
            var_by_day[day] = ((np.nanvar(bX) - np.nanvar(bX - bUd)) / np.nanvar(bX))
        var_by_rank[r] = var_by_day

    # get reconstruction error as variance explained per day per component
    var_by_rank = {}
    for r in V.results:
        var_by_fac = {}
        for fac_num in range(np.shape(V.results[r][0].factors[0][:, fac_num])[1]):
            a = V.results[r][0].factors[0][:, fac_num]
            b = V.results[r][0].factors[1][:, fac_num]
            c = V.results[r][0].factors[2][:, fac_num]
            ab = a[:, None] @ b[None, :]
            abc = ab[:, :, None] @ c[None, :]
            var_by_day = {}
            for day in np.unique(dates):
                day_bool = dates.isin(day)
                bU = abc[:, :, day_bool]
                bX = X[:, :, day_bool]
                var_by_day[day] = ((np.nanvar(bX) - np.nanvar(bX - bU)) / np.nanvar(bX))
            var_by_fac[fac_num] = var_by_day
        var_by_rank[r] = var_by_fac

    # get reconstruction error as variance explained per day per component
    var_by_rank = {}
    for r in V.results:
        var_by_fac = {}
        for fac_num in range(np.shape(V.results[r][0].factors[0][:, fac_num])[1]):
            a = V.results[r][0].factors[0][:, fac_num]
            b = V.results[r][0].factors[1][:, fac_num]
            c = V.results[r][0].factors[2][:, fac_num]
            ab = a[:, None] @ b[None, :]
            abc = ab[:, :, None] @ c[None, :]
            var_by_day = {}
            for day in np.unique(dates):
                day_bool = dates.isin(day)
                abcd = abc[:, :, day_bool]
                xxxx = X[:, :, day_bool]
                var_by_cell = {}
                for cell in range():
                    bX = xxxx[cell, :, :]
                    bU = abcd[cell, :, :]
                    var_by_cell[cell] = ((np.nanvar(bX) - np.nanvar(bX - bU)) / np.nanvar(bX))
                var_by_day[day] = var_by_cell
            var_by_fac[fac_num] = var_by_day
        var_by_rank[r] = var_by_fac


    # mean response of neuron across trials
    mU = np.nanmean(X, axis=2, keepdims=True) * np.ones((1, 1, np.shape(X)[2]))
    var_mean = (np.nanvar(X) - np.nanvar(X - mU)) / np.nanvar(X)

    # smoothed response of neuron across time
    smU = np.convolve(
        X.reshape((X.size)),
        np.ones(5, dtype=np.float64)/5, 'same').reshape(np.shape(X))
    var_smooth = (np.nanvar(X) - np.nanvar(X - smU)) / np.nanvar(X)

    # create figure and axes
    buffer = 5
    right_pad = 5
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(
        100, 100, figure=fig, left=0.05, right=.95, top=.95, bottom=0.05)
    ax = fig.add_subplot(gs[10:90-buffer, :90-right_pad])
    c = 0
    cmap = sns.color_palette(sns.cubehelix_palette(c+1))

    # plot
    R = np.max([r for r in V.results.keys()])
    ax.scatter(x_s, var_s, color=cmap[c], alpha=0.5)
    ax.scatter([R+2], var_mean, color=cmap[c], alpha=0.5)
    ax.scatter([R+4], var_smooth, color=cmap[c], alpha=0.5)
    ax.plot(x, var, label=('mouse ' + mouse), color=cmap[c])
    ax.plot([R+1.5, R+2.5], [var_mean, var_mean], color=cmap[c])
    ax.plot([R+3.5, R+4.5], [var_smooth, var_smooth], color=cmap[c])

    # add labels/titles
    x_labels = [str(R) for R in V.results]
    x_labels.extend(
        ['', 'mean\n cell\n response', '', 'smooth\n response\n (0.3s)'])
    ax.set_xticks(range(1, len(V.results) + 5))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('model rank')
    ax.set_ylabel('fractional variance explained')
    ax.set_title('Variance Explained: ' + str(method) + ', ' + mouse)
    ax.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)

    fig.savefig(var_path, bbox_inches='tight')
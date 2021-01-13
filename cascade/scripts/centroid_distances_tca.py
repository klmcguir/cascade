import numpy as np
import cascade as cas
import seaborn as sns
import matplotlib.pyplot as plt
import os


mice = cas.lookups.mice['all12']
words = ['respondent' if s in 'OA27' else 'computation' for s in mice]

for mouse, word in zip(mice, words):

    sort_ensemble, cell_ids, cell_clusters = cas.load.groupday_tca_model(
        mouse=mouse,
        trace_type='zscore_day',
        method='ncp_hals',
        cs='',
        warp=False,
        rank=15,
        word=word,
        group_by='all3',
        nan_thresh=0.95,
        score_threshold=0.8,
        full_output=True,
        unsorted=False,
        verbose=False)
    meta = cas.load.groupday_tca_meta(
        mouse=mouse,
        trace_type='zscore_day',
        method='ncp_hals',
        cs='',
        warp=False,
        word=word,
        group_by='all3',
        nan_thresh=0.95,
        score_threshold=0.8)

    centroids = np.loadtxt(f'S:\\twophoton_analysis\\Data\\data\\xday_registered_centroids\\{mouse}_centroids.txt')

    fig, ax = plt.subplots(1,2, figsize=(11, 5), sharex=True, sharey=True)

    # SELF-OTHER
    std_thresh = 1
    xlist = []
    ylist = []
    rank_num = 10

    tuning = cas.tuning.component_tuning(meta, sort_ensemble, rank_num, by_stage=False, by_reversal=True,
                                staging='parsed_11stage', tuning_type='initial')
    tuning = tuning.loc[tuning.staging_LR.isin(['learning'])]

    for cell_n in range(len(cell_ids[rank_num])):
        x, y = centroids[:, cell_ids[rank_num][cell_n] - 1]
        xlist.append(x)
        #     ylist.append(y*-1) # flip to match the way images are plotted
        ylist.append(y)

    all_distances = []
    for compi in range(rank_num):

        for compi2 in range(rank_num):

            # Skip self comparison
            if compi == compi2:
                continue

            hue = sort_ensemble.results[rank_num][0].factors[0][:, compi]
            thresh = hue > (np.std(hue) * std_thresh)
            x1 = np.array(xlist)[thresh]
            y1 = np.array(ylist)[thresh]

            hue = sort_ensemble.results[rank_num][0].factors[0][:, compi2]
            thresh = hue > (np.std(hue) * std_thresh)
            x2 = np.array(xlist)[thresh]
            y2 = np.array(ylist)[thresh]

            comp = []
            for offcell in zip(x2, y2):
                for stimcell in zip(x1, y1):
                    comp.append(np.linalg.norm(np.array(offcell) - np.array(stimcell)))

        all_distances.append(comp)

    for compi in range(rank_num):
        sns.ecdfplot(all_distances[compi], label=f'{compi + 1}', ax=ax[0])
    # ax[0].legend()
    ax[0].set_title('Centroid distances: comp n vs all other comps', size=16)
    ax[0].set_xlabel('distance between centroids\n(pixels)', size=14)
    ax[0].set_ylabel('Proportion of cells', size=14)

    # SELF-SELF
    std_thresh = 1
    xlist = []
    ylist = []
    rank_num = 10

    for cell_n in range(len(cell_ids[rank_num])):
        #     x, y = centroids[cell_ids[rank_num][cell_n], :]
        x, y = centroids[:, cell_ids[rank_num][cell_n] - 1]
        xlist.append(x)
        #     ylist.append(y*-1) # flip to match the way images are plotted
        ylist.append(y)  # flip to match the way images are plotted
    # plt.figure(figsize=(8, 6))

    all_distances = []
    for compi in range(rank_num):

        hue = sort_ensemble.results[rank_num][0].factors[0][:, compi]
        thresh = hue > (np.std(hue) * std_thresh)
        x1 = np.array(xlist)[thresh]
        y1 = np.array(ylist)[thresh]

        hue = sort_ensemble.results[rank_num][0].factors[0][:, compi]
        thresh = hue > (np.std(hue) * std_thresh)
        x2 = np.array(xlist)[thresh]
        y2 = np.array(ylist)[thresh]

        comp = []
        for offcell in zip(x2, y2):
            for stimcell in zip(x1, y1):
                comp.append(np.linalg.norm(np.array(offcell) - np.array(stimcell)))

        all_distances.append(comp)

    for compi in range(rank_num):
        #     for compi2 in range(rank_num):
        pref = tuning.loc[mouse, compi+1]['preferred tuning']
        offset = tuning.loc[mouse, compi+1]['offset component']
        sns.ecdfplot(all_distances[compi], label=f'{compi + 1}, offset={offset}, tuning={pref}', ax=ax[1])
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc=2)
    ax[1].set_title('Centroid distances: within comp', size=16)
    ax[1].set_xlabel('distance between centroids\n(pixels)', size=14)
    ax[1].set_ylabel('Proportion of cells', size=14)

    plt.savefig(os.path.join(cas.lookups.saveroot, f'{mouse}_centroid_distances.png'), bbox_inches='tight')

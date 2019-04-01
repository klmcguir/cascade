"""Experimental json metadata."""
import os
import flow


def dircreate(mouse, overwrite=False, update=False):
    """ Crawl a data directory mouse folder and create metadata for
    all existing dates and runs for that mouse.
    """

    # get your mouse data directory, check that it is a correctly formated dir
    if not mouse:
        mouse = raw_input('Enter mouse name: ')

    # add mouse metadata
    flow.metadata.add_mouse(mouse, tags=['kelly'], overwrite=overwrite, update=update)

    data_dir = '/Users/kelly_mcguire/Documents/Data/data'
    mouse_dir = os.path.join(data_dir, mouse)
    dates = sorted(os.listdir(mouse_dir))
    # check that yor dir names are all digits
    dates = [s for s in dates if s.isdigit()]
    # check that yor dir names are dirs
    dates = [s for s in dates if os.path.isdir(os.path.join(mouse_dir, s))]

    # ask important dates for mouse
    rev0 = raw_input('Enter first training/learning date: ')
    if rev0:
        rev0 = int(rev0)
    rev1 = raw_input('Enter reversal1 date: ')
    if rev1:
        rev1 = int(rev1)
    rev2 = raw_input('Enter reversal2 date: ')
    if rev2:
        rev2 = int(rev2)

    # add date to metadata get all runs for each data
    for i in dates:
        # get all runs for that data based on simpcell data
        simp_dir = sorted(os.listdir(os.path.join(mouse_dir, i)))
        simp_dir = [s for s in simp_dir if s.endswith('.simpcell')]
        run_nums = [int(s[-12:-9]) for s in simp_dir if s.endswith('.simpcell')]

        # define date_tags for training, reversal, etc.
        date_tags = None
        if int(i) < rev0:
            date_tags = 'naive'
        elif int(i) == rev0:
            date_tags = 'learning_start'
            rev0_run = int(raw_input('Enter 1st learning run: '+str(run_nums)))
        elif int(i) > rev0 and int(i) < rev1:
            date_tags = 'learning'
        elif int(i) == rev1:
            date_tags = 'reversal1_start'
            rev1_run = int(raw_input('Enter 1st reversal1 run: '+str(run_nums)))
        elif int(i) > rev1 and int(i) < rev2:
            date_tags = 'reversal1'
        elif int(i) == rev2:
            date_tags = 'reversal2_start'
            rev2_run = int(raw_input('Enter 1st reversal2 run: '+str(run_nums)))
        elif int(i) > rev2:
            date_tags = 'reversal2'

        # add date to metadata
        flow.metadata.add_date(mouse, int(i), tags=[date_tags],
                               photometry=None, overwrite=overwrite, update=update)

        # loop through adding runs and getting tags for days
        tag_options = ['hungry', 'sated', 'disengaged', 'contrast',
                       'orientation_mapping', 'retinotopy']

        for k in run_nums:
            # get correct run tags
            # run_type_options = ['naive', 'learning', 'reversal1', 'reversal2']
            if date_tags == 'learning_start':
                if int(k) < rev0_run:
                    run_tag1 = 'naive'
                else:
                    run_tag1 = 'learning'
            elif date_tags == 'reversal1_start':
                if int(k) < rev1_run:
                    run_tag1 = 'learning'
                else:
                    run_tag1 = 'reversal1'
            elif date_tags == 'reversal2_start':
                if int(k) < rev2_run:
                    run_tag1 = 'reversal1'
                else:
                    run_tag1 = 'reversal2'

            my_tags = raw_input('Enter tag#: ' + str(i) + ' run ' + str(k) +
                                ': 0=hungry, 1=sated, 2=disengaged, 3=contrast,' +
                                ' 4=orientation_mapping, 5=retinotopy: ')
            # if you just hit enter and leave ui input empty, defualt to 0=hungry
            if not my_tags:
                my_tags = 0
            run_tag2 = tag_options[int(my_tags)]

            # pick run_type based on tags
            all_tags = [run_tag1, run_tag2, date_tags]
            tag_flag = [False for s in range(len(all_tags)) if all_tags[s] == 'sated'
                        # or all_tags[s] == 'naive'
                        or all_tags[s] == 'orientation_mapping'
                        or all_tags[s] == 'contrast'
                        or all_tags[s] == 'disengaged'
                        or all_tags[s] == 'retinotopy']
            if not tag_flag:
                run_type = 'training'
            else:
                run_type = 'other'

            flow.metadata.add_run(mouse, int(i), int(k), run_type, tags=[run_tag1, run_tag2],
                                  overwrite=overwrite, update=update)


def dirupdate_training(mouse, update=True):
    """ Update existing json metadata to add tags or runtypes to match
    Arthur's and Jeff's data.
    """

    # get all runs for an existing mouse
    runs = flow.RunSorter.frommeta(mice=[mouse])

    for run in runs:

        # check for undesirable tags (runs where animal is not training)
        tag_flag = [False for s in range(len(run.tags)) if run.tags[s] == 'sated'
                    or run.tags[s] == 'naive'
                    or run.tags[s] == 'orientation_mapping'
                    or run.tags[s] == 'contrast'
                    or run.tags[s] == 'disengaged'
                    or run.tags[s] == 'retinotopy']

        # check for undesirable run_type (where animal is not training)
        type_flag = run.run_type == 'naive'

        # update metadata for all approved runs (tag_flag == empty)
        # add current run_type to tags and make update run_type
        if not tag_flag and not type_flag:
            run_type = 'training'
            flow.metadata.add_run(run.mouse, run.date, run.run, run_type,
                                  tags=[run.run_type], overwrite=False, update=update)
        else:
            run_type = 'other'
            flow.metadata.add_run(run.mouse, run.date, run.run, run_type,
                                  tags=[run.run_type], overwrite=False, update=update)

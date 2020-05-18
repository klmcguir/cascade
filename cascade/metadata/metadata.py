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
    # check that your dir names are all digits
    dates = [s for s in dates if s.isdigit()]
    # check that your dir names are dirs
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
                        or all_tags[s] == 'spontaneous'
                        or all_tags[s] == 'disengaged'
                        or all_tags[s] == 'retinotopy']
            if not tag_flag:
                run_type = 'training'
            else:
                run_type = 'other'

            flow.metadata.add_run(mouse, int(i), int(k), run_type, tags=[run_tag1, run_tag2],
                                  overwrite=overwrite, update=update)

def tag_dates_w_xday_files():
    """
    Helper function to crawl your flow datad directory and add 'xday' tag if
    crossday.txt file exists
    """

    # look for mouse directories
    data_dir = flow.paths.datad
    mice = sorted(os.listdir(data_dir))

    # check that your dir names are dirs
    mice = [s for s in mice if os.path.isdir(os.path.join(data_dir, s))]

    # loop over each mouse and get date directories
    for mouse in mice:

        # look for date directories
        mouse_dir = os.path.join(data_dir, mouse)
        dates = sorted(os.listdir(mouse_dir))

        # check that your dir names are all digits
        dates = [s for s in dates if s.isdigit()]

        # check that your dir names are dirs
        dates = [s for s in dates if os.path.isdir(os.path.join(mouse_dir, s))]

        # loop over each date directory looking for crossday.txt files
        for date in dates:
            xday_file = flow.xday._read_crossday_ids(mouse, date)

            # if the files returns an array, add the xday tag
            if len(xday_file) > 0:
                flow.metadata.add_date(mouse, int(date), tags=['xday'],
                                       overwrite=False, update=True)
            

def dircreate_from_AU(mouse, overwrite=False, update=True):
    """ Crawl a data directory mouse folder and create metadata for
    all existing dates and runs for that mouse. For use on Arthur's mice 
    with standardized run tags. 
    """

    # get your mouse data directory, check that it is a correctly formated dir
    if not mouse:
        mouse = raw_input('Enter mouse name: ')

    # add mouse metadata
    flow.metadata.add_mouse(mouse, tags=['kelly'], overwrite=overwrite, update=update)

    data_dir = flow.paths.datad
    mouse_dir = os.path.join(data_dir, mouse)
    dates = sorted(os.listdir(mouse_dir))
    # check that your dir names are all digits
    dates = [s for s in dates if s.isdigit()]
    # check that your dir names are dirs
    dates = [s for s in dates if os.path.isdir(os.path.join(mouse_dir, s))]

    # ask important dates for mouse
    rev0 = 0  # a very early date that does not actually exist
    if mouse in ['AS47']:
        rev0 = 180214 # training began
    if mouse in ['CB210']:
        rev0 = raw_input('Enter first training/learning date: ')
    rev0 = int(rev0)
    try:
        rev1 = flow.metadata.reversal(mouse)
    except:
        rev1 = 201231  # some date far in the unknown future
    rev1 = int(rev1)

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
        elif int(i) > rev0 and int(i) < rev1:
            date_tags = 'learning'
        elif int(i) == rev1:
            date_tags = 'reversal1_start'
        elif int(i) > rev1:
            date_tags = 'reversal1'

        # add date to metadata
        flow.metadata.add_date(mouse, int(i), tags=[date_tags],
                               overwrite=overwrite, update=update)

        for k in run_nums:
            # get correct run tags
            # run_type_options = ['naive', 'learning', 'reversal1', 'reversal2']
            first_run = True
            if date_tags in ['learning_start', 'learning']:
                run_tag1 = 'learning'
            elif date_tags in ['reversal1_start', 'reversal1']:
                run_tag1 = 'reversal1'
            elif date_tags in ['naive']:
                run_tag1 = 'naive'


            # set run tags based on run number
            if k in [1, 5]:
                all_run_tags = ['hungry', run_tag1]
            elif k in [2, 3, 4]:
                all_run_tags = ['hungry', run_tag1]
            elif k in [6, 7, 8, 9, 10, 11]:
                all_run_tags = ['sated', run_tag1]
            elif k in [7]:
                all_run_tags = ['sated', run_tag1]
            else:
                continue

            # set run type based on run number
            if k in [2, 3, 4, 7]:
                run_type = 'training'
            elif k in [1]:
                run_type = 'running'
            elif k in [1, 6, 8, 9, 10, 11]:
                run_type = 'spontaneous'
            else:
                run_type = 'other'

            flow.metadata.add_run(mouse, int(i), int(k), run_type,
                                  tags=all_run_tags,
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


def dirupdate_revdate(mouse, update=True):
    """ Update existing json metadata to add tags for learning state to
    Arthur's and Jeff's data.
    """

    # get reversal date
    reversal_date = flow.metadata.reversal(mouse)

    # get all days for an existing mouse
    days = flow.DateSorter.frommeta(mice=[mouse])

    for day in days:

        # check if date is pre- or post-reversal
        if int(day.date) < int(reversal_date):

            # add new tag for date to metadata
            flow.metadata.add_date(
                mouse, int(day.date), tags=['learning'], update=update)

        elif int(day.date) >= int(reversal_date):

            # add new tag for date to metadata
            flow.metadata.add_date(
                mouse, int(day.date), tags=['reversal1'], update=update)

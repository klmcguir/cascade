"""Walk anastasia directories for LM/V1 imaging rename all dirs and files for LM naming"""
import os

# forward slashes needed to recognize mounted NAS
root = os.path.abspath('//anastasia/data/2p/kelly')

# re_name are existing folders with subfolders that need renaming
old_name = ['OA26', 'OA67', 'VF226']
re_name = ['LM26', 'LM67', 'LM226']

# list all files and folder
dir_list_old, file_list_old = [], []
dir_list_new, file_list_new = [], []

# loop over new deep fields of view, renaming any folder or file within that has old mouse name
for old, new in zip(old_name, re_name):

    # folder to look inside of
    mouse_folder = os.path.join(root, new)

    # get files
    for dirpath, dirnames, filenames in os.walk(mouse_folder, topdown=False):
        for filename in filenames:
            if old in filename:
                file_list_old.append(os.path.join(dirpath, filename))
                file_list_new.append(os.path.join(dirpath, filename.replace(old, new)))

        for dirname in dirnames:
            if old in dirname:
                dir_list_old.append(os.path.join(dirpath, dirname))
                dir_list_new.append(os.path.join(dirpath, dirname.replace(old, new)))

# rename files
for old_f, new_f in zip(file_list_old, file_list_new):
    os.rename(old_f, new_f)

# rename folders
for old_dir, new_dir in zip(dir_list_old, dir_list_new):
    os.rename(old_dir, new_dir)

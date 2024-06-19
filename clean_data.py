import csv
import os
import shutil

from colorama import init, Fore

from exp.exp_basic import Exp_Basic
from hyper_parameter_optimizer import basic_settings

init(autoreset=True)

root_path = '.'
data_dir = 'data'

print('Start cleaning data...')

# build basic experiment
exp_basic = Exp_Basic(root_path=root_path, args=None, try_model=True, save_process=True, initialize_later=True)

# get all root folders
checkpoints_folder = exp_basic.root_checkpoints_path
process_folder = exp_basic.root_process_path
results_folder = exp_basic.root_results_path
test_results_folder = exp_basic.root_test_results_path
m4_results_folder = exp_basic.root_m4_results_path
prob_results_folder = exp_basic.root_prob_results_path

print('Root folders:')
print(f"\tCheckpoints folder: {checkpoints_folder}")
print(f"\tProcess folder: {process_folder}")
print(f"\tResults folder: {results_folder}")
print(f"\tTest results folder: {test_results_folder}")
print(f"\tM4 results folder: {m4_results_folder}")
print(f"\tProb results folder: {prob_results_folder}")

# get data folder
data_folder = os.path.join(root_path, data_dir)

print('Data folder:')
print(f"\tData folder: {data_folder}")

# get fieldnames
fieldnames = basic_settings.get_fieldnames('all')

print('Fieldnames:')
print(f"\tFieldnames: {fieldnames}")


def clean_blank_folder(_folders):
    # clean blank folder under root folder
    clean_number = 0
    blank_folders = []
    for folder in _folders:
        if os.path.exists(folder):
            for path in os.listdir(folder):
                setting_folder = os.path.join(folder, path)
                if os.path.isdir(setting_folder) and not os.listdir(setting_folder):
                    os.rmdir(setting_folder)
                    blank_folders.append(setting_folder)
                    clean_number += 1

    if clean_number != 0:
        print(Fore.RED + f"Cleaned {clean_number} blank folders")
        for folder in blank_folders:
            print(Fore.RED + f"\t{folder}")


clean_blank_folder([checkpoints_folder, process_folder, results_folder, test_results_folder, m4_results_folder,
                    prob_results_folder])


def clean_unrelated_folder(_folders):
    # scan all csv files under data folder
    file_paths = []
    for root, dirs, files in os.walk(str(data_folder)):
        for _file in files:
            if _file.endswith('.csv') and _file not in file_paths:
                _append_path = os.path.join(root, _file)
                file_paths.append(_append_path)
    print(f"Found {len(file_paths)} csv files under data folder")

    # read csv files
    setting_list = []
    for file_path in file_paths:
        with open(file_path, 'r') as csv_file:
            reader = csv.DictReader(csv_file, fieldnames=fieldnames)
            next(reader)  # skip the header
            for row in reader:
                setting = row['setting']
                if setting not in setting_list:
                    setting_list.append(setting)
    print(f"Found {len(setting_list)} settings in all csv files")

    # clean unrelated folders
    clean_number = 0
    unrelated_folders = []
    for folder in _folders:
        if os.path.exists(folder):
            for path in os.listdir(folder):
                setting_folder = os.path.join(folder, path)
                if os.path.isdir(setting_folder) and path not in setting_list:
                    shutil.rmtree(setting_folder)
                    unrelated_folders.append(setting_folder)
                    clean_number += 1

    if clean_number != 0:
        print(Fore.RED + f"Cleaned {clean_number} folders that are not experiments in csv files under data folder")
        for folder in unrelated_folders:
            print(Fore.RED + f"\t{folder}")


clean_unrelated_folder([checkpoints_folder, process_folder, results_folder, test_results_folder, m4_results_folder,
                        prob_results_folder])

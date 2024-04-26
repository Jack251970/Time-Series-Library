import os

from exp.exp_basic import Exp_Basic


def clean_blank_folder():
    # build basic experiment
    exp_basic = Exp_Basic(root_path='.', args=None, try_model=True, save_process=True, initialize_later=True)

    # get all root folders
    folders = [exp_basic.root_checkpoints_path, exp_basic.root_process_path, exp_basic.root_results_path,
               exp_basic.root_test_results_path, exp_basic.root_m4_results_path, exp_basic.root_prob_results_path]

    # clean blank folder under root folder
    clean_number = 0
    for folder in folders:
        # Delete blank folders under the folder
        if os.path.exists(folder):
            for path in os.listdir(folder):
                sub_folder = os.path.join(folder, path)
                if os.path.isdir(sub_folder) and not os.listdir(sub_folder):
                    os.rmdir(sub_folder)
                    clean_number += 1

    print(f"Cleaned {clean_number} blank folders")


clean_blank_folder()

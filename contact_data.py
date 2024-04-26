import os

from optimizer_script.run_optimizer_transformer import h


def contact_data_to_file(core_process_file, other_process_files):
    print(other_process_files + ' to ' + core_process_file)

    if os.path.exists(core_process_file):
        if os.path.exists(other_process_files):
            with open(other_process_files, "r") as f:
                with open(core_process_file, "a") as f1:
                    # skip the first line and write the remaining lines
                    f.readline()
                    for line in f:
                        f1.write(line)
            # delete file
            os.remove(other_process_files)
    else:
        if os.path.exists(other_process_files):
            with open(other_process_files, "r") as f:
                with open(core_process_file, "a") as f1:
                    for line in f:
                        f1.write(line)
            # delete file
            os.remove(other_process_files)


optimizer_settings = h.get_optimizer_settings()
task_names = h.get_all_task_names()
_max_process_index = optimizer_settings['process_number'] - 1
for task_name in task_names:
    for process_index in range(_max_process_index):
        if process_index == 0:
            continue
        contact_data_to_file(h.get_csv_file_path(task_name), h.get_csv_file_path(task_name, process_index))

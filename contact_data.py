import os

from run_optimizer import h


def contact_data_to_file(ori_file_format, dst_file, _process_number):
    _max_process_index = _process_number - 1
    for process_index in range(_max_process_index):
        file = ori_file_format.format(process_index + 1)
        if os.path.exists(file):
            with open(file, "r") as f:
                with open(dst_file, "a") as f1:
                    # skip the first line and write the remaining lines
                    f.readline()
                    for line in f:
                        f1.write(line)
            # delete file
            os.remove(file)


optimizer_settings = h.get_optimizer_settings()
contact_data_to_file(optimizer_settings['data_csv_file_format'], h.get_csv_file_path(),
                     optimizer_settings['process_number'])

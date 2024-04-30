from hyper_optimizer.optimizer import HyperParameterOptimizer

h = HyperParameterOptimizer(script_mode=True)
h.config_optimizer_settings(root_path='.',
                            data_dir='data',
                            jump_csv_file='jump_data.csv',
                            data_csv_file='data.csv',
                            data_csv_file_format='data_{}.csv',
                            scan_all_csv=True,
                            process_number=1,
                            save_process=True)

if __name__ == "__main__":
    h.start_search()

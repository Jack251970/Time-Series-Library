import csv
import math
import os
import random
import time
from itertools import product

import numpy as np
import torch
from tqdm import tqdm

from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from exp.exp_imputation import Exp_Imputation
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast


class HyperOptimizer(object):
    def __init__(self, get_fieldnames, get_search_space, prepare_config, build_setting, build_config_dict,
                 get_tags=None, check_jump_experiment=None,
                 jump_csv_file_path='jump_data.csv', csv_file_path_format="data_{}.csv",
                 process_number=1, random_seed=2021, save_process=True):
        # get fieldnames
        self.all_fieldnames = get_fieldnames('all')
        self.checked_fieldnames = get_fieldnames('checked')
        self.csv_data_fieldnames = get_fieldnames('csv_data')

        # function to get search space
        self.search_space = get_search_space()
        self._check_required_fieldnames(get_fieldnames('required'))

        # function to prepare config
        self.prepare_config = prepare_config

        # function to build setting, which is the unique identifier of the model
        self.build_setting = build_setting

        # function to build config dict, which is the data stored in the file
        self.build_config_dict_ori = build_config_dict

        # function to get tags
        self.get_tags = get_tags

        # function to check if we need to jump the experiment
        self.check_jump_experiment = check_jump_experiment

        # config data to be jumped
        self.jump_csv_file_path = jump_csv_file_path

        # config data to be stored in other processes
        self.csv_file_path_format = csv_file_path_format

        # the maximum index of the processes
        self.max_process_index = process_number - 1

        # random seed
        self.seed = random_seed

        # whether to save process
        self.save_process = save_process

        # init experiment
        self.Exp = None

    def _check_required_fieldnames(self, fieldnames):
        for fieldname in fieldnames:
            if fieldname not in self.all_fieldnames:
                raise ValueError(f'The fieldname {fieldname} is not in the all fieldnames!')

    def config_optimizer_settings(self, jump_csv_file_path=None, csv_file_path_format=None, max_process_index=None,
                                  seed=None):
        if jump_csv_file_path is not None:
            self.jump_csv_file_path = jump_csv_file_path
        if csv_file_path_format is not None:
            self.csv_file_path_format = csv_file_path_format
        if max_process_index is not None:
            self.max_process_index = max_process_index
        if seed is not None:
            self.seed = seed

    def get_optimizer_settings(self):
        return {
            'jump_csv_file_path': self.jump_csv_file_path,
            'csv_file_path_format': self.csv_file_path_format,
            'process_number': self.max_process_index + 1,
            'random_seed': self.seed,
            'search_space': self.search_space,
            'all_fieldnames': self.all_fieldnames,
            'checked_fieldnames': self.checked_fieldnames,
            'csv_data_fieldnames': self.csv_data_fieldnames,
        }

    def get_csv_file_path(self):
        """
        get the path of the specific data
        """
        csv_file_path = 'data.csv'

        fieldnames = self.csv_data_fieldnames
        search_space = self.search_space

        format_csv_file_path = csv_file_path[:-4]
        for key in fieldnames:
            if search_space[key]['_type'] == 'single':
                value = search_space[key]['_value']
            else:
                raise NotImplementedError
            format_csv_file_path = format_csv_file_path + '_' + str(value)
        format_csv_file_path = format_csv_file_path + '.csv'

        return format_csv_file_path

    def start_search(self, _process_index=0, _try_model=True, _force=False):
        # check the index of the process
        if _process_index > self.max_process_index or _process_index < 0:
            raise ValueError(f'The index of the process {_process_index} is out of range!')

        # init the search space
        search_space = self.search_space

        # init the name of local data file and jumped data file
        if _process_index == 0:
            _csv_file_path = self.get_csv_file_path()
        else:
            _csv_file_path = self.csv_file_path_format.format(_process_index)
        _jump_csv_file_path = self.jump_csv_file_path

        # init the head of local data file and jumped data file
        self._init_header(_csv_file_path)
        self._init_header(_jump_csv_file_path)

        # load config list in local data file and jumped data file
        if _process_index == 0:
            config_list = self._get_config_list(_csv_file_path)
        else:
            config_list = self._get_config_list(_csv_file_path)
            _ = self._get_config_list(self.get_csv_file_path())
            config_list.extend(_)
        jump_config_list = self._get_config_list(_jump_csv_file_path)

        # build parameters to be optimized from _search_space
        params = {}
        for key in search_space:
            params[key] = None  # _search_space[key]['_value']

        # build range for parameters from _search_space
        ranges = {}
        for key in search_space:
            if search_space[key]['_type'] == 'single':
                ranges[key] = [search_space[key]['_value']]
            elif search_space[key]['_type'] == 'choice':
                ranges[key] = search_space[key]['_value']
            else:
                raise ValueError(f'The type of {key} is not supported!')

        # generate all possible combinations of values within the specified ranges
        combinations = list(product(*[ranges[param] for param in params]))

        # filter combinations with the known rules and invert combination to parameters
        parameters = self._filter_combinations(combinations, params, jump_config_list, config_list, _jump_csv_file_path,
                                               _process_index, try_model=_try_model, force=_force)

        # equally distribute the parameters according to the number of processes
        # parameters = parameters[_process_index::(max_process_index + 1)]: It's in the order of the loops.
        # It's in the order in which they are arranged.
        parameters = self._distribute_parameters(parameters, _process_index)

        # find total times
        total_times = len(parameters)
        print(f'Start total {total_times} experiments:\n')

        # iterate through the combinations and start searching by enumeration
        _time = 1
        finish_time = 0
        for parameter in parameters:
            # prepare config: parse launch parameters and load default config
            args = self.prepare_config(parameter)

            # get the experiment type
            self._init_experiment(args.task_name)

            # create a dict to store the configuration values
            config = self._build_config_dict(args)

            # start experiment
            mse, mae, acc = self._start_experiment(args, parameter, config, try_model=False,
                                                   first_process_and_first_exp=(_process_index == 0 and _time == 1))

            # load criteria data
            config['mse'] = mse
            config['mae'] = mae
            config['acc'] = acc

            # save data if in training
            if parameter['is_training'] == 1:
                self._save_config_dict(_csv_file_path, config)

            print(f'>>>>>>> We have finished {_time}/{total_times}! >>>>>>>>>>>>>>>>>>>>>>>>>>\n')
            finish_time = finish_time + 1
            _time = _time + 1

        print(f"We have finished {finish_time} times, {total_times} times in total!")

    def _init_experiment(self, task_name):
        # fix random seed
        self._fix_random_seed()

        # build experiment
        if task_name == 'long_term_forecast':
            self.Exp = Exp_Long_Term_Forecast
        elif task_name == 'short_term_forecast':
            self.Exp = Exp_Short_Term_Forecast
        elif task_name == 'imputation':
            self.Exp = Exp_Imputation
        elif task_name == 'anomaly_detection':
            self.Exp = Exp_Anomaly_Detection
        elif task_name == 'classification':
            self.Exp = Exp_Classification

    def _start_experiment(self, _args, _parameter, _config, try_model, first_process_and_first_exp):
        """
        If try_model is True, we will just try this model:
            if this model can work, then return True.
        """
        # valid model if needed
        if try_model:
            exp = self.Exp(_args, try_model=True, save_process=False)
            setting = self.build_setting(_args, 0)
            valid = exp.train(setting, False)
            return valid

        _mse, _mae, _acc = math.inf, math.inf, 0
        if _args.is_training:
            for ii in range(_args.itr):
                # setting record of experiments
                setting = self.build_setting(_args, ii)

                # build the experiment
                exp = self.Exp(_args, try_model=False, save_process=self.save_process)

                # print info of the experiment
                exp.print_content(f'Optimizing params in experiment:{_parameter}')
                exp.print_content(f'Config in experiment:{_config}')

                # start training
                now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                exp.print_content('>>>>>>>{} - start training: {}<<<<<<<'.format(now, setting))
                exp.train(setting, check_folder=(first_process_and_first_exp and ii == 0))

                # start testing
                now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                exp.print_content('>>>>>>>{} - start testing: {}<<<<<<<'.format(now, setting))

                # get
                _mse, _mae, _acc = exp.test(setting, check_folder=(first_process_and_first_exp and ii == 0))

                torch.cuda.empty_cache()
        else:
            # setting record of experiments
            setting = self.build_setting(_args, 0)

            # build the experiment
            exp = self.Exp(_args, try_model=False, save_process=self.save_process)

            # print info of the experiment
            exp.print_content(f'Optimizing params in experiment:{_parameter}')
            exp.print_content(f'Config in experiment:{_config}')

            # start testing
            now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            exp.print_content('>>>>>>>{} - start testing: {}<<<<<<<'.format(now, setting))

            _mse, _mae, _acc = exp.test(setting, check_folder=first_process_and_first_exp)

            torch.cuda.empty_cache()

        return _mse, _mae, _acc

    def _fix_random_seed(self):
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def _filter_combinations(self, _combinations, _params, _jump_config_list, _config_list, _jump_csv_file_path,
                             _process_index, try_model=True, force=False):
        print(f"We are filtering the parameters, please wait util it done to start other processes!")

        if force:
            print(f'We are forced to run the experiments that we have done in the csv data!')

        jump_time = 0
        filtered_parameters = []
        for combination in tqdm(_combinations):
            # invert combination to parameter
            parameter = {param: value for param, value in zip(_params.keys(), combination)}

            # check if we need to jump this experiment according to the known rules
            if self.check_jump_experiment is not None and self.check_jump_experiment(parameter):
                continue

            # prepare config: parse launch parameters and load default config
            args = self.prepare_config(parameter)

            # create a dict to store the configuration values
            config = self._build_config_dict(args)

            # check if the parameters of this experiment need to be jumped
            if self._check_config_data(config, _jump_config_list) and not force:
                continue

            # check if the parameters of this experiment have been done
            if self._check_config_data(config, _config_list) and not force:
                continue

            # check if the model of this experiment can work
            if _process_index == 0 and try_model:
                # check if the parameters of this experiment is improper
                model_can_work = self._start_experiment(args, parameter, config, try_model=True,
                                                        first_process_and_first_exp=False)
                if not model_can_work:
                    self._save_config_dict(_jump_csv_file_path, config)
                    jump_time = jump_time + 1
                    continue

            filtered_parameters.append(parameter)

        if jump_time > 0:
            print(f"We found improper parameters and add {jump_time} experiments into {_jump_csv_file_path}!\n")

        return filtered_parameters

    def _build_config_dict(self, _args):
        config_dict = self.build_config_dict_ori(_args)
        config_dict['setting'] = self.build_setting(_args, 0)
        config_dict['seed'] = self.seed
        if self.get_tags is not None:
            config_dict['model_id'] = config_dict['model_id'] + self.get_tags(_args)

        return config_dict

    def _init_header(self, file_path):
        if not os.path.exists(file_path):
            with open(file_path, 'w', newline='') as csv_file:
                _writer = csv.DictWriter(csv_file, fieldnames=self.all_fieldnames)
                _writer.writeheader()

    def _get_config_list(self, file_path):
        _config_list = []
        with open(file_path, 'r') as csv_file:
            reader = csv.DictReader(csv_file, fieldnames=self.all_fieldnames)
            for row in reader:
                _config_list.append(row)
        return _config_list

    def _check_config_data(self, config_data, _config_list):
        for _config in _config_list:
            if all(_config[field] == str(config_data[field]) for field in self.checked_fieldnames):
                return True
        return False

    def _save_config_dict(self, file_path, _config):
        # delete the fieldnames in _config that not in _fieldnames
        for key in list(_config.keys()):
            if key not in self.all_fieldnames:
                del _config[key]

        with open(file_path, 'a', newline='') as csvfile:
            _writer = csv.DictWriter(csvfile, fieldnames=self.all_fieldnames)
            _writer.writerow(_config)

    def _distribute_parameters(self, _parameters, _process_index):
        # Calculate the number of parameters per process
        processes_number = self.max_process_index + 1
        combinations_per_process = len(_parameters) // processes_number
        remainder = len(_parameters) % processes_number

        # Initialize variables to keep track of the current index
        current_index = 0

        # Iterate over each process
        for i in range(processes_number):
            # Calculate the start and end indices for the current process
            start_index = current_index
            end_index = start_index + combinations_per_process + (1 if i < remainder else 0)

            # Get the parameters for the current process
            process_parameters = _parameters[start_index:end_index]

            # Update the current index for the next process
            current_index = end_index

            # Return parameters for the current process
            if i == _process_index:
                if _process_index != 0:
                    # Add process index into the checkpoint path to avoid the load dict conflict
                    for parameter in process_parameters:
                        # check if the checkpoints path is in the parameter
                        if 'checkpoints' in parameter.keys():
                            if parameter['checkpoints'].endswith('/'):
                                parameter['checkpoints'] = parameter['checkpoints'][:-1]
                                parameter['checkpoints'] = parameter['checkpoints'] + f'_{_process_index}/'
                            else:
                                parameter['checkpoints'] = parameter['checkpoints'] + f'_{_process_index}'
                        else:
                            parameter['checkpoints'] = f'./checkpoints_{_process_index}/'
                return process_parameters

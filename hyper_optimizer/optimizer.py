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
    def __init__(self, script_mode, models, prepare_config, build_setting, build_config_dict, get_fieldnames,
                 get_search_space, get_model_id_tags=None, check_jump_experiment=None):
        # core settings
        self.script_mode = script_mode  # script mode

        # core functions
        self.prepare_config = prepare_config  # function to prepare config
        self.build_setting = build_setting  # function to build setting, which is the unique identifier of the model
        self.build_config_dict_ori = build_config_dict  # function to build config dict - the data to be stored in files
        self.get_tags = get_model_id_tags  # function to get tags

        # all mode settings
        self.seed = 2021  # random seed
        self.add_tags = []  # added tags in the model id
        self.jump_csv_file = 'jump_data.csv'  # config data to be jumped
        self.data_csv_file_format = 'data_{}.csv'  # config data to be stored in other processes
        self.scan_all_csv = False
        self.max_process_index = 0  # the maximum index of the processes
        self.save_process = True  # whether to save process

        # init experiment and parameters
        self.Exp = None
        self.parameters = None

        if not self.script_mode:
            # non script mode settings
            # models
            self.models = models

            # search spaces
            self.search_spaces = {}
            for model in self.models:
                self.search_spaces[model] = get_search_space(model)
            self._check_required_fieldnames(get_fieldnames('required'))

            # fieldnames
            self.all_fieldnames = get_fieldnames('all')  # all fieldnames
            self.checked_fieldnames = get_fieldnames('checked')  # checked fieldnames

            # non script mode functions
            self.check_jump_experiment = check_jump_experiment  # check if we need to jump the experiment

    def _check_required_fieldnames(self, fieldnames):
        for model in self.models:
            search_space = self.search_spaces[model]
            # check if the required fieldnames are in the search space
            for fieldname in fieldnames:
                if fieldname == 'task_name':
                    if search_space[fieldname]['_type'] != 'single':
                        raise ValueError(f'The type of {fieldname} should be single!')
                if fieldname not in search_space.keys():
                    raise ValueError(f'The required fieldname {fieldname} is not in the search space!')

    def config_optimizer_settings(self, random_seed=None, add_tags=None, jump_csv_file=None, data_csv_file_format=None,
                                  scan_all_csv=None, process_number=None, save_process=None):
        if random_seed is not None:
            self.seed = random_seed
        if add_tags is not None:
            self.add_tags = add_tags
        if jump_csv_file is not None:
            self.jump_csv_file = jump_csv_file
        if data_csv_file_format is not None:
            self.data_csv_file_format = data_csv_file_format
        if scan_all_csv is not None:
            self.scan_all_csv = scan_all_csv
        if process_number is not None:
            self.max_process_index = process_number + 1
        if save_process is not None:
            self.save_process = save_process

    def get_optimizer_settings(self):
        core_setting = {
            'script_mode': self.script_mode,
        }

        all_mode_settings = {
            'random_seed': self.seed,
            'add_tags': self.add_tags,
            'jump_csv_file': self.jump_csv_file,
            'data_csv_file_format': self.data_csv_file_format,
            'scan_all_csv': self.scan_all_csv,
            'process_number': self.max_process_index + 1,
            'save_process': self.save_process,
        }

        non_script_mode_settings = {
            'models': self.models,
            'search_spaces': self.search_spaces,
            'all_fieldnames': self.all_fieldnames,
            'checked_fieldnames': self.checked_fieldnames,
        }

        if self.script_mode:
            return {**core_setting, **all_mode_settings}
        else:
            return {**core_setting, **all_mode_settings, **non_script_mode_settings}

    def get_csv_file_path(self, _process_index=0, _jump_data=False):
        """
        get the path of the specific data
        """
        # get all possible parameters
        if self.parameters is None:
            self.parameters = self._parse_search_space()
        parameters = self.parameters

        # get csv file name for other process under the root folder
        if not _jump_data and _process_index != 0:
            csv_file = self.data_csv_file_format.format(_process_index)
            return csv_file

        if not _jump_data:
            # get csv file name for core process
            csv_file = 'data.csv'
        else:
            # get csv file name for jump data
            csv_file = self.jump_csv_file

        # get the task name
        task_name = parameters[0]['task_name']
        file_path = f'./data/{task_name}'

        # check the folder path
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        csv_file_path = f'{file_path}/{csv_file}'
        return csv_file_path

    # noinspection DuplicatedCode
    def output_script(self, _data):
        # get all possible parameters
        if self.parameters is None:
            self.parameters = self._parse_search_space()
        parameters = self.parameters

        # get the task name
        task_name = parameters[0]['task_name']

        # filter the parameters according to the model
        model_parameters = {}
        for parameter in parameters:
            model = parameter['model']
            if model not in model_parameters:
                model_parameters[model] = []
            model_parameters[model].append(parameter)

        # save the script of each model
        for model in model_parameters:
            parameters = model_parameters[model]

            # get the path of the specific script
            script_path = f'./scripts/{task_name}/{_data}_script/{model}.sh'

            # create the folder of the specific script
            if not os.path.exists(os.path.dirname(script_path)):
                os.makedirs(os.path.dirname(script_path))

            # get the time
            t = time.localtime()
            _run_time = time.strftime('%Y-%m-%d %H:%M:%S', t)

            # write the script
            if not os.path.exists(script_path):
                with open(script_path, 'w') as f:
                    # write the header of the script
                    f.write(f'# This script is created by hyper_optimizer at {_run_time}.\n')
                    f.write('\n')
                    f.write('export CUDA_VISIBLE_DEVICES=1\n')
                    f.write('\n')
                    f.write('model_name=' + f'{model}' + '\n')
                    f.write('\n')
                    # write the content of the script
                    for parameter in parameters:
                        f.write(f'# This segment is writen at {_run_time}.\n')
                        f.write('python -u run.py \\\n')
                        for key in parameter:
                            if key == 'model':
                                f.write('\t--model $model_name\\\n')
                            f.write(f'\t--{key} {parameter[key]} \\\n')
                        f.write('\n')
            else:
                # write the content of the script
                with open(script_path, 'a') as f:
                    for parameter in parameters:
                        f.write(f'# This segment is writen at {_run_time}.\n')
                        f.write('python -u run.py \\\n')
                        for key in parameter:
                            if key == 'model':
                                f.write('\t--model $model_name\\\n')
                            f.write(f'\t--{key} {parameter[key]} \\\n')
                        f.write('\n')

        # print the info of the successful output
        print(f'We successfully output the scripts in ./scripts/{task_name}/{_data}_script/')

        # print the command to run the script
        print(f'You can run the following command to run the script:')
        print(f'bash ./scripts/{task_name}/{_data}_script/*.sh')

    def start_search(self, _process_index=0, _try_model=True, _force=False):
        # run directly under script mode
        if self.script_mode:
            # parse launch parameters and load default config
            args = self.prepare_config(None, True)

            # get the experiment type
            self._init_experiment(args.task_name)

            # create a dict to store the configuration values
            config = self._build_config_dict(args)

            # start experiment
            self._start_experiment(args, None, config, _try_model=False, _check_folder=False)

        # check the index of the process
        if _process_index > self.max_process_index or _process_index < 0:
            raise ValueError(f'The index of the process {_process_index} is out of range!')

        # init the name of local data file and jumped data file
        _csv_file_path = self.get_csv_file_path(_process_index=_process_index, _jump_data=False)
        _jump_csv_file_path = self.get_csv_file_path(_process_index=_process_index, _jump_data=True)

        # init the head of local data file and jumped data file
        self._init_header(_csv_file_path)
        self._init_header(_jump_csv_file_path)

        # load config list in local data file and jumped data file
        if _process_index == 0:
            config_list = self._get_config_list(_csv_file_path, scan_all_csv=self.scan_all_csv)
        else:
            config_list = self._get_config_list([_csv_file_path, self.get_csv_file_path(_process_index=0)],
                                                scan_all_csv=self.scan_all_csv)
        jump_config_list = self._get_config_list(_jump_csv_file_path)

        # get all possible parameters
        parameters = self.parameters

        # filter combinations with the known rules or trying models
        filtered_parameters = self._filter_parameters(parameters, jump_config_list, config_list, _jump_csv_file_path,
                                                      _process_index, try_model=_try_model, force=_force)

        # equally distribute the parameters according to the number of processes
        # parameters = parameters[_process_index::(max_process_index + 1)]: It's in the order of the loops.
        # It's in the order in which they are arranged.
        process_parameters = self._distribute_parameters(filtered_parameters, _process_index)

        # find total times
        total_times = len(process_parameters)
        print(f'Start total {total_times} experiments:\n')

        # iterate through the combinations and start searching by enumeration
        _time = 1
        finish_time = 0
        for parameter in process_parameters:
            # parse launch parameters and load default config
            args = self.prepare_config(parameter)

            # get the experiment type
            self._init_experiment(args.task_name)

            # create a dict to store the configuration values
            config = self._build_config_dict(args)

            # start experiment
            mse, mae, acc, run_time, setting = self._start_experiment(args, parameter, config, False,
                                                                      (_process_index == 0 and _time == 1))

            # load criteria data
            config['mse'] = mse
            config['mae'] = mae
            config['acc'] = acc

            # load setting and run time
            config['setting'] = setting
            config['run_time'] = run_time

            # save data if in training
            if parameter['is_training'] == 1:
                self._save_config_dict(_csv_file_path, config)

            print(f'>>>>>>> We have finished {_time}/{total_times}! >>>>>>>>>>>>>>>>>>>>>>>>>>\n')
            finish_time = finish_time + 1
            _time = _time + 1

        print(f"We have finished {finish_time} times, {total_times} times in total!")

    def _parse_search_space(self):
        _parameters = []

        for model in self.models:
            search_space = self.search_spaces[model]

            # build parameters to be optimized from _search_space
            _params = {}
            for key in search_space:
                _params[key] = None  # _search_space[key]['_value']

            # get range of parameters for parameters from _search_space
            _parameters_range = {}
            for key in search_space:
                if search_space[key]['_type'] == 'single':
                    _parameters_range[key] = [search_space[key]['_value']]
                elif search_space[key]['_type'] == 'choice':
                    _parameters_range[key] = search_space[key]['_value']
                else:
                    raise ValueError(f'The type of {key} is not supported!')

            # generate all possible combinations of parameters within the specified ranges
            _combinations = list(product(*[_parameters_range[param] for param in _params]))

            # invert combinations to parameters and collect them
            for combination in _combinations:
                parameter = {param: value for param, value in zip(_params.keys(), combination)}
                _parameters.append(parameter)

        return _parameters

    def _filter_parameters(self, _parameters, _jump_config_list, _config_list, _jump_csv_file_path,
                           _process_index, try_model=True, force=False, print_info=True):
        if print_info:
            print(f"We are filtering the parameters, please wait util it done to start other processes!")

            if force:
                print(f'We are forced to run the experiments that we have done in the csv data!')

        jump_time = 0
        filtered_parameters = []
        if try_model is True:
            _parameters = tqdm(_parameters)
        for parameter in _parameters:
            # check if we need to jump this experiment according to the known rules
            if self.check_jump_experiment is not None and self.check_jump_experiment(parameter):
                continue

            # parse launch parameters and load default config
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
                model_can_work = self._start_experiment(args, parameter, config, _try_model=True, _check_folder=False)
                if not model_can_work:
                    self._save_config_dict(_jump_csv_file_path, config)
                    jump_time = jump_time + 1
                    continue

            filtered_parameters.append(parameter)

        if jump_time > 0:
            print(f"We found improper parameters and add {jump_time} experiments into {_jump_csv_file_path}!\n")

        return filtered_parameters

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

    def _start_experiment(self, _args, _parameter, _config, _try_model, _check_folder):
        """
        If try_model is True, we will just try this model:
            if this model can work, then return True.
        """
        # init time and setting
        t = time.localtime()
        _run_time = time.strftime('%Y-%m-%d %H-%M-%S', t)
        _setting = self.build_setting(_args, _run_time)

        # try model if needed
        if _try_model:
            exp = self.Exp(_args, try_model=True, save_process=False)
            valid = exp.train(_setting, False)
            return valid

        _mse, _mae, _acc = math.inf, math.inf, 0
        if _args.is_training:
            # setting record of experiments
            _setting = self.build_setting(_args, _run_time)

            # build the experiment
            exp = self.Exp(_args, try_model=False, save_process=self.save_process)

            # print info of the experiment
            if _parameter is not None:
                exp.print_content(f'Optimizing params in experiment:{_parameter}')
            exp.print_content(f'Config in experiment:{_config}')

            # start training
            exp.print_content('>>>>>>>({}) start training: {}<<<<<<<'.format(_run_time, _setting))
            exp.train(_setting, check_folder=_check_folder)

            # start testing
            exp.print_content('>>>>>>>({}) start testing: {}<<<<<<<'.format(_run_time, _setting))
            _mse, _mae, _acc = exp.test(_setting, check_folder=_check_folder)

            # clean cuda cache
            torch.cuda.empty_cache()
        else:
            # build the experiment
            exp = self.Exp(_args, try_model=False, save_process=self.save_process)

            # print info of the experiment
            if _parameter is not None:
                exp.print_content(f'Optimizing params in experiment:{_parameter}')
            exp.print_content(f'Config in experiment:{_config}')

            # start testing
            exp.print_content('>>>>>>>({}) start testing: {}<<<<<<<'.format(_run_time, _setting))
            _mse, _mae, _acc = exp.test(_setting, check_folder=_check_folder)

            # clean cuda cache
            torch.cuda.empty_cache()

        return _mse, _mae, _acc, _run_time, _setting

    def _fix_random_seed(self):
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def _build_config_dict(self, _args):
        config_dict = self.build_config_dict_ori(_args)
        config_dict['seed'] = self.seed
        if self.get_tags is not None:
            config_dict['model_id'] = config_dict['model_id'] + self.get_tags(_args, self.add_tags)

        return config_dict

    def _init_header(self, file_path):
        if not os.path.exists(file_path):
            with open(file_path, 'w', newline='') as csv_file:
                _writer = csv.DictWriter(csv_file, fieldnames=self.all_fieldnames)
                _writer.writeheader()

    def _get_config_list(self, file_paths, scan_all_csv=False):
        if not isinstance(file_paths, list):
            file_paths = [file_paths]

        task_name = self.parameters[0]['task_name']
        root_path = f'./data/{task_name}'
        if scan_all_csv:
            # get all csv file under the path
            for root, dirs, files in os.walk(root_path):
                for file in files:
                    if file == self.jump_csv_file:
                        continue
                    if file.endswith('.csv') and file not in file_paths:
                        file_paths.append(f'{root}/{file}')

        _config_list = []
        for file_path in file_paths:
            _ = []
            with open(file_path, 'r') as csv_file:
                reader = csv.DictReader(csv_file, fieldnames=self.all_fieldnames)
                for row in reader:
                    _.append(row)
            _config_list.extend(_)

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

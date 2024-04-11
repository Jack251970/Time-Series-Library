from hyper_optimizer.basic_settings import prepare_config, build_setting, build_config_dict, set_args, get_fieldnames
from hyper_optimizer.optimizer import HyperOptimizer

if __name__ == "__main__":
    h = HyperOptimizer(True, None,
                       prepare_config, build_setting, build_config_dict, set_args, get_fieldnames, None)
    h.start_search()

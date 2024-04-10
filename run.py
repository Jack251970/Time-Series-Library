from hyper_optimizer.optimizer import HyperOptimizer
from run_optimizer import prepare_config, build_setting, build_config_dict, get_model_id_tags, set_args

if __name__ == "__main__":
    h = HyperOptimizer(True, None,
                       prepare_config, build_setting, build_config_dict, set_args, None, None,
                       get_model_id_tags=get_model_id_tags, check_jump_experiment=None)
    h.start_search()

from hyper_optimizer.optimizer import HyperOptimizer
from run_optimizer import prepare_config, build_setting, build_config_dict, get_model_id_tags

if __name__ == "__main__":
    h = HyperOptimizer(True, prepare_config, build_setting, build_config_dict, None, None,
                       get_model_id_tags=get_model_id_tags)
    h.start_search()

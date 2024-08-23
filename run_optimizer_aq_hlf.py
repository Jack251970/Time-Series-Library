from optimizers.optimizer_qf_hlf import h as ht
from utils.pass_parameter import save_parameter

process_index = 0

if __name__ == "__main__":
    for i in range(1, 10, 1):
        w_mql = i / 10
        save_parameter({'w_mql': w_mql})
        ht.start_search(process_index=process_index)

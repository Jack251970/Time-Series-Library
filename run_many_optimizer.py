from optimizers.optimizer_qsqf import h as hc
from optimizers.optimizer_qsqf_comp import h as hc1

process_index = 0

if __name__ == "__main__":
    hc.start_search(process_index=process_index, shutdown_after_done=False)
    hc1.start_search(process_index=process_index, shutdown_after_done=False)

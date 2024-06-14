from optimizers.optimizer_qf_comp import h
from optimizers.optimizer_qf_comp1 import h as h1

process_index = 0

if __name__ == "__main__":
    h.start_search(process_index=process_index)
    h1.start_search(process_index=process_index, shutdown_after_done=False)

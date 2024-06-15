from optimizers.optimizer_qf_test import h as ht
from optimizers.optimizer_qsqf_test import h as ht1

process_index = 0

if __name__ == "__main__":
    ht.start_search(process_index=process_index, force_test=True)
    ht1.start_search(process_index=process_index, force_test=True)

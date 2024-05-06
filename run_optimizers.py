from optimizers.optimizer_qf_test import h as ht

from optimizers.optimizer_qf_v2_1 import h as h1
from optimizers.optimizer_qf_v2_2 import h as h2
from optimizers.optimizer_qf_v2_3 import h as h3
from optimizers.optimizer_qf_v2_4 import h as h4
from optimizers.optimizer_qf_v2_5 import h as h5

process_index = 0

if __name__ == "__main__":
    ht.start_search(process_index=process_index)

    h1.start_search(process_index=process_index)
    h2.start_search(process_index=process_index)
    h3.start_search(process_index=process_index)
    h4.start_search(process_index=process_index)
    h5.start_search(process_index=process_index)

from optimizers.optimizer_qf_v2_1 import h as h1
from optimizers.optimizer_qf_v2_2 import h as h2
from optimizers.optimizer_qf_v2_3 import h as h3
from optimizers.optimizer_qf_v2_4 import h as h4
from optimizers.optimizer_qf_v2_5 import h as h5
from optimizers.optimizer_qf_v2_6 import h as h6
from optimizers.optimizer_qf_v2_7 import h as h7
from optimizers.optimizer_qf_v2_8 import h as h8

from optimizers.optimizer_transformer import h as hf
from optimizers.optimizer_qsqf_c import h as hc

process_index = 0

if __name__ == "__main__":
    h1.start_search(process_index=process_index)
    h2.start_search(process_index=process_index)
    # h3.start_search(process_index=process_index)
    # h4.start_search(process_index=process_index)
    # h5.start_search(process_index=process_index)
    # h6.start_search(process_index=process_index)
    # h7.start_search(process_index=process_index)
    # h8.start_search(process_index=process_index)

    # hf.start_search(process_index=process_index)
    # hc.start_search(process_index=process_index)

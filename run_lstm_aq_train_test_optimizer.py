from optimizers.optimizer_qf_train_test import h as h

process_index = 0

if __name__ == "__main__":
    h.start_search(process_index=process_index, shutdown_after_done=False)

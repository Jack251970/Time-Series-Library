from optimizers.optimizer_qf_hlf import h as ht

process_index = 0
file = 'temp'
w_mql = 0

if __name__ == "__main__":
    for i in range(1, 10, 1):
        w_mql = i / 10
        with open(file, 'w') as f:
            f.write(str(w_mql))
        ht.start_search(process_index=process_index)

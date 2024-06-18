def print_args(args, print_func=print):
    print_func("\033[1m" + "Basic Config" + "\033[0m")
    print_func(f'  {"Task Name:":<20}{args.task_name:<20}{"Is Training:":<20}{args.is_training:<20}')
    print_func(f'  {"Model ID:":<20}{args.model_id:<20}{"Model:":<20}{args.model:<20}')
    print_func()

    print_func("\033[1m" + "Data Loader" + "\033[0m")
    print_func(f'  {"Data:":<20}{args.data:<20}{"Root Path:":<20}{args.root_path:<20}')
    print_func(f'  {"Data Path:":<20}{args.data_path:<20}{"Features:":<20}{args.features:<20}')
    print_func(f'  {"Target:":<20}{args.target:<20}{"Freq:":<20}{args.freq:<20}')
    print_func(f'  {"Lag:":<20}{args.lag:<20}{"Scaler:":<20}{args.scaler:<20}')
    print_func(f'  {"Reindex:":<20}{args.reindex:<20}{"Reindex Tolerance:":<20}{args.reindex_tolerance:<20}')
    print_func(f'  {"Pin Memory:":<20}{args.pin_memory:<20}')
    print_func()

    if args.task_name in ['long_term_forecast', 'short_term_forecast']:
        print_func("\033[1m" + "Forecasting Task" + "\033[0m")
        print_func(f'  {"Seq Len:":<20}{args.seq_len:<20}{"Label Len:":<20}{args.label_len:<20}')
        print_func(f'  {"Pred Len:":<20}{args.pred_len:<20}{"Seasonal Patterns:":<20}{args.seasonal_patterns:<20}')
        print_func(f'  {"Inverse:":<20}{args.inverse:<20}')
        print_func()

    if args.task_name == 'imputation':
        print_func("\033[1m" + "Imputation Task" + "\033[0m")
        print_func(f'  {"Mask Rate:":<20}{args.mask_rate:<20}')
        print_func()

    if args.task_name == 'anomaly_detection':
        print_func("\033[1m" + "Anomaly Detection Task" + "\033[0m")
        print_func(f'  {"Anomaly Ratio:":<20}{args.anomaly_ratio:<20}')
        print_func()

    print_func("\033[1m" + "Model Parameters" + "\033[0m")
    print_func(f'  {"Top k:":<20}{args.top_k:<20}{"Num Kernels:":<20}{args.num_kernels:<20}')
    print_func(f'  {"Enc In:":<20}{args.enc_in:<20}{"Dec In:":<20}{args.dec_in:<20}')
    print_func(f'  {"C Out:":<20}{args.c_out:<20}{"d model:":<20}{args.d_model:<20}')
    print_func(f'  {"n heads:":<20}{args.n_heads:<20}{"e layers:":<20}{args.e_layers:<20}')
    print_func(f'  {"d layers:":<20}{args.d_layers:<20}{"d FF:":<20}{args.d_ff:<20}')
    print_func(f'  {"Moving Avg:":<20}{args.moving_avg:<20}{"Series Decomp Mode:":<20}{args.series_decomp_mode:<20}')
    print_func(f'  {"Factor:":<20}{args.factor:<20}{"Distil:":<20}{args.distil:<20}')
    print_func(f'  {"Dropout:":<20}{args.dropout:<20}{"Embed:":<20}{args.embed:<20}')
    print_func(f'  {"Activation:":<20}{args.activation:<20}{"Output Attention:":<20}{args.output_attention:<20}')
    print_func(f'  {"Channel Independence:":<20}{args.channel_independence:<20}')
    print_func()

    print_func("\033[1m" + "Run Parameters" + "\033[0m")
    print_func(f'  {"Num Workers:":<20}{args.num_workers:<20}{"Itr:":<20}{args.itr:<20}')
    print_func(f'  {"Train Epochs:":<20}{args.train_epochs:<20}{"Batch Size:":<20}{args.batch_size:<20}')
    print_func(f'  {"Patience:":<20}{args.patience:<20}{"Learning Rate:":<20}{args.learning_rate:<20}')
    print_func(f'  {"Des:":<20}{args.des:<20}{"Loss:":<20}{args.loss:<20}')
    print_func(f'  {"Lradj:":<20}{args.lradj:<20}{"Use Amp:":<20}{args.use_amp:<20}')
    print_func()

    print_func("\033[1m" + "GPU" + "\033[0m")
    print_func(f'  {"Use GPU:":<20}{args.use_gpu:<20}{"GPU:":<20}{args.gpu:<20}')
    print_func(f'  {"Use Multi GPU:":<20}{args.use_multi_gpu:<20}{"Devices:":<20}{args.devices:<20}')
    print_func()

    print_func("\033[1m" + "De-stationary Projector Params" + "\033[0m")
    p_hidden_dims_str = ', '.join(map(str, args.p_hidden_dims))
    print_func(f'  {"P Hidden Dims:":<20}{p_hidden_dims_str:<20}{"P Hidden Layers:":<20}{args.p_hidden_layers:<20}')
    print_func()

    print_func("\033[1m" + "LSTM Params" + "\033[0m")
    print_func(f'  {"LSTM Hidden Size:":<20}{args.lstm_hidden_size:<20}{"LSTM Layers:":<20}{args.lstm_layers:<20}')
    print_func()

    print_func("\033[1m" + "Spline Functions Params" + "\033[0m")
    print_func(f'  {"Num Spline:":<20}{args.num_spline:<20}{"Sample Times:":<20}{args.sample_times:<20}')
    print_func()

    print_func("\033[1m" + "Custom Params" + "\033[0m")
    print_func(f'  {"Custom Params:":<20}{args.custom_params:<20}')
    print_func()

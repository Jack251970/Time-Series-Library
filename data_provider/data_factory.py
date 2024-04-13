from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader
}


cached_data = {'train': None, 'val': None, 'test': None}


def build_argument(args):
    argument = {
        'data': args.data, 'embed': args.embed, 'task_name': args.task_name, 'batch_size': args.batch_size,
        'freq': args.freq, 'root_path': args.root_path, 'seq_len': args.seq_len, 'reindex': args.reindex,
        'reindex_tolerance': args.reindex_tolerance, 'num_workers': args.num_workers,
        'data_path': args.data_path, 'label_len': args.label_len, 'pred_len': args.pred_len,
        'features': args.features, 'target': args.target, 'scaler': args.scaler, 'lag': args.lag,
        'seasonal_patterns': args.seasonal_patterns
    }
    return argument


def cache_dataloader(flag, argument, data_set, data_loader, new_indexes):
    global cached_data
    cached_data[flag] = (argument, (data_set, data_loader, new_indexes))


def get_cached_dataloader(argument, flag):
    global cached_data
    if cached_data[flag] is None:
        return None

    # check if the arguments are the same
    cached_argument, cached_data_set_data_loader_new_indexes = cached_data[flag]
    for key in argument.keys():
        if argument[key] != cached_argument[key]:
            return None

    # return the cached dataloader with the same parameters
    return cached_data_set_data_loader_new_indexes


def data_provider(args, flag, new_indexes=None, cache_data=True):
    if cache_data:
        # build argument
        argument = build_argument(args)

        # check if the dataloader is cached
        _cached_data = get_cached_dataloader(argument, flag)
        if _cached_data is not None:
            data_set, data_loader, _new_indexes = _cached_data
            return data_set, data_loader, f"{args.data}: {flag} {len(data_set)} (cached)", _new_indexes

    # get data class
    Data = data_dict[args.data]

    # get data information
    timeenc = 0 if args.embed != 'timeF' else 1
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            # batch_size = 1  # bsz=1 for evaluation
            batch_size = args.batch_size  # fasten the test process
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    # return dataset, data loader and information
    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        # reindex if needed
        if args.reindex:
            if new_indexes is None:
                if flag == 'train':
                    new_indexes = data_set.get_new_indexes(tolerance=args.reindex_tolerance)
                else:
                    new_indexes = Data(
                        root_path=args.root_path,
                        win_size=args.seq_len,
                        flag='train',  # use train dataset to get more detailed information from more data
                    ).get_new_indexes(tolerance=args.reindex_tolerance)
            data_set.set_new_indexes(new_indexes)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True,
            persistent_workers=True)
        if cache_data:
            cache_dataloader(flag, argument, data_set, data_loader, new_indexes)
        return data_set, data_loader, f"{args.data}: {flag} {len(data_set)}", new_indexes
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )
        # reindex if needed
        if args.reindex:
            if new_indexes is None:
                if flag == 'train':
                    new_indexes = data_set.get_new_indexes(tolerance=args.reindex_tolerance)
                else:
                    new_indexes = Data(
                        root_path=args.root_path,
                        flag='train',
                    ).get_new_indexes(tolerance=args.reindex_tolerance)
            data_set.set_new_indexes(new_indexes)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len,),
            pin_memory=True,
            persistent_workers=True
        )
        if cache_data:
            cache_dataloader(flag, argument, data_set, data_loader, new_indexes)
        return data_set, data_loader, f"{args.data}: {flag} {len(data_set)}", new_indexes
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            scale=True,
            scaler=args.scaler,
            timeenc=timeenc,
            freq=freq,
            lag=args.lag,
            seasonal_patterns=args.seasonal_patterns
        )
        # reindex if needed
        if args.reindex:
            if new_indexes is None:
                if flag == 'train':
                    new_indexes = data_set.get_new_indexes(tolerance=args.reindex_tolerance)
                else:
                    new_indexes = Data(
                        root_path=args.root_path,
                        data_path=args.data_path,
                        flag='train',
                        size=[args.seq_len, args.label_len, args.pred_len],
                        features=args.features,
                        target=args.target,
                        scale=True,
                        scaler=args.scaler,
                        timeenc=timeenc,
                        freq=freq,
                        lag=args.lag,
                        seasonal_patterns=args.seasonal_patterns
                    ).get_new_indexes(tolerance=args.reindex_tolerance)
            data_set.set_new_indexes(new_indexes)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True,
            persistent_workers=True
        )
        if cache_data:
            cache_dataloader(flag, argument, data_set, data_loader, new_indexes)
        return data_set, data_loader, f"{args.data}: {flag} {len(data_set)}", new_indexes

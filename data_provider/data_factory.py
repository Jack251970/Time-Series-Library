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


cached_data = {'train': [], 'val': [], 'test': []}


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


def cache_dataloader(flag, argument, data_set, new_indexes):
    global cached_data
    cached_data[flag].append((argument, (data_set, new_indexes)))


def get_cached_dataloader(argument, flag):
    global cached_data
    if len(cached_data[flag]) == 0:
        return None

    # check if the arguments are the same
    for cached_argument, cached_data_set_new_indexes in cached_data[flag]:
        flag = True
        for key in argument.keys():
            if argument[key] != cached_argument[key]:
                flag = False
                break
        if flag:
            # return the cached dataset with the same parameters
            return cached_data_set_new_indexes

    return None


def data_provider(args, data_flag, enter_flag, new_indexes=None, cache_data=True):
    # prepare for cache
    argument = None
    cached_data_set = None
    cached_new_indexes = None
    if cache_data:
        # build argument
        argument = build_argument(args)

        # check if the dataloader is cached
        _cached_data = get_cached_dataloader(argument, data_flag)
        if _cached_data is not None:
            cached_data_set, cached_new_indexes = _cached_data

    # get data class
    Data = data_dict[args.data]

    # get data information
    timeenc = 0 if args.embed != 'timeF' else 1
    pin_memory = args.pin_memory

    shuffle_flag = False if data_flag == 'test' else True
    drop_last = True
    batch_size = args.batch_size
    freq = args.freq

    if enter_flag != 'train':
        pin_memory = False

    # return dataset, data loader and information
    if args.task_name == 'anomaly_detection':
        drop_last = False
        if cached_data_set is not None:
            data_set, new_indexes = cached_data_set, cached_new_indexes
        else:
            data_set = Data(
                args=args,
                root_path=args.root_path,
                win_size=args.seq_len,
                flag=data_flag,
            )
            # reindex if needed
            if args.reindex:
                if new_indexes is None:
                    if data_flag == 'train':
                        new_indexes = data_set.get_new_indexes(tolerance=args.reindex_tolerance)
                    else:
                        new_indexes = Data(
                            args=args,
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
            pin_memory=pin_memory,
            persistent_workers=True)
        if cache_data:
            cache_dataloader(data_flag, argument, data_set, new_indexes)
        if cached_data_set is not None:
            return data_set, data_loader, f"{args.data}: {data_flag} {len(data_set)} (cached)", new_indexes
        else:
            return data_set, data_loader, f"{args.data}: {data_flag} {len(data_set)}", new_indexes
    elif args.task_name == 'classification':
        drop_last = False
        if cached_data_set is not None:
            data_set, new_indexes = cached_data_set, cached_new_indexes
        else:
            data_set = Data(
                args=args,
                root_path=args.root_path,
                flag=data_flag,
            )
            # reindex if needed
            if args.reindex:
                if new_indexes is None:
                    if data_flag == 'train':
                        new_indexes = data_set.get_new_indexes(tolerance=args.reindex_tolerance)
                    else:
                        new_indexes = Data(
                            args=args,
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
            pin_memory=pin_memory,
            persistent_workers=True
        )
        if cache_data:
            cache_dataloader(data_flag, argument, data_set, new_indexes)
        if cached_data_set is not None:
            return data_set, data_loader, f"{args.data}: {data_flag} {len(data_set)} (cached)", new_indexes
        else:
            return data_set, data_loader, f"{args.data}: {data_flag} {len(data_set)}", new_indexes
    else:
        if args.data == 'm4':
            drop_last = False
        if cached_data_set is not None:
            data_set, new_indexes = cached_data_set, cached_new_indexes
        else:
            data_set = Data(
                args=args,
                root_path=args.root_path,
                data_path=args.data_path,
                flag=data_flag,
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
                    if data_flag == 'train':
                        new_indexes = data_set.get_new_indexes(tolerance=args.reindex_tolerance)
                    else:
                        new_indexes = Data(
                            args=args,
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
            pin_memory=pin_memory,
            persistent_workers=True
        )
        if cache_data:
            cache_dataloader(data_flag, argument, data_set, new_indexes)
        if cached_data_set is not None:
            return data_set, data_loader, f"{args.data}: {data_flag} {len(data_set)} (cached)", new_indexes
        else:
            return data_set, data_loader, f"{args.data}: {data_flag} {len(data_set)}", new_indexes

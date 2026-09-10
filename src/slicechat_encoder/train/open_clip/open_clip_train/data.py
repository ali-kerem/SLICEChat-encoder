import json
from dataclasses import dataclass
from multiprocessing import Value

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def wsi_collate_fn(batch):
    features = [item["features"] for item in batch]
    coords = [item["coords"] for item in batch]
    texts = [item["texts"] for item in batch]

    features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)
    coords = torch.nn.utils.rnn.pad_sequence(coords, batch_first=True, padding_value=-1)
    texts = torch.stack(texts)
    # Create padding_mask (True for padded, False for valid) with shape (B, N)
    # A patch is padded if any of its coordinate values are -1.
    padding_mask = (coords == -1).any(dim=-1).long()

    return {"features": features, "coords": coords, "texts": texts, "padding_masks": padding_mask}


class WSIDataset(Dataset):
    def __init__(self, data_path, data_dir, tokenizer):
        self.data_path = data_path
        self.data_dir = data_dir
        self.data = json.load(open(data_path, "r"))
        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        feature_path = f"{self.data_dir}/features/{data['filename']}.pt"
        coords_path = f"{self.data_dir}/coords/{data['filename']}.pt"
        text = data["caption"]
        
        feature = torch.load(feature_path)
        coords = torch.load(coords_path)
        text = self.preprocess_txt(text)
        return {"features": feature, "coords": coords, "texts": text}


def get_wsi_dataset(args, is_train, tokenizer=None):
    data_path = args.train_data if is_train else args.val_data
    data_dir = args.train_data_dir if is_train else args.val_data_dir
    dataset = WSIDataset(data_path, data_dir, tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset, seed=args.seed) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=wsi_collate_fn,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_data(args, tokenizer=None):
    data = {}

    if args.train_data:
        data["train"] = get_wsi_dataset(
            args, is_train=True, tokenizer=tokenizer)

    if args.val_data:
        data["val"] = get_wsi_dataset(
            args, is_train=False, tokenizer=tokenizer)

    return data

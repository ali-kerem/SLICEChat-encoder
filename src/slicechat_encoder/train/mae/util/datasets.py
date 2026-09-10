import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler


class WSIDataset(Dataset):
    def __init__(self, csv_path, data_path, features_subdir='features', coords_subdir='coords'):
        self.df = pd.read_csv(csv_path)
        self.data_path = data_path
        self.features_subdir = features_subdir
        self.coords_subdir = coords_subdir
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        filename = f"{self.df.iloc[idx]['filename']}.pt"
        
        feature_path = f"{self.data_path}/{self.features_subdir}/{filename}"
        coords_path = f"{self.data_path}/{self.coords_subdir}/{filename}"
        
        features = torch.load(feature_path)
        coords = torch.load(coords_path)
        
        return {"features": features, "coords": coords}


def wsi_mae_collate_fn(batch):
    features = [item["features"] for item in batch]
    coords = [item["coords"] for item in batch]
    
    features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)
    coords = torch.nn.utils.rnn.pad_sequence(coords, batch_first=True, padding_value=-1)
    
    # Create padding_mask (True for padded, False for valid)
    padding_mask = (coords == -1).any(dim=-1).long()
    
    return {
        "features": features, 
        "coords": coords, 
        "padding_masks": padding_mask
    }


def get_wsi_mae_dataset(args, is_train):
    dataset = WSIDataset(csv_path=args.csv_path, data_path=args.data_path)
    
    sampler = DistributedSampler(dataset, seed=args.seed) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=wsi_mae_collate_fn,
    )
    
    return dataloader

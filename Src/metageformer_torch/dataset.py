from torch.utils.data import Dataset, DataLoader
import anndata as ad
import os


class AnnDataset(Dataset):
    def __init__(self, adata):
        self.adata = adata

    def __len__(self):
        return self.adata.shape[0]

    def __getitem__(self, idx):
        return self.adata[idx]


def data_collection(data):
    return ad.concat(data)


def _load_dataset_from_adata(adata, batch_size=128, shuffle=False):
    dataset = AnnDataset(adata)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=data_collection)
    return dataloader


def load_dataset_from_dir_NMR(data_path: str, batch_size: int = 128, device: str = 'cpu'):
    train_data = ad.read_h5ad(os.path.join(data_path, 'train.h5ad'))
    val_data = ad.read_h5ad(os.path.join(data_path, 'val.h5ad'))

    train_dataloader = _load_dataset_from_adata(train_data, batch_size=batch_size, shuffle=False)
    val_dataloader = _load_dataset_from_adata(val_data, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, train_data, val_data


def load_dataset_from_adata_NMR(data_path: str, shuffle: bool = False, batch_size: int = 128, specify_eid=None, specific_col=None, device: str = 'cpu'):
    adata = ad.read_h5ad(data_path)
    if isinstance(specify_eid, list):
        adata = adata[specify_eid]
    if specific_col:
        adata = adata[:, specific_col]
    dataloader = _load_dataset_from_adata(adata, batch_size=batch_size, shuffle=shuffle)
    return dataloader, adata

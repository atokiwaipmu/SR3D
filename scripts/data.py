
import os
import glob
import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor, Normalize
import torch.utils.data as data

def get_minmax_transform(rangemin, rangemax):
    """
    Function to get a pair of transforms that normalize and denormalize tensors.
    """
    transform = Compose([lambda t: (t - rangemin) / (rangemax - rangemin) * 2 - 1])
    inverse_transform = Compose([lambda t: (t + 1) / 2 * (rangemax - rangemin) + rangemin])
    
    return transform, inverse_transform

class MapDataset(data.Dataset):
    """
    Class for the map dataset.
    """
    def __init__(self, mapdir, n_maps=100):
        self.maps = sorted(glob.glob(f'{mapdir}*.npy'))[:n_maps]

    def __getitem__(self, index):
        dmaps = np.array([np.load(map) for map in self.maps])  
        tensor_map = torch.from_numpy(dmaps).float()
        
        return tensor_map
    
def get_data(dir, n_maps=10):
    """
    Function to get the data.
    """
    data = MapDataset(dir, n_maps).__getitem__(0)
    return data

def get_norm_dataset(data_hr, data_lr):
    RANGE_MIN, RANGE_MAX = data_hr.min().clone().detach(), data_hr.max().clone().detach()
    transforms, inverse_transforms = get_minmax_transform(RANGE_MIN, RANGE_MAX)
    combined_dataset = data.TensorDataset(transforms(data_hr).unsqueeze(1), transforms(data_lr).unsqueeze(1))
    return combined_dataset, transforms, inverse_transforms, RANGE_MIN, RANGE_MAX
    
def get_dataloader(combined_dataset, rate_train, batch_size):
    """
    Function to get the dataloader.
    """
    len_train = int(rate_train * combined_dataset[:][0].shape[0])
    len_val = combined_dataset[:][0].shape[0] - len_train
    train, val = data.random_split(combined_dataset, [len_train, len_val])
    loaders = {x: data.DataLoader(ds, batch_size=batch_size, shuffle=x=='train', num_workers=os.cpu_count()) for x, ds in zip(('train', 'val'), (train, val))}
    return loaders['train'], loaders['val']
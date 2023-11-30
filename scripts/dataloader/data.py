
import os
import glob
import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor, Normalize
import torch.utils.data as data

class Transforms():
    def __init__(self, transform_type, range_min, range_max):
        self.transform_type = transform_type
        self.transform, self.inverse_transform = None, None
        self.range_min, self.range_max = range_min, range_max
        self.set_transforms()

    def set_transforms(self):
        if self.transform_type == 'minmax':
            self.transform = Compose([lambda t: (t - self.range_min) / (self.range_max - self.range_min) * 2 - 1])
            self.inverse_transform = Compose([lambda t: (t + 1) / 2 * (self.range_max - self.range_min) + self.range_min])
        elif self.transform_type == 'sigmoid':
            self.transform = Compose([lambda t: torch.sigmoid(t)])
            self.inverse_transform = Compose([lambda t: torch.logit((t))])
        elif self.transform_type == 'both':
            self.transform = Compose([lambda t: torch.sigmoid(t), lambda t: (t - self.range_min) / (self.range_max - self.range_min) * 2 - 1])
            self.inverse_transform = Compose([lambda t: (t + 1) / 2 * (self.range_max - self.range_min) + self.range_min, lambda t: torch.logit((t))])
        elif self.transform_type == 'log2linear':
            self.transform = Compose([lambda t: 10**t - 1])
            self.inverse_transform = Compose([lambda t: torch.log10(t + 1)])
        else:
            raise NotImplementedError()

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

def get_loaders(data_input, data_condition, rate_train, batch_size):
    """
    Function to get the loaders for training and validation datasets.
    """
    combined_dataset = data.TensorDataset(data_input, data_condition)
    len_train = int(rate_train * len(data_input))
    len_val = len(data_input) - len_train
    train, val = data.random_split(combined_dataset, [len_train, len_val])
    loaders = {x: data.DataLoader(ds, batch_size=batch_size, shuffle=x=='train', num_workers=os.cpu_count()) for x, ds in zip(('train', 'val'), (train, val))}
    
    return loaders['train'], loaders['val']

def get_normalized_data(data_loaded, transform_type='minmax'):
    range_min, range_max = data_loaded.min().clone().detach(), data_loaded.max().clone().detach()
    transform_class = Transforms(transform_type, range_min, range_max)
    data_normalized = transform_class.transform(data_loaded)
    return data_normalized, transform_class

def get_data_from_params(params):
    lr = get_data(params["data"]["LR_dir"], params["data"]["n_maps"])   
    print("LR data loaded from {}.  Number of maps: {}".format(params["data"]["LR_dir"], params["data"]["n_maps"]))
    hr = get_data(params["data"]["HR_dir"], params["data"]["n_maps"])
    print("HR data loaded from {}.  Number of maps: {}".format(params["data"]["HR_dir"], params["data"]["n_maps"]))
    return lr, hr

def get_normalized_from_params(lr, hr, params):
    data_cond, transforms_cond = get_normalized_data(lr, transform_type=params["data"]["transform_type"])
    print("LR data normalized to [{},{}] by {} transform.".format(data_cond.min(), data_cond.max(), params["data"]["transform_type"]))

    data_input, transforms_input  = get_normalized_data(hr, transform_type=params["data"]["transform_type"])
    print("HR data normalized to [{},{}] by {} transform.".format(data_input.min(), data_input.max(), params["data"]["transform_type"]))
    return data_input, data_cond, transforms_input, transforms_cond

def get_loaders_from_params(params):
    lr, hr = get_data_from_params(params)
    data_input, data_cond, _, _ = get_normalized_from_params(lr, hr, params)
    train_loader, val_loader = get_loaders(data_input, data_cond, params["train"]['train_rate'], params["train"]['batch_size'])
    print("train:validation = {}:{}, batch_size: {}".format(len(train_loader), len(val_loader), params["train"]['batch_size']))
    return train_loader, val_loader
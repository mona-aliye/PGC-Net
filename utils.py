import inspect
import h5py
import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import random
import albumentations as A
import pgcnet
from dataset import CellDataset


def calculate_val_mae(predict, counts):
    predict_sum = torch.sum(predict, dim=[1, 2, 3])
    counts_sum = torch.sum(counts, dim=[1, 2, 3])
    criterion = torch.nn.L1Loss()
    simple_mae = criterion(predict_sum, counts_sum)
    return simple_mae


def predict_calculate_val_mae(predict, counts):
    predict_sum = torch.sum(predict, dim=[1, 2, 3])
    counts_sum = torch.sum(counts, dim=[1, 2, 3])
    criterion = torch.nn.L1Loss()
    simple_mae = criterion(predict_sum, counts_sum)
    return simple_mae.item(), counts_sum.item(), predict_sum.item()


def change_type(dataset: dict) -> None:
    for key in dataset.keys():
        if key == 'imgs':
            dataset[key] = dataset[key].astype(np.float32) / 255.
        elif key == 'counts':
            dataset[key] = dataset[key].astype(np.float32)


def load_hdf5(data_filename, keys=None) -> dict:
    """
    assume all datasets are numpy arrays
    load hdf5 data to dictionary
    """
    dataset = {}
    with h5py.File(data_filename, 'a') as hf:
        if keys is None:
            for name in hf:
                dataset[name] = np.array(hf.get(name))
        else:
            for name in keys:
                dataset[name] = np.array(hf.get(name))
    return dataset


class CustomDataAugmentation:
    def __init__(self, crop_size):
        self.transform = A.Compose([
            A.RandomCrop(height=crop_size[0], width=crop_size[1], p=1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, p=0.5)
        ], additional_targets={'label': 'image'})
        self.test_transform = A.Compose([
            A.Crop(x_min=0, y_min=0, x_max=crop_size[0], y_max=crop_size[1])
        ], additional_targets={'label': 'image'})

    def call_train(self, sample):
        x, y = sample['image'], sample['label']
        transformed = self.transform(image=x, label=y)
        return transformed

    def call_test(self, sample):
        x, y = sample['image'], sample['label']
        transformed = self.test_transform(image=x, label=y)
        return transformed

    def __call__(self, sample, method):
        if method == 'train':
            return self.call_train(sample)
        elif method == 'test':
            return self.call_test(sample)
        else:
            raise ValueError("method error")


class RandomCrop(object):
    def __init__(self, crop_size):
        self.crop_height = crop_size[0]
        self.crop_width = crop_size[1]

    def __call__(self, sample):
        x, y = sample
        height, width = x.shape[:2]

        y_start = random.randint(0, height - self.crop_height - 1)
        x_start = random.randint(0, width - self.crop_width - 1)

        x_crop = x[y_start:y_start + self.crop_height, x_start:x_start + self.crop_width, ...].copy()
        y_crop = y[y_start:y_start + self.crop_height, x_start:x_start + self.crop_width, ...].copy()

        return x_crop, y_crop


def get_model(model_name, model_params: dict):
    if model_name == 'PGCNet':
        constructor_params = inspect.signature(pgcnet.PGCNet.__init__).parameters
        cons_params = [param for param in constructor_params]
        filter_model_params = {key: value for key, value in model_params.items() if key in cons_params}
        gradfunclist = get_gradfunlist(gradfunclist=model_params['gradient_func'])
        filter_model_params.update({'gradm_stategylist': gradfunclist})
        model = pgcnet.PGCNet(**filter_model_params)
        return model
    else:
        raise ValueError('Invalid model name: %s' % model_name)


def getgradfunc(symbol):
    if symbol == 1:
        gradfunc = None
    elif symbol == 2:
        gradfunc = None
    else:
        gradfunc = None
    return gradfunc


def get_gradfunlist(gradfunclist=None):
    if gradfunclist is not None and gradfunclist != 'None':
        for index, symbol in enumerate(gradfunclist):
            gradfunclist[index] = getgradfunc(symbol)
    return gradfunclist


def get_optimizer(optimizer_name, params, optimizer_params: dict):
    if optimizer_name == 'SGD':
        return optim.SGD(params, **optimizer_params)
    elif optimizer_name == 'Adam':
        return optim.Adam(params, **optimizer_params)
    else:
        raise ValueError('Invalid optimizer name: %s' % optimizer_name)


def get_lr_scheduler(scheduler_name, optimizer, scheduler_params: dict):
    if scheduler_name == 'CosineAnnealingWarmRestarts':
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **scheduler_params)
    elif scheduler_name == 'StepLR':
        return lr_scheduler.StepLR(optimizer, **scheduler_params)
    else:
        raise ValueError('Invalid scheduler name: %s' % scheduler_name)


def get_loss(loss_name, loss_params: dict):
    if loss_name == 'MSE':
        return nn.MSELoss()
    else:
        raise ValueError('Invalid loss name: %s' % loss_name)


def get_data_loaders(file_dir, batch_size, val_split: tuple = None, crop_size=None):
    if crop_size is None:
        crop_size = [224, 224]
    if val_split is None:
        val_split = [64, 136]
    data = load_hdf5(file_dir, keys=['counts', 'imgs'])
    dataset = CellDataset(data['imgs'], data['counts'], transform=None)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, val_split)
    train_data_augment = CustomDataAugmentation(crop_size=crop_size)
    train_dataset.dataset.transform = train_data_augment
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, validate_loader


def print_nested_dict(dictionary, indent=0):
    output = ""
    for key, value in dictionary.items():
        if isinstance(value, dict):
            output += "  " * indent + f"{key}:\n"
            output += print_nested_dict(value, indent + 1)
        else:
            output += "  " * indent + f"{key}: {value}\n"
    return output

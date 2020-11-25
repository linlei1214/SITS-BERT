import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from .pretrain_dataset import PretrainDataset

np.random.seed(0)

class DataSetWrapper(object):

    def __init__(self, batch_size, valid_size, data_path, num_features, max_length):
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.data_path = data_path
        self.num_features = num_features
        self.max_length = max_length

    def get_data_loaders(self):
        dataset = PretrainDataset(self.data_path, self.num_features, self.max_length)
        train_loader, valid_loader = self.get_train_validation_data_loaders(dataset)
        return train_loader, valid_loader

    def get_train_validation_data_loaders(self, dataset):
        num_train = len(dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        print('training samples: %d, validation samples: %d' % (num_train-split, split))
        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  drop_last=True)

        valid_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  drop_last=True)

        return train_loader, valid_loader

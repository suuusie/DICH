import torch
import os
import numpy as np
# import data.cifar10 as cifar10
# import data.nus_wide as nuswide
import data.MIRFlickr as MIRFlickr
import data.NUSWIDE as NUSWIDE

from data.transform import train_transform
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_data(dataset, root, num_seen, batch_size, num_workers):
    """
    Load dataset.

    Args
        dataset(str): Dataset name.
        root(str): Path of dataset.
        num_seen(int): Number of seen classes.
        num_workers(int): Number of loading data threads.

    Returns
        query_dataloader, seen_dataloader, unseen_dataloader(torch.utils.data.DataLoader): Data loader.
    """

    if dataset == 'MIRFlickr':
        query_dataloader, seen_dataloader, unseen_dataloader, retrieval_dataloader = MIRFlickr.load_data(root,
                                                                                                       num_seen,
                                                                                                       batch_size,
                                                                                                       num_workers,
                                                                                                        )
    elif dataset == 'NUSWIDE':
        query_dataloader, seen_dataloader, unseen_dataloader, retrieval_dataloader = NUSWIDE.load_data(root,
                                                                                                       num_seen,
                                                                                                       batch_size,
                                                                                                       num_workers,
                                                                                                        )
    else:
        raise ValueError("Invalid dataset name!")

    return query_dataloader, seen_dataloader, unseen_dataloader, retrieval_dataloader


def sample_dataloader(dataloader, num_samples, num_seen, batch_size, root, dataset):
    """
    Sample data from dataloder.

    Args
        dataloader(torch.utils.data.DataLoader): Dataloader.
        num_samples(int): Number of samples.
        num_seen(int): Number of seen data points.
        batch_size(int): Batch size.
        root(str): Path of dataset.
        dataset(str): Dataset name.

    Returns
        sample_loader(torch.utils.data.DataLoader): Sample dataloader.
        omega(torch.Tensor): Sample index.
        unseen_sample_in_unseen_index(torch.Tensor): Index of unseen samples in unseen dataset.
        unseen_sample_in_sample_index(torch.Tensor): Index of unseen samples in sampling dataset.
    """
    image = dataloader.dataset.image
    tag = dataloader.dataset.tag
    targets = dataloader.dataset.targets
    num_retrieval = len(image)

    omega = np.random.permutation(num_retrieval)[:num_samples]
    image = image[omega]
    tag = tag[omega]
    targets = targets[omega]
    sample_loader = wrap_data(image,tag, targets, batch_size, root, dataset)

    A = omega[omega > num_seen]
    B = A - num_seen
    C = (omega > num_seen)

    unseen_sample_in_unseen_index = omega[omega > num_seen] - num_seen
    unseen_sample_in_sample_index = (omega > num_seen).nonzero()[0]

    return sample_loader, omega, unseen_sample_in_unseen_index, unseen_sample_in_sample_index


def wrap_data(image, tag, targets, batch_size, root, dataset):
    """
    Wrap data into dataloader.

    Args
        data (np.ndarray): Data.
        targets (np.ndarray): Targets.
        batch_size (int): Batch size.
        root (str): Path of dataset.
        dataset(str): Dataset name.

    Returns
        dataloader (torch.utils.data.dataloader): Data loader.
    """
    class MyDataset(Dataset):
        def __init__(self, image,tag, targets, root, dataset):
            self.image = image
            self.tag = tag
            self.targets = targets
            self.root = root
            self.transform = train_transform()
            self.dataset = dataset
            self.onehot_targets = self.targets

        def __getitem__(self, index):
            if self.dataset == 'cifar-10':
                img = Image.fromarray(self.image[index])
            elif self.dataset == 'nus-wide-tc21':
                img = Image.open(os.path.join(self.root, self.data[index])).convert('RGB')
            elif self.dataset == 'MIRFlickr':
                img = Image.fromarray(self.image[index])
            elif self.dataset == 'NUSWIDE':
                img = Image.fromarray(self.image[index])
            else:
                raise ValueError('Invalid dataset name!')
            img = self.transform(img)
            return img, self.tag[index], self.targets[index], index

        def __len__(self):
            return self.image.shape[0]

        def get_onehot_targets(self):
            """
            Return one-hot encoding targets.
            """
            return torch.from_numpy(self.targets)

    dataset = MyDataset(image,tag, targets, root, dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
    )

    return dataloader

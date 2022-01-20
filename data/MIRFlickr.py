import torch
import numpy as np
from PIL import Image
import os

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from data.transform import train_transform, query_transform


def load_data(root, num_seen, batch_size, num_workers):
    """
    Load MIRFlickr dataset.

    Args
        root(str): Path of dataset.
        num_seen(int): Number of seen classes.
        batch_size(int): Batch size.
        num_workers(int): Number of loading data threads.

    Returns
        query_dataloader, seen_dataloader, unseen_dataloader, retrieval_dataloader(torch.evaluate.data.DataLoader): Data loader.
    """
    MIRFLICKR.init(root, num_seen)
    query_dataset = MIRFLICKR('query', transform=query_transform())
    seen_dataset = MIRFLICKR('seen', transform=train_transform())
    unseen_dataset = MIRFLICKR('unseen', transform=train_transform())
    retrieval_dataset = MIRFLICKR('retrieval', transform=train_transform())

    query_dataloader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=num_workers,

      )

    seen_dataloader = DataLoader(
        seen_dataset,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )

    unseen_dataloader = DataLoader(
        unseen_dataset,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )

    retrieval_dataloader = DataLoader(
        retrieval_dataset,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=num_workers,
    )

    return query_dataloader, seen_dataloader, unseen_dataloader, retrieval_dataloader


class MIRFLICKR(Dataset):
    """
    MIRFLICKR dataset.
    """
    @staticmethod
    def init(root, num_seen):
        # Load data
        # MIRFLICKR.QUERY_IMAGE= np.load(os.path.join(root, 'mir_2400_query_image.npy'))
        # MIRFLICKR.QUERY_TAG = np.load(os.path.join(root, 'mir_2400_query_tag.npy'))
        # MIRFLICKR.QUERY_TARGETS = np.load(os.path.join(root, 'mir_2400_query_label.npy'))
        # MIRFLICKR.RETRIEVAL_IMAGE = np.load(os.path.join(root, 'mir_17615_retrieval_image.npy'))
        # MIRFLICKR.RETRIEVAL_TAG = np.load(os.path.join(root, 'mir_17615_retrieval_tag.npy'))
        # MIRFLICKR.RETRIEVAL_TARGETS = np.load(os.path.join(root, 'mir_17615_retrieval_label.npy'))
        MIRFLICKR.QUERY_IMAGE= np.load(os.path.join(root, 'mir_2000_query_image.npy'))
        MIRFLICKR.QUERY_TAG = np.load(os.path.join(root, 'mir_2000_query_tag.npy'))
        MIRFLICKR.QUERY_TARGETS = np.load(os.path.join(root, 'mir_2000_query_label.npy'))
        MIRFLICKR.RETRIEVAL_IMAGE = np.load(os.path.join(root, 'mir_18015_retrieval_image.npy'))
        MIRFLICKR.RETRIEVAL_TAG = np.load(os.path.join(root, 'mir_18015_retrieval_tag.npy'))
        MIRFLICKR.RETRIEVAL_TARGETS = np.load(os.path.join(root, 'mir_18015_retrieval_label.npy'))
        # Split seen data
        L_unseen = MIRFLICKR.RETRIEVAL_TARGETS[:, num_seen:]
        temp = np.sum(L_unseen, axis=1)
        unseen_index = list((np.where(temp > 0))[0])
        seen_index = (list(set(range(MIRFLICKR.RETRIEVAL_TARGETS.shape[0])).difference(set(unseen_index))))

        MIRFLICKR.SEEN_IMAGE = MIRFLICKR.RETRIEVAL_IMAGE[seen_index, :]
        MIRFLICKR.SEEN_TAG = MIRFLICKR.RETRIEVAL_TAG[seen_index,:]
        MIRFLICKR.SEEN_TARGETS = MIRFLICKR.RETRIEVAL_TARGETS[seen_index, :]
        MIRFLICKR.UNSEEN_IMAGE = MIRFLICKR.RETRIEVAL_IMAGE[unseen_index, :]
        MIRFLICKR.UNSEEN_TAG = MIRFLICKR.RETRIEVAL_TAG[unseen_index,:]
        MIRFLICKR.UNSEEN_TARGETS = MIRFLICKR.RETRIEVAL_TARGETS[unseen_index, :]

        MIRFLICKR.RETRIEVAL_IMAGE = np.concatenate((MIRFLICKR.SEEN_IMAGE, MIRFLICKR.UNSEEN_IMAGE), axis=0)
        MIRFLICKR.RETRIEVAL_TAG = np.concatenate((MIRFLICKR.SEEN_TAG, MIRFLICKR.UNSEEN_TAG), axis=0)
        MIRFLICKR.RETRIEVAL_TARGETS = np.concatenate((MIRFLICKR.SEEN_TARGETS, MIRFLICKR.UNSEEN_TARGETS), axis=0)

        # unseen_index = np.array(unseen_index)
        # MIRFLICKR.UNSEEN_INDEX = unseen_index

    def __init__(self, mode,
                 transform=None, target_transform=None, tag_transform = None
                 ):
        self.transform = transform
        self.target_transform = target_transform
        self.tag_transform = tag_transform

        if mode == 'seen':
            self.image = MIRFLICKR.SEEN_IMAGE
            self.tag = MIRFLICKR.SEEN_TAG
            self.targets = MIRFLICKR.SEEN_TARGETS
        elif mode == 'unseen':
            self.image = MIRFLICKR.UNSEEN_IMAGE
            self.tag = MIRFLICKR.UNSEEN_TAG
            self.targets = MIRFLICKR.UNSEEN_TARGETS
        elif mode == 'query':
            self.image = MIRFLICKR.QUERY_IMAGE
            self.tag = MIRFLICKR.QUERY_TAG
            self.targets = MIRFLICKR.QUERY_TARGETS
        elif mode == 'retrieval':
            self.image = MIRFLICKR.RETRIEVAL_IMAGE
            self.tag = MIRFLICKR.RETRIEVAL_TAG
            self.targets = MIRFLICKR.RETRIEVAL_TARGETS
        else:
            raise ValueError('Mode error!')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        image, tag, target = self.image[index], self.tag[index],self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        if self.tag_transform is not None:
            tag = self.tag_transform(tag)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, tag, target, index

    def __len__(self):
        return len(self.image)

    def get_onehot_targets(self):
        """
        Return one-hot encoding targets.
        """
        return torch.from_numpy(self.targets)

    def get_tag(self):
        """

        Returns: tags

        """
        return torch.from_numpy(self.tag)



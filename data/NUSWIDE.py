import torch
import numpy as np
from PIL import Image
import os

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from data.transform import train_transform, query_transform


def load_data(root, num_seen, batch_size, num_workers):
    """
    Load NUSWIDE dataset.

    Args
        root(str): Path of dataset.
        num_seen(int): Number of seen classes.
        batch_size(int): Batch size.
        num_workers(int): Number of loading data threads.

    Returns
        query_dataloader, seen_dataloader, unseen_dataloader, retrieval_dataloader(torch.evaluate.data.DataLoader): Data loader.
    """
    NUSWIDE.init(root, num_seen)
    query_dataset = NUSWIDE('query', transform=query_transform())
    seen_dataset = NUSWIDE('seen', transform=train_transform())
    unseen_dataset = NUSWIDE('unseen', transform=train_transform())
    retrieval_dataset = NUSWIDE('retrieval', transform=train_transform())

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
        #num_workers=num_workers,
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
        #num_workers=num_workers,
    )

    return query_dataloader, seen_dataloader, unseen_dataloader, retrieval_dataloader


class NUSWIDE(Dataset):
    """
    NUSWIDE dataset.
    """
    @staticmethod
    def init(root, num_seen):
        # Load data
        NUSWIDE.QUERY_IMAGE= np.load(os.path.join(root, 'nus_2100_query_image.npy'))
        NUSWIDE.QUERY_TAG = np.load(os.path.join(root, 'nus_2100_query_tag.npy'))
        NUSWIDE.QUERY_TARGETS = np.load(os.path.join(root, 'nus_2100_query_label.npy'))
        NUSWIDE.RETRIEVAL_IMAGE = np.load(os.path.join(root, 'nus_193734_retrieval_image.npy'))
        NUSWIDE.RETRIEVAL_TAG = np.load(os.path.join(root, 'nus_193734_retrieval_tag.npy'))
        NUSWIDE.RETRIEVAL_TARGETS = np.load(os.path.join(root, 'nus_193734_retrieval_label.npy'))

        # Split seen data
        L_unseen = NUSWIDE.RETRIEVAL_TARGETS[:, num_seen:]
        temp = np.sum(L_unseen, axis=1)
        unseen_index = list((np.where(temp > 0))[0])
        seen_index = (list(set(range(NUSWIDE.RETRIEVAL_TARGETS.shape[0])).difference(set(unseen_index))))

        NUSWIDE.SEEN_IMAGE = NUSWIDE.RETRIEVAL_IMAGE[seen_index, :]
        NUSWIDE.SEEN_TAG = NUSWIDE.RETRIEVAL_TAG[seen_index,:]
        NUSWIDE.SEEN_TARGETS = NUSWIDE.RETRIEVAL_TARGETS[seen_index, :]
        NUSWIDE.UNSEEN_IMAGE = NUSWIDE.RETRIEVAL_IMAGE[unseen_index, :]
        NUSWIDE.UNSEEN_TAG = NUSWIDE.RETRIEVAL_TAG[unseen_index,:]
        NUSWIDE.UNSEEN_TARGETS = NUSWIDE.RETRIEVAL_TARGETS[unseen_index, :]

        del NUSWIDE.RETRIEVAL_IMAGE, NUSWIDE.RETRIEVAL_TAG, NUSWIDE.RETRIEVAL_TARGETS

        NUSWIDE.RETRIEVAL_IMAGE = np.concatenate((NUSWIDE.SEEN_IMAGE, NUSWIDE.UNSEEN_IMAGE), axis=0)
        NUSWIDE.RETRIEVAL_TAG = np.concatenate((NUSWIDE.SEEN_TAG, NUSWIDE.UNSEEN_TAG), axis=0)
        NUSWIDE.RETRIEVAL_TARGETS = np.concatenate((NUSWIDE.SEEN_TARGETS, NUSWIDE.UNSEEN_TARGETS), axis=0)

        # unseen_index = np.array(unseen_index)
        # NUSWIDE.UNSEEN_INDEX = unseen_index

    def __init__(self, mode,
                 transform=None, target_transform=None, tag_transform = None
                 ):
        self.transform = transform
        self.target_transform = target_transform
        self.tag_transform = tag_transform

        if mode == 'seen':
            self.image = NUSWIDE.SEEN_IMAGE
            self.tag = NUSWIDE.SEEN_TAG
            self.targets = NUSWIDE.SEEN_TARGETS
        elif mode == 'unseen':
            self.image = NUSWIDE.UNSEEN_IMAGE
            self.tag = NUSWIDE.UNSEEN_TAG
            self.targets = NUSWIDE.UNSEEN_TARGETS
        elif mode == 'query':
            self.image = NUSWIDE.QUERY_IMAGE
            self.tag = NUSWIDE.QUERY_TAG
            self.targets = NUSWIDE.QUERY_TARGETS
        elif mode == 'retrieval':
            self.image = NUSWIDE.RETRIEVAL_IMAGE
            self.tag = NUSWIDE.RETRIEVAL_TAG
            self.targets = NUSWIDE.RETRIEVAL_TARGETS
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



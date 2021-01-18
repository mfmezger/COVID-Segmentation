import os
import random
import re

import numpy as np
import torch
from torch.utils.data import Dataset




class TorchDataset(Dataset):
    """
    Loading the Datasets
    """

    def sorted_nicely(self, l):
        """ Sort the given iterable in the way that humans expect."""
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)



    def __init__(self, directory, augmentations=False):
        self.directory = directory
        self.augmentations = augmentations

        self.images = self.sorted_nicely(os.listdir(directory))



    def augment_gaussian_noise(self, data_sample, noise_variance=(0.001, 0.05)):
        # https://github.com/MIC-DKFZ/batchgenerators
        if noise_variance[0] == noise_variance[1]:
            variance = noise_variance[0]
        else:
            variance = random.uniform(noise_variance[0], noise_variance[1])
        data_sample = data_sample + np.random.normal(0.0, variance, size=data_sample.shape)
        return data_sample

    def __len__(self):
        return len(os.listdir(self.directory))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # load the image and the groundtruth
        name = self.images[idx]
        file = torch.load(os.path.join(self.directory, name))

        image = file["vol"]
        mask = file["mask"]
        image = image.to(torch.float32)
        mask = mask.to(torch.float32)
        # do augmentations
        if self.augmentations:
            random_number = random.randint(1, 10)
            image = image.numpy()
            if random_number >= 7:
                # do for each layer
                image = self.augment_gaussian_noise(image)
            image = torch.from_numpy(image)

        image = image.unsqueeze(0)
        image = image.float()
        mask = mask.to(dtype=torch.int64)

        return image, mask


if __name__ == '__main__':
    dataset = TorchDataset("/home/mfmezger/data/COVID/", augmentations=True)
    img, mask = dataset[1]

    from batchviewer import view_batch

    view_batch(img, mask, width=512, height=512)

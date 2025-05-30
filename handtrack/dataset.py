from .helpers import *
import cv2 
import os 
import sys
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image


class LoadStereoPairs(Dataset):
    """Generator for loading pairs of stereo image.
    
    This is a generator that inherits from Dataset class from PyTorch.

    This is further used in DataLoader:

    .. highlight:: python
    .. code-block:: python

        from torchvision import transforms

        data_transforms = transforms.Compose([
            transforms.Resize((244, 244))
        ])

        stereo_data_loader = torch.utils.data.DataLoader(LoadStereoPairs(left_dir, right_dir), batch_size=4, num_workers=4)

    """

    def __init__(self, left_dir, right_dir, transform=None):
        """Constructs a Loading stereo generator class.

        Parameters
        ----------
        left_dir: str
            Path to the left stereo images directory
        right_dir: str
            Path to the right stereo images directory

        """
        self.left_dir = left_dir
        self.right_dir = right_dir
        self.transform = transform

        self.left_names = sorted(list_files(left_dir), key=natural_sort_key)
        self.right_names = sorted(list_files(right_dir), key=natural_sort_key)

    def __len__(self):
        return len(self.left_names)

    def __getitem__(self, idx):
        left_image_name = list(self.left_names)[idx]
        right_image_name = list(self.right_names)[idx]

        left_image = read_image(os.path.join(self.left_dir, left_image_name))
        right_image = read_image(os.path.join(self.right_dir, right_image_name))

        if self.transform:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)

        return (left_image, right_image, left_image_name)


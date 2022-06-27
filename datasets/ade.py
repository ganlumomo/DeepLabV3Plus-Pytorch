import os
import numpy as np
import torch.utils.data as data
from PIL import Image

class ADE20KDataset(data.Dataset):

    def __init__(self, root, split='training', transform=None):
        self.root = os.path.expanduser(root)
        self.images_dir = os.path.join(self.root, 'images', split)
        self.targets_dir = os.path.join(self.root, 'annotations', split)
        self.transform = transform
        self.images = []
        self.targets = []

        for file_name in os.listdir(self.images_dir):
            self.images.append(os.path.join(self.images_dir, file_name))
            self.targets.append(os.path.join(self.targets_dir, file_name).replace('.jpg', '.png'))
        assert(len(self.images) == len(self.targets))

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        if self.transform:
            image, target = self.transform(image, target)
        return image, target

    def __len__(self):
        return len(self.images)

    def label2colormap(self, label):
        m = label.astype(np.uint8)
        r,c = m.shape
        cmap = np.zeros((r,c,3), dtype=np.uint8)
        cmap[:,:,0] = (m&1)<<7 | (m&8)<<3 | (m&64)>>1
        cmap[:,:,1] = (m&2)<<6 | (m&16)<<2 | (m&128)>>2
        cmap[:,:,2] = (m&4)<<5 | (m&32)<<1
        return cmap

    def decode_target(self, target):
        """decode semantic mask to RGB image"""
        return self.label2colormap(label=target)

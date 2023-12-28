import json, os
import torch
from collections import Counter
from PIL import Image
from torch.utils.data import Dataset


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageCaptionDataset(Dataset):
    def __init__(self, transform, data_path, split_type='train', fraction=1.0, bert=False):
        super(ImageCaptionDataset, self).__init__()
        self.split_type = split_type
        self.transform = transform
        self.img_paths = json.load(open(data_path + '/{}_img_paths.json'.format(split_type), 'r'))

        if bert==True:
            self.captions = json.load(open(data_path + '/{}_captions_bert.json'.format(split_type), 'r'))
        else:
            self.captions = json.load(open(data_path + '/{}_captions.json'.format(split_type), 'r'))

        # reduced dataset by fraction (for debugging)
        if fraction != 1.0:
            self.img_paths = self.img_paths[:int(len(self.img_paths) * fraction)]
            self.captions = self.captions[:int(len(self.captions) * fraction)]

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = pil_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)

        if self.split_type == 'train':
            return torch.FloatTensor(img), torch.tensor(self.captions[index])

        matching_idxs = [idx for idx, path in enumerate(self.img_paths) if path == img_path]
        all_captions = [self.captions[idx] for idx in matching_idxs]
        # TODO: check if self.captions[index] can be multiple captions???
        return torch.FloatTensor(img), torch.tensor(self.captions[index]), torch.tensor(all_captions)

    def __len__(self):
        return len(self.captions)
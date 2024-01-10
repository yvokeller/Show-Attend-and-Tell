import json
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from PIL import Image
import json


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageCaptionDataset(Dataset):
    def __init__(self, transform, data_path, split_type='train', fraction=1.0, bert=False):
        super(ImageCaptionDataset, self).__init__()
        self.transform = transform
        
        # Load image paths and captions
        img_paths = json.load(open(data_path + f'/{split_type}_img_paths.json', 'r'))
        if bert:
            captions = json.load(open(data_path + f'/{split_type}_captions_bert.json', 'r'))
        else:
            captions = json.load(open(data_path + f'/{split_type}_captions.json', 'r'))

        # Reduce dataset size if fraction is not 1.0
        if fraction != 1.0:
            img_paths = img_paths[:int(len(img_paths) * fraction)]
            captions = captions[:int(len(captions) * fraction)]

        # Preprocess and store data
        self.data = []
        all_captions = defaultdict(list)  # Store all captions for each image path

        for img_path, caption in zip(img_paths, captions):
            img = pil_loader(img_path)
            if self.transform is not None:
                img = self.transform(img)
            self.data.append((torch.FloatTensor(img), torch.tensor(caption)))
            all_captions[img_path].append(caption)

        # Convert all_captions dictionary to a list matching the order of images
        self.all_captions = [all_captions[path] for path in img_paths]

    def __getitem__(self, index):
        img_tensor, caption_tensor = self.data[index]
        all_captions_tensor = torch.tensor(self.all_captions[index])
        return img_tensor, caption_tensor, all_captions_tensor

    def __len__(self):
        return len(self.data)
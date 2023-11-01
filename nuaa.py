from torch.utils.data import Dataset
import os
import gdown
import zipfile
from glob import glob
from PIL import Image

class NUAA(Dataset):

    links = {
        'raw': 'https://drive.google.com/file/d/1-aSGKdAIK0YoKxQvnNx1KJvTm4zbwZLz/view?usp=sharing',
        'Detectedface': 'https://drive.google.com/file/d/1oE6yv-RYV5_4HDjUo6F8mJofzrW_tdTo/view?usp=sharing',
        'NormalizedFace': 'https://drive.google.com/file/d/1LT8LThFu3uJ3JdLRDb1c599YwzBEAcAS/view?usp=sharing'
    }


    def __init__(self, root, train=True, download=False, transform=None, target_transform=None, format='raw', verbose=False, normal_split=0.8, chosen_classes=0):
        self.root = os.path.join(root, 'nuaa')
        self.train = train
        self.download = download
        self.transform = transform
        self.target_transform = target_transform
        self.format = format
        self.verbose = verbose
        self.normal_split = normal_split
        if isinstance(chosen_classes, int):
            chosen_classes = [chosen_classes]
        self.chosen_classes = chosen_classes
        if download:
            self._download_and_extract()
        self.data, self.targets = self._load_data()

    def _download_and_extract(self):
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        
        file_path = os.path.join(self.root, f'{self.format}.zip')
        if not os.path.exists(file_path):
            gdown.download(self.links[self.format], file_path, fuzzy=True, quiet=not self.verbose)
        
        if not os.path.exists(os.path.join(self.root, self.format)):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(self.root)
    
    def _load_data(self):
        data = []
        targets = []
        data_path = os.path.join(self.root, self.format)
        search_dirs = glob(os.path.join(data_path, 'ClientRaw', '*')) + glob(os.path.join(data_path, 'ImposterRaw', '*'))
        chosen_dirs = list(filter(lambda path: int(os.path.basename(path)) - 1 in self.chosen_classes, search_dirs))
        for dir in chosen_dirs:
            target = os.path.basename(os.path.split(dir)[0]) == 'ImposterRaw'
            if self.train:
                file_slice = slice(0, int(len(os.listdir(dir)) * self.normal_split))
            else:
                file_slice = slice(int(len(os.listdir(dir)) * self.normal_split), None)
            files = glob(os.path.join(dir, '*.jpg'))[file_slice]
            for file in files:
                data.append(file)
                targets.append(target)
        return data, targets


    def __getitem__(self, index):
        image = Image.open(self.data[index]).convert('RGB')
        target = self.targets[index]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __len__(self):
        return len(self.data)

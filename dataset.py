import torch
from torch.utils.data import Dataset

class SpiderDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = torch.tensor(self.images[index], dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(self.masks[index], dtype=torch.float32).unsqueeze(0)
        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask
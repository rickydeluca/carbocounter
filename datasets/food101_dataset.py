from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

def Food101Dataset(root, transform=None):
    return ImageFolder(root=root, transform=transform)

class Food101Subset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample, label = self.dataset[self.indices[idx]]
        if self.transform:
            sample = self.transform(sample)
        return sample, label
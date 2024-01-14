from torchvision import transforms
from torch.utils.data import Dataset


class CellDataset(Dataset):
    """
    input array-list and optional transform, the sequence of imgs and counts must be correct
    """
    def __init__(self, images, counts, transform=None):
        self.images = images
        self.counts = counts
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, count = self.images[idx], self.counts[idx]
        if self.transform is not None:
            sample = {'image': image, 'label': count}
            transformed = self.transform(sample, method='train')
            image, count = transformed['image'], transformed['label']
        image = transforms.ToTensor()(image)
        count = transforms.ToTensor()(count)
        return image, count

    def get_fixed_item(self, img_index=0):
        image, count = self.images[img_index], self.counts[img_index]
        if self.transform is not None:
            sample = {'image': image, 'label': count}
            transformed = self.transform(sample, method='test')
            image, count = transformed['image'], transformed['label']
        image = transforms.ToTensor()(image)
        count = transforms.ToTensor()(count)
        return image, count

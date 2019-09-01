import torch
import random
from torchvision import transforms


def compute_mean_std(loader):
    mean_img = None
    for imgs, _ in loader:
        if mean_img is None:
            mean_img = torch.zeros_like(imgs[0])
        mean_img += imgs.sum(dim=0)
    mean_img /= len(loader.dataset)

    std_img = torch.zeros_like(mean_img)
    for imgs, _ in loader:
        std_img += ((imgs - mean_img) ** 2).sum(dim=0)
    std_img /= len(loader.dataset)
    std_img = torch.sqrt(std_img)

    std_img[std_img == 0] = 1

    return mean_img, std_img


class DatasetTransformer(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform, normalize, augment_prob=None):
        self.base_dataset = base_dataset
        self.transform = transform
        self.normalize = normalize
        self.augment_prob = augment_prob if augment_prob is not None else None
        self.to_tensor_transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        img, target = self.base_dataset[index]

        if self.augment_prob:
            if random.random() < self.augment_prob:
                img = self.transform(img)
            else:
                img = self.to_tensor_transform(img)

        else:
            img = self.transform(img)

        if self.normalize:
            return self.scale_tensor(img), target

        else:
            return img, target

    @staticmethod
    def scale_tensor(tensor):
        min = tensor.min()
        max = tensor.max()
        tensor = (tensor - min) / (max - min)
        return tensor

    def __len__(self):
        return len(self.base_dataset)


class CenterReduce:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x-self.mean)/self.std

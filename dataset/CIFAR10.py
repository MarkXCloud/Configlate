from configlate import registry
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from typing import Tuple, Any


class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        CIFAR10 fitted to albumentations.
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        image, target = self.data[index], self.targets[index]

        if self.transform is not None:
            image = self.transform(image=image)['image']

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target


@registry.dataset
def cifar10(root: str, img_size: tuple, num_classes: int, batch_size: int, **kwargs) -> tuple:
    train_set = CIFAR10(root=root,
                        train=True,
                        transform=A.Compose([
                            A.Resize(img_size[0], img_size[1]),
                            A.Normalize(mean=(0.485, 0.456, 0.406),
                                        std=(0.229, 0.224, 0.225)),
                            ToTensorV2()]),
                        target_transform=T.Compose([
                            lambda x: torch.LongTensor([x]),
                            lambda x: F.one_hot(x, num_classes).squeeze(0).to(torch.float32)]), )
    test_set = CIFAR10(root=root,
                       train=False,
                       transform=A.Compose([
                           A.Resize(img_size[0], img_size[1]),
                           A.Normalize(mean=(0.485, 0.456, 0.406),
                                       std=(0.229, 0.224, 0.225)),
                           ToTensorV2()]),
                       target_transform=lambda x: torch.tensor(x))
    return torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs), \
        torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)

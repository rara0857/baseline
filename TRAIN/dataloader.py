from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import TumorDataset
from randaugment import RandAugmentMC
from transform import transform_image_train

class TumorDataModule():
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.train_batch_size = config['train_batch_size']
        self.val_batch_size = config['val_batch_size']

        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transform_image_train(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])

        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])

        datasets = TumorDataset(self.config, self.train_transform, 'train')
        # total_len = len(datasets)
        self.train_dataset = datasets

        datasets = TumorDataset(self.config, self.val_transform, 'val')
        # total_len = len(datasets)
        self.valid_dataset = datasets

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, pin_memory=True, num_workers=8, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.val_batch_size, pin_memory=True, num_workers=8, drop_last=True, shuffle=True)







from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from aug_dataset import TumorDataset

class Aug_TumorDataModule():
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.unlabel_batch_size = config['unlabel_batch_size']
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])

        datasets = TumorDataset(self.config, self.train_transform, 'unlabel')
        # total_len = len(datasets)
        self.unlabel_dataset = datasets


    def unlabel_dataloader(self):
        return DataLoader(self.unlabel_dataset, batch_size=self.unlabel_batch_size, pin_memory=True, shuffle=True)
from torchvision import transforms
import torch
from prefect import task 
from typing import Optional
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets
from multiprocessing import cpu_count
import pytorch_lightning as pl

def process_data(data):
    return transforms.Compose(
        [transforms.PILToTensor(), transforms.Grayscale(), transforms.Resize((28, 28))]
    )(data).to(torch.float32)


@task(retries=3, retry_delay_seconds=60)
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = cpu_count()
        self.setup()

    def setup(self, stage: Optional[str] = None) -> None:
        self.mnist_test = datasets.MNIST(
            self.data_dir, train=False, download=True, transform=transforms.ToTensor()
        )
        mnist_full = datasets.MNIST(
            self.data_dir, train=True, download=True, transform=transforms.ToTensor()
        )
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        self.input_dim = self.mnist_test.data.shape[-1]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )
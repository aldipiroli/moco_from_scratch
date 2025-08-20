import os
from pathlib import Path

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import Flowers102
from torchvision.transforms import v2


class Flowers102Dataset(Dataset):
    def __init__(self, cfg, mode):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.root_dir = Path(cfg["DATA"]["root_dir"])
        self.img_size = self.cfg["MODEL"]["img_size"]
        self.crop_size = self.cfg["DATA"]["augm"]["crop_size"]

        self.dataset = Flowers102(
            root=self.root_dir,
            split=self.mode,
            download=True if not os.path.exists(self.root_dir / "flowers-102") else False,
        )

        self.transform_resize = v2.Resize((self.img_size[0], self.img_size[1]), antialias=True)
        self.transform_to_tensor = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = self.transform_resize(img)
        img = self.transform_to_tensor(img)
        return img

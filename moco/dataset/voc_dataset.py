import os
from pathlib import Path

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import VOCDetection
from torchvision.transforms import v2


class VOCDataset(Dataset):
    def __init__(self, cfg, mode):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.root_dir = Path(cfg["DATA"]["root_dir"])
        dest_path = os.path.join(self.root_dir, "VOCdevkit/VOC2012")
        self.img_size = self.cfg["MODEL"]["img_size"]
        self.crop_size = self.cfg["DATA"]["augm"]["crop_size"]
        self.dataset = VOCDetection(
            root=self.root_dir,
            year="2012",
            image_set=mode,
            download=True if not os.path.exists(dest_path) else False,
        )
        self.transform_resize = v2.Resize((self.img_size[0], self.img_size[1]), antialias=True)
        self.transform_to_tensor = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        img = self.transform_to_tensor(img)
        img = self.transform_resize(img)
        return img

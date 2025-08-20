from torchvision import transforms


class MoCoAugmentations:
    def __init__(self, config):
        self.config = config
        self.crop_size = config["DATA"]["augm"]["crop_size"]
        self.transforms = transforms.Compose([transforms.RandomCrop(size=self.crop_size)])

    def augment(self, x):
        x_t = self.transforms(x)
        return x_t

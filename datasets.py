import torch.utils.data as data

class Dataset(data.Dataset):
    """Dataset class for dsprites"""

    def __init__(self, data_root, normalize=True, rotate=False):

        # Recursively exract paths to all .png files in subdirectories
        self.file_paths = []
        for path, subdirs, files in os.walk(data_root):
            for name in files:
                if name.endswith(".png"):
                    self.file_paths.append(os.path.join(path, name))

        self.transform = self._set_transforms(normalize, rotate)

    def _set_transforms(self, normalize, rotate):
        """Decide transformations to data to be applied"""
        transforms_list = []

        # Normalize to the mean and standard deviation all pretrained
        # torchvision models expect
        # ImageNet mean=(0.485, 0.456, 0.406) and std=(0.229, 0.224, 0.225)
        # Other mean=(0.5, 0.5, 0.5) and std=(0.5, 0.5, 0.5)
        normalize = transforms.Normalize(mean=(0.5,),
                                         std=(0.5,))

        # 1) transforms PIL image in range [0,255] to [0,1],
        # 2) transposes [H, W, C] to [C, H, W]
        if normalize:
            transforms_list += [transforms.ToTensor(), normalize]
        else:
            transforms_list += [transforms.ToTensor()]

        # Applies a random rotation augmentation
        if rotate:
            transforms_list += [transforms.RandomRotation(90)]

        transform = transforms.Compose([t for t in transforms_list if t])
        return transform

    def __len__(self):
        """Required: specify dataset length for dataloader"""
        return len(self.file_paths)

    def __getitem__(self, index):
        """Required: specify what each iteration in dataloader yields"""
        img = Image.open(self.file_paths[index])
        img = self.transform(img)
        return img
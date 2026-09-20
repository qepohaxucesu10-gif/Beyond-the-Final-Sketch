import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Fallback configuration to prevent import errors if config.py is missing
try:
    from config import config
except ImportError:
    class Config:
        crop_size = 256  # Default image size for the model input


    config = Config()


class face2sketch(Dataset):
    """
    Custom PyTorch Dataset class for the Face-to-Sketch synthesis task.
    Handles loading and preprocessing of images from the specified directory.
    """

    def __init__(self, image_path, sort):
        """
        Initialize the dataset.

        Args:
            image_path (str): Root directory of the dataset (e.g., './data/').
            sort (str): Specific sub-folder name (e.g., 'trainA', 'trainB7').
        """
        self.path = os.path.join(image_path, sort)

        # Filter valid image formats to prevent interruption from hidden/system files
        self.images = [
            x for x in sorted(os.listdir(self.path))
            if x.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))
        ]

        # Define the image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize(config.crop_size),  # Resize to model's expected input size
            transforms.ToTensor(),  # Convert PIL Image to PyTorch Tensor
            transforms.Normalize(mean=(0.5, 0.5, 0.5),  # Normalize to [-1, 1] for GAN stability
                                 std=(0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        """
        Retrieve and preprocess a single image given its index.
        """
        image_path = os.path.join(self.path, self.images[index])
        try:
            # Convert to RGB to ensure 3-channel input, even if the image is grayscale
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
            return image
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {image_path} | Error: {str(e)}")

    def __len__(self):
        """
        Return the total number of images in this dataset split.
        """
        return len(self.images)


def get_face2sketch_loader(purpose, batch_size):
    """
    Constructs and returns DataLoaders for all generative stages.

    Args:
        purpose (str): 'train' for training set, 'test' for evaluation set.
        batch_size (int): Number of images per batch.

    Returns:
        tuple: (loader_face, loader_sketch1, loader_sketch2, loader_sketch3)
               corresponding to input face, stage 1 (B7), stage 2 (B19), and stage 3 (B25).
    """
    if purpose == 'train':
        # Initialize training datasets for Real Face and 3 intermediate sketch stages
        train_face = face2sketch('./data/', 'trainA')
        train_sketch1 = face2sketch('./data/', 'trainB7')
        train_sketch2 = face2sketch('./data/', 'trainB19')
        train_sketch3 = face2sketch('./data/', 'trainB25')

        # Create DataLoaders
        # drop_last=True is used for sketch targets to ensure consistent batch sizes during training
        trainloader_face = DataLoader(train_face, batch_size=batch_size)
        trainloader_sketch1 = DataLoader(train_sketch1, batch_size=batch_size, drop_last=True)
        trainloader_sketch2 = DataLoader(train_sketch2, batch_size=batch_size, drop_last=True)
        trainloader_sketch3 = DataLoader(train_sketch3, batch_size=batch_size, drop_last=True)

        return trainloader_face, trainloader_sketch1, trainloader_sketch2, trainloader_sketch3

    elif purpose == 'test':
        # Initialize testing datasets for evaluation
        test_face = face2sketch('./data/', 'testA')
        test_sketch1 = face2sketch('./data/', 'testB7')
        test_sketch2 = face2sketch('./data/', 'testB19')
        test_sketch3 = face2sketch('./data/', 'testB25')

        # Create DataLoaders
        testloader_face = DataLoader(test_face, batch_size=batch_size)
        testloader_sketch1 = DataLoader(test_sketch1, batch_size=batch_size, drop_last=True)
        testloader_sketch2 = DataLoader(test_sketch2, batch_size=batch_size, drop_last=True)
        testloader_sketch3 = DataLoader(test_sketch3, batch_size=batch_size, drop_last=True)

        return testloader_face, testloader_sketch1, testloader_sketch2, testloader_sketch3

    else:
        raise NameError("Purpose should be either 'train' or 'test'.")
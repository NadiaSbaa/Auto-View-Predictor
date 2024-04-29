from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import get_image_filepaths


class PredictionCustomDataset(Dataset):
    """
    A custom dataset class for prediction perspective scores of cars.
    """
    def __init__(self, data_path, transform=None):
        """
        Initializes the PredictionCustomDataset.

        Args:
            data_path (str): The path to the directory containing images.
            transform (callable, optional): A function/transform to apply to the images. Default is None.
        """
        self.data = get_image_filepaths(data_path)
        if not transform:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        """
        Returns the total number of images in the dataset.

        Returns:
            int: The number of images in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves an image and its associated file path from the dataset.

        Args:
            idx (int): The index of the image to retrieve.

        Returns:
            Tuple[str, Tensor]: A tuple containing the file path of the image and the transformed image tensor.
        """
        image_filepath = self.data[idx]
        image = Image.open(image_filepath)
        if self.transform:
            image = self.transform(image)
        return image_filepath, image

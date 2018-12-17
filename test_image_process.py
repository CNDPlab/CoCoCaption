import torchvision as tv
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms


image = Image.open(os.path.join('image_folder/cat-3846780__340.jpg')).convert('RGB')

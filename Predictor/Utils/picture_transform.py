import torchvision.transforms as transforms
import ipdb
from PIL import Image


picture_tranform_func = transforms.Compose(
            [
                transforms.Resize((224, 224), Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

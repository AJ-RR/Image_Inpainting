import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from utils import make_mask
from PIL import Image

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.ppm', '.pgm')

def find_images_recursively(root):
    image_paths = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(IMG_EXTENSIONS) and not fname.startswith('.'):
                image_paths.append(os.path.join(dirpath, fname))
    return sorted(image_paths)

def apply_clahe_transform(img):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to a PIL Image.
    Args:
        img (PIL.Image): Input PIL Image.
    Returns:
        PIL.Image: CLAHE enhanced PIL Image.
    """
    # Convert PIL Image to OpenCV format (NumPy array)
    img_np = np.array(img)

    # Convert RGB to LAB color space
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)

    # Split LAB image into L, A, B channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L-channel with original A and B channels
    limg = cv2.merge((cl, a, b))

    # Convert LAB back to RGB
    final_img_np = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    # Convert NumPy array back to PIL Image
    return Image.fromarray(final_img_np)

class InpaintingDataset(Dataset):
    def __init__(self, root, dataset='celeba', image_size=128, mask_size=64, mask_type='mixed', transform=None):
        self.image_size = image_size
        self.mask_size = mask_size
        self.mask_type = mask_type
        if transform:
            transform_list = [transforms.Resize((image_size, image_size)), transform]
        else:
            transform_list = [
                transforms.CenterCrop(min(image_size, image_size)),
                transforms.Resize((image_size, image_size)),
                transforms.Lambda(apply_clahe_transform),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        self.transform = transforms.Compose(transform_list)

        if dataset == 'celeba':
            self.samples = find_images_recursively(root)
            if len(self.samples) == 0:
                raise RuntimeError(f"No images found in {root} (searched recursively). Supported extensions: {IMG_EXTENSIONS}")
            print(f"[INFO] Found {len(self.samples)} images for 'celeba' in {root} (recursively, single class)")
            self.use_flat = True
        elif dataset == 'imagenet':
            self.dataset = datasets.ImageFolder(root=os.path.join(root, 'train'), transform=self.transform)
            self.use_flat = False
        else:
            raise ValueError(f"Unsupported dataset {dataset}")

    def __len__(self):
        if hasattr(self, 'use_flat') and self.use_flat:
            return len(self.samples)
        return len(self.dataset)

    def __getitem__(self, idx):
        if hasattr(self, 'use_flat') and self.use_flat:
            img_path = self.samples[idx]
            img = Image.open(img_path) #.convert('RGB')
            img = self.transform(img)
        else:
            img, _ = self.dataset[idx] if isinstance(self.dataset[idx], tuple) else (self.dataset[idx], None)
        mask = make_mask(self.image_size, self.mask_size, self.mask_type)
        masked_img = img.clone()
        masked_img = masked_img * (1 - mask)
        return masked_img, img, mask

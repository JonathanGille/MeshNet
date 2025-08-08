from PIL import Image
import torch
import torchvision.transforms as transforms


def synthetic_transform(image):
    synth_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ])
    
    return synth_transform(image)
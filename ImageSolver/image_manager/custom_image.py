from PIL import Image
import torch
import torchvision.transforms as transforms

class CustomImage():
    def __init__(self, path, name, load_image=True):
        if load_image:
            self.pil_image = Image.open(path)
            self.image = self.transform(self.pil_image)
        else:
            self.image = None
            self.pil_image = None

        self.name = name
        self.path = path

    def transform(self, image):
        # bilder transformieren
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image = image.convert("RGB")
        return transform(image).unsqueeze(0).to(device)  # Batch-Dimension + GPU


import os
from torch.utils.data import Dataset
from ImageSolver import get_image

type_to_index_map = {
    "IfcAirTerminal": 1,
    "IfcBeam": 2,
    "IfcCableCarrierFitting": 3,
    "IfcCableCarrierSegment": 4,
    "IfcDoor": 5,
    "IfcDuctFitting": 6,
    "IfcDuctSegment": 7,
    "IfcFurniture": 8,
    "IfcLamp": 9,
    "IfcOutlet": 10,
    "IfcPipeFitting": 11,
    "IfcPipeSegment": 12,
    "IfcPlate": 13,
    "IfcRailing": 14,
    "IfcSanitaryTerminal": 15,
    "IfcSlab": 16,
    "IfcSpaceHeater": 17,
    "IfcStair": 18,
    "IfcValve": 19,
    "IfcWall": 20
}

class ImagesDataset(Dataset):
    def __init__(self, part='train'):
        self.data = []

        root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(root_path, 'dataset', 'IFCNetCorePng')
        for classification in os.listdir(data_dir):
            label = type_to_index_map[classification]
            part_root = os.path.join(data_dir, classification, part)
            for img in os.listdir(part_root):
                self.data.append((os.path.join(part_root, img), label))

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx][0]
        img = get_image(img_path).image # return image attribute of c_image object
        label = self.data[idx][1]
        return img, label  
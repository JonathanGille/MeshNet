import numpy as np
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as data
from MeshSolver.config import get_emb_config, get_pcim_config
from MeshSolver.data import ModelNet40, EmbSet, IFCNetCore
from MeshSolver.models import MeshEmb
from utils.retrival import append_feature, calculate_map

from ImageSolver import load_custom_model, get_image, ImagesDataset

cfg = get_pcim_config()
os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_devices']

obj_set = IFCNetCore(cfg=cfg['dataset'], part='train')
#print('Length of obj-dataset',len(obj_set))
obj_loader = data.DataLoader(obj_set, batch_size=cfg['batch_size'], num_workers=4, shuffle=True, pin_memory=False)

img_set = ImagesDataset(part='train')
#print('Length of img-dataset:',len(img_set))
img_loader = data.DataLoader(img_set, batch_size=128, num_workers=4, shuffle=True, pin_memory=False)

def load_obj_embs(model):
    correct_num = 0
    ft_all, lbl_all = None, None

    n = 0
    with torch.no_grad():
        for i, (centers, corners, normals, neighbor_index, targets) in enumerate(obj_loader):
            centers = centers.cuda()
            corners = corners.cuda()
            normals = normals.cuda()
            neighbor_index = neighbor_index.cuda()
            #targets = targets.cuda()

            feas = model(centers, corners, normals, neighbor_index)
            print(feas.size())
            n += 1
    print(n)

def load_img_embs(model):
    n=0
    for i, (img, label) in enumerate(img_loader):
        embedding = model.generate_embedding(img)
        print(embedding.size())
        n+=1
    print(n)


if __name__ == '__main__':

    model = MeshEmb(cfg=cfg['MeshNet'])
    model.cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(cfg['load_model']))
    model.eval()

    #load_obj_embs(model)

    # Load the base model with timm
    c_model = load_custom_model(base_model_name='convnext_base')
    load_img_embs(c_model)
    
    # # c_image of the input image
    # img_path = os.path.join("dataset", "IFCNetCorePng", "IfcAirTerminal", "train", "0a59ac7a7fa04a9dafe8c425bab881f7.0.png")
    # input_image = get_image(img_path)
    # input_image.embedding = c_model.generate_embedding(input_image.image)
    # print(input_image.embedding.size())
